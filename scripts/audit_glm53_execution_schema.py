#!/usr/bin/env python3
"""Audit every GLM-5.3 text tensor against its executable PyTree mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jaxsft.lora import format_parameter_path
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    Glm53TextConfig,
    SafetensorsIndex,
    SafetensorsTensorRange,
    checkpoint_text_tensor_specs,
)
from scripts.audit_glm53_checkpoint_headers import _category, _read_header, _scale_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_digest(rows: list[dict[str, Any]]) -> str:
    encoded = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_all_headers(
    index: SafetensorsIndex,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, SafetensorsTensorRange], int]:
    headers: dict[str, SafetensorsTensorRange] = {}
    network_bytes = 0
    indexed = dict(index.tensor_files)
    for shard in index.shard_names:
        header, result = _read_header(shard, timeout_seconds=timeout_seconds)
        network_bytes += result["header_network_bytes"]
        expected_names = {name for name, filename in index.tensor_files if filename == shard}
        actual_names = {tensor.name for tensor in header.tensors}
        if actual_names != expected_names:
            raise ValueError(f"{shard} header names do not exactly match the pinned index")
        for tensor in header.tensors:
            if tensor.name in headers or indexed[tensor.name] != shard:
                raise ValueError(f"duplicate or mis-sharded tensor {tensor.name!r}")
            headers[tensor.name] = tensor
    if set(headers) != set(indexed):
        raise ValueError("combined headers do not exactly cover the pinned index")
    return headers, network_bytes


def audit_execution_schema(
    config: Glm53TextConfig,
    index: SafetensorsIndex,
    headers: dict[str, SafetensorsTensorRange],
    *,
    source_revision: str,
    header_network_bytes: int,
) -> dict[str, Any]:
    specs = checkpoint_text_tensor_specs(config)
    specs_by_name = {spec.source_name: spec for spec in specs}
    logical_names = set(specs_by_name)
    actual_text_names = {name for name in headers if _category(name) == "text"}
    scale_names = {_scale_name(name) for name in logical_names if _scale_name(name) in headers}
    expected_text_names = logical_names | scale_names
    if actual_text_names != expected_text_names:
        missing = sorted(expected_text_names - actual_text_names)
        unexpected = sorted(actual_text_names - expected_text_names)
        raise ValueError(
            "executable text schema does not exactly cover checkpoint text tensors: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    transform_counts = Counter()
    dtype_counts = Counter()
    payload_by_role = Counter()
    target_members: dict[tuple[str | int, ...], list[Any]] = defaultdict(list)
    mapping_rows: list[dict[str, Any]] = []
    quantized_target_paths: set[tuple[str | int, ...]] = set()
    scale_payload_bytes = 0
    for spec in specs:
        tensor = headers[spec.source_name]
        if tensor.shape != spec.source_shape:
            raise ValueError(
                f"source shape for {spec.source_name!r} is {tensor.shape}, expected {spec.source_shape}"
            )
        scale_name = _scale_name(spec.source_name)
        scale = headers.get(scale_name)
        if tensor.dtype == "F8_E4M3":
            if scale is None:
                raise ValueError(f"FP8 executable tensor {spec.source_name!r} has no scale grid")
            if len(tensor.shape) != 2 or any(size % 128 for size in tensor.shape):
                raise ValueError(f"FP8 executable tensor {spec.source_name!r} is not a 128x128 matrix")
            expected_scale_shape = tuple(size // 128 for size in tensor.shape)
            if scale.dtype != "F32" or scale.shape != expected_scale_shape:
                raise ValueError(
                    f"FP8 scale for {spec.source_name!r} is {scale.dtype}{scale.shape}, "
                    f"expected F32{expected_scale_shape}"
                )
            role = "fp8_expert_pack" if spec.transform == "expert_transpose" else "fp8_linear"
            quantized_target_paths.add(spec.target_path)
            scale_payload_bytes += scale.nbytes
        else:
            if scale is not None:
                raise ValueError(f"non-FP8 executable tensor {spec.source_name!r} has a scale grid")
            if spec.transform == "expert_transpose":
                raise ValueError(f"expert tensor {spec.source_name!r} is not block-FP8")
            role = {
                "identity": "direct_array",
                "transpose": "dense_transpose",
                "squeeze_conv": "depthwise_conv",
            }[spec.transform]
        transform_counts[role] += 1
        dtype_counts[(role, tensor.dtype)] += 1
        payload_by_role[role] += tensor.nbytes + (0 if scale is None else scale.nbytes)
        target_members[spec.target_path].append(spec)
        mapping_rows.append(
            {
                "dtype": tensor.dtype,
                "pack_index": spec.pack_index,
                "role": role,
                "scale": None if scale is None else scale_name,
                "shape": tensor.shape,
                "source": spec.source_name,
                "target": format_parameter_path(spec.target_path),
                "transform": spec.transform,
            }
        )

    packed_groups: list[dict[str, Any]] = []
    maximum_packed_device_buffer = 0
    maximum_packed_target = ""
    for target_path, members in sorted(target_members.items(), key=lambda item: item[0]):
        if members[0].transform != "expert_transpose":
            if len(members) != 1:
                raise ValueError(f"non-expert target {format_parameter_path(target_path)!r} has multiple sources")
            continue
        indices = sorted(member.pack_index for member in members)
        if indices != list(range(config.n_routed_experts)):
            raise ValueError(f"expert pack {format_parameter_path(target_path)!r} has incomplete indices")
        if len({member.source_shape for member in members}) != 1:
            raise ValueError(f"expert pack {format_parameter_path(target_path)!r} has mixed source shapes")
        source_shape = members[0].source_shape
        source_bytes = sum(headers[member.source_name].nbytes for member in members)
        per_device_bytes = source_bytes // 16
        if source_bytes % 16:
            raise ValueError(f"expert pack {format_parameter_path(target_path)!r} is not 16-way shardable")
        if per_device_bytes > maximum_packed_device_buffer:
            maximum_packed_device_buffer = per_device_bytes
            maximum_packed_target = format_parameter_path(target_path)
        packed_groups.append(
            {
                "target": format_parameter_path(target_path),
                "expert_count": len(members),
                "source_matrix_shape": source_shape,
                "packed_source_shape": [len(members), *source_shape],
                "source_bytes": source_bytes,
                "per_device_bytes": per_device_bytes,
            }
        )

    text_payload_bytes = sum(headers[name].nbytes for name in actual_text_names)
    if text_payload_bytes != 319_706_118_392:
        raise ValueError(f"text payload byte count drifted to {text_payload_bytes}")
    if len(specs) != 37_534 or len(scale_names) != 36_467 or len(actual_text_names) != 74_001:
        raise ValueError("logical/scale/text tensor counts drifted from the G4 audit")
    if len(packed_groups) != 42 * 3:
        raise ValueError("expert tensors did not form exactly three packs in each sparse MLP layer")
    if len(target_members) != 1_372 or len(quantized_target_paths) != 305:
        raise ValueError("executable target or quantized target count drifted")
    if sum(payload_by_role.values()) != text_payload_bytes:
        raise ValueError("execution roles do not account for every text payload byte")

    dtype_summary = {
        f"{role}:{dtype}": count for (role, dtype), count in sorted(dtype_counts.items())
    }
    mapping_rows.sort(key=lambda row: row["source"])
    packed_projection_counts = Counter(group["target"].rsplit("_", 1)[-1] for group in packed_groups)
    return {
        "schema_version": 1,
        "test": "glm53_g5_execution_schema_audit",
        "source_revision": source_revision,
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "config_sha256": OFFICIAL_CHECKPOINT.config_sha256,
            "index_sha256": index.sha256,
            "text_payload_bytes": text_payload_bytes,
        },
        "coverage": {
            "logical_tensor_count": len(specs),
            "scale_tensor_count": len(scale_names),
            "text_tensor_count": len(actual_text_names),
            "all_text_tensors_mapped_exactly_once": True,
            "mapping_sha256": _mapping_digest(mapping_rows),
            "header_network_bytes": header_network_bytes,
        },
        "execution": {
            "target_group_count": len(target_members),
            "quantized_target_group_count": len(quantized_target_paths),
            "role_counts": dict(sorted(transform_counts.items())),
            "role_dtype_counts": dtype_summary,
            "payload_bytes_by_role": dict(sorted(payload_by_role.items())),
            "scale_payload_bytes": scale_payload_bytes,
        },
        "expert_packing": {
            "group_count": len(packed_groups),
            "all_groups_cover_experts_exactly_once": True,
            "group_counts_by_projection": dict(sorted(packed_projection_counts.items())),
            "groups_sha256": _mapping_digest(packed_groups),
            "source_bytes": sum(group["source_bytes"] for group in packed_groups),
            "per_device_bytes": sum(group["per_device_bytes"] for group in packed_groups),
            "maximum_device_staging_buffer_bytes": maximum_packed_device_buffer,
            "maximum_device_staging_target": maximum_packed_target,
            "sample_groups": packed_groups[:3],
        },
        "gate": {
            "g5a_execution_schema": "passed",
            "full_model_runnable": False,
            "remaining_blockers": [
                "Block-FP8 wrappers and expert-pack contractions have not passed reduced parity.",
                "The complete mapped PyTree has not been range-loaded on v4-32.",
                "Whole-model compilation, HBM, and forward numerics remain unmeasured.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if _sha256(args.config) != OFFICIAL_CHECKPOINT.config_sha256:
        raise ValueError("config SHA-256 does not match the pinned checkpoint")
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    index.verify(OFFICIAL_CHECKPOINT)
    started = time.monotonic()
    headers, network_bytes = _read_all_headers(index, timeout_seconds=args.timeout_seconds)
    result = audit_execution_schema(
        config,
        index,
        headers,
        source_revision=args.source_revision,
        header_network_bytes=network_bytes,
    )
    result["elapsed_seconds"] = time.monotonic() - started
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
