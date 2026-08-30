#!/usr/bin/env python3
"""Audit every pinned GLM-5.3 shard header without reading weight payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from jaxsft.checkpoint import StrictHTTPRangeReader
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    SafetensorsIndex,
    SafetensorsShardHeader,
    SafetensorsTensorRange,
)


_LAYER_NAME = re.compile(r"model\.language_model\.layers\.([0-9]+)\.")
_DTYPE_WIDTH = {"F8_E4M3": 1, "BF16": 2, "F32": 4}
_TEXT_ROOTS = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
}


def _source_url(shard: str) -> str:
    return (
        f"https://huggingface.co/{OFFICIAL_CHECKPOINT.repo_id}/resolve/"
        f"{OFFICIAL_CHECKPOINT.revision}/{shard}"
    )


def _sha256_names(names: list[str]) -> str:
    payload = "".join(name + "\n" for name in sorted(names)).encode()
    return hashlib.sha256(payload).hexdigest()


def _category(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision_excluded"
    match = _LAYER_NAME.match(name)
    if match:
        layer = int(match.group(1))
        if 0 <= layer < 45:
            return "text"
        if layer == 45:
            return "mtp_excluded"
        raise ValueError(f"unexpected language-model layer index in {name!r}")
    if name in _TEXT_ROOTS:
        return "text"
    raise ValueError(f"checkpoint tensor has no explicit inclusion/exclusion rule: {name!r}")


def _scale_name(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".weight_scale_inv"
    return weight_name + "_scale_inv"


def _placement_policy(name: str, tensor: SafetensorsTensorRange) -> str:
    if name.endswith("weight_scale_inv"):
        return "replicated_scale_metadata"
    if tensor.nbytes <= 1024 * 1024 or len(tensor.shape) < 2:
        return "replicated_small"
    if tensor.shape[0] % 16 == 0:
        return "axis0_model_16"
    return "unsupported"


def _read_header(shard: str, *, timeout_seconds: float) -> tuple[SafetensorsShardHeader, dict[str, Any]]:
    reader = StrictHTTPRangeReader(
        _source_url(shard),
        timeout_seconds=timeout_seconds,
        maximum_request_bytes=2 * 1024 * 1024,
    )
    length_bytes = reader.read(0, 7)
    header_length = int.from_bytes(length_bytes, "little", signed=False)
    payload = reader.read(8, 7 + header_length)
    header = SafetensorsShardHeader.from_json_bytes(payload, header_length=header_length)
    payload_end = max(tensor.relative_end for tensor in header.tensors)
    expected_file_bytes = header.data_section_start + payload_end
    if reader.total_size_bytes != expected_file_bytes:
        raise ValueError(
            f"{shard} HTTP size {reader.total_size_bytes} does not match header-derived {expected_file_bytes}"
        )
    return header, {
        "name": shard,
        "header_length": header_length,
        "header_sha256": hashlib.sha256(payload).hexdigest(),
        "tensor_count": len(header.tensors),
        "file_bytes": reader.total_size_bytes,
        "header_network_bytes": reader.bytes_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    started = time.monotonic()
    index = SafetensorsIndex.from_path(args.index)
    index.verify(OFFICIAL_CHECKPOINT)
    indexed = dict(index.tensor_files)
    headers: dict[str, SafetensorsTensorRange] = {}
    shard_results: list[dict[str, Any]] = []
    for shard in index.shard_names:
        header, shard_result = _read_header(shard, timeout_seconds=args.timeout_seconds)
        expected_names = {name for name, filename in index.tensor_files if filename == shard}
        header_names = {tensor.name for tensor in header.tensors}
        if header_names != expected_names:
            missing = sorted(expected_names - header_names)
            unexpected = sorted(header_names - expected_names)
            raise ValueError(
                f"{shard} header/index mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        for tensor in header.tensors:
            if tensor.name in headers:
                raise ValueError(f"tensor {tensor.name!r} occurs in more than one shard header")
            if indexed[tensor.name] != shard:
                raise ValueError(f"tensor {tensor.name!r} mapped to the wrong source shard")
            headers[tensor.name] = tensor
        shard_results.append(shard_result)
    if set(headers) != set(indexed):
        raise ValueError("combined shard headers do not exactly cover the pinned index")

    element_counts = Counter()
    payload_bytes = Counter()
    category_names: dict[str, list[str]] = {
        "text": [],
        "mtp_excluded": [],
        "vision_excluded": [],
    }
    category_payload = Counter()
    placement_counts = Counter()
    placement_payload = Counter()
    placement_per_device_bytes = Counter()
    maximum_range_bytes = 0
    maximum_range_tensor: dict[str, Any] | None = None
    unsupported: list[str] = []
    for name, tensor in headers.items():
        elements = math.prod(tensor.shape)
        element_counts[tensor.dtype] += elements
        payload_bytes[tensor.dtype] += tensor.nbytes
        category = _category(name)
        category_names[category].append(name)
        category_payload[category] += tensor.nbytes
        if category != "text":
            continue
        policy = _placement_policy(name, tensor)
        placement_counts[policy] += 1
        placement_payload[policy] += tensor.nbytes
        if policy == "axis0_model_16":
            per_device = tensor.nbytes // 16
        elif policy.startswith("replicated"):
            per_device = tensor.nbytes
        else:
            unsupported.append(name)
            continue
        placement_per_device_bytes[policy] += per_device
        if per_device > maximum_range_bytes:
            maximum_range_bytes = per_device
            maximum_range_tensor = {
                "name": name,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "source_bytes": tensor.nbytes,
                "device_range_bytes": per_device,
                "policy": policy,
            }

    expected_counts = dict(OFFICIAL_CHECKPOINT.serialized_element_counts_by_dtype)
    if dict(element_counts) != expected_counts:
        raise ValueError(f"dtype element counts {dict(element_counts)} do not match {expected_counts}")
    if sum(payload_bytes.values()) != OFFICIAL_CHECKPOINT.total_size_bytes:
        raise ValueError("aggregate tensor payload bytes do not match the pinned checkpoint contract")

    fp8_names = {name for name, tensor in headers.items() if tensor.dtype == "F8_E4M3"}
    scale_names = {name for name in headers if name.endswith("weight_scale_inv")}
    paired_scales: set[str] = set()
    for weight_name in fp8_names:
        weight = headers[weight_name]
        if len(weight.shape) < 2:
            raise ValueError(f"FP8 tensor {weight_name!r} is not matrix-like")
        scale_name = _scale_name(weight_name)
        if scale_name not in headers:
            raise ValueError(f"FP8 tensor {weight_name!r} has no scale companion {scale_name!r}")
        scale = headers[scale_name]
        if scale.dtype != "F32":
            raise ValueError(f"FP8 scale {scale_name!r} has dtype {scale.dtype}, expected F32")
        rows, columns = weight.shape[-2:]
        if rows % 128 or columns % 128:
            raise ValueError(f"FP8 tensor {weight_name!r} is not divisible into 128x128 blocks")
        expected_shape = (*weight.shape[:-2], rows // 128, columns // 128)
        if scale.shape != expected_shape:
            raise ValueError(
                f"FP8 scale {scale_name!r} has shape {scale.shape}, expected {expected_shape}"
            )
        paired_scales.add(scale_name)
    orphan_scales = sorted(scale_names - paired_scales)
    if orphan_scales:
        raise ValueError(f"orphan FP8 scale tensors: {orphan_scales[:20]}")
    if unsupported:
        raise ValueError(f"text tensors have no bounded placement policy: {unsupported[:20]}")

    category_summary = {
        name: {
            "tensor_count": len(names),
            "payload_bytes": category_payload[name],
            "names_sha256": _sha256_names(names),
        }
        for name, names in category_names.items()
    }
    per_device_bytes = sum(placement_per_device_bytes.values())
    if maximum_range_tensor is None:
        raise ValueError("placement audit produced no loadable text tensors")
    per_host_streamed_bytes = (
        placement_payload["axis0_model_16"] // 4
        + placement_payload["replicated_scale_metadata"]
        + placement_payload["replicated_small"]
    )
    result = {
        "schema_version": 1,
        "test": "glm53_all_shard_header_and_placement_audit",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "index_sha256": index.sha256,
            "tensor_count": len(headers),
            "shard_count": len(shard_results),
            "payload_bytes": sum(payload_bytes.values()),
            "element_counts_by_dtype": dict(sorted(element_counts.items())),
            "payload_bytes_by_dtype": dict(sorted(payload_bytes.items())),
        },
        "header_audit": {
            "all_index_tensors_covered_once": True,
            "all_file_sizes_match_header_offsets": True,
            "network_bytes": sum(shard["header_network_bytes"] for shard in shard_results),
            "maximum_header_length": max(shard["header_length"] for shard in shard_results),
            "source_file_bytes_including_headers": sum(shard["file_bytes"] for shard in shard_results),
            "shards": shard_results,
        },
        "scope_audit": {
            **category_summary,
            "unknown_tensor_count": 0,
        },
        "fp8_pair_audit": {
            "fp8_weight_count": len(fp8_names),
            "scale_count": len(scale_names),
            "all_fp8_weights_have_exact_f32_scale_grids": True,
            "orphan_scale_count": 0,
        },
        "placement_plan": {
            "mesh_axis": "model",
            "mesh_size": 16,
            "large_tensor_policy": "contiguous source axis 0 -> PartitionSpec('model', ...)",
            "scale_policy": "replicate small F32 block metadata",
            "small_tensor_replication_limit_bytes": 1024 * 1024,
            "counts_by_policy": dict(sorted(placement_counts.items())),
            "source_payload_bytes_by_policy": dict(sorted(placement_payload.items())),
            "per_device_bytes_by_policy": dict(sorted(placement_per_device_bytes.items())),
            "estimated_text_base_bytes_per_device": per_device_bytes,
            "maximum_single_device_range_bytes": maximum_range_bytes,
            "maximum_single_device_range_tensor": maximum_range_tensor,
            "estimated_streamed_payload_bytes_per_host": per_host_streamed_bytes,
            "unsupported_tensor_count": 0,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
