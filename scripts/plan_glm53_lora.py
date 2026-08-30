#!/usr/bin/env python3
"""Inspect the pinned GLM-5.3-Flash metadata and emit a fail-closed v4-32 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jaxsft.models.glm5_3_flash import (
    GIB,
    OFFICIAL_CHECKPOINT,
    Glm53TextConfig,
    SafetensorsIndex,
    v4_32_lora_preflight,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_gib_fields(plan: dict) -> dict:
    result = dict(plan)
    result["hbm_per_device_gib"] = result["hbm_per_device_bytes"] / GIB
    result["used_per_device_gib"] = result["used_per_device_bytes"] / GIB
    result["free_per_device_gib"] = result["free_per_device_bytes"] / GIB
    result["staging_per_host_gib"] = result["staging_per_host_bytes"] / GIB
    result["memory"] = [
        {
            **line,
            "aggregate_gib": line["aggregate_bytes"] / GIB,
            "per_device_gib": line["per_device_bytes"] / GIB,
        }
        for line in result["memory"]
    ]
    return result


def validate_kernel_evidence(path: Path) -> dict:
    """Validate tracked G3 evidence rather than accepting a boolean override."""

    payload_bytes = path.read_bytes()
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in kernel evidence")
            result[key] = value
        return result

    payload = json.loads(payload_bytes, object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError("kernel evidence must be a JSON object")
    mismatches: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    model = payload.get("model", {})
    cpu = payload.get("cpu_oracle", {})
    tpu = payload.get("tpu", {})
    topology = tpu.get("topology", {})
    hlo_shapes = tpu.get("tiled_full_weight_hlo_shape_mentions", {})
    memory = tpu.get("compiler_memory_bytes_per_process", {})
    gate = payload.get("gate", {})
    source_revision = payload.get("source_revision")
    require(payload.get("schema_version") == 1, "schema_version must be 1")
    require(
        payload.get("test") == "glm53_real_block_fp8_tiled_contraction",
        "unexpected evidence test identity",
    )
    require(
        isinstance(source_revision, str)
        and len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source_revision must be a full lowercase Git hash",
    )
    require(model.get("repo_id") == OFFICIAL_CHECKPOINT.repo_id, "model repo_id mismatch")
    require(model.get("revision") == OFFICIAL_CHECKPOINT.revision, "model revision mismatch")
    require(model.get("weight_shape") == [1536, 4096], "probe weight shape mismatch")
    require(model.get("scale_shape") == [12, 32], "probe scale shape mismatch")
    require(model.get("block_shape") == [128, 128], "probe block shape mismatch")
    require(
        model.get("weight_http_range_inclusive") == [2_941_704_672, 2_947_996_127],
        "probe weight HTTP range mismatch",
    )
    require(
        model.get("scale_http_range_inclusive") == [883_976, 885_511],
        "probe scale HTTP range mismatch",
    )
    require(model.get("weight_bytes") == 6_291_456, "probe weight byte count mismatch")
    require(model.get("scale_bytes") == 1_536, "probe scale byte count mismatch")
    require(
        model.get("weight_sha256")
        == "d79be6a957e1c23680665a68e4bbc9ffaf71a01bb7dc540e40140c6af9a3b3bc",
        "probe weight hash mismatch",
    )
    require(
        model.get("scale_sha256")
        == "165bb5ed26c4a904ba915d5bd22657560e019041ccb0f13868ddd811e3c429dd",
        "probe scale hash mismatch",
    )
    require(model.get("http_status") == 206, "probe was not an HTTP partial-content read")
    require(model.get("full_source_shard_downloaded") is False, "full source shard was downloaded")
    require(topology.get("accelerator_type") == "v4-32", "accelerator topology mismatch")
    require(topology.get("process_count") == 4, "TPU process count mismatch")
    require(topology.get("local_device_count") == 4, "TPU local device count mismatch")
    require(topology.get("global_device_count") == 16, "TPU global device count mismatch")
    require(tpu.get("precision") == "HIGHEST", "TPU contraction precision was not HIGHEST")
    require(tpu.get("compute_dtype") == "bfloat16", "TPU compute dtype was not bfloat16")
    require(tpu.get("all_process_outputs_finite") is True, "a TPU output was non-finite")
    require(tpu.get("all_process_hlo_sha256_equal") is True, "TPU HLO differed across processes")
    require(tpu.get("clean_distributed_shutdown") is True, "distributed shutdown was not clean")
    for dtype in ("bfloat16", "float32", "float8_e4m3fn"):
        require(hlo_shapes.get(dtype) == 0, f"optimized HLO contains a full {dtype} weight")
    temporary_bytes = memory.get("temporaries")
    require(
        isinstance(temporary_bytes, int) and 0 <= temporary_bytes < 1536 * 4096 * 2,
        "TPU temporary memory is not below one full BF16 probe weight",
    )
    require(
        cpu.get("float32_tiled_vs_transformers", {}).get("relative_l2", float("inf")) < 1e-5,
        "float32 JAX/Transformers error exceeds the G3 threshold",
    )
    require(
        tpu.get("tpu_bfloat16_vs_cpu_bfloat16", {}).get("relative_l2", float("inf")) < 1e-5,
        "TPU/CPU BF16 error exceeds the G3 threshold",
    )
    require(gate.get("g3_block_fp8_primitive") == "passed", "G3 is not marked passed")
    require(gate.get("full_model_runnable") is False, "G3 evidence must not claim a runnable full model")
    if mismatches:
        raise ValueError("invalid GLM-5.3 kernel evidence: " + "; ".join(mismatches))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "source_revision": source_revision,
        "test": payload["test"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="pinned config.json")
    parser.add_argument("--index", type=Path, required=True, help="pinned model.safetensors.index.json")
    parser.add_argument("--rank", type=int, default=8, help="attention-only LoRA rank")
    parser.add_argument(
        "--execution-weight-format",
        choices=("fp8_blockwise", "bfloat16"),
        default="fp8_blockwise",
    )
    parser.add_argument(
        "--kernel-evidence",
        type=Path,
        help="tracked G3 JSON evidence; validated fail-closed before marking the kernel proven",
    )
    args = parser.parse_args()

    config_hash = _sha256(args.config)
    if config_hash != OFFICIAL_CHECKPOINT.config_sha256:
        raise ValueError(
            f"config hash {config_hash} does not match pinned {OFFICIAL_CHECKPOINT.config_sha256}"
        )
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    index.verify(OFFICIAL_CHECKPOINT)
    kernel_evidence = validate_kernel_evidence(args.kernel_evidence) if args.kernel_evidence else None
    plan = v4_32_lora_preflight(
        config,
        index,
        rank=args.rank,
        execution_weight_format=args.execution_weight_format,
        executable_kernel_proven=kernel_evidence is not None,
    )
    payload = {
        "schema_version": 1,
        "experiment": "glm53_flash_attention_lora_v4_32",
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "config_sha256": config_hash,
            "index_sha256": index.sha256,
            "logical_parameter_count": OFFICIAL_CHECKPOINT.logical_parameter_count,
            "checkpoint_bytes": OFFICIAL_CHECKPOINT.total_size_bytes,
            "tensor_count": index.tensor_count,
            "source_shard_count": index.shard_count,
        },
        "preflight": _with_gib_fields(plan.to_dict()),
        "evidence": {"kernel": kernel_evidence, "direct_loader": None},
        "warning": (
            "G3 proves one real block-FP8 contraction, not the full model; runnable remains false "
            "until the direct-to-final-shard loader and whole-model HBM gates pass"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
