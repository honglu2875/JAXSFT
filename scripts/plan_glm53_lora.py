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
    args = parser.parse_args()

    config_hash = _sha256(args.config)
    if config_hash != OFFICIAL_CHECKPOINT.config_sha256:
        raise ValueError(
            f"config hash {config_hash} does not match pinned {OFFICIAL_CHECKPOINT.config_sha256}"
        )
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    index.verify(OFFICIAL_CHECKPOINT)
    plan = v4_32_lora_preflight(
        config,
        index,
        rank=args.rank,
        execution_weight_format=args.execution_weight_format,
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
        "warning": (
            "static_fit is only a byte lower bound; runnable remains false until the TPU-v4 "
            "block-FP8 kernel and direct loader gates produce measured evidence"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
