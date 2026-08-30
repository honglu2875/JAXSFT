#!/usr/bin/env python3
"""Inspect the pinned GLM-5.3-Flash metadata and emit a fail-closed v4-32 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _load_unique_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload_bytes = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    payload = json.loads(payload_bytes, object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, payload_bytes


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

    payload, payload_bytes = _load_unique_json(path, label="kernel evidence")
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


def validate_loader_evidence(path: Path) -> dict:
    """Validate tracked G4 evidence and its separately committed header audit."""

    payload, payload_bytes = _load_unique_json(path, label="loader evidence")
    mismatches: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    source_revision = payload.get("source_revision")
    require(payload.get("schema_version") == 1, "schema_version must be 1")
    require(
        payload.get("test") == "glm53_g4_direct_sharded_loader_acceptance",
        "unexpected evidence test identity",
    )
    require(
        isinstance(source_revision, str)
        and len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source_revision must be a full lowercase Git hash",
    )

    header = payload.get("header_audit", {})
    header_file = header.get("file")
    require(
        isinstance(header_file, str) and Path(header_file).name == header_file,
        "header audit reference must be a sibling basename",
    )
    header_path = path.parent / header_file if isinstance(header_file, str) else path
    header_payload: dict[str, Any] = {}
    header_bytes = b""
    if header_path != path and header_path.is_file():
        header_payload, header_bytes = _load_unique_json(header_path, label="checkpoint header audit")
    else:
        mismatches.append("referenced checkpoint header audit is missing")
    require(
        bool(header_bytes) and hashlib.sha256(header_bytes).hexdigest() == header.get("sha256"),
        "checkpoint header audit SHA-256 mismatch",
    )
    require(header.get("tensor_count") == 76_108, "header tensor count mismatch")
    require(header.get("shard_count") == 62, "header shard count mismatch")
    require(header.get("header_network_bytes") == 10_684_096, "header range-read byte count mismatch")
    require(header.get("fp8_weight_scale_pairs") == 37_338, "FP8 weight/scale pair count mismatch")

    require(
        header_payload.get("test") == "glm53_all_shard_header_and_placement_audit",
        "unexpected checkpoint header audit identity",
    )
    require(
        header_payload.get("source_revision") == source_revision,
        "checkpoint header audit source revision mismatch",
    )
    header_model = header_payload.get("model", {})
    require(header_model.get("repo_id") == OFFICIAL_CHECKPOINT.repo_id, "header model repo mismatch")
    require(header_model.get("revision") == OFFICIAL_CHECKPOINT.revision, "header model revision mismatch")
    require(
        header_model.get("index_sha256") == OFFICIAL_CHECKPOINT.index_sha256,
        "header index SHA-256 mismatch",
    )
    require(
        header_model.get("payload_bytes") == OFFICIAL_CHECKPOINT.total_size_bytes,
        "header payload byte count mismatch",
    )
    require(
        header_model.get("element_counts_by_dtype")
        == dict(OFFICIAL_CHECKPOINT.serialized_element_counts_by_dtype),
        "header dtype element counts mismatch",
    )
    require(
        header_payload.get("header_audit", {}).get("all_index_tensors_covered_once") is True,
        "header/index coverage is incomplete",
    )
    require(
        header_payload.get("fp8_pair_audit", {}).get(
            "all_fp8_weights_have_exact_f32_scale_grids"
        )
        is True,
        "header FP8 scale pairing is incomplete",
    )

    placement = payload.get("placement_plan", {})
    require(
        placement.get("estimated_text_base_bytes_per_device") == 20_234_287_352,
        "text base placement byte count mismatch",
    )
    require(
        placement.get("maximum_single_device_range_bytes") == 79_298_560,
        "maximum host staging range mismatch",
    )
    require(
        placement.get("estimated_streamed_payload_bytes_per_host") == 80_128_653_560,
        "per-host streamed payload byte count mismatch",
    )
    require(placement.get("unsupported_tensor_count") == 0, "placement contains unsupported tensors")

    sample = payload.get("sample_loader", {})
    require(sample.get("process_count") == 4, "loader process count mismatch")
    require(sample.get("global_device_count") == 16, "loader device count mismatch")
    require(sample.get("device_range_count") == 16, "loader device-range count mismatch")
    require(
        sample.get("global_fingerprint_uint32") == [1_028_930_362, 72, 2_258_651_919, 1_881_823_194],
        "loader global TPU fingerprint mismatch",
    )
    require(
        sample.get("weight_full_sha256")
        == "d79be6a957e1c23680665a68e4bbc9ffaf71a01bb7dc540e40140c6af9a3b3bc",
        "loader source weight SHA-256 mismatch",
    )
    require(
        sample.get("scale_sha256")
        == "165bb5ed26c4a904ba915d5bd22657560e019041ccb0f13868ddd811e3c429dd",
        "loader scale SHA-256 mismatch",
    )
    require(
        sample.get("weight_payload_bytes_downloaded_across_hosts") == 6_291_456,
        "loader did not download the weight payload exactly once across hosts",
    )
    require(sample.get("largest_http_range_bytes") == 393_216, "loader HTTP range bound mismatch")
    require(
        isinstance(sample.get("maximum_process_vmhwm_bytes"), int)
        and 0 < sample["maximum_process_vmhwm_bytes"] <= 6 * GIB,
        "loader process high-water memory exceeds 6 GiB",
    )
    require(
        isinstance(sample.get("maximum_shm_used_delta_bytes"), int)
        and 0 <= sample["maximum_shm_used_delta_bytes"] <= 1024**2,
        "loader /dev/shm delta exceeds 1 MiB",
    )
    for claim in (
        "device_ranges_cover_source_exactly_once",
        "no_full_weight_replica_on_host_or_device",
        "all_global_fingerprints_equal",
        "all_distributed_shutdowns_complete",
    ):
        require(sample.get(claim) is True, f"loader claim {claim} is not true")
    gate = payload.get("gate", {})
    require(gate.get("g4_direct_loader") == "passed", "G4 is not marked passed")
    require(gate.get("full_model_runnable") is False, "G4 evidence must not claim a runnable full model")
    if mismatches:
        raise ValueError("invalid GLM-5.3 loader evidence: " + "; ".join(mismatches))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "source_revision": source_revision,
        "test": payload["test"],
        "header_audit": {
            "path": str(header_path),
            "sha256": hashlib.sha256(header_bytes).hexdigest(),
        },
        "placed_base_per_device_bytes": placement["estimated_text_base_bytes_per_device"],
        "staging_per_host_bytes": placement["maximum_single_device_range_bytes"],
    }


def validate_execution_schema_evidence(path: Path) -> dict:
    """Validate G5a's complete text-tensor to executable-target mapping."""

    payload, payload_bytes = _load_unique_json(path, label="execution schema evidence")
    mismatches: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    source_revision = payload.get("source_revision")
    require(payload.get("schema_version") == 1, "schema_version must be 1")
    require(payload.get("test") == "glm53_g5_execution_schema_audit", "unexpected test identity")
    require(
        isinstance(source_revision, str)
        and len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source_revision must be a full lowercase Git hash",
    )
    model = payload.get("model", {})
    require(model.get("repo_id") == OFFICIAL_CHECKPOINT.repo_id, "model repo mismatch")
    require(model.get("revision") == OFFICIAL_CHECKPOINT.revision, "model revision mismatch")
    require(model.get("config_sha256") == OFFICIAL_CHECKPOINT.config_sha256, "config hash mismatch")
    require(model.get("index_sha256") == OFFICIAL_CHECKPOINT.index_sha256, "index hash mismatch")
    require(model.get("text_payload_bytes") == 319_706_118_392, "text payload bytes mismatch")
    coverage = payload.get("coverage", {})
    require(coverage.get("logical_tensor_count") == 37_534, "logical tensor count mismatch")
    require(coverage.get("scale_tensor_count") == 36_467, "scale tensor count mismatch")
    require(coverage.get("text_tensor_count") == 74_001, "text tensor count mismatch")
    require(coverage.get("header_network_bytes") == 10_684_096, "header byte count mismatch")
    require(
        coverage.get("mapping_sha256")
        == "be4e5e87c71f8f51c65d41cd1f57e6cd1e0b90f7e37367b9eddce852c8112b36",
        "execution mapping digest mismatch",
    )
    require(
        coverage.get("all_text_tensors_mapped_exactly_once") is True,
        "execution mapping coverage is incomplete",
    )
    execution = payload.get("execution", {})
    require(execution.get("target_group_count") == 1_372, "executable target count mismatch")
    require(execution.get("quantized_target_group_count") == 305, "quantized target count mismatch")
    require(
        execution.get("role_counts")
        == {
            "dense_transpose": 494,
            "depthwise_conv": 102,
            "direct_array": 471,
            "fp8_expert_pack": 36_288,
            "fp8_linear": 179,
        },
        "execution role counts mismatch",
    )
    require(execution.get("scale_payload_bytes") == 74_956_800, "scale payload bytes mismatch")
    packing = payload.get("expert_packing", {})
    require(packing.get("group_count") == 126, "expert pack group count mismatch")
    require(
        packing.get("group_counts_by_projection") == {"down": 42, "gate": 42, "up": 42},
        "expert projection pack counts mismatch",
    )
    require(
        packing.get("groups_sha256")
        == "a1677002f0a90fb025bd4df36720eb1abcd0890e25cdd7586d24d96023a3f2c3",
        "expert pack manifest digest mismatch",
    )
    require(packing.get("source_bytes") == 304_405_807_104, "expert source byte count mismatch")
    require(packing.get("per_device_bytes") == 19_025_362_944, "expert device byte count mismatch")
    require(
        packing.get("maximum_device_staging_buffer_bytes") == 150_994_944,
        "expert staging buffer bound mismatch",
    )
    require(
        packing.get("all_groups_cover_experts_exactly_once") is True,
        "expert pack coverage is incomplete",
    )
    gate = payload.get("gate", {})
    require(gate.get("g5a_execution_schema") == "passed", "G5a is not marked passed")
    require(gate.get("full_model_runnable") is False, "G5a must not claim a runnable model")
    if mismatches:
        raise ValueError("invalid GLM-5.3 execution schema evidence: " + "; ".join(mismatches))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "source_revision": source_revision,
        "test": payload["test"],
        "staging_per_host_bytes": packing["maximum_device_staging_buffer_bytes"],
    }


def validate_expert_kernel_evidence(path: Path) -> dict:
    """Validate G5b's official-size packed-expert v4-32 acceptance."""

    payload, payload_bytes = _load_unique_json(path, label="expert kernel evidence")
    mismatches: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    source_revision = payload.get("source_revision")
    require(payload.get("schema_version") == 1, "schema_version must be 1")
    require(
        payload.get("test") == "glm53_g5b_official_expert_acceptance",
        "unexpected test identity",
    )
    require(
        isinstance(source_revision, str)
        and len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source_revision must be a full lowercase Git hash",
    )
    require(
        payload.get("topology")
        == {
            "accelerator_type": "v4-32",
            "global_device_count": 16,
            "host_count": 4,
            "process_count": 4,
        },
        "expert topology mismatch",
    )
    expert = payload.get("expert", {})
    expected_scalars = {
        "source_fp8_bytes_global": 7_247_757_312,
        "source_fp8_bytes_per_device": 452_984_832,
        "selected_bf16_weight_bytes_global": 402_653_184,
        "compiler_argument_bytes_per_device": 455_361_024,
        "compiler_temporary_bytes_per_device": 75_884_544,
        "maximum_device_bytes_in_use": 457_928_704,
        "maximum_device_peak_bytes_in_use": 457_929_216,
        "maximum_process_vmhwm_bytes": 6_532_288_512,
        "maximum_shm_used_delta_bytes": 0,
    }
    for name, expected in expected_scalars.items():
        require(expert.get(name) == expected, f"expert {name} mismatch")
    require(
        expert.get("statistics_float32_sha256")
        == "97effd6c04ae3afcba21d068f829dec80eda6f9b70957949f105596dc133626b",
        "expert output hash mismatch",
    )
    require(
        expert.get("optimized_hlo_sha256")
        == "e3608a6f69bbde3ede1f3e747488fb260522f77a1a1124c346540173ecb7d502",
        "expert optimized HLO hash mismatch",
    )
    mentions = expert.get("optimized_hlo_shape_mentions", {})
    for prefix in ("local_gate_up_expert_bank", "local_down_expert_bank"):
        require(mentions.get(prefix + ":bf16") == 0, f"{prefix} has a persistent BF16 shape")
        require(mentions.get(prefix + ":f32") == 0, f"{prefix} has a persistent F32 shape")
        require(mentions.get(prefix + ":u8", 0) > 0, f"{prefix} has no sharded uint8 source")
    for claim in (
        "all_outputs_equal",
        "all_optimized_hlo_equal",
        "no_persistent_bf16_or_f32_expert_bank_shape",
        "all_distributed_shutdowns_complete",
    ):
        require(expert.get(claim) is True, f"expert claim {claim} is not true")
    gate = payload.get("gate", {})
    require(gate.get("g5b_official_expert_kernel") == "passed", "G5b is not marked passed")
    require(gate.get("full_model_runnable") is False, "G5b must not claim a runnable model")
    if mismatches:
        raise ValueError("invalid GLM-5.3 expert kernel evidence: " + "; ".join(mismatches))
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
    parser.add_argument(
        "--loader-evidence",
        type=Path,
        help="tracked G4 JSON evidence; validated with its sibling header audit",
    )
    parser.add_argument(
        "--execution-schema-evidence",
        type=Path,
        help="tracked G5a JSON evidence covering every text tensor and expert pack",
    )
    parser.add_argument(
        "--expert-kernel-evidence",
        type=Path,
        help="tracked G5b JSON evidence for one official-size packed expert layer",
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
    loader_evidence = validate_loader_evidence(args.loader_evidence) if args.loader_evidence else None
    schema_evidence = (
        validate_execution_schema_evidence(args.execution_schema_evidence)
        if args.execution_schema_evidence
        else None
    )
    expert_evidence = (
        validate_expert_kernel_evidence(args.expert_kernel_evidence)
        if args.expert_kernel_evidence
        else None
    )
    staging_bounds = [
        evidence["staging_per_host_bytes"]
        for evidence in (loader_evidence, schema_evidence)
        if evidence is not None
    ]
    plan = v4_32_lora_preflight(
        config,
        index,
        rank=args.rank,
        execution_weight_format=args.execution_weight_format,
        executable_kernel_proven=kernel_evidence is not None,
        direct_loader_proven=loader_evidence is not None,
        execution_schema_proven=schema_evidence is not None,
        official_expert_kernel_proven=expert_evidence is not None,
        placed_base_per_device_bytes=(
            loader_evidence["placed_base_per_device_bytes"] if loader_evidence else None
        ),
        staging_per_host_bytes=max(staging_bounds) if staging_bounds else None,
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
        "evidence": {
            "kernel": kernel_evidence,
            "direct_loader": loader_evidence,
            "execution_schema": schema_evidence,
            "expert_kernel": expert_evidence,
        },
        "warning": (
            "G3/G4/G5a/G5b evidence proves real block-FP8 contractions, bounded direct sharding, "
            "complete schema coverage, and one official-size expert layer, not the full model; "
            "runnable remains false until the whole-model forward/HBM gate passes"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
