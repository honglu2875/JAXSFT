#!/usr/bin/env python3
"""Validate four-host header-only GLM attention-LoRA backward compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


GIB = 1024**3
SOURCE_TEST = "glm53_full_attention_lora_backward_header_only_compile_v4_probe"
EXPECTED_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
EXPECTED_INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
EXPECTED_HLO_SHA256 = "19702470dc779b6609734f91aeea9e9849e81580622fc4d24ddaf5c7168b0bdc"
EXPECTED_HLO_BYTES = 163_191_375
EXPECTED_COMPILER_MEMORY = {
    "alias_size_in_bytes": 0,
    "argument_size_in_bytes": 20_270_928_384,
    "generated_code_size_in_bytes": 211_123_200,
    "host_argument_size_in_bytes": 0,
    "host_output_size_in_bytes": 0,
    "host_temp_size_in_bytes": 0,
    "output_size_in_bytes": 8_718_336,
    "temp_size_in_bytes": 1_248_113_152,
}
EXPECTED_EXECUTION_GATE = {
    "compiler_working_set_upper_bound_bytes_per_device": 21_527_759_872,
    "full_checkpoint_execution_authorized": True,
    "headroom_before_safety_margin_bytes_per_device": 11_486_647_296,
    "measured_hbm_limit_bytes_per_device": 33_014_407_168,
    "required_safety_margin_bytes_per_device": GIB,
}
EXPECTED_SHAPE_MENTIONS = {
    **{
        f"{prefix}:{dtype}": 0
        for prefix in (
            "all_assignment_down_dense",
            "all_assignment_gate_dense",
            "local_all_assignment_down_dense",
            "local_all_assignment_gate_dense",
            "local_token_topk_down_dense",
            "local_token_topk_gate_dense",
            "token_topk_down_dense",
            "token_topk_gate_dense",
        )
        for dtype in ("bf16", "f32", "f8e4m3fn", "u8")
    },
    "local_bounded_down_dense:bf16": 252,
    "local_bounded_down_dense:f32": 252,
    "local_bounded_down_dense:f8e4m3fn": 126,
    "local_bounded_down_dense:u8": 126,
    "local_bounded_gate_dense:bf16": 504,
    "local_bounded_gate_dense:f32": 504,
    "local_bounded_gate_dense:f8e4m3fn": 252,
    "local_bounded_gate_dense:u8": 252,
}


def _load(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def summarize(rank_paths: list[Path], *, source_revision: str) -> dict[str, Any]:
    if len(rank_paths) != 4:
        raise ValueError("exactly four LoRA backward compile results are required")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source_revision must be a full lowercase Git hash")

    values_with_hashes = [_load(path) for path in rank_paths]
    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    maximum_header_seconds = 0.0
    maximum_compile_seconds = 0.0
    maximum_elapsed_seconds = 0.0
    maximum_process_vmhwm = 0
    expected_model = {
        "alpha": 4.0,
        "attention_lora_target_count": 191,
        "batch_size": 1,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "loss_token_count": 1,
        "num_hidden_layers": 45,
        "rank": 4,
        "rematerialize_each_decoder_layer": True,
        "repo_id": "zai-org/GLM-5.3-Flash",
        "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "sequence_length": 2,
    }
    expected_loader = {
        "bytes_by_category": {"header": 10_684_096},
        "bytes_read_including_resolves": 10_684_158,
        "checkpoint_payload_bytes_read": 0,
        "largest_request_bytes": 180_384,
        "loaded_logical_tensor_count": 0,
        "loaded_scale_tensor_count": 0,
        "loaded_target_count": 0,
        "maximum_expert_host_buffer_bytes": 0,
        "prepared_shard_count": 62,
        "request_count_including_resolves": 186,
        "requests_by_category": {"header": 124},
    }
    expected_adapter_common = {
        "a_partition_spec": [],
        "b_partition_spec": [None, "model"],
        "global_parameter_count": 10_289_152,
        "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
        "target_count": 191,
    }

    for value, _ in values_with_hashes:
        if value.get("schema_version") != 1 or value.get("test") != SOURCE_TEST:
            raise ValueError("LoRA backward compile schema or test identity drifted")
        if value.get("source_revision") != source_revision or value.get("model") != expected_model:
            raise ValueError("LoRA backward compile source or model identity drifted")
        runtime = value.get("runtime", {})
        if (
            runtime.get("jax_version") != "0.11.0"
            or runtime.get("backend") != "tpu"
            or runtime.get("device_kinds") != ["TPU v4"]
            or runtime.get("process_count") != 4
            or runtime.get("local_device_count") != 4
            or runtime.get("global_device_count") != 16
            or runtime.get("mesh_shape") != {"model": 16}
            or runtime.get("precision") != "HIGHEST"
            or runtime.get("distributed_initialized") is not True
            or runtime.get("distributed_shutdown_complete") is not True
        ):
            raise ValueError("LoRA backward compile topology, precision, or lifecycle drifted")
        hostname = runtime.get("hostname")
        process_index = runtime.get("process_index")
        if not isinstance(hostname, str) or not isinstance(process_index, int):
            raise ValueError("LoRA backward compile runtime identity is malformed")
        hostnames.add(hostname)
        process_indexes.add(process_index)

        loader = value.get("header_only_loader", {})
        if {key: loader.get(key) for key in expected_loader} != expected_loader:
            raise ValueError("header-only loader accounting drifted or read checkpoint payloads")
        header_seconds = loader.get("header_seconds")
        if not isinstance(header_seconds, (int, float)) or not 0 < header_seconds <= 120:
            raise ValueError("header-only loader duration is invalid")
        maximum_header_seconds = max(maximum_header_seconds, header_seconds)

        placement = value.get("adapter_placement", {})
        if {key: placement.get(key) for key in expected_adapter_common} != expected_adapter_common:
            raise ValueError("LoRA adapter shape or sharding contract drifted")
        per_device = placement.get("parameter_bytes_by_device")
        if (
            not isinstance(per_device, dict)
            or {int(device_id) for device_id in per_device} != set(range(16))
            or set(per_device.values()) != {8_705_024}
        ):
            raise ValueError("LoRA adapter placement does not cover 16 chips uniformly")
        if value.get("compiler_memory") != EXPECTED_COMPILER_MEMORY:
            raise ValueError("LoRA backward compiler memory drifted")
        if value.get("execution_gate") != EXPECTED_EXECUTION_GATE:
            raise ValueError("LoRA backward execution safety gate drifted")
        if (
            value.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256
            or value.get("optimized_hlo_bytes") != EXPECTED_HLO_BYTES
            or value.get("optimized_hlo_shape_mentions") != EXPECTED_SHAPE_MENTIONS
        ):
            raise ValueError("LoRA backward HLO identity or bounded expert shapes drifted")
        if value.get("shm", {}).get("used_delta_bytes") != 0:
            raise ValueError("LoRA backward compile unexpectedly consumed RAMFS payload space")
        vmhwm = value.get("host_memory", {}).get("after_compile", {}).get("vmhwm_bytes")
        if not isinstance(vmhwm, int) or not 0 < vmhwm <= 20 * GIB:
            raise ValueError("LoRA backward compiler process HWM exceeds 20 GiB")
        maximum_process_vmhwm = max(maximum_process_vmhwm, vmhwm)
        timing = value.get("timing", {})
        compile_seconds = timing.get("compile_seconds")
        elapsed_seconds = timing.get("elapsed_seconds_before_shutdown")
        if (
            not isinstance(compile_seconds, (int, float))
            or not 0 < compile_seconds <= 1_200
            or not isinstance(elapsed_seconds, (int, float))
            or not 0 < elapsed_seconds <= 1_500
        ):
            raise ValueError("LoRA backward compile timing is invalid")
        maximum_compile_seconds = max(maximum_compile_seconds, compile_seconds)
        maximum_elapsed_seconds = max(maximum_elapsed_seconds, elapsed_seconds)

    if len(hostnames) != 4 or process_indexes != set(range(4)):
        raise ValueError("LoRA backward compile results do not cover four hosts and ranks")

    return {
        "schema_version": 1,
        "test": "glm53_g6b0_full_attention_lora_backward_compile_acceptance",
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
            "physical_hostnames": sorted(hostnames),
            "physical_hostname_order_independent_of_process_index": True,
        },
        "header_only_loader": {
            "header_bytes_per_host": 10_684_096,
            "checkpoint_payload_bytes_per_host": 0,
            "maximum_header_seconds": maximum_header_seconds,
            "maximum_shm_used_delta_bytes": 0,
        },
        "adapter": {
            **expected_adapter_common,
            "parameter_bytes_per_device": 8_705_024,
        },
        "compilation": {
            **EXPECTED_COMPILER_MEMORY,
            **EXPECTED_EXECUTION_GATE,
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "optimized_hlo_bytes": EXPECTED_HLO_BYTES,
            "all_rank_hlo_equal": True,
            "no_assignment_wide_dense_weight_in_optimized_hlo": True,
            "maximum_process_vmhwm_bytes": maximum_process_vmhwm,
            "maximum_compile_seconds": maximum_compile_seconds,
            "maximum_elapsed_seconds": maximum_elapsed_seconds,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256": [digest for _, digest in values_with_hashes],
        "gate": {
            "g6b0_full_attention_lora_backward_compile": "passed",
            "full_checkpoint_backward_execution_authorized": True,
            "full_model_lora_backward_proven": False,
            "remaining_blockers": [
                "The complete checkpoint has not executed this adapter-gradient program.",
                "Finite/nonzero full-model loss and adapter gradients remain unmeasured.",
                "Optimizer update, adapter-only checkpoint restore, and multi-step stability remain untested.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-results", type=Path, nargs=4, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(list(args.rank_results), source_revision=args.source_revision)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
