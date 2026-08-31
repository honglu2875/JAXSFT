#!/usr/bin/env python3
"""Validate and summarize a four-host complete GLM-5.3 frozen forward."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


GIB = 1024**3
EXPECTED_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
EXPECTED_INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
EXPECTED_HLO_SHA256 = "3b116a177e03dc5372c03a471c6c99dd185134cad5a6572b4d4b503ad6bd1429"
EXPECTED_OUTPUT_SHA256 = "2d27e8296bffd4a841dabc1630408f9a6db30ca6930e889b75363a5cf977dc67"
EXPECTED_BASE_BYTES_PER_DEVICE = 20_234_287_352
EXPECTED_DEVICE_LIMIT_BYTES = 33_014_407_168
EXPECTED_PLACED_RUNTIME_BYTES = 20_262_216_192
EXPECTED_LOADER_BYTES_PER_HOST = 80_141_139_062
EXPECTED_LOADER_REQUESTS_PER_HOST = 38_847
EXPECTED_OUTPUT = {
    "finite": True,
    "statistics": [
        -46_584.71875,
        388_708.03125,
        8.375,
        -8.0,
        -0.30077943205833435,
        0.96484375,
        1.9140625,
        0.765625,
        -2.21875,
        0.173828125,
        0.08935546875,
        -0.390625,
        -2.421875,
        -1.8828125,
    ],
    "statistics_float32_sha256": EXPECTED_OUTPUT_SHA256,
    "summary_names": [
        "logits_sum",
        "logits_square_sum",
        "logits_max",
        "logits_min",
        "logits_mean",
        "logit_token_0",
        "logit_token_1",
        "logit_token_2",
        "logit_token_42",
        "logit_token_1024",
        "logit_token_8192",
        "logit_token_65536",
        "logit_token_131072",
        "logit_token_154420",
    ],
    "two_executions_bitwise_equal": True,
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
        raise ValueError("exactly four complete-forward TPU results are required")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source_revision must be a full lowercase Git hash")

    values_with_hashes = [_load(path) for path in rank_paths]
    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    device_ids: set[int] = set()
    maximum_device_peak = 0
    minimum_execution_free_block = EXPECTED_DEVICE_LIMIT_BYTES
    maximum_host_vmhwm = 0
    maximum_load_seconds = 0.0
    maximum_compile_seconds = 0.0
    maximum_first_execute_seconds = 0.0
    maximum_second_execute_seconds = 0.0
    maximum_elapsed_seconds = 0.0

    expected_loader = {
        "bytes_by_category": {
            "axis0_tensor": 3_757_703_168,
            "expert_tensor": 76_101_451_776,
            "header": 10_684_096,
            "replicated_tensor": 193_428_312,
            "scale_envelope": 77_871_648,
        },
        "bytes_read_including_resolves": EXPECTED_LOADER_BYTES_PER_HOST,
        "largest_request_bytes": 79_298_560,
        "loaded_logical_tensor_count": 37_534,
        "loaded_scale_tensor_count": 36_467,
        "loaded_target_count": 1_372,
        "maximum_expert_host_buffer_bytes": 603_979_776,
        "prepared_shard_count": 62,
        "request_count_including_resolves": EXPECTED_LOADER_REQUESTS_PER_HOST,
        "requests_by_category": {
            "axis0_tensor": 1_796,
            "expert_tensor": 36_288,
            "header": 124,
            "replicated_tensor": 516,
            "scale_envelope": 61,
        },
    }
    expected_placement_common = {
        "all_local_devices_match_header_audit": True,
        "array_leaf_count": 1_677,
        "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
        "global_leaf_elements_by_dtype": {
            "bfloat16": 6_303_463_936,
            "float32": 19_034_430,
            "uint8": 307_023_052_800,
        },
        "global_leaf_elements_including_scale_metadata": 313_345_551_166,
        "local_leaf_shard_count": 6_708,
    }
    expected_compiler = {
        "alias_size_in_bytes": 0,
        "argument_size_in_bytes": 20_262_202_880,
        "generated_code_size_in_bytes": 41_561_600,
        "host_argument_size_in_bytes": 0,
        "host_output_size_in_bytes": 0,
        "host_temp_size_in_bytes": 0,
        "output_size_in_bytes": 512,
        "temp_size_in_bytes": 225_031_168,
    }

    for value, _ in values_with_hashes:
        if value.get("schema_version") != 1:
            raise ValueError("complete-forward schema version drifted")
        if value.get("test") != "glm53_complete_text_streaming_one_token_v4_forward":
            raise ValueError("unexpected complete-forward test identity")
        if value.get("source_revision") != source_revision:
            raise ValueError("complete-forward source revision mismatch")
        if value.get("model") != {
            "config_path_sha256": EXPECTED_CONFIG_SHA256,
            "index_sha256": EXPECTED_INDEX_SHA256,
            "input_token_id": 1,
            "num_hidden_layers": 45,
            "selected_logit_ids": [0, 1, 2, 42, 1024, 8192, 65536, 131072, 154420],
            "vocab_size": 154_880,
        }:
            raise ValueError("complete-forward model identity drifted")

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
            raise ValueError("complete-forward topology, precision, or lifecycle drifted")
        hostname = runtime.get("hostname")
        process_index = runtime.get("process_index")
        if not isinstance(hostname, str) or not isinstance(process_index, int):
            raise ValueError("complete-forward runtime identity is malformed")
        hostnames.add(hostname)
        process_indexes.add(process_index)

        loader = value.get("loader", {})
        placement = loader.get("parameter_placement", {})
        loader_without_variable_fields = {
            key: loader.get(key) for key in expected_loader
        }
        if loader_without_variable_fields != expected_loader:
            raise ValueError("complete-forward loader accounting drifted")
        load_seconds = loader.get("load_seconds")
        if not isinstance(load_seconds, (int, float)) or not 0 < load_seconds <= 3_600:
            raise ValueError("complete-forward loader duration is invalid")
        if placement.get("local_bytes_by_device") is None:
            raise ValueError("complete-forward local placement is missing")
        placement_common = {
            key: placement.get(key) for key in expected_placement_common
        }
        if placement_common != expected_placement_common:
            raise ValueError("complete-forward parameter placement drifted")
        local_bytes = placement["local_bytes_by_device"]
        if not isinstance(local_bytes, dict) or len(local_bytes) != 4:
            raise ValueError("complete-forward placement must cover four local devices")
        local_device_ids = {int(key) for key in local_bytes}
        if set(local_bytes.values()) != {EXPECTED_BASE_BYTES_PER_DEVICE}:
            raise ValueError("complete-forward base bytes per device drifted")

        device_memory = value.get("device_memory", {})
        placed_records = device_memory.get("after_full_base_placement", [])
        execution_records = device_memory.get("after_second_execute", [])
        if len(placed_records) != 4 or len(execution_records) != 4:
            raise ValueError("complete-forward result must report four local TPU memory records")
        if {record.get("device_id") for record in placed_records} != local_device_ids:
            raise ValueError("complete-forward placement device IDs drifted")
        for record in placed_records:
            stats = record.get("stats", {})
            if (
                stats.get("bytes_in_use") != EXPECTED_PLACED_RUNTIME_BYTES
                or stats.get("bytes_limit") != EXPECTED_DEVICE_LIMIT_BYTES
                or stats.get("largest_alloc_size") != 150_994_944
            ):
                raise ValueError("complete-forward placed TPU memory drifted")
        for record in execution_records:
            device_id = record.get("device_id")
            if device_id in device_ids:
                raise ValueError(f"device {device_id} appears in multiple complete-forward results")
            device_ids.add(device_id)
            stats = record.get("stats", {})
            peak = stats.get("peak_bytes_in_use")
            free_block = stats.get("largest_free_block_bytes")
            if (
                stats.get("bytes_limit") != EXPECTED_DEVICE_LIMIT_BYTES
                or not isinstance(peak, int)
                or not EXPECTED_PLACED_RUNTIME_BYTES <= peak <= 21 * GIB
                or not isinstance(free_block, int)
                or free_block < 11 * GIB
            ):
                raise ValueError("complete-forward execution exceeds its TPU memory bounds")
            maximum_device_peak = max(maximum_device_peak, peak)
            minimum_execution_free_block = min(minimum_execution_free_block, free_block)

        if value.get("compiler_memory") != expected_compiler:
            raise ValueError("complete-forward compiler memory drifted")
        if value.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256:
            raise ValueError("complete-forward optimized HLO hash drifted")
        if value.get("output") != EXPECTED_OUTPUT:
            raise ValueError("complete-forward output was non-finite, nondeterministic, or drifted")
        if (
            value.get("shm", {}).get("used_delta_during_load_bytes") != 0
            or value.get("shm", {}).get("used_delta_total_bytes") != 0
        ):
            raise ValueError("complete-forward unexpectedly consumed RAMFS payload space")

        vmhwm = value.get("host_memory", {}).get("after_second_execute", {}).get("vmhwm_bytes")
        if not isinstance(vmhwm, int) or not 0 < vmhwm <= 10 * GIB:
            raise ValueError("complete-forward host high-water memory exceeds 10 GiB")
        maximum_host_vmhwm = max(maximum_host_vmhwm, vmhwm)
        timing = value.get("timing", {})
        timing_values = [
            timing.get("load_seconds"),
            timing.get("compile_seconds"),
            timing.get("first_execute_seconds"),
            timing.get("second_execute_seconds"),
            timing.get("elapsed_seconds_before_shutdown"),
        ]
        if any(not isinstance(item, (int, float)) or not 0 < item <= 4_000 for item in timing_values):
            raise ValueError("complete-forward timing is invalid")
        maximum_load_seconds = max(maximum_load_seconds, float(timing_values[0]))
        maximum_compile_seconds = max(maximum_compile_seconds, float(timing_values[1]))
        maximum_first_execute_seconds = max(maximum_first_execute_seconds, float(timing_values[2]))
        maximum_second_execute_seconds = max(maximum_second_execute_seconds, float(timing_values[3]))
        maximum_elapsed_seconds = max(maximum_elapsed_seconds, float(timing_values[4]))

    if len(hostnames) != 4 or process_indexes != set(range(4)) or device_ids != set(range(16)):
        raise ValueError("complete-forward results do not cover four hosts, ranks, and 16 devices")

    return {
        "schema_version": 1,
        "test": "glm53_g5c2_complete_frozen_forward_acceptance",
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
            "physical_hostnames": sorted(hostnames),
            "physical_hostname_order_independent_of_process_index": True,
        },
        "streaming": {
            "prepared_shard_count": 62,
            "loaded_logical_tensor_count": 37_534,
            "loaded_scale_tensor_count": 36_467,
            "loaded_target_count": 1_372,
            "network_bytes_per_host": EXPECTED_LOADER_BYTES_PER_HOST,
            "network_bytes_across_hosts": 4 * EXPECTED_LOADER_BYTES_PER_HOST,
            "requests_per_host_including_resolves": EXPECTED_LOADER_REQUESTS_PER_HOST,
            "largest_range_bytes": 79_298_560,
            "maximum_expert_host_buffer_bytes": 603_979_776,
            "maximum_process_vmhwm_bytes": maximum_host_vmhwm,
            "maximum_load_seconds": maximum_load_seconds,
            "maximum_shm_used_delta_bytes": 0,
        },
        "execution": {
            "parameter_array_leaf_count": 1_677,
            "base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
            "compiler_argument_bytes_per_device": expected_compiler["argument_size_in_bytes"],
            "compiler_temporary_bytes_per_device": expected_compiler["temp_size_in_bytes"],
            "compiler_output_bytes_per_device": expected_compiler["output_size_in_bytes"],
            "generated_code_bytes": expected_compiler["generated_code_size_in_bytes"],
            "hbm_limit_bytes_per_device": EXPECTED_DEVICE_LIMIT_BYTES,
            "maximum_device_peak_bytes_in_use": maximum_device_peak,
            "headroom_after_peak_bytes_per_device": EXPECTED_DEVICE_LIMIT_BYTES - maximum_device_peak,
            "minimum_largest_free_block_after_execution_bytes": minimum_execution_free_block,
            "maximum_compile_seconds": maximum_compile_seconds,
            "maximum_first_execute_seconds": maximum_first_execute_seconds,
            "maximum_second_execute_seconds": maximum_second_execute_seconds,
            "maximum_elapsed_seconds": maximum_elapsed_seconds,
            "statistics": EXPECTED_OUTPUT["statistics"],
            "statistics_float32_sha256": EXPECTED_OUTPUT_SHA256,
            "all_outputs_finite": True,
            "two_executions_bitwise_equal_on_every_rank": True,
            "all_rank_outputs_equal": True,
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "all_optimized_hlo_equal": True,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256": [digest for _, digest in values_with_hashes],
        "gate": {
            "g5c2_full_frozen_forward": "passed",
            "full_model_frozen_forward_proven": True,
            "bounded_lora_sft_runnable": False,
            "remaining_blockers": [
                "The selected-expert temporary still scales with tokens times top-k.",
                "A capacity-bounded expert-dispatch kernel is required before long sequences.",
                "Adapter backward, optimizer, loss, and update HBM remain unmeasured on the full model.",
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
