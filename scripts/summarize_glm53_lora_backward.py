#!/usr/bin/env python3
"""Validate one complete-checkpoint GLM attention-LoRA backward on four TPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_EXECUTION_GATE,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
)


GIB = 1024**3
EXPECTED_HLO_SHA256 = "3062ea4ba05cb62b59132751e854cb17ae753829067286fa956e3113945c5fad"
EXPECTED_HLO_BYTES = 163_191_482
EXPECTED_BASE_BYTES_PER_DEVICE = 20_234_287_352
EXPECTED_LOADER_BYTES_PER_HOST = 80_141_139_062
EXPECTED_INITIALIZER_MEMORY = {
    "alias_size_in_bytes": 0,
    "argument_size_in_bytes": 0,
    "generated_code_size_in_bytes": 5_957_632,
    "host_argument_size_in_bytes": 0,
    "host_output_size_in_bytes": 0,
    "host_temp_size_in_bytes": 0,
    "output_size_in_bytes": 8_717_824,
    "temp_size_in_bytes": 0,
}
EXPECTED_INITIALIZATION_STATISTICS = [
    1.0,
    254.57638549804688,
    0.0,
    0.044189453125,
    0.0,
    3_956_735.0,
    0.0,
]
EXPECTED_GRADIENT_STATISTICS = [
    1.0,
    0.0,
    7.319513320922852,
    0.0,
    1466.92578125,
    0.0,
    0.1455078125,
    0.0,
    4_821_488.0,
    0.0,
    169.0,
]
EXPECTED_OUTPUT = {
    "b_gradients_nonzero": True,
    "finite_loss_and_gradients": True,
    "gradient_statistic_names": [
        "all_finite",
        "a_l2_squared",
        "b_l2_squared",
        "a_l1",
        "b_l1",
        "a_max_abs",
        "b_max_abs",
        "a_nonzero_elements",
        "b_nonzero_elements",
        "a_nonzero_leaves",
        "b_nonzero_leaves",
    ],
    "gradient_statistics": EXPECTED_GRADIENT_STATISTICS,
    "gradient_statistics_float32_sha256": (
        "ef3fad7a12d9e968494bc1790d84cb424d8f48212247b8e1f75e00ee61ad9207"
    ),
    "loss": 12.241244316101074,
    "loss_float32_sha256": "ab4404e2ca8a94e73b62eef6c38d4a2e54f41f21d346d58230ba935c099c9101",
    "zero_initialized_b_gives_exact_zero_a_gradients": True,
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
        raise ValueError("exactly four full LoRA backward results are required")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source_revision must be a full lowercase Git hash")

    values_with_hashes = [_load(path) for path in rank_paths]
    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    device_ids: set[int] = set()
    maximum_process_vmhwm = 0
    maximum_device_peak = 0
    minimum_free_block = 33_014_407_168
    maxima = {
        "header_seconds": 0.0,
        "compile_seconds": 0.0,
        "initializer_seconds": 0.0,
        "load_seconds": 0.0,
        "backward_execute_seconds": 0.0,
        "gradient_statistics_seconds": 0.0,
        "elapsed_seconds_before_shutdown": 0.0,
    }
    expected_model = {
        "alpha": 4.0,
        "attention_lora_target_count": 191,
        "batch_size": 1,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "input_ids": [[1, 2]],
        "loss_token_count": 1,
        "loss_weights": [[0.0, 1.0]],
        "num_hidden_layers": 45,
        "rank": 4,
        "rematerialize_each_decoder_layer": True,
        "repo_id": "zai-org/GLM-5.3-Flash",
        "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "seed": 0,
        "sequence_length": 2,
    }
    expected_header_network = {
        "bytes_by_category": {"header": 10_684_096},
        "bytes_read_including_resolves": 10_684_158,
        "largest_request_bytes": 180_384,
        "loaded_logical_tensor_count": 0,
        "loaded_scale_tensor_count": 0,
        "loaded_target_count": 0,
        "maximum_expert_host_buffer_bytes": 0,
        "prepared_shard_count": 62,
        "request_count_including_resolves": 186,
        "requests_by_category": {"header": 124},
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
    }
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
        "request_count_including_resolves": 38_847,
        "requests_by_category": {
            "axis0_tensor": 1_796,
            "expert_tensor": 36_288,
            "header": 124,
            "replicated_tensor": 516,
            "scale_envelope": 61,
        },
    }
    expected_adapter_placement = {
        "a_partition_spec": [],
        "b_partition_spec": [None, "model"],
        "global_parameter_count": 10_289_152,
        "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
        "target_count": 191,
    }

    for value, _ in values_with_hashes:
        if (
            value.get("schema_version") != 1
            or value.get("test") != "glm53_complete_text_rank4_attention_lora_backward_v4_probe"
            or value.get("source_revision") != source_revision
            or value.get("model") != expected_model
        ):
            raise ValueError("full LoRA backward schema, source, or model identity drifted")
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
            raise ValueError("full LoRA backward topology, precision, or lifecycle drifted")
        hostname = runtime.get("hostname")
        process_index = runtime.get("process_index")
        if not isinstance(hostname, str) or not isinstance(process_index, int):
            raise ValueError("full LoRA backward runtime identity is malformed")
        hostnames.add(hostname)
        process_indexes.add(process_index)

        preflight = value.get("compile_preflight", {})
        if (
            preflight.get("checkpoint_payload_bytes_read") != 0
            or preflight.get("header_network") != expected_header_network
            or preflight.get("compiler_memory") != EXPECTED_COMPILER_MEMORY
            or preflight.get("execution_gate") != EXPECTED_EXECUTION_GATE
            or preflight.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256
            or preflight.get("optimized_hlo_bytes") != EXPECTED_HLO_BYTES
            or preflight.get("optimized_hlo_shape_mentions") != EXPECTED_SHAPE_MENTIONS
        ):
            raise ValueError("full LoRA backward compile preflight drifted")

        adapter = value.get("adapter", {})
        placement = adapter.get("placement", {})
        if {key: placement.get(key) for key in expected_adapter_placement} != expected_adapter_placement:
            raise ValueError("full LoRA backward adapter placement drifted")
        adapter_bytes = placement.get("parameter_bytes_by_device")
        if (
            not isinstance(adapter_bytes, dict)
            or {int(device_id) for device_id in adapter_bytes} != set(range(16))
            or set(adapter_bytes.values()) != {8_705_024}
            or adapter.get("initializer_compiler_memory") != EXPECTED_INITIALIZER_MEMORY
            or adapter.get("initialization_statistic_names")
            != [
                "all_finite",
                "a_l2_squared",
                "b_l2_squared",
                "a_max_abs",
                "b_max_abs",
                "a_nonzero_elements",
                "b_nonzero_elements",
            ]
            or adapter.get("initialization_statistics") != EXPECTED_INITIALIZATION_STATISTICS
        ):
            raise ValueError("full LoRA backward initialization drifted")

        loader = value.get("loader", {})
        if {key: loader.get(key) for key in expected_loader} != expected_loader:
            raise ValueError("full LoRA backward loader accounting drifted")
        parameter_placement = loader.get("parameter_placement", {})
        if (
            {key: parameter_placement.get(key) for key in expected_placement_common}
            != expected_placement_common
        ):
            raise ValueError("full LoRA backward base placement drifted")
        local_bytes = parameter_placement.get("local_bytes_by_device")
        if (
            not isinstance(local_bytes, dict)
            or len(local_bytes) != 4
            or set(local_bytes.values()) != {EXPECTED_BASE_BYTES_PER_DEVICE}
        ):
            raise ValueError("full LoRA backward local base bytes drifted")

        if value.get("output") != EXPECTED_OUTPUT:
            raise ValueError("full LoRA backward loss or gradient diagnostics drifted")
        memory = value.get("device_memory", {})
        placed = memory.get("after_full_base_placement", [])
        backward = memory.get("after_backward", [])
        final = memory.get("after_gradient_statistics", [])
        local_device_ids = {int(device_id) for device_id in local_bytes}
        if any(len(records) != 4 for records in (placed, backward, final)):
            raise ValueError("full LoRA backward must report four local device records per phase")
        if any({record.get("device_id") for record in records} != local_device_ids for records in (placed, backward, final)):
            raise ValueError("full LoRA backward device IDs drifted between memory phases")
        for record in placed:
            stats = record.get("stats", {})
            if (
                stats.get("bytes_limit") != 33_014_407_168
                or stats.get("bytes_in_use") != 20_270_932_480
                or stats.get("peak_bytes_in_use") != 20_270_932_480
                or stats.get("largest_alloc_size") != 150_994_944
                or not isinstance(stats.get("largest_free_block_bytes"), int)
                or stats["largest_free_block_bytes"] < 11 * GIB
            ):
                raise ValueError("full LoRA backward placed-device memory drifted")
        for record in backward + final:
            stats = record.get("stats", {})
            peak = stats.get("peak_bytes_in_use")
            free_block = stats.get("largest_free_block_bytes")
            if (
                stats.get("bytes_limit") != 33_014_407_168
                or stats.get("largest_alloc_size") != 211_123_200
                or not isinstance(peak, int)
                or not 20_270_932_480 <= peak <= 20 * GIB
                or not isinstance(free_block, int)
                or free_block < 10 * GIB
            ):
                raise ValueError("full LoRA backward execution memory exceeded its bounds")
            maximum_device_peak = max(maximum_device_peak, peak)
            minimum_free_block = min(minimum_free_block, free_block)
        for record in final:
            device_id = record["device_id"]
            if device_id in device_ids:
                raise ValueError(f"device {device_id} appears in multiple LoRA backward results")
            device_ids.add(device_id)

        if (
            value.get("shm", {}).get("used_delta_during_load_bytes") != 0
            or value.get("shm", {}).get("used_delta_total_bytes") != 0
        ):
            raise ValueError("full LoRA backward unexpectedly consumed RAMFS payload space")
        process_hwm = max(
            phase.get("vmhwm_bytes", 0) for phase in value.get("host_memory", {}).values()
        )
        if not isinstance(process_hwm, int) or not 0 < process_hwm <= 20 * GIB:
            raise ValueError("full LoRA backward process HWM exceeds 20 GiB")
        maximum_process_vmhwm = max(maximum_process_vmhwm, process_hwm)
        timing = value.get("timing", {})
        bounds = {
            "header_seconds": 120,
            "compile_seconds": 1_200,
            "initializer_seconds": 120,
            "load_seconds": 3_600,
            "backward_execute_seconds": 1_200,
            "gradient_statistics_seconds": 120,
            "elapsed_seconds_before_shutdown": 4_000,
        }
        for name, bound in bounds.items():
            item = timing.get(name)
            if not isinstance(item, (int, float)) or not 0 < item <= bound:
                raise ValueError(f"full LoRA backward {name} is invalid")
            maxima[name] = max(maxima[name], item)

    if (
        len(hostnames) != 4
        or process_indexes != set(range(4))
        or device_ids != set(range(16))
    ):
        raise ValueError("full LoRA backward does not cover four hosts, ranks, and 16 devices")

    return {
        "schema_version": 1,
        "test": "glm53_g6b1_full_attention_lora_backward_acceptance",
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
            "network_bytes_per_host": EXPECTED_LOADER_BYTES_PER_HOST,
            "network_bytes_across_hosts": 4 * EXPECTED_LOADER_BYTES_PER_HOST,
            "loaded_target_count": 1_372,
            "loaded_logical_tensor_count": 37_534,
            "loaded_scale_tensor_count": 36_467,
            "base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
            "maximum_load_seconds": maxima["load_seconds"],
            "maximum_shm_used_delta_bytes": 0,
        },
        "adapter": {
            **expected_adapter_placement,
            "parameter_bytes_per_device": 8_705_024,
            "initialization_statistics": EXPECTED_INITIALIZATION_STATISTICS,
            "maximum_initializer_seconds": maxima["initializer_seconds"],
        },
        "execution": {
            **EXPECTED_COMPILER_MEMORY,
            "compiler_working_set_upper_bound_bytes_per_device": EXPECTED_EXECUTION_GATE[
                "compiler_working_set_upper_bound_bytes_per_device"
            ],
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "loss": EXPECTED_OUTPUT["loss"],
            "loss_float32_sha256": EXPECTED_OUTPUT["loss_float32_sha256"],
            "gradient_statistics": EXPECTED_GRADIENT_STATISTICS,
            "gradient_statistics_float32_sha256": EXPECTED_OUTPUT[
                "gradient_statistics_float32_sha256"
            ],
            "all_rank_loss_and_gradient_statistics_equal": True,
            "all_loss_and_gradients_finite": True,
            "zero_initialized_b_gives_exact_zero_a_gradients": True,
            "nonzero_b_gradient_elements": 4_821_488,
            "nonzero_b_gradient_leaves": 169,
            "maximum_device_peak_bytes_in_use": maximum_device_peak,
            "headroom_after_peak_bytes_per_device": 33_014_407_168 - maximum_device_peak,
            "minimum_largest_free_block_after_execution_bytes": minimum_free_block,
            "maximum_process_vmhwm_bytes": maximum_process_vmhwm,
            **{f"maximum_{name}": value for name, value in maxima.items()},
            "all_rank_hlo_equal": True,
            "no_assignment_wide_dense_weight_in_optimized_hlo": True,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256": [digest for _, digest in values_with_hashes],
        "gate": {
            "g6b1_full_attention_lora_backward": "passed",
            "full_model_lora_backward_proven": True,
            "three_step_optimizer_checkpoint_probe_authorized": True,
            "remaining_blockers": [
                "No optimizer update has executed with the complete checkpoint.",
                "Adapter-only checkpoint save/restore has not passed on the sharded state.",
                "Three-step and 10--50-step loss/gradient/memory stability remain unmeasured.",
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
