#!/usr/bin/env python3
"""Validate real-checkpoint expert streaming on four TPU hosts plus CPU oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_FINGERPRINTS = {
    "down": {
        "weight_bits_uint32": [2547530454, 234, 4069122540, 2591194520],
        "weight_scale_inv_bits_uint32": [205523066, 4194314, 1618318052, 2641454218],
    },
    "gate": {
        "weight_bits_uint32": [2532608034, 14, 3442219676, 6421482],
        "weight_scale_inv_bits_uint32": [2475241188, 12582922, 3777101266, 982101497],
    },
    "up": {
        "weight_bits_uint32": [2537129703, 109, 636097807, 107821607],
        "weight_scale_inv_bits_uint32": [3011513093, 2396747, 763879255, 4034090950],
    },
}
EXPECTED_OUTPUT_SHA256 = "046a54bf22934b0271d5dd30e8cfd5349190147d5252f7d5310e1ae0a0b2bbf7"
EXPECTED_HLO_SHA256 = "cfc69a283323a02652a2b5cfe026b6da9f430fa487b2d4f3433237adeda58f24"
EXPERT_BYTES_PER_HOST = 1_811_939_328
LOADER_BYTES_PER_HOST = 1_814_863_370


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


def summarize(
    rank_paths: list[Path],
    oracle_path: Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    if len(rank_paths) != 4:
        raise ValueError("exactly four real-expert TPU results are required")
    values_with_hashes = [_load(path) for path in rank_paths]
    values = [value for value, _ in values_with_hashes]
    oracle, oracle_hash = _load(oracle_path)
    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    device_ids: set[int] = set()
    maximum_device_peak = 0
    maximum_host_vmhwm = 0
    maximum_elapsed = 0.0
    for value in values:
        if value.get("schema_version") != 1:
            raise ValueError("real-expert schema version drifted")
        if value.get("test") != "glm53_real_checkpoint_expert_streaming_v4_probe":
            raise ValueError("unexpected real-expert TPU identity")
        if value.get("source_revision") != source_revision:
            raise ValueError("real-expert TPU source revision mismatch")
        if value.get("selection") != {
            "expert_indices": [0, 17, 63, 95, 127, 191, 255, 287],
            "experts": 288,
            "global_source_fp8_bytes": 7_247_757_312,
            "hidden_size": 4096,
            "layer": 3,
            "moe_intermediate_size": 2048,
            "top_k": 8,
        }:
            raise ValueError("real-expert selection contract drifted")
        runtime = value.get("runtime", {})
        if (
            runtime.get("backend") != "tpu"
            or runtime.get("device_kinds") != ["TPU v4"]
            or runtime.get("process_count") != 4
            or runtime.get("local_device_count") != 4
            or runtime.get("global_device_count") != 16
            or runtime.get("mesh_shape") != {"model": 16}
            or runtime.get("precision") != "HIGHEST"
            or runtime.get("distributed_initialized") is not True
            or runtime.get("distributed_shutdown_complete") is not True
        ):
            raise ValueError("real-expert topology, precision, or lifecycle drifted")
        hostnames.add(runtime["hostname"])
        process_indexes.add(runtime["process_index"])
        loader = value.get("loader", {})
        if loader != {
            "bytes_by_category": {
                "expert_tensor": EXPERT_BYTES_PER_HOST,
                "header": 350_808,
                "scale_envelope": 2_573_232,
            },
            "bytes_read_including_resolves": LOADER_BYTES_PER_HOST,
            "largest_request_bytes": 2_097_152,
            "loaded_logical_tensor_count": 864,
            "loaded_scale_tensor_count": 864,
            "loaded_target_count": 3,
            "maximum_expert_host_buffer_bytes": 603_979_776,
            "prepared_shard_count": 2,
            "request_count_including_resolves": 872,
            "requests_by_category": {
                "expert_tensor": 864,
                "header": 4,
                "scale_envelope": 2,
            },
        }:
            raise ValueError("real-expert loader accounting drifted")
        if value.get("selected_source_fingerprints") != EXPECTED_SOURCE_FINGERPRINTS:
            raise ValueError("real-expert source fingerprints drifted")
        output = value.get("output", {})
        if output.get("finite") is not True or output.get("statistics_float32_sha256") != EXPECTED_OUTPUT_SHA256:
            raise ValueError("real-expert output was non-finite or drifted")
        compiler = value.get("compiler_memory", {})
        if compiler.get("argument_size_in_bytes") != 455_361_024:
            raise ValueError("real-expert compiler argument size drifted")
        if compiler.get("temp_size_in_bytes") != 75_884_544:
            raise ValueError("real-expert compiler temporary size drifted")
        if value.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256:
            raise ValueError("real-expert optimized HLO hash drifted")
        placement = value.get("device_memory", {}).get("after_real_expert_placement", [])
        execution = value.get("device_memory", {}).get("after_execute", [])
        if len(placement) != 4 or len(execution) != 4:
            raise ValueError("real-expert result must report four local TPU memory records")
        for record in placement:
            if record.get("stats", {}).get("bytes_in_use") != 455_357_952:
                raise ValueError("real-expert placed bytes per device drifted")
        for record in execution:
            device_id = record.get("device_id")
            if device_id in device_ids:
                raise ValueError(f"device {device_id} appears in multiple real-expert results")
            device_ids.add(device_id)
            peak = record.get("stats", {}).get("peak_bytes_in_use")
            if not isinstance(peak, int) or not 455_357_952 <= peak <= 512 * 1024**2:
                raise ValueError("real-expert execution exceeds the 512 MiB device bound")
            maximum_device_peak = max(maximum_device_peak, peak)
        vmhwm = value.get("host_memory", {}).get("after_execute", {}).get("vmhwm_bytes")
        if not isinstance(vmhwm, int) or not 0 < vmhwm <= 7 * 1024**3:
            raise ValueError("real-expert host high-water memory exceeds 7 GiB")
        maximum_host_vmhwm = max(maximum_host_vmhwm, vmhwm)
        if value.get("shm", {}).get("used_delta_bytes") != 0:
            raise ValueError("real-expert run unexpectedly consumed RAMFS payload space")
        elapsed = value.get("elapsed_seconds_before_shutdown")
        if not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 300:
            raise ValueError("real-expert elapsed time is invalid")
        maximum_elapsed = max(maximum_elapsed, float(elapsed))
    if len(hostnames) != 4 or process_indexes != set(range(4)) or device_ids != set(range(16)):
        raise ValueError("real-expert results do not cover four hosts, ranks, and 16 devices")

    if oracle.get("schema_version") != 1:
        raise ValueError("real-expert CPU oracle schema drifted")
    if oracle.get("test") != "glm53_real_expert_transformers_cpu_oracle":
        raise ValueError("unexpected real-expert CPU oracle identity")
    if oracle.get("source_revision") != source_revision:
        raise ValueError("real-expert CPU oracle source revision mismatch")
    if oracle.get("selected_source_fingerprints") != EXPECTED_SOURCE_FINGERPRINTS:
        raise ValueError("CPU oracle source fingerprints do not match TPU fingerprints")
    comparison = oracle.get("comparison", {})
    if (
        comparison.get("passed") is not True
        or comparison.get("maximum_relative_l2_tolerance") != 0.02
        or comparison.get("maximum_absolute_tolerance") != 2e-5
        or not 0 <= comparison.get("relative_l2", -1) <= 0.02
        or not 0 <= comparison.get("maximum_absolute", -1) <= 2e-5
    ):
        raise ValueError("real-expert Transformers CPU comparison is outside its explicit bounds")
    runtime = oracle.get("runtime", {})
    if runtime.get("torch_version") != "2.10.0+cpu" or runtime.get("transformers_version") != "5.16.1":
        raise ValueError("real-expert CPU oracle dependency versions drifted")

    return {
        "schema_version": 1,
        "test": "glm53_g5c_real_expert_streaming_acceptance",
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
        },
        "streaming": {
            "real_expert_fp8_bytes_global": 7_247_757_312,
            "real_expert_fp8_bytes_per_host": EXPERT_BYTES_PER_HOST,
            "real_expert_fp8_bytes_per_device": 452_984_832,
            "total_network_bytes_across_hosts": 4 * LOADER_BYTES_PER_HOST,
            "source_payload_downloaded_exactly_once_across_slice": True,
            "maximum_single_range_bytes": 2_097_152,
            "maximum_expert_host_buffer_bytes": 603_979_776,
            "selected_source_fingerprints": EXPECTED_SOURCE_FINGERPRINTS,
            "all_tpu_source_fingerprints_equal": True,
            "transformers_cpu_source_fingerprints_equal": True,
        },
        "execution": {
            "placed_bytes_per_device": 455_357_952,
            "compiler_argument_bytes_per_device": 455_361_024,
            "compiler_temporary_bytes_per_device": 75_884_544,
            "maximum_device_peak_bytes_in_use": maximum_device_peak,
            "maximum_process_vmhwm_bytes": maximum_host_vmhwm,
            "maximum_elapsed_seconds": maximum_elapsed,
            "statistics_float32_sha256": EXPECTED_OUTPUT_SHA256,
            "all_outputs_equal": True,
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "all_optimized_hlo_equal": True,
            "maximum_shm_used_delta_bytes": 0,
            "all_distributed_shutdowns_complete": True,
        },
        "transformers_cpu_oracle": {
            "torch_version": runtime["torch_version"],
            "transformers_version": runtime["transformers_version"],
            "statistics": comparison["transformers_cpu_statistics"],
            "tpu_statistics": comparison["tpu_statistics"],
            "relative_l2": comparison["relative_l2"],
            "maximum_absolute": comparison["maximum_absolute"],
            "relative_l2_tolerance": comparison["maximum_relative_l2_tolerance"],
            "maximum_absolute_tolerance": comparison["maximum_absolute_tolerance"],
            "passed": True,
        },
        "rank_result_sha256": [digest for _, digest in values_with_hashes],
        "oracle_result_sha256": oracle_hash,
        "gate": {
            "g5c_real_expert_streaming": "passed",
            "full_model_runnable": False,
            "remaining_blockers": [
                "The complete 1,372-target text PyTree has not been streamed onto v4-32.",
                "Whole-model compilation, HBM, and a one-token frozen forward remain unmeasured.",
                "Long-sequence SFT still requires a capacity-bounded expert-dispatch kernel.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-results", type=Path, nargs=4, required=True)
    parser.add_argument("--cpu-oracle", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        list(args.rank_results),
        args.cpu_oracle,
        source_revision=args.source_revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
