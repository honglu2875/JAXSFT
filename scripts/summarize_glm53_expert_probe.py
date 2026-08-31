#!/usr/bin/env python3
"""Validate four official-size GLM-5.3 expert probe results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_OUTPUT_SHA256 = "97effd6c04ae3afcba21d068f829dec80eda6f9b70957949f105596dc133626b"
EXPECTED_HLO_SHA256 = "e3608a6f69bbde3ede1f3e747488fb260522f77a1a1124c346540173ecb7d502"
EXPECTED_SHAPE_MENTIONS = {
    "full_down_expert_bank:bf16": 0,
    "full_down_expert_bank:f32": 0,
    "full_down_expert_bank:f8e4m3fn": 0,
    "full_down_expert_bank:u8": 0,
    "full_gate_up_expert_bank:bf16": 0,
    "full_gate_up_expert_bank:f32": 0,
    "full_gate_up_expert_bank:f8e4m3fn": 0,
    "full_gate_up_expert_bank:u8": 0,
    "local_down_expert_bank:bf16": 0,
    "local_down_expert_bank:f32": 0,
    "local_down_expert_bank:f8e4m3fn": 0,
    "local_down_expert_bank:u8": 5,
    "local_gate_up_expert_bank:bf16": 0,
    "local_gate_up_expert_bank:f32": 0,
    "local_gate_up_expert_bank:f8e4m3fn": 0,
    "local_gate_up_expert_bank:u8": 10,
    "local_selected_down_dense:bf16": 3,
    "local_selected_down_dense:f32": 0,
    "local_selected_down_dense:f8e4m3fn": 1,
    "local_selected_down_dense:u8": 3,
    "local_selected_gate_up_dense:bf16": 6,
    "local_selected_gate_up_dense:f32": 0,
    "local_selected_gate_up_dense:f8e4m3fn": 2,
    "local_selected_gate_up_dense:u8": 6,
    "selected_down_dense:bf16": 0,
    "selected_down_dense:f32": 0,
    "selected_down_dense:f8e4m3fn": 0,
    "selected_down_dense:u8": 0,
    "selected_gate_up_dense:bf16": 0,
    "selected_gate_up_dense:f32": 0,
    "selected_gate_up_dense:f8e4m3fn": 0,
    "selected_gate_up_dense:u8": 0,
}
SOURCE_BYTES_PER_DEVICE = 452_984_832
MAXIMUM_COMPILER_TEMP = 128 * 1024**2
MAXIMUM_DEVICE_PEAK = 512 * 1024**2
MAXIMUM_HOST_VMHWM = 7 * 1024**3


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


def summarize(paths: list[Path], *, source_revision: str) -> dict[str, Any]:
    if len(paths) != 4:
        raise ValueError("exactly four expert probe results are required")
    values_with_hashes = [_load(path) for path in paths]
    values = [value for value, _ in values_with_hashes]
    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    device_ids: set[int] = set()
    maximum_device_peak = 0
    maximum_device_in_use = 0
    maximum_host_vmhwm = 0
    maximum_shm_delta = 0
    for value in values:
        if value.get("schema_version") != 1:
            raise ValueError("expert probe schema version drifted")
        if value.get("test") != "glm53_official_size_packed_expert_v4_probe":
            raise ValueError("unexpected expert probe identity")
        if value.get("source_revision") != source_revision:
            raise ValueError("expert probe source revision mismatch")
        contract = value.get("contract", {})
        if contract != {
            "block_shape": [128, 128],
            "down_shape": [288, 4096, 2048],
            "experts": 288,
            "gate_shape": [288, 2048, 4096],
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "selected_bf16_weight_bytes_global": 402_653_184,
            "source_fp8_bytes": 7_247_757_312,
            "source_fp8_bytes_per_device": SOURCE_BYTES_PER_DEVICE,
            "top_k": 8,
        }:
            raise ValueError("expert probe contract drifted")
        runtime = value.get("runtime", {})
        if (
            runtime.get("backend") != "tpu"
            or runtime.get("process_count") != 4
            or runtime.get("local_device_count") != 4
            or runtime.get("global_device_count") != 16
            or runtime.get("mesh_shape") != {"model": 16}
            or runtime.get("expert_partition_spec") != [None, "model", None]
            or runtime.get("precision") != "HIGHEST"
            or runtime.get("distributed_initialized") is not True
            or runtime.get("distributed_shutdown_complete") is not True
        ):
            raise ValueError("expert probe topology, sharding, precision, or lifecycle drifted")
        hostnames.add(runtime["hostname"])
        process_indexes.add(runtime["process_index"])
        memory = value.get("compiler_memory", {})
        temporary = memory.get("temp_size_in_bytes")
        arguments = memory.get("argument_size_in_bytes")
        if not isinstance(temporary, int) or not 0 < temporary <= MAXIMUM_COMPILER_TEMP:
            raise ValueError("expert compiler temporary exceeds the G5b bound")
        if not isinstance(arguments, int) or not SOURCE_BYTES_PER_DEVICE <= arguments <= 512 * 1024**2:
            raise ValueError("expert compiler arguments do not match one FP8 device shard")
        if value.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256:
            raise ValueError("expert optimized HLO hash drifted")
        mentions = value.get("optimized_hlo_shape_mentions", {})
        if mentions != EXPECTED_SHAPE_MENTIONS:
            raise ValueError("expert optimized HLO local/global shape inventory drifted")
        output = value.get("output", {})
        if output.get("finite") is not True or output.get("statistics_float32_sha256") != EXPECTED_OUTPUT_SHA256:
            raise ValueError("expert output was non-finite or drifted")
        local_device_memory = value.get("device_memory", {}).get("after_execute", [])
        if len(local_device_memory) != 4:
            raise ValueError("expert probe did not report four local device memory records")
        for record in local_device_memory:
            device_id = record.get("device_id")
            if device_id in device_ids:
                raise ValueError(f"device {device_id} appears in multiple expert results")
            device_ids.add(device_id)
            stats = record.get("stats", {})
            peak = stats.get("peak_bytes_in_use")
            in_use = stats.get("bytes_in_use")
            if not isinstance(peak, int) or not SOURCE_BYTES_PER_DEVICE <= peak <= MAXIMUM_DEVICE_PEAK:
                raise ValueError("expert device peak memory exceeds the G5b bound")
            if not isinstance(in_use, int) or not SOURCE_BYTES_PER_DEVICE <= in_use <= peak:
                raise ValueError("expert device memory-in-use is inconsistent")
            maximum_device_peak = max(maximum_device_peak, peak)
            maximum_device_in_use = max(maximum_device_in_use, in_use)
        vmhwm = value.get("host_memory", {}).get("after_execute", {}).get("vmhwm_bytes")
        if not isinstance(vmhwm, int) or not 0 < vmhwm <= MAXIMUM_HOST_VMHWM:
            raise ValueError("expert host high-water memory exceeds 7 GiB")
        maximum_host_vmhwm = max(maximum_host_vmhwm, vmhwm)
        shm_delta = value.get("shm", {}).get("used_delta_bytes")
        if not isinstance(shm_delta, int) or not 0 <= shm_delta <= 1024**2:
            raise ValueError("expert /dev/shm delta exceeds 1 MiB")
        maximum_shm_delta = max(maximum_shm_delta, shm_delta)
    if (
        len(hostnames) != 4
        or process_indexes != set(range(4))
        or device_ids != set(range(16))
    ):
        raise ValueError("expert results do not cover four hosts, ranks, and 16 devices")
    return {
        "schema_version": 1,
        "test": "glm53_g5b_official_expert_acceptance",
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
        },
        "expert": {
            "source_fp8_bytes_global": 7_247_757_312,
            "source_fp8_bytes_per_device": SOURCE_BYTES_PER_DEVICE,
            "selected_bf16_weight_bytes_global": 402_653_184,
            "compiler_argument_bytes_per_device": values[0]["compiler_memory"][
                "argument_size_in_bytes"
            ],
            "compiler_temporary_bytes_per_device": values[0]["compiler_memory"][
                "temp_size_in_bytes"
            ],
            "maximum_device_bytes_in_use": maximum_device_in_use,
            "maximum_device_peak_bytes_in_use": maximum_device_peak,
            "maximum_process_vmhwm_bytes": maximum_host_vmhwm,
            "maximum_shm_used_delta_bytes": maximum_shm_delta,
            "all_outputs_equal": True,
            "statistics_float32_sha256": EXPECTED_OUTPUT_SHA256,
            "all_optimized_hlo_equal": True,
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "optimized_hlo_shape_mentions": EXPECTED_SHAPE_MENTIONS,
            "no_persistent_bf16_or_f32_expert_bank_shape": True,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256": [digest for _, digest in values_with_hashes],
        "gate": {
            "g5b_official_expert_kernel": "passed",
            "full_model_runnable": False,
            "remaining_blockers": [
                "The complete mapped text PyTree has not been range-loaded on v4-32.",
                "Whole-model compilation, HBM, and one-token forward numerics remain unmeasured.",
                "Long-sequence SFT still requires a capacity-bounded expert-dispatch kernel.",
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
