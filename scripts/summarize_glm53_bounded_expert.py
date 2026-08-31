#!/usr/bin/env python3
"""Validate bounded official-size GLM expert forward/backward on four TPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CASES = {
    1: {
        "assignment_count": 8,
        "argument_size_in_bytes": 455_361_024,
        "temp_size_in_bytes": 774_144,
        "generated_code_size_in_bytes": 2_506_240,
        "hlo_sha256": "192c994184c605154bd62fd66f639b9cab9b6572f33238e392848e849bf5b850",
        "output_sha256": "405fdfb2c9a73568ed3b9297db98d50737e9f6285a7111b974b6c12a9c5a6eac",
        "statistics": [
            1.1974419976468198e-05,
            14.173828125,
            0.04904722422361374,
            0.003460407257080078,
            0.003460407257080078,
            0.00482177734375,
            5.676156433764845e-09,
            1.1771917343139648e-06,
            1.1771917343139648e-06,
        ],
    },
    4: {
        "assignment_count": 32,
        "argument_size_in_bytes": 455_378_944,
        "temp_size_in_bytes": 741_888,
        "generated_code_size_in_bytes": 2_631_680,
        "hlo_sha256": "8f6a7a7d691ec5bb7f0bdca709d7de4f012bd61514f18338bc3ed0ec3da60172",
        "output_sha256": "dd21c6d661268dab5814d45c2627a7c5b3136a710848dca4c6fae62ccfc31873",
        "statistics": [
            1.1974419976468198e-05,
            56.6953125,
            0.19618889689445496,
            0.003460407257080078,
            0.003460407257080078,
            0.00482177734375,
            1.4190391084412113e-09,
            2.942979335784912e-07,
            2.942979335784912e-07,
        ],
    },
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


def _validate_shape_mentions(mentions: Any) -> None:
    if not isinstance(mentions, dict):
        raise ValueError("bounded expert HLO shape mentions are missing")
    for prefix in (
        "all_assignment_gate_dense",
        "all_assignment_down_dense",
        "token_topk_gate_dense",
        "bounded_gate_dense",
        "bounded_down_dense",
    ):
        for dtype in ("u8", "f8e4m3fn", "bf16", "f32"):
            if mentions.get(f"{prefix}:{dtype}") != 0:
                raise ValueError(f"bounded expert HLO contains unsharded or assignment-wide {prefix}")
    expected_local = {
        "local_bounded_gate_dense:u8": 4,
        "local_bounded_gate_dense:f8e4m3fn": 4,
        "local_bounded_gate_dense:bf16": 10,
        "local_bounded_gate_dense:f32": 12,
        "local_bounded_down_dense:u8": 2,
        "local_bounded_down_dense:f8e4m3fn": 2,
        "local_bounded_down_dense:bf16": 5,
        "local_bounded_down_dense:f32": 6,
    }
    for name, expected in expected_local.items():
        if mentions.get(name) != expected:
            raise ValueError(f"bounded expert local HLO shape count drifted for {name}")


def summarize(rank_paths: list[Path], *, source_revision: str) -> dict[str, Any]:
    if len(rank_paths) != 4:
        raise ValueError("exactly four bounded-expert TPU results are required")
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
    maximum_host_vmhwm = 0
    maximum_elapsed = 0.0
    maximum_compile = {token_count: 0.0 for token_count in EXPECTED_CASES}
    maximum_execute = {token_count: 0.0 for token_count in EXPECTED_CASES}

    expected_contract = {
        "backward_chunk_rematerialized": True,
        "block_shape": [128, 128],
        "experts": 288,
        "hidden_size": 4096,
        "maximum_dequantized_weight_bytes_per_projection_global": 16_777_216,
        "moe_intermediate_size": 2048,
        "selected_weight_batch_size": 1,
        "source_fp8_bytes": 7_247_757_312,
        "source_fp8_bytes_per_device": 452_984_832,
        "top_k": 8,
        "weight_workspace_independent_of_token_count": True,
    }
    for value, _ in values_with_hashes:
        if value.get("schema_version") != 1:
            raise ValueError("bounded expert schema version drifted")
        if value.get("test") != "glm53_bounded_official_expert_forward_backward_v4_probe":
            raise ValueError("unexpected bounded expert test identity")
        if value.get("source_revision") != source_revision:
            raise ValueError("bounded expert source revision mismatch")
        if value.get("contract") != expected_contract:
            raise ValueError("bounded expert contract drifted")
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
            or runtime.get("expert_partition_spec") != [None, "model", None]
            or runtime.get("distributed_initialized") is not True
            or runtime.get("distributed_shutdown_complete") is not True
        ):
            raise ValueError("bounded expert topology, precision, or lifecycle drifted")
        hostname = runtime.get("hostname")
        process_index = runtime.get("process_index")
        if not isinstance(hostname, str) or not isinstance(process_index, int):
            raise ValueError("bounded expert runtime identity is malformed")
        hostnames.add(hostname)
        process_indexes.add(process_index)

        cases = value.get("cases")
        if not isinstance(cases, list) or [case.get("token_count") for case in cases] != [1, 4]:
            raise ValueError("bounded expert cases must be the one- and four-token probes")
        for case in cases:
            token_count = case["token_count"]
            expected = EXPECTED_CASES[token_count]
            if (
                case.get("assignment_count") != expected["assignment_count"]
                or case.get("selected_weight_batch_size") != 1
                or case.get("optimized_hlo_sha256") != expected["hlo_sha256"]
                or case.get("statistics_float32_sha256") != expected["output_sha256"]
                or case.get("statistics") != expected["statistics"]
            ):
                raise ValueError(f"bounded expert case {token_count} identity or output drifted")
            memory = case.get("compiler_memory", {})
            if memory != {
                "alias_size_in_bytes": 0,
                "argument_size_in_bytes": expected["argument_size_in_bytes"],
                "generated_code_size_in_bytes": expected["generated_code_size_in_bytes"],
                "host_argument_size_in_bytes": 0,
                "host_output_size_in_bytes": 0,
                "host_temp_size_in_bytes": 0,
                "output_size_in_bytes": 512,
                "temp_size_in_bytes": expected["temp_size_in_bytes"],
            }:
                raise ValueError(f"bounded expert case {token_count} compiler memory drifted")
            _validate_shape_mentions(case.get("optimized_hlo_shape_mentions"))
            compile_seconds = case.get("compile_seconds")
            execute_seconds = case.get("execute_seconds")
            if (
                not isinstance(compile_seconds, (int, float))
                or not 0 < compile_seconds <= 10
                or not isinstance(execute_seconds, (int, float))
                or not 0 < execute_seconds <= 10
            ):
                raise ValueError(f"bounded expert case {token_count} timing is invalid")
            maximum_compile[token_count] = max(maximum_compile[token_count], compile_seconds)
            maximum_execute[token_count] = max(maximum_execute[token_count], execute_seconds)

        memory_records = value.get("device_memory", {}).get("after_cases", [])
        if len(memory_records) != 4:
            raise ValueError("bounded expert result must report four local TPU memory records")
        for record in memory_records:
            device_id = record.get("device_id")
            if device_id in device_ids:
                raise ValueError(f"device {device_id} appears in multiple bounded expert results")
            device_ids.add(device_id)
            stats = record.get("stats", {})
            peak = stats.get("peak_bytes_in_use")
            if (
                stats.get("bytes_limit") != 33_014_407_168
                or stats.get("largest_alloc_size") != 150_994_944
                or not isinstance(peak, int)
                or not 455_357_952 <= peak <= 512 * 1024**2
            ):
                raise ValueError("bounded expert execution exceeds the 512 MiB device bound")
            maximum_device_peak = max(maximum_device_peak, peak)
        vmhwm = value.get("host_memory", {}).get("after_cases", {}).get("vmhwm_bytes")
        if not isinstance(vmhwm, int) or not 0 < vmhwm <= 7 * 1024**3:
            raise ValueError("bounded expert process high-water memory exceeds 7 GiB")
        maximum_host_vmhwm = max(maximum_host_vmhwm, vmhwm)
        if value.get("shm", {}).get("used_delta_bytes") != 0:
            raise ValueError("bounded expert run unexpectedly consumed RAMFS payload space")
        elapsed = value.get("elapsed_seconds_before_shutdown")
        if not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 60:
            raise ValueError("bounded expert elapsed time is invalid")
        maximum_elapsed = max(maximum_elapsed, elapsed)

    if len(hostnames) != 4 or process_indexes != set(range(4)) or device_ids != set(range(16)):
        raise ValueError("bounded expert results do not cover four hosts, ranks, and 16 devices")
    if EXPECTED_CASES[4]["temp_size_in_bytes"] > EXPECTED_CASES[1]["temp_size_in_bytes"]:
        raise ValueError("bounded expert compiler temporary grew with assignment count")

    return {
        "schema_version": 1,
        "test": "glm53_g6a_bounded_expert_forward_backward_acceptance",
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
        },
        "expert": {
            **expected_contract,
            "maximum_dequantized_weight_bytes_per_projection_per_device": 1_048_576,
            "one_token_assignment_count": 8,
            "four_token_assignment_count": 32,
            "one_token_compiler_temporary_bytes_per_device": EXPECTED_CASES[1][
                "temp_size_in_bytes"
            ],
            "four_token_compiler_temporary_bytes_per_device": EXPECTED_CASES[4][
                "temp_size_in_bytes"
            ],
            "compiler_temporary_did_not_grow_with_assignments": True,
            "maximum_device_peak_bytes_in_use": maximum_device_peak,
            "maximum_process_vmhwm_bytes": maximum_host_vmhwm,
            "maximum_shm_used_delta_bytes": 0,
            "maximum_elapsed_seconds": maximum_elapsed,
            "maximum_compile_seconds_by_token_count": {
                str(key): value for key, value in maximum_compile.items()
            },
            "maximum_execute_seconds_by_token_count": {
                str(key): value for key, value in maximum_execute.items()
            },
            "optimized_hlo_sha256_by_token_count": {
                str(key): value["hlo_sha256"] for key, value in EXPECTED_CASES.items()
            },
            "statistics_float32_sha256_by_token_count": {
                str(key): value["output_sha256"] for key, value in EXPECTED_CASES.items()
            },
            "all_rank_outputs_equal": True,
            "all_rank_hlo_equal": True,
            "no_assignment_wide_dense_weight_in_optimized_hlo": True,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256": [digest for _, digest in values_with_hashes],
        "gate": {
            "g6a_bounded_expert_input_gradient": "passed",
            "full_model_lora_step_runnable": False,
            "remaining_blockers": [
                "The bounded primitive is not yet selected by the complete model.",
                "Full-checkpoint attention-LoRA backward and optimizer HBM remain unmeasured.",
                "SFT loss, adapter update, checkpoint restore, and multi-step stability remain untested.",
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
