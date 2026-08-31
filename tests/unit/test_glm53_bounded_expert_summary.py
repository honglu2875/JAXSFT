import copy
import json

import pytest

from scripts.summarize_glm53_bounded_expert import EXPECTED_CASES, summarize


SOURCE_REVISION = "b" * 40


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _shape_mentions():
    result = {}
    for prefix in (
        "all_assignment_gate_dense",
        "all_assignment_down_dense",
        "token_topk_gate_dense",
        "bounded_gate_dense",
        "bounded_down_dense",
    ):
        for dtype in ("u8", "f8e4m3fn", "bf16", "f32"):
            result[f"{prefix}:{dtype}"] = 0
    result.update(
        {
            "local_bounded_gate_dense:u8": 4,
            "local_bounded_gate_dense:f8e4m3fn": 4,
            "local_bounded_gate_dense:bf16": 10,
            "local_bounded_gate_dense:f32": 12,
            "local_bounded_down_dense:u8": 2,
            "local_bounded_down_dense:f8e4m3fn": 2,
            "local_bounded_down_dense:bf16": 5,
            "local_bounded_down_dense:f32": 6,
        }
    )
    return result


def _rank(process_index):
    cases = []
    for token_count, expected in EXPECTED_CASES.items():
        cases.append(
            {
                "token_count": token_count,
                "assignment_count": expected["assignment_count"],
                "selected_weight_batch_size": 1,
                "compiler_memory": {
                    "alias_size_in_bytes": 0,
                    "argument_size_in_bytes": expected["argument_size_in_bytes"],
                    "generated_code_size_in_bytes": expected["generated_code_size_in_bytes"],
                    "host_argument_size_in_bytes": 0,
                    "host_output_size_in_bytes": 0,
                    "host_temp_size_in_bytes": 0,
                    "output_size_in_bytes": 512,
                    "temp_size_in_bytes": expected["temp_size_in_bytes"],
                },
                "optimized_hlo_sha256": expected["hlo_sha256"],
                "optimized_hlo_shape_mentions": _shape_mentions(),
                "statistics": expected["statistics"],
                "statistics_float32_sha256": expected["output_sha256"],
                "compile_seconds": 0.8,
                "execute_seconds": 0.2,
            }
        )
    device_ids = range(process_index * 4, process_index * 4 + 4)
    return {
        "schema_version": 1,
        "test": "glm53_bounded_official_expert_forward_backward_v4_probe",
        "source_revision": SOURCE_REVISION,
        "contract": {
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
        },
        "runtime": {
            "jax_version": "0.11.0",
            "backend": "tpu",
            "device_kinds": ["TPU v4"],
            "hostname": f"host-{process_index}",
            "process_index": process_index,
            "process_count": 4,
            "local_device_count": 4,
            "global_device_count": 16,
            "mesh_shape": {"model": 16},
            "precision": "HIGHEST",
            "expert_partition_spec": [None, "model", None],
            "distributed_initialized": True,
            "distributed_shutdown_complete": True,
        },
        "cases": cases,
        "device_memory": {
            "after_cases": [
                {
                    "device_id": device_id,
                    "stats": {
                        "bytes_limit": 33_014_407_168,
                        "largest_alloc_size": 150_994_944,
                        "peak_bytes_in_use": 460_946_432,
                    },
                }
                for device_id in device_ids
            ]
        },
        "host_memory": {"after_cases": {"vmhwm_bytes": 6_720_000_000}},
        "shm": {"used_delta_bytes": 0},
        "elapsed_seconds_before_shutdown": 6.0,
    }


def test_bounded_expert_summary_accepts_assignment_independent_workspace(tmp_path):
    paths = [_write(tmp_path / f"rank-{index}.json", _rank(index)) for index in range(4)]
    result = summarize(paths, source_revision=SOURCE_REVISION)
    assert result["gate"]["g6a_bounded_expert_input_gradient"] == "passed"
    assert result["expert"]["compiler_temporary_did_not_grow_with_assignments"] is True
    assert result["expert"]["maximum_dequantized_weight_bytes_per_projection_per_device"] == 1_048_576


def test_bounded_expert_summary_rejects_assignment_wide_weight_or_memory_drift(tmp_path):
    ranks = [_rank(index) for index in range(4)]
    ranks[0]["cases"][1]["optimized_hlo_shape_mentions"][
        "all_assignment_gate_dense:bf16"
    ] = 1
    paths = [_write(tmp_path / f"hlo-{index}.json", rank) for index, rank in enumerate(ranks)]
    with pytest.raises(ValueError, match="assignment-wide"):
        summarize(paths, source_revision=SOURCE_REVISION)

    ranks = [_rank(index) for index in range(4)]
    ranks[2]["cases"][1]["compiler_memory"]["temp_size_in_bytes"] += 1
    paths = [_write(tmp_path / f"memory-{index}.json", rank) for index, rank in enumerate(ranks)]
    with pytest.raises(ValueError, match="compiler memory"):
        summarize(paths, source_revision=SOURCE_REVISION)
