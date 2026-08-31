import copy
import json

import pytest

from scripts.summarize_glm53_full_forward import (
    EXPECTED_BASE_BYTES_PER_DEVICE,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_DEVICE_LIMIT_BYTES,
    EXPECTED_HLO_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_LOADER_BYTES_PER_HOST,
    EXPECTED_LOADER_REQUESTS_PER_HOST,
    EXPECTED_OUTPUT,
    EXPECTED_PLACED_RUNTIME_BYTES,
    summarize,
)


SOURCE_REVISION = "a" * 40


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _rank(process_index):
    device_ids = range(process_index * 4, process_index * 4 + 4)
    placed = [
        {
            "device_id": device_id,
            "stats": {
                "bytes_in_use": EXPECTED_PLACED_RUNTIME_BYTES,
                "bytes_limit": EXPECTED_DEVICE_LIMIT_BYTES,
                "largest_alloc_size": 150_994_944,
            },
        }
        for device_id in device_ids
    ]
    executed = [
        {
            "device_id": device_id,
            "stats": {
                "bytes_limit": EXPECTED_DEVICE_LIMIT_BYTES,
                "largest_free_block_bytes": 12_558_333_440,
                "peak_bytes_in_use": 20_303_898_624,
            },
        }
        for device_id in device_ids
    ]
    return {
        "schema_version": 1,
        "test": "glm53_complete_text_streaming_one_token_v4_forward",
        "source_revision": SOURCE_REVISION,
        "model": {
            "config_path_sha256": EXPECTED_CONFIG_SHA256,
            "index_sha256": EXPECTED_INDEX_SHA256,
            "input_token_id": 1,
            "num_hidden_layers": 45,
            "selected_logit_ids": [0, 1, 2, 42, 1024, 8192, 65536, 131072, 154420],
            "vocab_size": 154_880,
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
            "distributed_initialized": True,
            "distributed_shutdown_complete": True,
        },
        "loader": {
            "bytes_by_category": {
                "axis0_tensor": 3_757_703_168,
                "expert_tensor": 76_101_451_776,
                "header": 10_684_096,
                "replicated_tensor": 193_428_312,
                "scale_envelope": 77_871_648,
            },
            "bytes_read_including_resolves": EXPECTED_LOADER_BYTES_PER_HOST,
            "largest_request_bytes": 79_298_560,
            "load_seconds": 1_426.0,
            "loaded_logical_tensor_count": 37_534,
            "loaded_scale_tensor_count": 36_467,
            "loaded_target_count": 1_372,
            "maximum_expert_host_buffer_bytes": 603_979_776,
            "parameter_placement": {
                "all_local_devices_match_header_audit": True,
                "array_leaf_count": 1_677,
                "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
                "global_leaf_elements_by_dtype": {
                    "bfloat16": 6_303_463_936,
                    "float32": 19_034_430,
                    "uint8": 307_023_052_800,
                },
                "global_leaf_elements_including_scale_metadata": 313_345_551_166,
                "local_bytes_by_device": {
                    str(device_id): EXPECTED_BASE_BYTES_PER_DEVICE for device_id in device_ids
                },
                "local_leaf_shard_count": 6_708,
            },
            "prepared_shard_count": 62,
            "request_count_including_resolves": EXPECTED_LOADER_REQUESTS_PER_HOST,
            "requests_by_category": {
                "axis0_tensor": 1_796,
                "expert_tensor": 36_288,
                "header": 124,
                "replicated_tensor": 516,
                "scale_envelope": 61,
            },
        },
        "compiler_memory": {
            "alias_size_in_bytes": 0,
            "argument_size_in_bytes": 20_262_202_880,
            "generated_code_size_in_bytes": 41_561_600,
            "host_argument_size_in_bytes": 0,
            "host_output_size_in_bytes": 0,
            "host_temp_size_in_bytes": 0,
            "output_size_in_bytes": 512,
            "temp_size_in_bytes": 225_031_168,
        },
        "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
        "output": copy.deepcopy(EXPECTED_OUTPUT),
        "device_memory": {
            "after_full_base_placement": placed,
            "after_second_execute": executed,
        },
        "host_memory": {"after_second_execute": {"vmhwm_bytes": 7_900_000_000}},
        "shm": {"used_delta_during_load_bytes": 0, "used_delta_total_bytes": 0},
        "timing": {
            "load_seconds": 1_426.0,
            "compile_seconds": 86.0,
            "first_execute_seconds": 134.0,
            "second_execute_seconds": 124.0,
            "elapsed_seconds_before_shutdown": 1_774.0,
        },
    }


def test_full_forward_summary_accepts_exact_four_rank_contract(tmp_path):
    rank_paths = [_write(tmp_path / f"rank-{index}.json", _rank(index)) for index in range(4)]
    result = summarize(rank_paths, source_revision=SOURCE_REVISION)
    assert result["gate"]["g5c2_full_frozen_forward"] == "passed"
    assert result["execution"]["headroom_after_peak_bytes_per_device"] == 12_710_508_544
    assert result["streaming"]["network_bytes_across_hosts"] == 320_564_556_248


def test_full_forward_summary_rejects_output_or_memory_drift(tmp_path):
    ranks = [_rank(index) for index in range(4)]
    ranks[0]["output"]["statistics"][0] += 1.0
    paths = [_write(tmp_path / f"output-{index}.json", rank) for index, rank in enumerate(ranks)]
    with pytest.raises(ValueError, match="output"):
        summarize(paths, source_revision=SOURCE_REVISION)

    ranks = [_rank(index) for index in range(4)]
    ranks[3]["device_memory"]["after_second_execute"][0]["stats"][
        "largest_free_block_bytes"
    ] = 10 * 1024**3
    paths = [_write(tmp_path / f"memory-{index}.json", rank) for index, rank in enumerate(ranks)]
    with pytest.raises(ValueError, match="memory bounds"):
        summarize(paths, source_revision=SOURCE_REVISION)
