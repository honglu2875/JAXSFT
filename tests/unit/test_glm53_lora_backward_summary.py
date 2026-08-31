import copy
import json

import pytest

from scripts.summarize_glm53_lora_backward import (
    EXPECTED_GRADIENT_STATISTICS,
    EXPECTED_HLO_BYTES,
    EXPECTED_HLO_SHA256,
    EXPECTED_INITIALIZATION_STATISTICS,
    EXPECTED_INITIALIZER_MEMORY,
    EXPECTED_LOADER_BYTES_PER_HOST,
    EXPECTED_OUTPUT,
    summarize,
)
from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_EXECUTION_GATE,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
)


REVISION = "ccd541611ec5b1d671c7d7c6cf4362671ccc2ba3"
BASE_BYTES = 20_234_287_352


def _memory_record(device_id, *, phase):
    if phase == "placed":
        stats = {
            "bytes_limit": 33_014_407_168,
            "bytes_in_use": 20_270_932_480,
            "peak_bytes_in_use": 20_270_932_480,
            "largest_alloc_size": 150_994_944,
            "largest_free_block_bytes": 12_724_616_704,
        }
    else:
        stats = {
            "bytes_limit": 33_014_407_168,
            "bytes_in_use": 20_490_830_336,
            "peak_bytes_in_use": 20_512_925_696 if phase == "final" else 20_490_885_120,
            "largest_alloc_size": 211_123_200,
            "largest_free_block_bytes": 11_769_517_568,
        }
    return {"device_id": device_id, "stats": stats}


def _rank(process_index):
    device_ids = range(process_index * 4, process_index * 4 + 4)
    header_network = {
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
    return {
        "schema_version": 1,
        "test": "glm53_complete_text_rank4_attention_lora_backward_v4_probe",
        "source_revision": REVISION,
        "model": {
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
        },
        "runtime": {
            "jax_version": "0.11.0",
            "backend": "tpu",
            "device_kinds": ["TPU v4"],
            "process_count": 4,
            "local_device_count": 4,
            "global_device_count": 16,
            "mesh_shape": {"model": 16},
            "precision": "HIGHEST",
            "distributed_initialized": True,
            "distributed_shutdown_complete": True,
            "hostname": f"host-{process_index}",
            "process_index": process_index,
        },
        "compile_preflight": {
            "checkpoint_payload_bytes_read": 0,
            "header_network": header_network,
            "compiler_memory": copy.deepcopy(EXPECTED_COMPILER_MEMORY),
            "execution_gate": copy.deepcopy(EXPECTED_EXECUTION_GATE),
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "optimized_hlo_bytes": EXPECTED_HLO_BYTES,
            "optimized_hlo_shape_mentions": copy.deepcopy(EXPECTED_SHAPE_MENTIONS),
        },
        "adapter": {
            "placement": {
                "a_partition_spec": [],
                "b_partition_spec": [None, "model"],
                "global_parameter_count": 10_289_152,
                "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
                "parameter_bytes_by_device": {
                    str(device): 8_705_024 for device in range(16)
                },
                "target_count": 191,
            },
            "initializer_compiler_memory": copy.deepcopy(EXPECTED_INITIALIZER_MEMORY),
            "initialization_statistic_names": [
                "all_finite",
                "a_l2_squared",
                "b_l2_squared",
                "a_max_abs",
                "b_max_abs",
                "a_nonzero_elements",
                "b_nonzero_elements",
            ],
            "initialization_statistics": list(EXPECTED_INITIALIZATION_STATISTICS),
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
            "parameter_placement": {
                "all_local_devices_match_header_audit": True,
                "array_leaf_count": 1_677,
                "expected_base_bytes_per_device": BASE_BYTES,
                "global_leaf_elements_by_dtype": {
                    "bfloat16": 6_303_463_936,
                    "float32": 19_034_430,
                    "uint8": 307_023_052_800,
                },
                "global_leaf_elements_including_scale_metadata": 313_345_551_166,
                "local_bytes_by_device": {
                    str(device): BASE_BYTES for device in device_ids
                },
            },
        },
        "output": copy.deepcopy(EXPECTED_OUTPUT),
        "device_memory": {
            "after_full_base_placement": [
                _memory_record(device, phase="placed") for device in device_ids
            ],
            "after_backward": [
                _memory_record(device, phase="backward") for device in device_ids
            ],
            "after_gradient_statistics": [
                _memory_record(device, phase="final") for device in device_ids
            ],
        },
        "host_memory": {"after_backward": {"vmhwm_bytes": 18_493_476_864}},
        "shm": {"used_delta_during_load_bytes": 0, "used_delta_total_bytes": 0},
        "timing": {
            "header_seconds": 38.0,
            "compile_seconds": 582.0,
            "initializer_seconds": 32.0,
            "load_seconds": 1_286.0,
            "backward_execute_seconds": 249.0,
            "gradient_statistics_seconds": 18.0,
            "elapsed_seconds_before_shutdown": 2_216.0,
        },
    }


def _write_ranks(tmp_path, values):
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f"rank-{index}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    return paths


def test_lora_backward_summary_accepts_exact_four_rank_gate(tmp_path):
    result = summarize(
        _write_ranks(tmp_path, [_rank(index) for index in range(4)]),
        source_revision=REVISION,
    )
    assert result["gate"]["g6b1_full_attention_lora_backward"] == "passed"
    assert result["gate"]["full_model_lora_backward_proven"] is True
    assert result["gate"]["three_step_optimizer_checkpoint_probe_authorized"] is True
    assert result["execution"]["gradient_statistics"] == EXPECTED_GRADIENT_STATISTICS
    assert result["execution"]["no_assignment_wide_dense_weight_in_optimized_hlo"] is True


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value["compile_preflight"].__setitem__(
                "checkpoint_payload_bytes_read", 1
            ),
            "preflight",
        ),
        (
            lambda value: value["compile_preflight"]["optimized_hlo_shape_mentions"].__setitem__(
                "local_all_assignment_gate_dense:bf16", 1
            ),
            "preflight",
        ),
        (
            lambda value: value["output"]["gradient_statistics"].__setitem__(2, 0.0),
            "gradient",
        ),
        (
            lambda value: value["shm"].__setitem__("used_delta_total_bytes", 4096),
            "RAMFS",
        ),
    ],
)
def test_lora_backward_summary_rejects_evidence_drift(tmp_path, mutation, error):
    values = [_rank(index) for index in range(4)]
    mutation(values[0])
    with pytest.raises(ValueError, match=error):
        summarize(_write_ranks(tmp_path, values), source_revision=REVISION)
