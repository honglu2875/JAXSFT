import json

import pytest

from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_EXECUTION_GATE,
    EXPECTED_HLO_BYTES,
    EXPECTED_HLO_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
    SOURCE_TEST,
    summarize,
)


REVISION = "56cd8b7a7fab16009f74614e6605d19298c5a0b4"


def _rank(process_index):
    return {
        "schema_version": 1,
        "test": SOURCE_TEST,
        "source_revision": REVISION,
        "model": {
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
        "header_only_loader": {
            "bytes_by_category": {"header": 10_684_096},
            "bytes_read_including_resolves": 10_684_158,
            "checkpoint_payload_bytes_read": 0,
            "header_seconds": 40.0,
            "largest_request_bytes": 180_384,
            "loaded_logical_tensor_count": 0,
            "loaded_scale_tensor_count": 0,
            "loaded_target_count": 0,
            "maximum_expert_host_buffer_bytes": 0,
            "prepared_shard_count": 62,
            "request_count_including_resolves": 186,
            "requests_by_category": {"header": 124},
        },
        "adapter_placement": {
            "a_partition_spec": [],
            "b_partition_spec": [None, "model"],
            "global_parameter_count": 10_289_152,
            "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
            "parameter_bytes_by_device": {str(device): 8_705_024 for device in range(16)},
            "target_count": 191,
        },
        "compiler_memory": EXPECTED_COMPILER_MEMORY,
        "execution_gate": EXPECTED_EXECUTION_GATE,
        "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
        "optimized_hlo_bytes": EXPECTED_HLO_BYTES,
        "optimized_hlo_shape_mentions": EXPECTED_SHAPE_MENTIONS,
        "host_memory": {"after_compile": {"vmhwm_bytes": 18_000_000_000}},
        "shm": {"used_delta_bytes": 0},
        "timing": {
            "compile_seconds": 600.0,
            "elapsed_seconds_before_shutdown": 650.0,
        },
    }


def _write_ranks(tmp_path, values):
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f"rank-{index}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    return paths


def test_lora_backward_compile_summary_accepts_exact_four_rank_gate(tmp_path):
    result = summarize(_write_ranks(tmp_path, [_rank(index) for index in range(4)]), source_revision=REVISION)
    assert result["gate"]["g6b0_full_attention_lora_backward_compile"] == "passed"
    assert result["gate"]["full_checkpoint_backward_execution_authorized"] is True
    assert result["gate"]["full_model_lora_backward_proven"] is False
    assert result["compilation"]["no_assignment_wide_dense_weight_in_optimized_hlo"] is True


def test_lora_backward_compile_summary_rejects_payload_reads_and_hlo_drift(tmp_path):
    values = [_rank(index) for index in range(4)]
    values[0]["header_only_loader"]["checkpoint_payload_bytes_read"] = 1
    with pytest.raises(ValueError, match="payloads"):
        summarize(_write_ranks(tmp_path, values), source_revision=REVISION)

    values = [_rank(index) for index in range(4)]
    values[0]["optimized_hlo_shape_mentions"] = dict(EXPECTED_SHAPE_MENTIONS)
    values[0]["optimized_hlo_shape_mentions"]["local_all_assignment_gate_dense:bf16"] = 1
    with pytest.raises(ValueError, match="HLO"):
        summarize(_write_ranks(tmp_path, values), source_revision=REVISION)
