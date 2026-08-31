import copy
import json

import pytest

from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
)
from scripts.summarize_glm53_lora_optimizer_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_EXECUTION_GATE,
    EXPECTED_HLO_BYTES,
    EXPECTED_HLO_SHA256,
    SOURCE_TEST,
    summarize,
)


REVISION = "088c24a4232306606a0801675a11b70492e36ebd"


def _per_device(value):
    return {str(device): value for device in range(16)}


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
        "optimizer": {
            "adapter_dtype": "bfloat16",
            "beta1": 0.9,
            "beta2": 0.95,
            "donated_argument_numbers": [1, 2],
            "epsilon": 1e-8,
            "learning_rate": 1e-5,
            "max_grad_norm": 1.0,
            "name": "AdamW",
            "weight_decay": 0.1,
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
            "header_seconds": 38.0,
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
            "parameter_bytes_by_device": _per_device(8_705_024),
            "target_count": 191,
        },
        "optimizer_placement": {
            "first_moment_bytes_by_device": _per_device(17_410_048),
            "moment_dtype": "float32",
            "moment_global_elements_per_slot": 10_289_152,
            "moment_slot_count": 2,
            "optimizer_state_bytes_by_device": _per_device(34_820_100),
            "second_moment_bytes_by_device": _per_device(17_410_048),
            "step_bytes_by_device": _per_device(4),
            "step_dtype": "int32",
            "step_global_elements": 1,
        },
        "compiler_memory": copy.deepcopy(EXPECTED_COMPILER_MEMORY),
        "execution_gate": copy.deepcopy(EXPECTED_EXECUTION_GATE),
        "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
        "optimized_hlo_bytes": EXPECTED_HLO_BYTES,
        "optimized_hlo_shape_mentions": copy.deepcopy(EXPECTED_SHAPE_MENTIONS),
        "host_memory": {"after_compile": {"vmhwm_bytes": 19_003_092_992}},
        "shm": {"used_delta_bytes": 0},
        "timing": {
            "compile_seconds": 622.0,
            "elapsed_seconds_before_shutdown": 681.0,
        },
    }


def _write_ranks(tmp_path, values):
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f"rank-{index}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    return paths


def test_optimizer_compile_summary_accepts_exact_four_rank_gate(tmp_path):
    result = summarize(
        _write_ranks(tmp_path, [_rank(index) for index in range(4)]),
        source_revision=REVISION,
    )
    assert result["gate"]["g6c0_full_attention_lora_optimizer_compile"] == "passed"
    assert result["gate"]["full_checkpoint_three_step_execution_authorized"] is True
    assert result["gate"]["full_model_optimizer_update_proven"] is False
    assert result["optimizer"]["optimizer_state_bytes_per_device"] == 34_820_100
    assert result["compilation"]["alias_bytes_not_subtracted_from_conservative_bound"] is True


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value["header_only_loader"].__setitem__(
                "checkpoint_payload_bytes_read", 1
            ),
            "payloads",
        ),
        (
            lambda value: value["optimizer_placement"][
                "second_moment_bytes_by_device"
            ].__setitem__("0", 1),
            "placement",
        ),
        (
            lambda value: value["optimized_hlo_shape_mentions"].__setitem__(
                "local_all_assignment_gate_dense:bf16", 1
            ),
            "HLO",
        ),
        (
            lambda value: value["execution_gate"].__setitem__(
                "full_checkpoint_optimizer_execution_authorized", False
            ),
            "safety",
        ),
    ],
)
def test_optimizer_compile_summary_rejects_evidence_drift(tmp_path, mutation, error):
    values = [_rank(index) for index in range(4)]
    mutation(values[0])
    with pytest.raises(ValueError, match=error):
        summarize(_write_ranks(tmp_path, values), source_revision=REVISION)
