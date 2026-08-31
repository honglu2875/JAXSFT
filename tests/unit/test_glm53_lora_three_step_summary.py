import copy
import json

import pytest

import scripts.summarize_glm53_lora_three_step as summary
from scripts.probe_glm53_lora_three_step import TRAINING_STATISTIC_NAMES
from scripts.summarize_glm53_lora_backward import (
    EXPECTED_BASE_BYTES_PER_DEVICE,
    EXPECTED_INITIALIZER_MEMORY,
    EXPECTED_LOADER_BYTES_PER_HOST,
)
from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
)
from scripts.summarize_glm53_lora_optimizer_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_EXECUTION_GATE,
)


REVISION = "543350c630da2c6e298194f8f10417e42ccbf4c0"
RESULT_HASH_BY_RANK = {
    0: "ca81c774917abe2477edaefe040e432e8b3e63fe61c2ce246dcf0b87d620d08d",
    1: "c4007df084f4059419126e09b8103d6a4b0040d3809c3a5b887897045adb6723",
    2: "e955d48fb9a587c7f8f88d1e331f92923b7effb5e7bb6669101c63edafad6eaf",
    3: "083fd01391e03a3ceac678bc8650c1fc861f78e1601bc98b23cdfa87ed0d241c",
}


def _memory_records(rank, phase):
    phase_peaks = {
        "after_checkpoint_restore": 20_601_651_712,
        "after_checkpoint_save": 20_601_651_712,
        "after_distributed_init": 13_824,
        "after_full_base_placement": 20_380_723_712,
        "after_optimizer_compile": 13_824,
        "after_step_1": 20_601_598_464,
        "after_step_2": 20_601_598_464,
        "after_step_3": 20_601_684_480,
        "after_training_state_initialization": 118_521_856,
        "after_training_state_release": 20_601_651_712,
    }
    peak = phase_peaks[phase]
    execution = phase.startswith("after_step_") or phase.startswith("after_checkpoint_")
    bytes_in_use = peak
    if phase == "after_training_state_release":
        bytes_in_use = 20_558_000_000
    return [
        {
            "device_id": rank * 4 + offset,
            "stats": {
                "bytes_limit": 33_014_407_168,
                "bytes_in_use": bytes_in_use,
                "peak_bytes_in_use": peak,
                "largest_free_block_bytes": (
                    11_615_084_032 if execution else 12_523_058_688
                ),
                "largest_alloc_size": (
                    150_994_944
                    if phase == "after_full_base_placement"
                    else 220_750_848
                ),
            },
        }
        for offset in range(4)
    ]


def _rank(rank):
    artifact = summary.EXPECTED_CHECKPOINT_ARTIFACTS[rank]
    checkpoint_identity = {
        "format_purpose": "GLM-5.3-Flash rank-4 attention-LoRA adapter-only AdamW",
        "model_repo_id": "zai-org/GLM-5.3-Flash",
        "model_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "source_revision": REVISION,
        "step": 2,
    }
    phases = (
        "after_checkpoint_restore",
        "after_checkpoint_save",
        "after_distributed_init",
        "after_full_base_placement",
        "after_optimizer_compile",
        "after_step_1",
        "after_step_2",
        "after_step_3",
        "after_training_state_initialization",
        "after_training_state_release",
    )
    return {
        "schema_version": 1,
        "test": summary.SOURCE_TEST,
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
        "optimizer": {
            "adapter_dtype": "bfloat16",
            "beta1": 0.9,
            "beta2": 0.95,
            "donated_argument_numbers": [1, 2],
            "epsilon": 1e-8,
            "learning_rate": 1e-5,
            "max_grad_norm": 1.0,
            "moment_dtype": "float32",
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
            "hostname": f"host-{rank}",
            "process_index": rank,
        },
        "compile_preflight": {
            "authorizing_evidence_sha256": summary.EXPECTED_AUTHORIZING_EVIDENCE_SHA256,
            "header_network": {
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
            },
            "checkpoint_payload_bytes_read": 0,
            "compiler_memory": copy.deepcopy(EXPECTED_COMPILER_MEMORY),
            "execution_gate": copy.deepcopy(EXPECTED_EXECUTION_GATE),
            "optimized_hlo_sha256": summary.EXPECTED_HLO_SHA256,
            "optimized_hlo_bytes": summary.EXPECTED_HLO_BYTES,
            "optimized_hlo_shape_mentions": copy.deepcopy(EXPECTED_SHAPE_MENTIONS),
        },
        "training_state": {
            "adapter_placement": {
                "a_partition_spec": [],
                "b_partition_spec": [None, "model"],
                "global_parameter_count": 10_289_152,
                "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
                "parameter_bytes_by_device": {
                    str(device): 8_705_024 for device in range(16)
                },
                "target_count": 191,
            },
            "optimizer_placement": {
                "first_moment_bytes_by_device": {
                    str(device): 17_410_048 for device in range(16)
                },
                "moment_dtype": "float32",
                "moment_global_elements_per_slot": 10_289_152,
                "moment_slot_count": 2,
                "optimizer_state_bytes_by_device": {
                    str(device): 34_820_100 for device in range(16)
                },
                "second_moment_bytes_by_device": {
                    str(device): 17_410_048 for device in range(16)
                },
                "step_bytes_by_device": {str(device): 4 for device in range(16)},
                "step_dtype": "int32",
                "step_global_elements": 1,
            },
            "adapter_initializer_compiler_memory": copy.deepcopy(EXPECTED_INITIALIZER_MEMORY),
            "optimizer_initializer_compiler_memory": copy.deepcopy(
                summary.EXPECTED_OPTIMIZER_INITIALIZER_MEMORY
            ),
            "initialization_statistic_names": list(TRAINING_STATISTIC_NAMES),
            "initialization_statistics": list(summary.EXPECTED_INITIALIZATION_STATISTICS),
            "initialization_statistics_float32_sha256": (
                summary.EXPECTED_INITIALIZATION_SHA256
            ),
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
                "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
                "global_leaf_elements_by_dtype": {
                    "bfloat16": 6_303_463_936,
                    "float32": 19_034_430,
                    "uint8": 307_023_052_800,
                },
                "global_leaf_elements_including_scale_metadata": 313_345_551_166,
                "local_bytes_by_device": {
                    str(rank * 4 + offset): EXPECTED_BASE_BYTES_PER_DEVICE
                    for offset in range(4)
                },
            },
        },
        "steps": [
            {**copy.deepcopy(record), "execute_seconds": 249.0, "diagnostics_seconds": 0.02}
            for record in summary.EXPECTED_STEP_RECORDS
        ],
        "checkpoint": {
            "schema_version": 1,
            "format": "jaxsft_rank_local_sharded_pytree_npz",
            "identity": checkpoint_identity,
            "process_index": rank,
            "process_count": 4,
            "root_keys": ["adapters", "optimizer"],
            "leaf_count": 1_147,
            "global_elements_including_replicas_once": 30_867_457,
            "global_logical_bytes_including_replicas_once": 102_891_524,
            "local_unique_shard_count": 2_866,
            "local_unique_tensor_bytes": 55_398_404,
            "local_device_resident_bytes": 174_100_496,
            "replicated_payload_sha256": summary.EXPECTED_REPLICATED_CHECKPOINT_SHA256,
            "npz_file_bytes": 56_172_246,
            "manifest_file": f"rank-{rank:03d}.json",
            "directory": "/tmp/jaxsft-glm53-g6c1-543350c/step-00000002",
            "checkpoint_step": 2,
            "frozen_base_included": False,
            "base_leaf_count": 0,
            "pre_save_statistics_float32_sha256": summary.EXPECTED_STEP_RECORDS[1][
                "training_statistics_float32_sha256"
            ],
            "restored_statistics_float32_sha256": summary.EXPECTED_STEP_RECORDS[1][
                "training_statistics_float32_sha256"
            ],
            "pre_save_and_restored_statistics_equal": True,
            **artifact,
            "restore": {
                "all_local_shards_byte_exact": True,
                "leaf_count": 1_147,
                "local_payload_sha256": artifact["local_payload_sha256"],
                "local_unique_shard_count": 2_866,
                "manifest_sha256": artifact["manifest_sha256"],
                "npz_sha256": artifact["npz_sha256"],
            },
        },
        "device_memory": {phase: _memory_records(rank, phase) for phase in phases},
        "host_memory": {"after_step_3": {"vmhwm_bytes": 19_000_000_000}},
        "shm": {"used_delta_during_load_bytes": 0, "used_delta_total_bytes": 0},
        "timing": {
            "header_seconds": 38.0,
            "compile_seconds": 627.0,
            "adapter_initializer_seconds": 25.0,
            "optimizer_initializer_seconds": 2.0,
            "load_seconds": 1_214.0,
            "checkpoint_save_seconds": 4.0,
            "checkpoint_restore_seconds": 3.0,
            "elapsed_seconds_before_shutdown": 2_724.0,
        },
    }


def _install_artifact_stub(monkeypatch):
    manifests = {
        rank: {
            "npz_sha256": artifact["npz_sha256"],
            "local_payload_sha256": artifact["local_payload_sha256"],
        }
        for rank, artifact in summary.EXPECTED_CHECKPOINT_ARTIFACTS.items()
    }
    checkpoint = {
        "global_payload_sha256": summary.EXPECTED_GLOBAL_CHECKPOINT_SHA256,
        "all_npz_member_hashes_verified": True,
        "artifact_sha256_by_process_index": {
            str(rank): {
                "manifest_sha256": artifact["manifest_sha256"],
                "npz_sha256": artifact["npz_sha256"],
                "local_payload_sha256": artifact["local_payload_sha256"],
            }
            for rank, artifact in summary.EXPECTED_CHECKPOINT_ARTIFACTS.items()
        },
    }
    monkeypatch.setattr(
        summary,
        "_validate_checkpoint_artifacts",
        lambda *args, **kwargs: (checkpoint, manifests),
    )


def _install_result_loader(monkeypatch):
    def load(path):
        value = json.loads(path.read_text())
        rank = value["runtime"]["process_index"]
        return value, RESULT_HASH_BY_RANK[rank], path.stat().st_size

    monkeypatch.setattr(summary, "_load", load)


def _write_ranks(tmp_path, values):
    paths = []
    for rank, value in enumerate(values):
        path = tmp_path / f"rank-{rank}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    return paths


def test_three_step_summary_accepts_exact_trajectory_and_checkpoint_gate(tmp_path, monkeypatch):
    _install_artifact_stub(monkeypatch)
    _install_result_loader(monkeypatch)
    result = summary.summarize(
        _write_ranks(tmp_path, [_rank(rank) for rank in range(4)]),
        [tmp_path / "unused"] * 4,
        [tmp_path / "unused"] * 4,
        source_revision=REVISION,
    )
    assert result["gate"]["g6c1_three_step_optimizer_checkpoint"] == "passed"
    assert result["gate"]["ten_step_resume_probe_authorized"] is True
    assert result["gate"]["fifty_step_probe_authorized"] is False
    assert result["trajectory"]["loss_monotonic_decrease"] is False
    assert result["memory"]["step_one_to_step_three_peak_slope_bytes"] == 86_016
    assert result["checkpoint"]["artifact_sha256_by_process_index"]["0"][
        "manifest_sha256"
    ] == summary.EXPECTED_CHECKPOINT_ARTIFACTS[0]["manifest_sha256"]


def test_three_step_summary_rejects_trajectory_drift(tmp_path, monkeypatch):
    _install_artifact_stub(monkeypatch)
    _install_result_loader(monkeypatch)
    values = [_rank(rank) for rank in range(4)]
    values[0]["steps"][2]["loss"] += 1
    with pytest.raises(ValueError, match="trajectory"):
        summary.summarize(
            _write_ranks(tmp_path, values),
            [tmp_path / "unused"] * 4,
            [tmp_path / "unused"] * 4,
            source_revision=REVISION,
        )


def test_expected_shard_contract_covers_replicated_and_model_partitioned_arrays():
    index, shape = summary._expected_shard((4, 8192), (None, "model"), 3)
    assert index == [
        {"kind": "slice", "start": None, "stop": None},
        {"kind": "slice", "start": 1536, "stop": 2048},
    ]
    assert shape == [4, 512]
    index, shape = summary._expected_shard((), (), 7)
    assert index == []
    assert shape == []
