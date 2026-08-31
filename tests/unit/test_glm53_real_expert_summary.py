import copy
import json

import pytest

from scripts.summarize_glm53_real_expert import (
    EXPECTED_HLO_SHA256,
    EXPECTED_OUTPUT_SHA256,
    EXPECTED_SOURCE_FINGERPRINTS,
    EXPERT_BYTES_PER_HOST,
    LOADER_BYTES_PER_HOST,
    summarize,
)


SOURCE_REVISION = "1" * 40


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _rank(process_index):
    device_records = [
        {
            "device_id": process_index * 4 + local_index,
            "stats": {
                "bytes_in_use": 455_357_952,
                "peak_bytes_in_use": 459_828_736,
            },
        }
        for local_index in range(4)
    ]
    return {
        "schema_version": 1,
        "test": "glm53_real_checkpoint_expert_streaming_v4_probe",
        "source_revision": SOURCE_REVISION,
        "selection": {
            "expert_indices": [0, 17, 63, 95, 127, 191, 255, 287],
            "experts": 288,
            "global_source_fp8_bytes": 7_247_757_312,
            "hidden_size": 4096,
            "layer": 3,
            "moe_intermediate_size": 2048,
            "top_k": 8,
        },
        "runtime": {
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
        },
        "selected_source_fingerprints": EXPECTED_SOURCE_FINGERPRINTS,
        "output": {
            "finite": True,
            "statistics_float32_sha256": EXPECTED_OUTPUT_SHA256,
        },
        "compiler_memory": {
            "argument_size_in_bytes": 455_361_024,
            "temp_size_in_bytes": 75_884_544,
        },
        "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
        "device_memory": {
            "after_real_expert_placement": device_records,
            "after_execute": device_records,
        },
        "host_memory": {"after_execute": {"vmhwm_bytes": 5_600_000_000}},
        "shm": {"used_delta_bytes": 0},
        "elapsed_seconds_before_shutdown": 40.0,
    }


def _oracle():
    return {
        "schema_version": 1,
        "test": "glm53_real_expert_transformers_cpu_oracle",
        "source_revision": SOURCE_REVISION,
        "selected_source_fingerprints": EXPECTED_SOURCE_FINGERPRINTS,
        "runtime": {
            "torch_version": "2.10.0+cpu",
            "transformers_version": "5.16.1",
        },
        "comparison": {
            "passed": True,
            "maximum_relative_l2_tolerance": 0.02,
            "maximum_absolute_tolerance": 2e-5,
            "relative_l2": 0.015,
            "maximum_absolute": 1.5e-5,
            "transformers_cpu_statistics": [1.0, 2.0, 3.0, 4.0],
            "tpu_statistics": [1.0, 2.0, 3.0, 4.0],
        },
    }


def test_real_expert_summary_requires_exact_tpu_and_cpu_source_fingerprints(tmp_path):
    rank_paths = [_write(tmp_path / f"rank-{index}.json", _rank(index)) for index in range(4)]
    oracle_path = _write(tmp_path / "oracle.json", _oracle())
    result = summarize(rank_paths, oracle_path, source_revision=SOURCE_REVISION)
    assert result["gate"]["g5c_real_expert_streaming"] == "passed"
    assert result["streaming"]["source_payload_downloaded_exactly_once_across_slice"] is True

    tampered = copy.deepcopy(_oracle())
    tampered["selected_source_fingerprints"]["gate"]["weight_bits_uint32"][0] += 1
    oracle_path = _write(tmp_path / "oracle-tampered.json", tampered)
    with pytest.raises(ValueError, match="fingerprints"):
        summarize(rank_paths, oracle_path, source_revision=SOURCE_REVISION)
