import json

import pytest

from scripts.summarize_glm53_loader_probe import (
    EXPECTED_FINGERPRINT,
    EXPECTED_MAXIMUM_DEVICE_RANGE,
    EXPECTED_STREAMED_PAYLOAD_PER_HOST,
    EXPECTED_TEXT_BASE_PER_DEVICE,
    EXPECTED_WEIGHT_RANGE,
    summarize,
)


SOURCE_REVISION = "1" * 40


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _header():
    placement = {
        "estimated_text_base_bytes_per_device": EXPECTED_TEXT_BASE_PER_DEVICE,
        "maximum_single_device_range_bytes": EXPECTED_MAXIMUM_DEVICE_RANGE,
        "estimated_streamed_payload_bytes_per_host": EXPECTED_STREAMED_PAYLOAD_PER_HOST,
        "unsupported_tensor_count": 0,
    }
    return {
        "schema_version": 1,
        "test": "glm53_all_shard_header_and_placement_audit",
        "source_revision": SOURCE_REVISION,
        "model": {
            "repo_id": "zai-org/GLM-5.3-Flash",
            "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
            "tensor_count": 76_108,
            "shard_count": 62,
            "payload_bytes": 328_326_771_576,
        },
        "header_audit": {
            "all_index_tensors_covered_once": True,
            "network_bytes": 10_684_096,
        },
        "fp8_pair_audit": {
            "all_fp8_weights_have_exact_f32_scale_grids": True,
            "fp8_weight_count": 37_338,
        },
        "scope_audit": {"unknown_tensor_count": 0},
        "placement_plan": placement,
    }


def _loader(process_index):
    local_ranges = []
    for device_id in range(process_index * 4, process_index * 4 + 4):
        start = EXPECTED_WEIGHT_RANGE[0] + device_id * 393_216
        local_ranges.append(
            {
                "device_id": device_id,
                "device_process_index": process_index,
                "rows": [device_id * 96, (device_id + 1) * 96],
                "source_http_range_inclusive": [start, start + 393_216 - 1],
                "bytes": 393_216,
                "sha256": f"{device_id:064x}",
            }
        )
    return {
        "schema_version": 1,
        "test": "glm53_direct_to_final_named_sharding_probe",
        "source_revision": SOURCE_REVISION,
        "model": {
            "repo_id": "zai-org/GLM-5.3-Flash",
            "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
            "source_shard": "model-00032-of-00062.safetensors",
            "source_shard_total_bytes": 5_363_915_232,
            "source_header_length": 177_792,
            "source_header_sha256": (
                "d2a826379afa4a6ffb5dc5d6df2aef080afdaed432fdcebccc19360597fd285e"
            ),
            "tensor": "model.language_model.layers.3.self_attn.q_a_proj.weight",
            "weight_shape": [1536, 4096],
            "scale_shape": [12, 32],
            "block_shape": [128, 128],
            "weight_full_sha256": (
                "d79be6a957e1c23680665a68e4bbc9ffaf71a01bb7dc540e40140c6af9a3b3bc"
            ),
            "scale_sha256": (
                "165bb5ed26c4a904ba915d5bd22657560e019041ccb0f13868ddd811e3c429dd"
            ),
        },
        "runtime": {
            "hostname": f"host-{process_index}",
            "backend": "tpu",
            "process_index": process_index,
            "process_count": 4,
            "local_device_count": 4,
            "global_device_count": 16,
            "distributed_initialized": True,
            "distributed_shutdown_complete": True,
        },
        "sharding": {
            "mesh_shape": {"model": 16},
            "weight_partition_spec": ["model", None],
            "global_fingerprint_uint32": EXPECTED_FINGERPRINT,
            "local_addressable_weight_bytes": 1_572_864,
            "local_device_weight_bytes": [393_216] * 4,
        },
        "network": {
            "request_count": 7,
            "bytes_read": 1_752_200,
            "largest_request_bytes": 393_216,
            "local_weight_ranges": local_ranges,
        },
        "host_memory": {"after_global_fingerprint": {"vmhwm_bytes": 5_000_000_000}},
        "shm": {"used_delta_bytes": 28_672},
    }


def test_loader_summary_requires_exact_four_host_range_coverage(tmp_path):
    header_path = _write(tmp_path / "header.json", _header())
    loader_paths = [
        _write(tmp_path / f"host-{process}.json", _loader(process)) for process in range(4)
    ]
    result = summarize(header_path, loader_paths, source_revision=SOURCE_REVISION)
    assert result["gate"]["g4_direct_loader"] == "passed"
    assert result["gate"]["full_model_runnable"] is False
    assert result["header_audit"]["file"] == "header.json"
    assert result["sample_loader"]["weight_payload_bytes_downloaded_across_hosts"] == 6_291_456

    tampered = _loader(3)
    tampered["network"]["local_weight_ranges"][0]["rows"] = [1153, 1249]
    loader_paths[3] = _write(tmp_path / "host-3-tampered.json", tampered)
    with pytest.raises(ValueError, match="contiguous partition"):
        summarize(header_path, loader_paths, source_revision=SOURCE_REVISION)
