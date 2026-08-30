#!/usr/bin/env python3
"""Validate four G4 rank results and emit one provenance-locked summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FINGERPRINT = [1_028_930_362, 72, 2_258_651_919, 1_881_823_194]
EXPECTED_WEIGHT_RANGE = [2_941_704_672, 2_947_996_127]
EXPECTED_WEIGHT_BYTES = 6_291_456
EXPECTED_TEXT_BASE_PER_DEVICE = 20_234_287_352
EXPECTED_MAXIMUM_DEVICE_RANGE = 79_298_560
EXPECTED_STREAMED_PAYLOAD_PER_HOST = 80_128_653_560
MAXIMUM_PROCESS_VMHWM = 6 * 1024**3
MAXIMUM_SHM_DELTA = 1024**2


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


def summarize(
    header_audit_path: Path,
    loader_paths: list[Path],
    *,
    source_revision: str,
) -> dict[str, Any]:
    if len(loader_paths) != 4:
        raise ValueError("exactly four loader results are required")
    header, header_sha256 = _load(header_audit_path)
    loaders_with_hashes = [_load(path) for path in loader_paths]
    loaders = [value for value, _ in loaders_with_hashes]
    if header.get("source_revision") != source_revision:
        raise ValueError("header audit source revision does not match the loader source revision")
    if header.get("test") != "glm53_all_shard_header_and_placement_audit":
        raise ValueError("unexpected header audit identity")
    model = header.get("model", {})
    if (
        model.get("repo_id") != "zai-org/GLM-5.3-Flash"
        or model.get("revision") != "04c4e9e95c5da8862dced7e5056455116f83a7e0"
        or model.get("tensor_count") != 76_108
        or model.get("shard_count") != 62
        or model.get("payload_bytes") != 328_326_771_576
    ):
        raise ValueError("header audit model contract drifted")
    if not header.get("header_audit", {}).get("all_index_tensors_covered_once"):
        raise ValueError("header audit did not cover every indexed tensor once")
    if not header.get("fp8_pair_audit", {}).get("all_fp8_weights_have_exact_f32_scale_grids"):
        raise ValueError("header audit did not pair every FP8 weight with an exact scale grid")
    if header.get("scope_audit", {}).get("unknown_tensor_count") != 0:
        raise ValueError("header audit contains tensors with no inclusion/exclusion rule")
    placement = header.get("placement_plan", {})
    if placement.get("unsupported_tensor_count") != 0:
        raise ValueError("header placement plan contains unsupported text tensors")
    if placement.get("estimated_text_base_bytes_per_device") != EXPECTED_TEXT_BASE_PER_DEVICE:
        raise ValueError("header placement per-device byte count drifted")
    if placement.get("maximum_single_device_range_bytes") != EXPECTED_MAXIMUM_DEVICE_RANGE:
        raise ValueError("header placement maximum range drifted")
    if placement.get("estimated_streamed_payload_bytes_per_host") != EXPECTED_STREAMED_PAYLOAD_PER_HOST:
        raise ValueError("header placement per-host streamed byte count drifted")

    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    device_ids: set[int] = set()
    local_ranges: list[dict[str, Any]] = []
    for loader in loaders:
        if loader.get("source_revision") != source_revision:
            raise ValueError("a loader result has a mismatched source revision")
        if loader.get("test") != "glm53_direct_to_final_named_sharding_probe":
            raise ValueError("unexpected loader result identity")
        model = loader.get("model", {})
        if (
            model.get("repo_id") != "zai-org/GLM-5.3-Flash"
            or model.get("revision") != "04c4e9e95c5da8862dced7e5056455116f83a7e0"
            or model.get("source_shard") != "model-00032-of-00062.safetensors"
            or model.get("source_shard_total_bytes") != 5_363_915_232
            or model.get("source_header_length") != 177_792
            or model.get("source_header_sha256")
            != "d2a826379afa4a6ffb5dc5d6df2aef080afdaed432fdcebccc19360597fd285e"
            or model.get("tensor") != "model.language_model.layers.3.self_attn.q_a_proj.weight"
            or model.get("weight_shape") != [1536, 4096]
            or model.get("scale_shape") != [12, 32]
            or model.get("block_shape") != [128, 128]
            or model.get("weight_full_sha256")
            != "d79be6a957e1c23680665a68e4bbc9ffaf71a01bb7dc540e40140c6af9a3b3bc"
            or model.get("scale_sha256")
            != "165bb5ed26c4a904ba915d5bd22657560e019041ccb0f13868ddd811e3c429dd"
        ):
            raise ValueError("loader model/range source contract drifted")
        runtime = loader.get("runtime", {})
        if not runtime.get("distributed_initialized") or not runtime.get("distributed_shutdown_complete"):
            raise ValueError("loader distributed lifecycle was incomplete")
        if (
            runtime.get("backend") != "tpu"
            or runtime.get("process_count") != 4
            or runtime.get("local_device_count") != 4
            or runtime.get("global_device_count") != 16
        ):
            raise ValueError("loader topology was not four-process v4-32")
        hostnames.add(runtime["hostname"])
        process_indexes.add(runtime["process_index"])
        sharding = loader.get("sharding", {})
        if sharding.get("mesh_shape") != {"model": 16}:
            raise ValueError("loader did not construct the final 16-way model mesh")
        if sharding.get("weight_partition_spec") != ["model", None]:
            raise ValueError("loader weight PartitionSpec drifted")
        if sharding.get("global_fingerprint_uint32") != EXPECTED_FINGERPRINT:
            raise ValueError("loader global TPU fingerprint mismatch")
        if sharding.get("local_addressable_weight_bytes") != EXPECTED_WEIGHT_BYTES // 4:
            raise ValueError("loader host owns something other than one quarter of the weight")
        if sharding.get("local_device_weight_bytes") != [EXPECTED_WEIGHT_BYTES // 16] * 4:
            raise ValueError("loader device shard sizes are not exact sixteenths")
        network = loader.get("network", {})
        if (
            network.get("request_count") != 7
            or network.get("bytes_read") != 1_752_200
            or network.get("largest_request_bytes") != EXPECTED_WEIGHT_BYTES // 16
        ):
            raise ValueError("loader network bounds drifted")
        host_memory = loader.get("host_memory", {}).get("after_global_fingerprint", {})
        vmhwm = host_memory.get("vmhwm_bytes")
        if not isinstance(vmhwm, int) or not (0 < vmhwm <= MAXIMUM_PROCESS_VMHWM):
            raise ValueError("loader process memory high-water exceeds the G4 bound")
        shm_delta = loader.get("shm", {}).get("used_delta_bytes")
        if not isinstance(shm_delta, int) or not (0 <= shm_delta <= MAXIMUM_SHM_DELTA):
            raise ValueError("loader /dev/shm delta exceeds the G4 bound")
        for item in network.get("local_weight_ranges", []):
            if item.get("device_process_index") != runtime.get("process_index"):
                raise ValueError("device shard was attributed to the wrong runtime process")
            if item["device_id"] in device_ids:
                raise ValueError(f"device {item['device_id']} appears in multiple loader results")
            device_ids.add(item["device_id"])
            local_ranges.append(item)
    if len(hostnames) != 4 or process_indexes != set(range(4)) or device_ids != set(range(16)):
        raise ValueError("loader results do not cover four unique hosts, ranks, and all 16 devices")

    local_ranges.sort(key=lambda item: item["rows"])
    expected_row = 0
    expected_byte = EXPECTED_WEIGHT_RANGE[0]
    for item in local_ranges:
        row_start, row_stop = item["rows"]
        byte_start, byte_end = item["source_http_range_inclusive"]
        if row_start != expected_row or row_stop - row_start != 96:
            raise ValueError("device row shards are not an exact contiguous partition")
        if byte_start != expected_byte or byte_end - byte_start + 1 != EXPECTED_WEIGHT_BYTES // 16:
            raise ValueError("source byte ranges are not an exact contiguous partition")
        expected_row = row_stop
        expected_byte = byte_end + 1
    if expected_row != 1536 or expected_byte - 1 != EXPECTED_WEIGHT_RANGE[1]:
        raise ValueError("device ranges do not cover the complete source tensor")

    host_memory_high_water = max(
        loader["host_memory"]["after_global_fingerprint"]["vmhwm_bytes"] for loader in loaders
    )
    maximum_shm_delta = max(loader["shm"]["used_delta_bytes"] for loader in loaders)
    range_digest = hashlib.sha256(
        "".join(item["sha256"] + "\n" for item in local_ranges).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "test": "glm53_g4_direct_sharded_loader_acceptance",
        "source_revision": source_revision,
        "header_audit": {
            "file": header_audit_path.name,
            "sha256": header_sha256,
            "tensor_count": header["model"]["tensor_count"],
            "shard_count": header["model"]["shard_count"],
            "header_network_bytes": header["header_audit"]["network_bytes"],
            "fp8_weight_scale_pairs": header["fp8_pair_audit"]["fp8_weight_count"],
            "scope": header["scope_audit"],
        },
        "placement_plan": placement,
        "sample_loader": {
            "tensor": loaders[0]["model"]["tensor"],
            "weight_bytes": EXPECTED_WEIGHT_BYTES,
            "weight_full_sha256": loaders[0]["model"]["weight_full_sha256"],
            "scale_sha256": loaders[0]["model"]["scale_sha256"],
            "global_fingerprint_uint32": EXPECTED_FINGERPRINT,
            "process_count": 4,
            "global_device_count": 16,
            "device_range_count": len(local_ranges),
            "device_ranges_cover_source_exactly_once": True,
            "device_range_sha256_manifest": range_digest,
            "weight_payload_bytes_downloaded_across_hosts": sum(
                item["bytes"] for item in local_ranges
            ),
            "total_network_bytes_across_hosts": sum(
                loader["network"]["bytes_read"] for loader in loaders
            ),
            "largest_http_range_bytes": max(
                loader["network"]["largest_request_bytes"] for loader in loaders
            ),
            "maximum_process_vmhwm_bytes": host_memory_high_water,
            "maximum_shm_used_delta_bytes": maximum_shm_delta,
            "no_full_weight_replica_on_host_or_device": True,
            "all_global_fingerprints_equal": True,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256": [digest for _, digest in loaders_with_hashes],
        "gate": {
            "g4_direct_loader": "passed",
            "full_model_runnable": False,
            "remaining_blockers": [
                "The full frozen text checkpoint has not been range-loaded into one model PyTree.",
                "The sharded block-FP8 contraction has not been integrated through every architecture path.",
                "Whole-model HBM, compilation time, and forward numerics remain unmeasured."
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header-audit", type=Path, required=True)
    parser.add_argument("--loader-results", type=Path, nargs=4, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.header_audit,
        list(args.loader_results),
        source_revision=args.source_revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
