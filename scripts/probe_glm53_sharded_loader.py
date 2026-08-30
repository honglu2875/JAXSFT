#!/usr/bin/env python3
"""Range-load one real GLM-5.3 tensor directly into final v4-32 shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from jaxsft.checkpoint import StrictHTTPRangeReader
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    SafetensorsShardHeader,
    SafetensorsTensorRange,
)


SOURCE_SHARD = "model-00032-of-00062.safetensors"
TENSOR_NAME = "model.language_model.layers.3.self_attn.q_a_proj.weight"
SCALE_NAME = TENSOR_NAME + "_scale_inv"
WEIGHT_SHAPE = (1536, 4096)
SCALE_SHAPE = (12, 32)
BLOCK_SHAPE = (128, 128)
EXPECTED_WEIGHT_SHA256 = "d79be6a957e1c23680665a68e4bbc9ffaf71a01bb7dc540e40140c6af9a3b3bc"
EXPECTED_SCALE_SHA256 = "165bb5ed26c4a904ba915d5bd22657560e019041ccb0f13868ddd811e3c429dd"
EXPECTED_FINGERPRINT = (1_028_930_362, 72, 2_258_651_919, 1_881_823_194)


def _source_url() -> str:
    return f"https://huggingface.co/{OFFICIAL_REPO_ID}/resolve/{OFFICIAL_REVISION}/{SOURCE_SHARD}"


def _initialize_distributed() -> bool:
    coordinator = os.environ.get("JAXSFT_COORDINATOR_ADDRESS")
    count = os.environ.get("JAXSFT_PROCESS_COUNT")
    process_id = os.environ.get("JAXSFT_PROCESS_ID")
    supplied = (coordinator is not None, count is not None, process_id is not None)
    if any(supplied) and not all(supplied):
        raise ValueError(
            "JAXSFT_COORDINATOR_ADDRESS, JAXSFT_PROCESS_COUNT, and JAXSFT_PROCESS_ID "
            "must be supplied together"
        )
    if not all(supplied):
        return False
    assert coordinator is not None and count is not None and process_id is not None
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=int(count),
        process_id=int(process_id),
        initialization_timeout=300,
        heartbeat_timeout_seconds=60,
    )
    return True


def _process_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, raw = line.partition(":")
        if name in {"VmRSS", "VmHWM"}:
            fields = raw.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"unexpected /proc/self/status memory line: {line!r}")
            values[name.lower() + "_bytes"] = int(fields[0]) * 1024
    if set(values) != {"vmrss_bytes", "vmhwm_bytes"}:
        raise ValueError("/proc/self/status did not expose VmRSS and VmHWM")
    return values


def _shm_usage() -> dict[str, int]:
    usage = shutil.disk_usage("/dev/shm")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _read_header(reader: StrictHTTPRangeReader) -> tuple[SafetensorsShardHeader, str]:
    header_length_bytes = reader.read(0, 7)
    header_length = int.from_bytes(header_length_bytes, "little", signed=False)
    if header_length <= 0 or header_length > 100 * 1024 * 1024:
        raise ValueError(f"unsafe remote safetensors header length: {header_length}")
    payload = reader.read(8, 7 + header_length)
    header = SafetensorsShardHeader.from_json_bytes(payload, header_length=header_length)
    return header, hashlib.sha256(payload).hexdigest()


def _normalize_slice(value: slice, size: int) -> tuple[int, int]:
    start = 0 if value.start is None else value.start
    stop = size if value.stop is None else value.stop
    if value.step not in (None, 1) or not (0 <= start < stop <= size):
        raise ValueError(f"loader requires a positive contiguous slice, got {value!r} for size {size}")
    return start, stop


def _axis0_payload_range(
    tensor: SafetensorsTensorRange,
    row_start: int,
    row_stop: int,
) -> tuple[int, int]:
    if tensor.dtype != "F8_E4M3" or tensor.shape != WEIGHT_SHAPE:
        raise ValueError(f"unexpected source tensor contract: {tensor}")
    if not (0 <= row_start < row_stop <= tensor.shape[0]):
        raise ValueError(f"invalid row interval {(row_start, row_stop)}")
    row_bytes = tensor.shape[1]
    return (
        tensor.absolute_start + row_start * row_bytes,
        tensor.absolute_start + row_stop * row_bytes - 1,
    )


def _fingerprint(weight: jax.Array) -> jax.Array:
    values = weight.reshape((-1,)).astype(jnp.uint32)
    positions = jnp.arange(values.size, dtype=jnp.uint32)
    return jnp.stack(
        (
            jnp.sum(values, dtype=jnp.uint32),
            jnp.bitwise_xor.reduce(values),
            jnp.sum(values * positions, dtype=jnp.uint32),
            jnp.sum(values * values, dtype=jnp.uint32),
        )
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    memory = {"after_distributed_init": _process_memory()}
    shm = {"before": _shm_usage()}
    reader = StrictHTTPRangeReader(
        _source_url(),
        timeout_seconds=args.timeout_seconds,
        maximum_request_bytes=args.maximum_range_bytes,
    )
    header, header_sha256 = _read_header(reader)
    weight_record = header.tensor(TENSOR_NAME)
    scale_record = header.tensor(SCALE_NAME)
    if weight_record.shape != WEIGHT_SHAPE or weight_record.dtype != "F8_E4M3":
        raise ValueError(f"unexpected weight metadata: {weight_record}")
    if scale_record.shape != SCALE_SHAPE or scale_record.dtype != "F32":
        raise ValueError(f"unexpected scale metadata: {scale_record}")
    scale_payload = reader.read(*scale_record.http_range)
    scale_sha256 = hashlib.sha256(scale_payload).hexdigest()
    if scale_sha256 != EXPECTED_SCALE_SHA256:
        raise ValueError(f"scale hash {scale_sha256} does not match {EXPECTED_SCALE_SHA256}")
    host_scale = np.frombuffer(scale_payload, dtype=np.dtype("<f4")).reshape(SCALE_SHAPE)
    if not np.isfinite(host_scale).all() or not np.all(host_scale > 0):
        raise ValueError("source scale grid must contain finite positive values")
    memory["after_header_and_scale"] = _process_memory()

    global_devices = np.asarray(jax.devices(), dtype=object)
    mesh = Mesh(global_devices, ("model",))
    weight_sharding = NamedSharding(mesh, PartitionSpec("model", None))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    indices = weight_sharding.addressable_devices_indices_map(WEIGHT_SHAPE)
    if len(indices) != jax.local_device_count():
        raise ValueError(
            f"addressable shard count {len(indices)} does not equal local devices {jax.local_device_count()}"
        )

    device_arrays: list[jax.Array] = []
    local_ranges: list[dict[str, Any]] = []
    for device, index in indices.items():
        if not isinstance(index, tuple) or len(index) != 2 or not all(
            isinstance(part, slice) for part in index
        ):
            raise ValueError(f"unexpected NamedSharding index for {device}: {index!r}")
        row_start, row_stop = _normalize_slice(index[0], WEIGHT_SHAPE[0])
        column_start, column_stop = _normalize_slice(index[1], WEIGHT_SHAPE[1])
        if (column_start, column_stop) != (0, WEIGHT_SHAPE[1]):
            raise ValueError(f"axis-0 loader received non-full column slice {index[1]!r}")
        byte_start, byte_end = _axis0_payload_range(weight_record, row_start, row_stop)
        payload = reader.read(byte_start, byte_end)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        expected_bytes = (row_stop - row_start) * WEIGHT_SHAPE[1]
        if len(payload) != expected_bytes:
            raise ValueError(f"device payload has {len(payload)} bytes, expected {expected_bytes}")
        host_array = np.frombuffer(payload, dtype=np.uint8).reshape(
            row_stop - row_start, WEIGHT_SHAPE[1]
        )
        device_array = jax.device_put(host_array, SingleDeviceSharding(device))
        device_array.block_until_ready()
        device_arrays.append(device_array)
        local_ranges.append(
            {
                "device_id": device.id,
                "device_process_index": device.process_index,
                "rows": [row_start, row_stop],
                "source_http_range_inclusive": [byte_start, byte_end],
                "bytes": len(payload),
                "sha256": payload_sha256,
            }
        )
        del host_array, payload
        gc.collect()

    weight = jax.make_array_from_single_device_arrays(WEIGHT_SHAPE, weight_sharding, device_arrays)
    weight.block_until_ready()
    scale = jax.make_array_from_callback(
        SCALE_SHAPE,
        replicated_sharding,
        lambda index: host_scale[index],
    )
    scale.block_until_ready()
    memory["after_device_placement"] = _process_memory()

    compiled_fingerprint = jax.jit(
        _fingerprint,
        in_shardings=(weight_sharding,),
        out_shardings=replicated_sharding,
    )
    fingerprint = tuple(
        int(value)
        for value in np.asarray(
            compiled_fingerprint(weight).block_until_ready()
        )
    )
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"global TPU fingerprint {fingerprint} does not match {EXPECTED_FINGERPRINT}")
    memory["after_global_fingerprint"] = _process_memory()
    shm["after"] = _shm_usage()

    local_ranges.sort(key=lambda item: item["rows"])
    addressable_shards = sorted(weight.addressable_shards, key=lambda shard: shard.device.id)
    local_device_bytes = [int(shard.data.size * shard.data.dtype.itemsize) for shard in addressable_shards]
    return {
        "schema_version": 1,
        "test": "glm53_direct_to_final_named_sharding_probe",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "source_shard": SOURCE_SHARD,
            "source_shard_total_bytes": reader.total_size_bytes,
            "source_header_length": header.header_length,
            "source_header_sha256": header_sha256,
            "tensor": TENSOR_NAME,
            "weight_shape": WEIGHT_SHAPE,
            "scale_shape": SCALE_SHAPE,
            "block_shape": BLOCK_SHAPE,
            "weight_full_sha256": EXPECTED_WEIGHT_SHA256,
            "scale_sha256": scale_sha256,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "global_device_count": jax.device_count(),
            "device_kinds": sorted({device.device_kind for device in jax.devices()}),
        },
        "sharding": {
            "mesh_shape": {"model": mesh.size},
            "weight_partition_spec": ["model", None],
            "scale_partition_spec": [],
            "global_weight_bytes": int(weight.size * weight.dtype.itemsize),
            "local_device_weight_bytes": local_device_bytes,
            "local_addressable_weight_bytes": sum(local_device_bytes),
            "local_scale_replica_bytes_per_device": int(scale.size * scale.dtype.itemsize),
            "global_fingerprint_uint32": fingerprint,
        },
        "network": {
            "request_count": len(reader.records),
            "bytes_read": reader.bytes_read,
            "largest_request_bytes": max(record.bytes_read for record in reader.records),
            "records": [
                {
                    "range_inclusive": [record.start, record.end],
                    "bytes": record.bytes_read,
                    "elapsed_seconds": record.elapsed_seconds,
                }
                for record in reader.records
            ],
            "local_weight_ranges": local_ranges,
        },
        "host_memory": memory,
        "shm": {
            **shm,
            "used_delta_bytes": shm["after"]["used_bytes"] - shm["before"]["used_bytes"],
        },
        "elapsed_seconds_before_shutdown": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-range-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")

    distributed = _initialize_distributed()
    shutdown_complete = False
    try:
        result = _run(args)
    finally:
        if distributed:
            jax.distributed.shutdown()
            shutdown_complete = True
    result["runtime"]["distributed_initialized"] = distributed
    result["runtime"]["distributed_shutdown_complete"] = shutdown_complete
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
