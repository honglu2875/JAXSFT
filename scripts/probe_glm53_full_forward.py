#!/usr/bin/env python3
"""Stream the complete GLM-5.3 text base and run a one-token v4-32 forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxsft.models.glm5_3_flash import Glm53TextConfig, SafetensorsIndex, forward
from jaxsft.models.glm5_3_streaming import Glm53StreamingLoader


EXPECTED_BASE_BYTES_PER_DEVICE = 20_234_287_352
INPUT_TOKEN_ID = 1
SELECTED_LOGIT_IDS = (0, 1, 2, 42, 1024, 8192, 65536, 131072, 154420)


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
                raise ValueError(f"unexpected process memory line: {line!r}")
            values[name.lower() + "_bytes"] = int(fields[0]) * 1024
    if set(values) != {"vmrss_bytes", "vmhwm_bytes"}:
        raise ValueError("/proc/self/status did not expose VmRSS and VmHWM")
    return values


def _shm_usage() -> dict[str, int]:
    usage = shutil.disk_usage("/dev/shm")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _device_memory() -> list[dict[str, Any]]:
    results = []
    for device in jax.local_devices():
        raw = device.memory_stats() or {}
        stats = {
            key: value
            for key, value in raw.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        results.append({"device_id": device.id, "process_index": device.process_index, "stats": stats})
    return results


def _memory_analysis(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {name: getattr(analysis, name, None) for name in names}


def _parameter_placement(params: Mapping[str, Any]) -> dict[str, Any]:
    leaves = jax.tree.leaves(params)
    if not leaves or any(not isinstance(leaf, jax.Array) for leaf in leaves):
        raise ValueError("complete executable parameter tree must contain only JAX array leaves")
    per_device = Counter()
    local_leaf_shards = 0
    dtype_global_elements = Counter()
    global_elements = 0
    for leaf in leaves:
        elements = int(leaf.size)
        global_elements += elements
        dtype_global_elements[str(leaf.dtype)] += elements
        for shard in leaf.addressable_shards:
            shard_bytes = int(shard.data.size * shard.data.dtype.itemsize)
            per_device[shard.device.id] += shard_bytes
            local_leaf_shards += 1
    local_device_ids = {device.id for device in jax.local_devices()}
    if set(per_device) != local_device_ids:
        raise ValueError("parameter placement does not cover every local device exactly")
    if set(per_device.values()) != {EXPECTED_BASE_BYTES_PER_DEVICE}:
        raise ValueError(
            f"placed base bytes per local device are {dict(per_device)}, "
            f"expected {EXPECTED_BASE_BYTES_PER_DEVICE}"
        )
    return {
        "array_leaf_count": len(leaves),
        "local_leaf_shard_count": local_leaf_shards,
        "global_leaf_elements_including_scale_metadata": global_elements,
        "global_leaf_elements_by_dtype": dict(sorted(dtype_global_elements.items())),
        "local_bytes_by_device": {str(key): value for key, value in sorted(per_device.items())},
        "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
        "all_local_devices_match_header_audit": True,
    }


def _forward_statistics(
    params: Mapping[str, Any],
    config: Glm53TextConfig,
    input_ids: jax.Array,
) -> jax.Array:
    logits = forward(params, config, input_ids, remat=False).astype(jnp.float32)
    last = logits[:, -1, :]
    selected = last[:, jnp.asarray(SELECTED_LOGIT_IDS, jnp.int32)].reshape(-1)
    summary = jnp.stack(
        (
            jnp.sum(last, dtype=jnp.float32),
            jnp.sum(jnp.square(last), dtype=jnp.float32),
            jnp.max(last),
            jnp.min(last),
            jnp.mean(last, dtype=jnp.float32),
        )
    )
    return jnp.concatenate((summary, selected))


def _progress(event: Mapping[str, Any]) -> None:
    kind = event.get("event")
    should_print = kind == "header_ready" and (
        event.get("prepared_shards") in {1, 10, 20, 30, 40, 50, 60, 62}
    )
    if kind == "target_ready":
        index = event.get("target_index")
        should_print = index in {1, 1372} or (isinstance(index, int) and index % 25 == 0)
    if should_print:
        print(json.dumps(dict(event), sort_keys=True), file=sys.stderr, flush=True)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    host_memory = {"after_distributed_init": _process_memory()}
    shm = {"before": _shm_usage()}
    device_memory = {"after_distributed_init": _device_memory()}
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    if mesh.size != 16:
        raise ValueError("full GLM-5.3 forward requires a 16-chip model mesh")
    replicated = NamedSharding(mesh, PartitionSpec())

    loader = Glm53StreamingLoader(
        config,
        index,
        mesh,
        connections_per_shard=args.connections_per_shard,
        worker_threads=args.worker_threads,
        timeout_seconds=args.timeout_seconds,
        progress=_progress,
    )
    load_started = time.monotonic()
    try:
        params = loader.load_parameters()
        load_seconds = time.monotonic() - load_started
        host_memory["after_full_base_placement"] = _process_memory()
        device_memory["after_full_base_placement"] = _device_memory()
        loader_network = loader.network_summary()
        parameter_placement = _parameter_placement(params)
        loader.release_host_cache()
        host_memory["after_host_cache_release"] = _process_memory()
    finally:
        loader.close()
    shm["after_load"] = _shm_usage()

    input_ids = jax.device_put(jnp.asarray([[INPUT_TOKEN_ID]], jnp.int32), replicated)
    lowered = jax.jit(
        lambda current_params, tokens: _forward_statistics(current_params, config, tokens),
        out_shardings=replicated,
    ).lower(params, input_ids)
    compile_started = time.monotonic()
    compiled = lowered.compile()
    compile_seconds = time.monotonic() - compile_started
    del lowered
    host_memory["after_compile"] = _process_memory()
    device_memory["after_compile"] = _device_memory()

    execute_started = time.monotonic()
    first = np.asarray(compiled(params, input_ids).block_until_ready(), dtype=np.float32)
    first_seconds = time.monotonic() - execute_started
    host_memory["after_first_execute"] = _process_memory()
    device_memory["after_first_execute"] = _device_memory()
    execute_started = time.monotonic()
    second = np.asarray(compiled(params, input_ids).block_until_ready(), dtype=np.float32)
    second_seconds = time.monotonic() - execute_started
    host_memory["after_second_execute"] = _process_memory()
    device_memory["after_second_execute"] = _device_memory()
    if not np.isfinite(first).all() or not np.array_equal(first, second):
        raise ValueError("full frozen forward was non-finite or non-deterministic")
    optimized_hlo = compiled.as_text()
    shm["after"] = _shm_usage()
    output_sha256 = hashlib.sha256(first.astype("<f4").tobytes()).hexdigest()
    return {
        "schema_version": 1,
        "test": "glm53_complete_text_streaming_one_token_v4_forward",
        "source_revision": args.source_revision,
        "model": {
            "config_path_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "index_sha256": index.sha256,
            "num_hidden_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
            "input_token_id": INPUT_TOKEN_ID,
            "selected_logit_ids": list(SELECTED_LOGIT_IDS),
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
            "mesh_shape": {"model": mesh.size},
            "precision": str(jax.lax.Precision.HIGHEST),
        },
        "loader": {
            **loader_network,
            "load_seconds": load_seconds,
            "parameter_placement": parameter_placement,
        },
        "compiler_memory": _memory_analysis(compiled),
        "optimized_hlo_sha256": hashlib.sha256(optimized_hlo.encode()).hexdigest(),
        "output": {
            "summary_names": [
                "logits_sum",
                "logits_square_sum",
                "logits_max",
                "logits_min",
                "logits_mean",
                *[f"logit_token_{token_id}" for token_id in SELECTED_LOGIT_IDS],
            ],
            "statistics": first.tolist(),
            "statistics_float32_sha256": output_sha256,
            "finite": True,
            "two_executions_bitwise_equal": True,
        },
        "timing": {
            "load_seconds": load_seconds,
            "compile_seconds": compile_seconds,
            "first_execute_seconds": first_seconds,
            "second_execute_seconds": second_seconds,
            "elapsed_seconds_before_shutdown": time.monotonic() - started,
        },
        "host_memory": host_memory,
        "device_memory": device_memory,
        "shm": {
            **shm,
            "used_delta_during_load_bytes": (
                shm["after_load"]["used_bytes"] - shm["before"]["used_bytes"]
            ),
            "used_delta_total_bytes": shm["after"]["used_bytes"] - shm["before"]["used_bytes"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--connections-per-shard", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
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
