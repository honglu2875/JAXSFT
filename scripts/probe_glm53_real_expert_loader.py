#!/usr/bin/env python3
"""Stream and execute one real GLM-5.3 expert layer on a v4-32."""

from __future__ import annotations

import argparse
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxsft.models.glm5_3_flash import (
    BatchedBlockFP8LinearKernel,
    Glm53TextConfig,
    SafetensorsIndex,
    selected_block_fp8_linear,
)
from jaxsft.models.glm5_3_streaming import Glm53StreamingLoader


EXPERTS = 288
HIDDEN_SIZE = 4096
MOE_SIZE = 2048
TOP_K = 8
EXPERT_INDICES = (0, 17, 63, 95, 127, 191, 255, 287)


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


def _expert_forward(
    inputs: jax.Array,
    indices: jax.Array,
    gate_bits: jax.Array,
    gate_scales: jax.Array,
    up_bits: jax.Array,
    up_scales: jax.Array,
    down_bits: jax.Array,
    down_scales: jax.Array,
) -> jax.Array:
    gate_kernel = BatchedBlockFP8LinearKernel(gate_bits, gate_scales)
    up_kernel = BatchedBlockFP8LinearKernel(up_bits, up_scales)
    down_kernel = BatchedBlockFP8LinearKernel(down_bits, down_scales)
    gate = selected_block_fp8_linear(inputs, indices, gate_kernel)
    up = selected_block_fp8_linear(inputs, indices, up_kernel)
    activated = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    routed = selected_block_fp8_linear(
        activated.reshape(-1, MOE_SIZE).astype(jnp.bfloat16),
        indices.reshape(-1, 1),
        down_kernel,
        output_dtype=jnp.float32,
    ).reshape(inputs.shape[0], TOP_K, HIDDEN_SIZE)
    weights = jnp.arange(1, TOP_K + 1, dtype=jnp.float32)
    weights /= jnp.sum(weights)
    output = jnp.sum(routed * weights[None, :, None], axis=1)
    return jnp.stack(
        (
            jnp.sum(output, dtype=jnp.float32),
            jnp.sum(jnp.square(output), dtype=jnp.float32),
            jnp.max(output),
            jnp.min(output),
        )
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    host_memory = {"after_distributed_init": _process_memory()}
    shm = {"before": _shm_usage()}
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    if (
        config.hidden_size != HIDDEN_SIZE
        or config.moe_intermediate_size != MOE_SIZE
        or config.n_routed_experts != EXPERTS
        or config.num_experts_per_tok != TOP_K
        or config.mlp_layer_types[args.layer] != "sparse"
    ):
        raise ValueError("real expert probe requires the pinned official sparse-layer dimensions")

    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    target_names = ("experts_gate", "experts_up", "experts_down")
    loader = Glm53StreamingLoader(
        config,
        index,
        mesh,
        connections_per_shard=args.connections_per_shard,
        worker_threads=args.worker_threads,
        timeout_seconds=args.timeout_seconds,
    )
    groups = {
        path: specs
        for path, specs in loader.target_groups()
        if path[:3] == ("layers", args.layer, "mlp") and path[-1] in target_names
    }
    expected_paths = {
        ("layers", args.layer, "mlp", target_name) for target_name in target_names
    }
    if set(groups) != expected_paths:
        raise ValueError(f"expert source schema did not expose the three required targets: {set(groups)}")

    try:
        kernels = {
            path[-1]: loader.load_target(groups[path]).value for path in sorted(groups)
        }
        host_memory["after_real_expert_placement"] = _process_memory()
        device_memory = {"after_real_expert_placement": _device_memory()}
        network = loader.network_summary()
        loader.release_host_cache()
        host_memory["after_host_cache_release"] = _process_memory()
    finally:
        loader.close()
    gate = kernels["experts_gate"]
    up = kernels["experts_up"]
    down = kernels["experts_down"]
    if not all(isinstance(kernel, BatchedBlockFP8LinearKernel) for kernel in (gate, up, down)):
        raise TypeError("real checkpoint expert targets were not block-FP8 kernels")
    arrays = (
        gate.weight_bits,
        gate.weight_scale_inv,
        up.weight_bits,
        up.weight_scale_inv,
        down.weight_bits,
        down.weight_scale_inv,
    )
    for value in arrays:
        value.block_until_ready()
    expert_sharding = arrays[0].sharding
    if not all(arrays[index].sharding == expert_sharding for index in (0, 2, 4)):
        raise ValueError("real expert weight banks do not share one final sharding")
    if not all(arrays[index].sharding == replicated for index in (1, 3, 5)):
        raise ValueError("real expert scale grids are not replicated")

    inputs = jax.device_put(jnp.full((1, HIDDEN_SIZE), 0.01, jnp.bfloat16), replicated)
    indices = jax.device_put(jnp.asarray([EXPERT_INDICES], jnp.int32), replicated)
    compiled = jax.jit(
        _expert_forward,
        in_shardings=(
            replicated,
            replicated,
            expert_sharding,
            replicated,
            expert_sharding,
            replicated,
            expert_sharding,
            replicated,
        ),
        out_shardings=replicated,
    ).lower(inputs, indices, *arrays).compile()
    host_memory["after_compile"] = _process_memory()
    statistics = np.asarray(compiled(inputs, indices, *arrays).block_until_ready(), dtype=np.float32)
    if not np.isfinite(statistics).all():
        raise ValueError("real expert probe produced non-finite statistics")
    host_memory["after_execute"] = _process_memory()
    device_memory["after_execute"] = _device_memory()
    optimized_hlo = compiled.as_text()
    shm["after"] = _shm_usage()
    return {
        "schema_version": 1,
        "test": "glm53_real_checkpoint_expert_streaming_v4_probe",
        "source_revision": args.source_revision,
        "selection": {
            "layer": args.layer,
            "experts": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_SIZE,
            "top_k": TOP_K,
            "expert_indices": list(EXPERT_INDICES),
            "global_source_fp8_bytes": int(
                2 * EXPERTS * MOE_SIZE * HIDDEN_SIZE
                + EXPERTS * HIDDEN_SIZE * MOE_SIZE
            ),
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
        "loader": network,
        "compiler_memory": _memory_analysis(compiled),
        "optimized_hlo_sha256": hashlib.sha256(optimized_hlo.encode()).hexdigest(),
        "output": {
            "statistics": statistics.tolist(),
            "statistics_float32_sha256": hashlib.sha256(
                statistics.astype("<f4").tobytes()
            ).hexdigest(),
            "finite": True,
        },
        "host_memory": host_memory,
        "device_memory": device_memory,
        "shm": {
            **shm,
            "used_delta_bytes": shm["after"]["used_bytes"] - shm["before"]["used_bytes"],
        },
        "elapsed_seconds_before_shutdown": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--layer", type=int, default=3)
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
