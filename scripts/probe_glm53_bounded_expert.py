#!/usr/bin/env python3
"""Compile bounded official-size GLM expert forward/backward cases on v4-32."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    bounded_selected_block_fp8_linear,
)


EXPERTS = 288
HIDDEN_SIZE = 4096
MOE_SIZE = 2048
TOP_K = 8
BLOCK_SHAPE = (128, 128)
GATE_SHAPE = (EXPERTS, MOE_SIZE, HIDDEN_SIZE)
DOWN_SHAPE = (EXPERTS, HIDDEN_SIZE, MOE_SIZE)
GATE_SCALE_SHAPE = (EXPERTS, MOE_SIZE // 128, HIDDEN_SIZE // 128)
DOWN_SCALE_SHAPE = (EXPERTS, HIDDEN_SIZE // 128, MOE_SIZE // 128)
RAW_FP8_VALUE = 32  # E4M3FN encoding of +0.125.
SCALE_VALUE = 0.01


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


def _local_shape(index: tuple[slice, ...], global_shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = []
    for part, size in zip(index, global_shape, strict=True):
        if not isinstance(part, slice) or part.step not in (None, 1):
            raise ValueError(f"probe requires contiguous slice indices, got {index!r}")
        start = 0 if part.start is None else part.start
        stop = size if part.stop is None else part.stop
        shape.append(stop - start)
    return tuple(shape)


def _constant_global_array(
    shape: tuple[int, ...],
    sharding: NamedSharding,
    value: int | float,
    dtype: Any,
) -> jax.Array:
    dtype = np.dtype(dtype)
    return jax.make_array_from_callback(
        shape,
        sharding,
        lambda index: np.full(_local_shape(index, shape), value, dtype=dtype),
    )


def _shape_mentions(
    hlo: str,
    *,
    token_count: int,
    selected_weight_batch_size: int,
) -> dict[str, int]:
    assignments = token_count * TOP_K
    shapes = {
        "all_assignment_gate_dense": (assignments, MOE_SIZE, HIDDEN_SIZE),
        "all_assignment_down_dense": (assignments, HIDDEN_SIZE, MOE_SIZE),
        "token_topk_gate_dense": (token_count, TOP_K, MOE_SIZE, HIDDEN_SIZE),
        "bounded_gate_dense": (selected_weight_batch_size, MOE_SIZE, HIDDEN_SIZE),
        "bounded_down_dense": (selected_weight_batch_size, HIDDEN_SIZE, MOE_SIZE),
        "local_bounded_gate_dense": (
            selected_weight_batch_size,
            MOE_SIZE // 16,
            HIDDEN_SIZE,
        ),
        "local_bounded_down_dense": (
            selected_weight_batch_size,
            HIDDEN_SIZE // 16,
            MOE_SIZE,
        ),
    }
    result: dict[str, int] = {}
    for name, shape in shapes.items():
        dimensions = ",".join(str(size) for size in shape)
        for dtype in ("u8", "f8e4m3fn", "bf16", "f32"):
            result[f"{name}:{dtype}"] = len(re.findall(rf"{dtype}\[{dimensions}\]", hlo))
    return result


def _routed_output(
    inputs: jax.Array,
    indices: jax.Array,
    gate_bits: jax.Array,
    gate_scales: jax.Array,
    up_bits: jax.Array,
    up_scales: jax.Array,
    down_bits: jax.Array,
    down_scales: jax.Array,
    *,
    selected_weight_batch_size: int,
) -> jax.Array:
    gate_kernel = BatchedBlockFP8LinearKernel(gate_bits, gate_scales)
    up_kernel = BatchedBlockFP8LinearKernel(up_bits, up_scales)
    down_kernel = BatchedBlockFP8LinearKernel(down_bits, down_scales)
    gate = bounded_selected_block_fp8_linear(
        inputs,
        indices,
        gate_kernel,
        selected_weight_batch_size=selected_weight_batch_size,
    )
    up = bounded_selected_block_fp8_linear(
        inputs,
        indices,
        up_kernel,
        selected_weight_batch_size=selected_weight_batch_size,
    )
    activated = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    routed = bounded_selected_block_fp8_linear(
        activated.reshape(-1, MOE_SIZE).astype(jnp.bfloat16),
        indices.reshape(-1, 1),
        down_kernel,
        selected_weight_batch_size=selected_weight_batch_size,
        output_dtype=jnp.float32,
    ).reshape(inputs.shape[0], TOP_K, HIDDEN_SIZE)
    weights = jnp.arange(1, TOP_K + 1, dtype=jnp.float32)
    weights /= jnp.sum(weights)
    return jnp.sum(routed * weights[None, :, None], axis=1)


def _forward_backward_statistics(
    inputs: jax.Array,
    indices: jax.Array,
    gate_bits: jax.Array,
    gate_scales: jax.Array,
    up_bits: jax.Array,
    up_scales: jax.Array,
    down_bits: jax.Array,
    down_scales: jax.Array,
    *,
    selected_weight_batch_size: int,
) -> jax.Array:
    arrays = (gate_bits, gate_scales, up_bits, up_scales, down_bits, down_scales)

    def loss_with_output(current_inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        output = _routed_output(
            current_inputs,
            indices,
            *arrays,
            selected_weight_batch_size=selected_weight_batch_size,
        ).astype(jnp.float32)
        return jnp.mean(jnp.square(output), dtype=jnp.float32), output

    (loss, output), input_gradient = jax.value_and_grad(loss_with_output, has_aux=True)(inputs)
    input_gradient = input_gradient.astype(jnp.float32)
    return jnp.stack(
        (
            loss,
            jnp.sum(output, dtype=jnp.float32),
            jnp.sum(jnp.square(output), dtype=jnp.float32),
            jnp.max(output),
            jnp.min(output),
            jnp.sum(input_gradient, dtype=jnp.float32),
            jnp.sum(jnp.square(input_gradient), dtype=jnp.float32),
            jnp.max(input_gradient),
            jnp.min(input_gradient),
        )
    )


def _indices(token_count: int) -> np.ndarray:
    base = np.asarray([0, 17, 63, 95, 127, 191, 255, 287], dtype=np.int32)
    return np.stack([np.roll(base, token % TOP_K) for token in range(token_count)])


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    host_memory = {"after_distributed_init": _process_memory()}
    shm = {"before": _shm_usage()}
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    if mesh.size != 16:
        raise ValueError("bounded official-size expert probe requires a 16-chip model mesh")
    expert_sharding = NamedSharding(mesh, PartitionSpec(None, "model", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    gate_bits = _constant_global_array(GATE_SHAPE, expert_sharding, RAW_FP8_VALUE, np.uint8)
    up_bits = _constant_global_array(GATE_SHAPE, expert_sharding, RAW_FP8_VALUE, np.uint8)
    down_bits = _constant_global_array(DOWN_SHAPE, expert_sharding, RAW_FP8_VALUE, np.uint8)
    gate_scales = _constant_global_array(GATE_SCALE_SHAPE, replicated, SCALE_VALUE, np.float32)
    up_scales = _constant_global_array(GATE_SCALE_SHAPE, replicated, SCALE_VALUE, np.float32)
    down_scales = _constant_global_array(DOWN_SCALE_SHAPE, replicated, SCALE_VALUE, np.float32)
    arrays = (gate_bits, gate_scales, up_bits, up_scales, down_bits, down_scales)
    for value in arrays:
        value.block_until_ready()
    host_memory["after_expert_bank_placement"] = _process_memory()
    device_memory = {"after_expert_bank_placement": _device_memory()}

    probe = lambda inputs, indices, *current_arrays: _forward_backward_statistics(
        inputs,
        indices,
        *current_arrays,
        selected_weight_batch_size=args.selected_weight_batch_size,
    )
    in_shardings = (
        replicated,
        replicated,
        expert_sharding,
        replicated,
        expert_sharding,
        replicated,
        expert_sharding,
        replicated,
    )
    cases = []
    for token_count in args.token_counts:
        inputs = jax.device_put(
            jnp.full((token_count, HIDDEN_SIZE), 0.01, jnp.bfloat16),
            replicated,
        )
        indices = jax.device_put(jnp.asarray(_indices(token_count)), replicated)
        compile_started = time.monotonic()
        compiled = jax.jit(
            probe,
            in_shardings=in_shardings,
            out_shardings=replicated,
        ).lower(inputs, indices, *arrays).compile()
        compile_seconds = time.monotonic() - compile_started
        execute_started = time.monotonic()
        statistics = np.asarray(
            compiled(inputs, indices, *arrays).block_until_ready(),
            dtype=np.float32,
        )
        execute_seconds = time.monotonic() - execute_started
        if not np.isfinite(statistics).all():
            raise ValueError("bounded official-size expert probe produced non-finite statistics")
        hlo = compiled.as_text()
        cases.append(
            {
                "token_count": token_count,
                "assignment_count": token_count * TOP_K,
                "selected_weight_batch_size": args.selected_weight_batch_size,
                "compiler_memory": _memory_analysis(compiled),
                "optimized_hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                "optimized_hlo_shape_mentions": _shape_mentions(
                    hlo,
                    token_count=token_count,
                    selected_weight_batch_size=args.selected_weight_batch_size,
                ),
                "statistics": statistics.tolist(),
                "statistics_float32_sha256": hashlib.sha256(
                    statistics.astype("<f4").tobytes()
                ).hexdigest(),
                "compile_seconds": compile_seconds,
                "execute_seconds": execute_seconds,
            }
        )
        del compiled
    host_memory["after_cases"] = _process_memory()
    device_memory["after_cases"] = _device_memory()
    shm["after"] = _shm_usage()
    return {
        "schema_version": 1,
        "test": "glm53_bounded_official_expert_forward_backward_v4_probe",
        "source_revision": args.source_revision,
        "contract": {
            "experts": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_SIZE,
            "top_k": TOP_K,
            "block_shape": BLOCK_SHAPE,
            "source_fp8_bytes": int(2 * np.prod(GATE_SHAPE) + np.prod(DOWN_SHAPE)),
            "source_fp8_bytes_per_device": int(
                (2 * np.prod(GATE_SHAPE) + np.prod(DOWN_SHAPE)) // 16
            ),
            "selected_weight_batch_size": args.selected_weight_batch_size,
            "maximum_dequantized_weight_bytes_per_projection_global": int(
                args.selected_weight_batch_size * MOE_SIZE * HIDDEN_SIZE * 2
            ),
            "weight_workspace_independent_of_token_count": True,
            "backward_chunk_rematerialized": True,
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
            "precision": str(jax.lax.Precision.HIGHEST),
            "mesh_shape": {"model": mesh.size},
            "expert_partition_spec": [None, "model", None],
        },
        "cases": cases,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--token-counts", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--selected-weight-batch-size", type=int, default=1)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if (
        not args.token_counts
        or any(token_count <= 0 for token_count in args.token_counts)
        or tuple(sorted(set(args.token_counts))) != tuple(args.token_counts)
    ):
        raise ValueError("token-counts must be positive, unique, and increasing")
    if args.selected_weight_batch_size <= 0:
        raise ValueError("selected-weight-batch-size must be positive")
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
