#!/usr/bin/env python3
"""Probe one real GLM-5.3 FP8 weight without downloading its source shard.

The two inputs are exact HTTP range responses for the pinned q_a projection
and its scale grid.  This script performs strict size/hash checks, optionally
uses Transformers' own dequantization operator as the CPU oracle, compiles the
tiled JAX contraction, and emits compiler-memory/HLO evidence as JSON.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxsft.models.glm5_3_flash import block_fp8_linear, dequantize_block_fp8


TENSOR_NAME = "model.language_model.layers.3.self_attn.q_a_proj.weight"
SOURCE_SHARD = "model-00032-of-00062.safetensors"
WEIGHT_SHAPE = (1536, 4096)
SCALE_SHAPE = (12, 32)
BLOCK_SHAPE = (128, 128)
WEIGHT_HTTP_RANGE = (2_941_704_672, 2_947_996_127)
SCALE_HTTP_RANGE = (883_976, 885_511)
WEIGHT_SHA256 = "d79be6a957e1c23680665a68e4bbc9ffaf71a01bb7dc540e40140c6af9a3b3bc"
SCALE_SHA256 = "165bb5ed26c4a904ba915d5bd22657560e019041ccb0f13868ddd811e3c429dd"
RNG_SEED = 20_260_830


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
    atexit.register(jax.distributed.shutdown)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exact(path: Path, *, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} has {actual_bytes} bytes, expected exactly {expected_bytes}")
    value = np.fromfile(path, dtype=dtype)
    return value.reshape(shape)


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    difference = actual - expected
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
    }


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


def _shape_mentions(hlo: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for dtype in ("bf16", "f32", "f8e4m3fn", "u8"):
        # Optimized HLO may insert layout annotations after the dimensions.
        pattern = rf"{dtype}\[1536,4096\]"
        result[f"{dtype}_full_weight"] = len(re.findall(pattern, hlo))
    return result


def _transformers_oracle(inputs: np.ndarray, bits: np.ndarray, scales: np.ndarray) -> np.ndarray:
    import torch
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    quantized = torch.from_numpy(np.array(bits, copy=True)).view(torch.float8_e4m3fn)
    torch_scales = torch.from_numpy(np.array(scales, copy=True))
    dense = Fp8Dequantize(None)._dequantize_one(
        quantized,
        torch_scales,
        output_dtype=torch.float32,
    )
    return (torch.from_numpy(inputs) @ dense.T).numpy(force=True)


def _dense_jax_linear(
    inputs: jax.Array,
    bits: jax.Array,
    scales: jax.Array,
    *,
    compute_dtype: Any,
) -> jax.Array:
    dense = dequantize_block_fp8(bits, scales, block_shape=BLOCK_SHAPE, dtype=compute_dtype)
    return jax.lax.dot_general(
        inputs.astype(compute_dtype),
        dense,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-bits", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-array", type=Path)
    parser.add_argument("--hlo", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--transformers-oracle", action="store_true")
    parser.add_argument("--compile-dense-reference", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    distributed_initialized = _initialize_distributed()

    bits = _load_exact(args.weight_bits, dtype=np.uint8, shape=WEIGHT_SHAPE)
    scales = _load_exact(args.scales, dtype=np.dtype("<f4"), shape=SCALE_SHAPE)
    weight_sha256 = _sha256(args.weight_bits)
    scale_sha256 = _sha256(args.scales)
    if weight_sha256 != WEIGHT_SHA256 or scale_sha256 != SCALE_SHA256:
        raise ValueError(
            "probe payload hash mismatch: "
            f"weight={weight_sha256}, expected {WEIGHT_SHA256}; "
            f"scale={scale_sha256}, expected {SCALE_SHA256}"
        )
    if not np.isfinite(scales).all() or not np.all(scales > 0):
        raise ValueError("weight_scale_inv must contain finite positive values")
    inputs = np.random.default_rng(RNG_SEED).standard_normal(
        (args.batch_size, WEIGHT_SHAPE[1]), dtype=np.float32
    )
    compute_dtype = jnp.float32 if args.compute_dtype == "float32" else jnp.bfloat16

    jax_inputs = jnp.asarray(inputs)
    jax_bits = jnp.asarray(bits)
    jax_scales = jnp.asarray(scales)
    tiled = jax.jit(
        lambda x, q, s: block_fp8_linear(
            x,
            q,
            s,
            block_shape=BLOCK_SHAPE,
            compute_dtype=compute_dtype,
            output_dtype=jnp.float32,
        )
    )
    lowered = tiled.lower(jax_inputs, jax_bits, jax_scales)
    compiled = lowered.compile()
    actual = np.asarray(compiled(jax_inputs, jax_bits, jax_scales).block_until_ready())
    optimized_hlo = compiled.as_text()
    canonical_output = np.asarray(actual, dtype=np.dtype("<f4"), order="C")
    if args.output_array:
        args.output_array.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_array, canonical_output, allow_pickle=False)
    if args.hlo:
        args.hlo.parent.mkdir(parents=True, exist_ok=True)
        args.hlo.write_text(optimized_hlo)

    result: dict[str, Any] = {
        "probe_contract": {
            "tensor": TENSOR_NAME,
            "source_shard": SOURCE_SHARD,
            "weight_shape": WEIGHT_SHAPE,
            "scale_shape": SCALE_SHAPE,
            "block_shape": BLOCK_SHAPE,
            "weight_http_range_inclusive": WEIGHT_HTTP_RANGE,
            "scale_http_range_inclusive": SCALE_HTTP_RANGE,
            "weight_bytes": args.weight_bits.stat().st_size,
            "scale_bytes": args.scales.stat().st_size,
            "weight_sha256": weight_sha256,
            "scale_sha256": scale_sha256,
        },
        "runtime": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "distributed_initialized": distributed_initialized,
            "local_device_count": jax.local_device_count(),
            "global_device_count": jax.device_count(),
            "device_kinds": sorted({device.device_kind for device in jax.devices()}),
            "compute_dtype": args.compute_dtype,
            "batch_size": args.batch_size,
            "precision": str(jax.lax.Precision.HIGHEST),
        },
        "source_scales": {"minimum": float(scales.min()), "maximum": float(scales.max())},
        "tiled_compiler_memory": _memory_analysis(compiled),
        "tiled_optimized_hlo_sha256": hashlib.sha256(optimized_hlo.encode()).hexdigest(),
        "tiled_hlo_full_weight_shape_mentions": _shape_mentions(optimized_hlo),
        "output": {
            "finite": bool(np.isfinite(actual).all()),
            "float32_sha256": hashlib.sha256(canonical_output.tobytes()).hexdigest(),
            "minimum": float(actual.min()),
            "maximum": float(actual.max()),
            "l2": float(np.linalg.norm(actual.astype(np.float64))),
        },
    }

    if args.transformers_oracle:
        reference = _transformers_oracle(inputs, bits, scales)
        result["transformers_float32_error"] = _error(actual, reference)

    if args.compile_dense_reference:
        dense = jax.jit(lambda x, q, s: _dense_jax_linear(x, q, s, compute_dtype=compute_dtype))
        dense_compiled = dense.lower(jax_inputs, jax_bits, jax_scales).compile()
        dense_output = np.asarray(dense_compiled(jax_inputs, jax_bits, jax_scales).block_until_ready())
        dense_hlo = dense_compiled.as_text()
        result["dense_reference"] = {
            "error_from_tiled": _error(actual, dense_output),
            "compiler_memory": _memory_analysis(dense_compiled),
            "hlo_full_weight_shape_mentions": _shape_mentions(dense_hlo),
        }

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
