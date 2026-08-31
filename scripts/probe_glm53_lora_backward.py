#!/usr/bin/env python3
"""Execute one smallest full-checkpoint GLM-5.3 attention-LoRA backward on v4-32."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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

from jaxsft.lora import LoRAConfig, init_lora_adapters
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    Glm53TextConfig,
    SafetensorsIndex,
    attention_lora_target_paths,
)
from jaxsft.models.glm5_3_streaming import Glm53StreamingLoader
from scripts.probe_glm53_lora_backward_compile import (
    BATCH_SIZE,
    HBM_LIMIT_BYTES_PER_DEVICE,
    SEQUENCE_LENGTH,
    _abstract_attention_adapters,
    _adapter_placement,
    _execution_gate,
    _initialize_distributed,
    _loss_and_adapter_gradients,
    _memory_analysis,
    _process_memory,
    _shape_mentions,
    _shm_usage,
)


EXPECTED_BASE_BYTES_PER_DEVICE = 20_234_287_352
INPUT_IDS = ((1, 2),)
LOSS_WEIGHTS = ((0.0, 1.0),)
GRADIENT_STATISTIC_NAMES = (
    "all_finite",
    "a_l2_squared",
    "b_l2_squared",
    "a_l1",
    "b_l1",
    "a_max_abs",
    "b_max_abs",
    "a_nonzero_elements",
    "b_nonzero_elements",
    "a_nonzero_leaves",
    "b_nonzero_leaves",
)
INITIALIZATION_STATISTIC_NAMES = (
    "all_finite",
    "a_l2_squared",
    "b_l2_squared",
    "a_max_abs",
    "b_max_abs",
    "a_nonzero_elements",
    "b_nonzero_elements",
)


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


def _parameter_placement(params: Mapping[str, Any]) -> dict[str, Any]:
    leaves = jax.tree.leaves(params)
    if not leaves or any(not isinstance(leaf, jax.Array) for leaf in leaves):
        raise ValueError("complete executable parameter tree must contain only JAX array leaves")
    per_device = Counter()
    dtype_global_elements = Counter()
    global_elements = 0
    for leaf in leaves:
        global_elements += int(leaf.size)
        dtype_global_elements[str(leaf.dtype)] += int(leaf.size)
        for shard in leaf.addressable_shards:
            per_device[shard.device.id] += int(shard.data.size * shard.data.dtype.itemsize)
    local_device_ids = {device.id for device in jax.local_devices()}
    if set(per_device) != local_device_ids or set(per_device.values()) != {
        EXPECTED_BASE_BYTES_PER_DEVICE
    }:
        raise ValueError("complete base placement does not match the header-audited bytes per chip")
    return {
        "array_leaf_count": len(leaves),
        "global_leaf_elements_including_scale_metadata": global_elements,
        "global_leaf_elements_by_dtype": dict(sorted(dtype_global_elements.items())),
        "local_bytes_by_device": {str(key): value for key, value in sorted(per_device.items())},
        "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
        "all_local_devices_match_header_audit": True,
    }


def _tree_scalar_statistics(
    adapters: Mapping[str, Mapping[str, jax.Array]],
    *,
    include_l1_and_leaf_counts: bool,
) -> jax.Array:
    a_values = [pair["a"] for _, pair in sorted(adapters.items())]
    b_values = [pair["b"] for _, pair in sorted(adapters.items())]

    def squared_sum(values: list[jax.Array]) -> jax.Array:
        return sum(
            (jnp.sum(jnp.square(value.astype(jnp.float32)), dtype=jnp.float32) for value in values),
            start=jnp.asarray(0.0, jnp.float32),
        )

    def max_abs(values: list[jax.Array]) -> jax.Array:
        return jnp.max(jnp.stack([jnp.max(jnp.abs(value.astype(jnp.float32))) for value in values]))

    def nonzero_elements(values: list[jax.Array]) -> jax.Array:
        return sum(
            (jnp.count_nonzero(value) for value in values),
            start=jnp.asarray(0, jnp.int32),
        ).astype(jnp.float32)

    all_values = a_values + b_values
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in all_values])).astype(
        jnp.float32
    )
    a_squared = squared_sum(a_values)
    b_squared = squared_sum(b_values)
    a_max = max_abs(a_values)
    b_max = max_abs(b_values)
    a_nonzero = nonzero_elements(a_values)
    b_nonzero = nonzero_elements(b_values)
    if not include_l1_and_leaf_counts:
        return jnp.stack((finite, a_squared, b_squared, a_max, b_max, a_nonzero, b_nonzero))

    def l1(values: list[jax.Array]) -> jax.Array:
        return sum(
            (jnp.sum(jnp.abs(value.astype(jnp.float32)), dtype=jnp.float32) for value in values),
            start=jnp.asarray(0.0, jnp.float32),
        )

    def nonzero_leaves(values: list[jax.Array]) -> jax.Array:
        return jnp.sum(
            jnp.stack([jnp.any(value != 0) for value in values]).astype(jnp.float32),
            dtype=jnp.float32,
        )

    return jnp.stack(
        (
            finite,
            a_squared,
            b_squared,
            l1(a_values),
            l1(b_values),
            a_max,
            b_max,
            a_nonzero,
            b_nonzero,
            nonzero_leaves(a_values),
            nonzero_leaves(b_values),
        )
    )


def _initialize_adapters(
    abstract_params: Mapping[str, Any],
    config: Glm53TextConfig,
    mesh: Mesh,
    *,
    rank: int,
    seed: int,
) -> tuple[dict[str, dict[str, jax.Array]], dict[str, int | None], float]:
    lora_config = LoRAConfig(rank=rank, alpha=float(rank))
    targets = attention_lora_target_paths(config)
    abstract_adapters = _abstract_attention_adapters(abstract_params, config, mesh, rank=rank)
    adapter_shardings = jax.tree.map(lambda value: value.sharding, abstract_adapters)
    initializer = jax.jit(
        lambda: init_lora_adapters(
            jax.random.key(seed),
            abstract_params,
            targets,
            eligible_paths=targets,
            config=lora_config,
            dtype=jnp.bfloat16,
        ),
        out_shardings=adapter_shardings,
    )
    started = time.monotonic()
    lowered = initializer.lower()
    compiled = lowered.compile()
    memory = _memory_analysis(compiled)
    del lowered
    adapters = compiled()
    jax.block_until_ready(adapters)
    elapsed = time.monotonic() - started
    return adapters, memory, elapsed


def _progress(event: Mapping[str, Any]) -> None:
    kind = event.get("event")
    should_print = kind == "header_ready" and event.get("prepared_shards") in {
        1,
        10,
        20,
        30,
        40,
        50,
        60,
        62,
    }
    if kind == "target_ready":
        index = event.get("target_index")
        should_print = index in {1, 1372} or (isinstance(index, int) and index % 25 == 0)
    if should_print:
        print(json.dumps(dict(event), sort_keys=True), file=sys.stderr, flush=True)


def _require_execution_headroom(records: list[dict[str, Any]], compiler_memory: Mapping[str, Any]) -> None:
    required_free_block = (
        int(compiler_memory["temp_size_in_bytes"])
        + int(compiler_memory["output_size_in_bytes"])
        + 1024**3
    )
    if len(records) != 4:
        raise ValueError("base placement did not report four local TPU memory records")
    for record in records:
        stats = record.get("stats", {})
        if (
            stats.get("bytes_limit") != HBM_LIMIT_BYTES_PER_DEVICE
            or not isinstance(stats.get("largest_free_block_bytes"), int)
            or stats["largest_free_block_bytes"] < required_free_block
        ):
            raise ValueError("placed base does not leave the compiler-predicted backward safety margin")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    host_memory = {"after_distributed_init": _process_memory()}
    device_memory = {"after_distributed_init": _device_memory()}
    shm = {"before": _shm_usage()}
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if config_sha256 != OFFICIAL_CHECKPOINT.config_sha256:
        raise ValueError("config SHA-256 does not match the pinned GLM-5.3 checkpoint")
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    if mesh.size != 16:
        raise ValueError("full GLM-5.3 LoRA backward requires a 16-chip model mesh")
    replicated = NamedSharding(mesh, PartitionSpec())
    lora_config = LoRAConfig(rank=args.rank, alpha=float(args.rank))
    loader = Glm53StreamingLoader(
        config,
        index,
        mesh,
        connections_per_shard=args.connections_per_shard,
        worker_threads=args.worker_threads,
        timeout_seconds=args.timeout_seconds,
        progress=_progress,
    )
    try:
        header_started = time.monotonic()
        abstract_params = loader.abstract_parameters()
        header_seconds = time.monotonic() - header_started
        header_network = loader.network_summary()
        if set(header_network["requests_by_category"]) != {"header"}:
            raise ValueError("abstract compile preflight fetched checkpoint tensor payloads")

        abstract_adapters = _abstract_attention_adapters(
            abstract_params,
            config,
            mesh,
            rank=args.rank,
        )
        adapter_shardings = jax.tree.map(lambda value: value.sharding, abstract_adapters)
        abstract_input_ids = jax.ShapeDtypeStruct(
            (BATCH_SIZE, SEQUENCE_LENGTH),
            jnp.int32,
            sharding=replicated,
        )
        abstract_loss_weights = jax.ShapeDtypeStruct(
            (BATCH_SIZE, SEQUENCE_LENGTH),
            jnp.float32,
            sharding=replicated,
        )
        compile_started = time.monotonic()
        lowered = jax.jit(
            lambda current_params, current_adapters, tokens, weights: _loss_and_adapter_gradients(
                current_params,
                current_adapters,
                tokens,
                weights,
                config=config,
                lora_config=lora_config,
            ),
            out_shardings=(replicated, adapter_shardings),
        ).lower(
            abstract_params,
            abstract_adapters,
            abstract_input_ids,
            abstract_loss_weights,
        )
        compiled = lowered.compile()
        compile_seconds = time.monotonic() - compile_started
        del lowered
        compiler_memory = _memory_analysis(compiled)
        execution_gate = _execution_gate(compiler_memory)
        if execution_gate["full_checkpoint_execution_authorized"] is not True:
            raise ValueError("compile-only safety gate rejected full-checkpoint execution")
        hlo = compiled.as_text()
        hlo_sha256 = hashlib.sha256(hlo.encode()).hexdigest()
        hlo_bytes = len(hlo.encode())
        shape_mentions = _shape_mentions(hlo, config)
        for prefix in (
            "all_assignment_gate_dense",
            "all_assignment_down_dense",
            "local_all_assignment_gate_dense",
            "local_all_assignment_down_dense",
            "token_topk_gate_dense",
            "token_topk_down_dense",
            "local_token_topk_gate_dense",
            "local_token_topk_down_dense",
        ):
            if any(
                shape_mentions[f"{prefix}:{dtype}"]
                for dtype in ("u8", "f8e4m3fn", "bf16", "f32")
            ):
                raise ValueError(f"optimized HLO contains unbounded selected-weight shape {prefix}")
        del hlo
        gc.collect()
        host_memory["after_backward_compile"] = _process_memory()
        device_memory["after_backward_compile"] = _device_memory()

        adapters, initializer_memory, initializer_seconds = _initialize_adapters(
            abstract_params,
            config,
            mesh,
            rank=args.rank,
            seed=args.seed,
        )
        initialization = np.asarray(
            jax.jit(
                lambda values: _tree_scalar_statistics(
                    values,
                    include_l1_and_leaf_counts=False,
                ),
                out_shardings=replicated,
            )(adapters).block_until_ready(),
            dtype=np.float32,
        )
        if (
            not np.isfinite(initialization).all()
            or initialization[0] != 1
            or initialization[1] <= 0
            or initialization[2] != 0
            or initialization[3] <= 0
            or initialization[4] != 0
            or initialization[5] <= 0
            or initialization[6] != 0
        ):
            raise ValueError("sharded LoRA initialization violated finite-A/zero-B invariants")
        actual_adapter_placement = _adapter_placement(adapters)
        host_memory["after_adapter_initialization"] = _process_memory()
        device_memory["after_adapter_initialization"] = _device_memory()

        load_started = time.monotonic()
        params = loader.load_parameters()
        load_seconds = time.monotonic() - load_started
        loader_network = loader.network_summary()
        parameter_placement = _parameter_placement(params)
        loader.release_host_cache()
        host_memory["after_full_base_placement"] = _process_memory()
        device_memory["after_full_base_placement"] = _device_memory()
    finally:
        loader.close()
    shm["after_load"] = _shm_usage()
    _require_execution_headroom(device_memory["after_full_base_placement"], compiler_memory)

    input_ids = jax.device_put(jnp.asarray(INPUT_IDS, jnp.int32), replicated)
    loss_weights = jax.device_put(jnp.asarray(LOSS_WEIGHTS, jnp.float32), replicated)
    execute_started = time.monotonic()
    loss, gradients = compiled(params, adapters, input_ids, loss_weights)
    jax.block_until_ready((loss, gradients))
    execute_seconds = time.monotonic() - execute_started
    loss_value = float(np.asarray(loss, dtype=np.float32))
    host_memory["after_backward"] = _process_memory()
    device_memory["after_backward"] = _device_memory()

    statistics_started = time.monotonic()
    gradient_statistics = np.asarray(
        jax.jit(
            lambda values: _tree_scalar_statistics(values, include_l1_and_leaf_counts=True),
            out_shardings=replicated,
        )(gradients).block_until_ready(),
        dtype=np.float32,
    )
    statistics_seconds = time.monotonic() - statistics_started
    if (
        not np.isfinite(loss_value)
        or loss_value <= 0
        or not np.isfinite(gradient_statistics).all()
        or gradient_statistics[0] != 1
        or gradient_statistics[1] != 0
        or gradient_statistics[2] <= 0
        or gradient_statistics[3] != 0
        or gradient_statistics[4] <= 0
        or gradient_statistics[5] != 0
        or gradient_statistics[6] <= 0
        or gradient_statistics[7] != 0
        or gradient_statistics[8] <= 0
        or gradient_statistics[9] != 0
        or gradient_statistics[10] <= 0
    ):
        raise ValueError("full attention-LoRA backward produced invalid zero-B gradient diagnostics")
    host_memory["after_gradient_statistics"] = _process_memory()
    device_memory["after_gradient_statistics"] = _device_memory()
    shm["after"] = _shm_usage()
    return {
        "schema_version": 1,
        "test": "glm53_complete_text_rank4_attention_lora_backward_v4_probe",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "config_sha256": config_sha256,
            "index_sha256": index.sha256,
            "num_hidden_layers": config.num_hidden_layers,
            "rank": args.rank,
            "alpha": lora_config.alpha,
            "seed": args.seed,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "input_ids": [list(row) for row in INPUT_IDS],
            "loss_weights": [list(row) for row in LOSS_WEIGHTS],
            "loss_token_count": 1,
            "attention_lora_target_count": len(adapters),
            "rematerialize_each_decoder_layer": True,
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
        "compile_preflight": {
            "header_network": header_network,
            "header_seconds": header_seconds,
            "checkpoint_payload_bytes_read": 0,
            "compiler_memory": compiler_memory,
            "execution_gate": execution_gate,
            "optimized_hlo_sha256": hlo_sha256,
            "optimized_hlo_bytes": hlo_bytes,
            "optimized_hlo_shape_mentions": shape_mentions,
            "compile_seconds": compile_seconds,
        },
        "adapter": {
            "placement": actual_adapter_placement,
            "initializer_compiler_memory": initializer_memory,
            "initializer_seconds": initializer_seconds,
            "initialization_statistic_names": list(INITIALIZATION_STATISTIC_NAMES),
            "initialization_statistics": initialization.tolist(),
        },
        "loader": {
            **loader_network,
            "load_seconds": load_seconds,
            "parameter_placement": parameter_placement,
        },
        "output": {
            "loss": loss_value,
            "loss_float32_sha256": hashlib.sha256(
                np.asarray([loss_value], dtype="<f4").tobytes()
            ).hexdigest(),
            "gradient_statistic_names": list(GRADIENT_STATISTIC_NAMES),
            "gradient_statistics": gradient_statistics.tolist(),
            "gradient_statistics_float32_sha256": hashlib.sha256(
                gradient_statistics.astype("<f4").tobytes()
            ).hexdigest(),
            "finite_loss_and_gradients": True,
            "zero_initialized_b_gives_exact_zero_a_gradients": True,
            "b_gradients_nonzero": True,
        },
        "timing": {
            "header_seconds": header_seconds,
            "compile_seconds": compile_seconds,
            "initializer_seconds": initializer_seconds,
            "load_seconds": load_seconds,
            "backward_execute_seconds": execute_seconds,
            "gradient_statistics_seconds": statistics_seconds,
            "elapsed_seconds_before_shutdown": time.monotonic() - started,
        },
        "host_memory": host_memory,
        "device_memory": device_memory,
        "shm": {
            **shm,
            "used_delta_during_load_bytes": shm["after_load"]["used_bytes"]
            - shm["before"]["used_bytes"],
            "used_delta_total_bytes": shm["after"]["used_bytes"] - shm["before"]["used_bytes"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--connections-per-shard", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if args.rank <= 0 or args.seed < 0:
        raise ValueError("rank must be positive and seed must be non-negative")
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
