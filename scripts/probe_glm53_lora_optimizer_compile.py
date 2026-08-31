#!/usr/bin/env python3
"""Compile a donated full GLM-5.3 attention-LoRA AdamW step from headers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxsft.lora import LoRAConfig
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    Glm53TextConfig,
    SafetensorsIndex,
    attention_lora_parameter_count,
)
from jaxsft.models.glm5_3_streaming import Glm53StreamingLoader
from jaxsft.optim import AdamWHyperparameters, AdamWState, adamw_update
from scripts.probe_glm53_lora_backward_compile import (
    BATCH_SIZE,
    EXECUTION_SAFETY_MARGIN_BYTES_PER_DEVICE,
    HBM_LIMIT_BYTES_PER_DEVICE,
    SEQUENCE_LENGTH,
    _abstract_attention_adapters,
    _adapter_placement,
    _initialize_distributed,
    _loss_and_adapter_gradients,
    _memory_analysis,
    _process_memory,
    _progress,
    _shape_mentions,
    _shm_usage,
)


DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_HYPERPARAMETERS = AdamWHyperparameters(
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    weight_decay=0.1,
    max_grad_norm=1.0,
)


def _abstract_adamw_state(
    adapters: Mapping[str, Mapping[str, jax.ShapeDtypeStruct]],
    replicated: NamedSharding,
) -> AdamWState:
    def moment(value: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(value.shape, jnp.float32, sharding=value.sharding)

    return AdamWState(
        jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated),
        jax.tree.map(moment, adapters),
        jax.tree.map(moment, adapters),
    )


def _local_bytes_by_device(tree: object) -> tuple[dict[str, int], Counter[str], int]:
    per_device = Counter()
    global_elements_by_dtype: Counter[str] = Counter()
    global_elements = 0
    for value in jax.tree.leaves(tree):
        global_elements += int(value.size)
        global_elements_by_dtype[str(value.dtype)] += int(value.size)
        itemsize = np.dtype(value.dtype).itemsize
        for device, index in value.sharding.devices_indices_map(value.shape).items():
            local_elements = 1
            for part, size in zip(index, value.shape, strict=True):
                if not isinstance(part, slice) or part.step not in (None, 1):
                    raise ValueError("optimizer state contains a non-contiguous shard")
                start = 0 if part.start is None else part.start
                stop = size if part.stop is None else part.stop
                local_elements *= stop - start
            per_device[str(device.id)] += local_elements * itemsize
    return dict(sorted(per_device.items())), global_elements_by_dtype, global_elements


def _optimizer_placement(state: AdamWState) -> dict[str, Any]:
    first_bytes, first_dtypes, first_elements = _local_bytes_by_device(state.first_moment)
    second_bytes, second_dtypes, second_elements = _local_bytes_by_device(state.second_moment)
    step_bytes, step_dtypes, step_elements = _local_bytes_by_device(state.step)
    device_ids = {str(device.id) for device in jax.devices()}
    if (
        set(first_bytes) != device_ids
        or set(second_bytes) != device_ids
        or set(step_bytes) != device_ids
        or first_dtypes != Counter({"float32": first_elements})
        or second_dtypes != Counter({"float32": second_elements})
        or step_dtypes != Counter({"int32": step_elements})
        or first_elements != second_elements
        or step_elements != 1
    ):
        raise ValueError("abstract AdamW state violates its FP32-moment/replicated-step contract")
    total_bytes = {
        device_id: first_bytes[device_id] + second_bytes[device_id] + step_bytes[device_id]
        for device_id in sorted(device_ids, key=int)
    }
    return {
        "moment_global_elements_per_slot": first_elements,
        "moment_slot_count": 2,
        "moment_dtype": "float32",
        "step_global_elements": step_elements,
        "step_dtype": "int32",
        "first_moment_bytes_by_device": first_bytes,
        "second_moment_bytes_by_device": second_bytes,
        "step_bytes_by_device": step_bytes,
        "optimizer_state_bytes_by_device": total_bytes,
    }


def _loss_and_adamw_step(
    params: Mapping[str, Any],
    adapters: Mapping[str, Mapping[str, jax.Array]],
    optimizer: AdamWState,
    input_ids: jax.Array,
    loss_weights: jax.Array,
    *,
    config: Glm53TextConfig,
    lora_config: LoRAConfig,
    learning_rate: float,
    hyperparameters: AdamWHyperparameters,
) -> tuple[jax.Array, object, AdamWState, jax.Array]:
    loss, gradients = _loss_and_adapter_gradients(
        params,
        adapters,
        input_ids,
        loss_weights,
        config=config,
        lora_config=lora_config,
    )
    updated, optimizer, gradient_norm = adamw_update(
        adapters,
        gradients,
        optimizer,
        learning_rate=learning_rate,
        hyperparameters=hyperparameters,
    )
    return loss, updated, optimizer, gradient_norm


def _optimizer_execution_gate(memory: Mapping[str, int | None]) -> dict[str, Any]:
    required = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    if any(not isinstance(memory.get(name), int) for name in required):
        raise ValueError("compiler memory analysis omitted a required device byte count")
    # Deliberately retain both argument and output bytes even when XLA reports
    # donation aliases. This is a conservative authorization bound.
    working_set = sum(int(memory[name]) for name in required)
    limit_with_margin = HBM_LIMIT_BYTES_PER_DEVICE - EXECUTION_SAFETY_MARGIN_BYTES_PER_DEVICE
    return {
        "compiler_working_set_upper_bound_bytes_per_device": working_set,
        "measured_hbm_limit_bytes_per_device": HBM_LIMIT_BYTES_PER_DEVICE,
        "required_safety_margin_bytes_per_device": EXECUTION_SAFETY_MARGIN_BYTES_PER_DEVICE,
        "headroom_before_safety_margin_bytes_per_device": HBM_LIMIT_BYTES_PER_DEVICE - working_set,
        "full_checkpoint_optimizer_execution_authorized": working_set <= limit_with_margin,
        "alias_bytes_not_subtracted_from_conservative_bound": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    shm_before = _shm_usage()
    host_memory = {"after_distributed_init": _process_memory()}
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if config_sha256 != OFFICIAL_CHECKPOINT.config_sha256:
        raise ValueError("config SHA-256 does not match the pinned GLM-5.3 checkpoint")
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    if mesh.size != 16:
        raise ValueError("full GLM-5.3 LoRA optimizer compile requires a 16-chip model mesh")
    replicated = NamedSharding(mesh, PartitionSpec())
    loader = Glm53StreamingLoader(
        config,
        index,
        mesh,
        connections_per_shard=args.connections_per_shard,
        timeout_seconds=args.timeout_seconds,
        progress=_progress,
    )
    header_started = time.monotonic()
    try:
        params = loader.abstract_parameters()
        header_seconds = time.monotonic() - header_started
        network = loader.network_summary()
    finally:
        loader.close()
    if set(network["requests_by_category"]) != {"header"}:
        raise ValueError("header-only optimizer preflight fetched checkpoint tensor payloads")
    host_memory["after_abstract_tree"] = _process_memory()

    lora_config = LoRAConfig(rank=args.rank, alpha=float(args.rank))
    adapters = _abstract_attention_adapters(params, config, mesh, rank=args.rank)
    adapter_placement = _adapter_placement(adapters)
    if adapter_placement["global_parameter_count"] != attention_lora_parameter_count(
        config, rank=args.rank
    ):
        raise ValueError("abstract adapter count disagrees with the architecture contract")
    optimizer = _abstract_adamw_state(adapters, replicated)
    optimizer_placement = _optimizer_placement(optimizer)
    input_ids = jax.ShapeDtypeStruct(
        (BATCH_SIZE, SEQUENCE_LENGTH), jnp.int32, sharding=replicated
    )
    loss_weights = jax.ShapeDtypeStruct(
        (BATCH_SIZE, SEQUENCE_LENGTH), jnp.float32, sharding=replicated
    )
    adapter_shardings = jax.tree.map(lambda value: value.sharding, adapters)
    optimizer_shardings = jax.tree.map(lambda value: value.sharding, optimizer)

    compile_started = time.monotonic()
    lowered = jax.jit(
        lambda current_params, current_adapters, current_optimizer, tokens, weights: (
            _loss_and_adamw_step(
                current_params,
                current_adapters,
                current_optimizer,
                tokens,
                weights,
                config=config,
                lora_config=lora_config,
                learning_rate=args.learning_rate,
                hyperparameters=DEFAULT_HYPERPARAMETERS,
            )
        ),
        out_shardings=(replicated, adapter_shardings, optimizer_shardings, replicated),
        donate_argnums=(1, 2),
    ).lower(params, adapters, optimizer, input_ids, loss_weights)
    compiled = lowered.compile()
    compile_seconds = time.monotonic() - compile_started
    del lowered
    memory = _memory_analysis(compiled)
    gate = _optimizer_execution_gate(memory)
    hlo = compiled.as_text()
    mentions = _shape_mentions(hlo, config)
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
        if any(mentions[f"{prefix}:{dtype}"] for dtype in ("u8", "f8e4m3fn", "bf16", "f32")):
            raise ValueError(f"optimized HLO contains unbounded selected-weight shape {prefix}")
    host_memory["after_compile"] = _process_memory()
    shm_after = _shm_usage()
    return {
        "schema_version": 1,
        "test": "glm53_full_attention_lora_adamw_header_only_compile_v4_probe",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "config_sha256": config_sha256,
            "index_sha256": index.sha256,
            "num_hidden_layers": config.num_hidden_layers,
            "attention_lora_target_count": adapter_placement["target_count"],
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "loss_token_count": 1,
            "rank": args.rank,
            "alpha": lora_config.alpha,
            "rematerialize_each_decoder_layer": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "beta1": DEFAULT_HYPERPARAMETERS.beta1,
            "beta2": DEFAULT_HYPERPARAMETERS.beta2,
            "epsilon": DEFAULT_HYPERPARAMETERS.epsilon,
            "weight_decay": DEFAULT_HYPERPARAMETERS.weight_decay,
            "max_grad_norm": DEFAULT_HYPERPARAMETERS.max_grad_norm,
            "adapter_dtype": "bfloat16",
            "donated_argument_numbers": [1, 2],
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
        "header_only_loader": {
            **network,
            "header_seconds": header_seconds,
            "checkpoint_payload_bytes_read": 0,
        },
        "adapter_placement": adapter_placement,
        "optimizer_placement": optimizer_placement,
        "compiler_memory": memory,
        "execution_gate": gate,
        "optimized_hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "optimized_hlo_bytes": len(hlo.encode()),
        "optimized_hlo_shape_mentions": mentions,
        "timing": {
            "header_seconds": header_seconds,
            "compile_seconds": compile_seconds,
            "elapsed_seconds_before_shutdown": time.monotonic() - started,
        },
        "host_memory": host_memory,
        "shm": {
            "before": shm_before,
            "after": shm_after,
            "used_delta_bytes": shm_after["used_bytes"] - shm_before["used_bytes"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--connections-per-shard", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if args.rank <= 0 or not 0 < args.learning_rate < 1:
        raise ValueError("rank and learning-rate must be positive")
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
