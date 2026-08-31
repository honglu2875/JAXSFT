#!/usr/bin/env python3
"""Compile the smallest full GLM-5.3 attention-LoRA backward from headers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxsft.lora import LoRAConfig, format_parameter_path, parameter_at_path
from jaxsft.loss import causal_loss_statistics, normalized_loss
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    Glm53TextConfig,
    SafetensorsIndex,
    attention_lora_parameter_count,
    attention_lora_target_paths,
    forward,
)
from jaxsft.models.glm5_3_streaming import Glm53StreamingLoader


HBM_LIMIT_BYTES_PER_DEVICE = 33_014_407_168
EXECUTION_SAFETY_MARGIN_BYTES_PER_DEVICE = 1024**3
BATCH_SIZE = 1
SEQUENCE_LENGTH = 2


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


def _abstract_attention_adapters(
    params: Mapping[str, Any],
    config: Glm53TextConfig,
    mesh: Mesh,
    *,
    rank: int,
) -> dict[str, dict[str, jax.ShapeDtypeStruct]]:
    """Build the rank-local signature: replicated A and output-sharded B."""

    replicated = NamedSharding(mesh, PartitionSpec())
    output_sharded = NamedSharding(mesh, PartitionSpec(None, "model"))
    adapters: dict[str, dict[str, jax.ShapeDtypeStruct]] = {}
    targets = attention_lora_target_paths(config)
    for path in targets:
        kernel = parameter_at_path(params, path)
        if kernel.ndim != 2:
            raise ValueError(f"attention LoRA target {format_parameter_path(path)!r} is not rank two")
        input_size, output_size = (int(size) for size in kernel.shape)
        if rank > min(input_size, output_size):
            raise ValueError(f"rank {rank} exceeds {format_parameter_path(path)!r} dimensions")
        if output_size % mesh.size:
            raise ValueError(
                f"LoRA B output {output_size} at {format_parameter_path(path)!r} "
                f"is not divisible by mesh size {mesh.size}"
            )
        adapters[format_parameter_path(path)] = {
            "a": jax.ShapeDtypeStruct((input_size, rank), jnp.bfloat16, sharding=replicated),
            "b": jax.ShapeDtypeStruct(
                (rank, output_size),
                jnp.bfloat16,
                sharding=output_sharded,
            ),
        }
    return adapters


def _adapter_placement(adapters: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_device = Counter()
    global_elements = 0
    by_factor = Counter()
    for name, pair in adapters.items():
        for factor, value in pair.items():
            global_elements += int(value.size)
            by_factor[factor] += int(value.size)
            itemsize = np.dtype(value.dtype).itemsize
            for device, index in value.sharding.devices_indices_map(value.shape).items():
                local_elements = 1
                for part, size in zip(index, value.shape, strict=True):
                    if not isinstance(part, slice) or part.step not in (None, 1):
                        raise ValueError(f"adapter {name}.{factor} has a non-contiguous shard")
                    start = 0 if part.start is None else part.start
                    stop = size if part.stop is None else part.stop
                    local_elements *= stop - start
                per_device[device.id] += local_elements * itemsize
    mesh_size = len(jax.devices())
    if len(per_device) != mesh_size:
        raise ValueError(f"adapter placement covers {len(per_device)} devices, expected {mesh_size}")
    return {
        "target_count": len(adapters),
        "global_parameter_count": global_elements,
        "global_parameter_count_by_factor": dict(sorted(by_factor.items())),
        "parameter_bytes_by_device": {
            str(device_id): value for device_id, value in sorted(per_device.items())
        },
        "a_partition_spec": [],
        "b_partition_spec": [None, "model"],
    }


def _loss_and_adapter_gradients(
    params: Mapping[str, Any],
    adapters: Mapping[str, Mapping[str, jax.Array]],
    input_ids: jax.Array,
    loss_weights: jax.Array,
    *,
    config: Glm53TextConfig,
    lora_config: LoRAConfig,
) -> tuple[jax.Array, Mapping[str, Mapping[str, jax.Array]]]:
    def adapter_loss(trainable: Mapping[str, Mapping[str, jax.Array]]) -> jax.Array:
        logits = forward(
            params,
            config,
            input_ids,
            adapters=trainable,
            lora_config=lora_config,
            remat=True,
        )
        return normalized_loss(causal_loss_statistics(logits, input_ids, loss_weights))

    return jax.value_and_grad(adapter_loss)(adapters)


def _shape_mentions(hlo: str, config: Glm53TextConfig) -> dict[str, int]:
    tokens = BATCH_SIZE * SEQUENCE_LENGTH
    assignments = tokens * config.num_experts_per_tok
    local_moe = config.moe_intermediate_size // 16
    local_hidden = config.hidden_size // 16
    shapes = {
        "all_assignment_gate_dense": (
            assignments,
            config.moe_intermediate_size,
            config.hidden_size,
        ),
        "all_assignment_down_dense": (
            assignments,
            config.hidden_size,
            config.moe_intermediate_size,
        ),
        "local_all_assignment_gate_dense": (assignments, local_moe, config.hidden_size),
        "local_all_assignment_down_dense": (assignments, local_hidden, config.moe_intermediate_size),
        "token_topk_gate_dense": (
            tokens,
            config.num_experts_per_tok,
            config.moe_intermediate_size,
            config.hidden_size,
        ),
        "token_topk_down_dense": (
            tokens,
            config.num_experts_per_tok,
            config.hidden_size,
            config.moe_intermediate_size,
        ),
        "local_token_topk_gate_dense": (
            tokens,
            config.num_experts_per_tok,
            local_moe,
            config.hidden_size,
        ),
        "local_token_topk_down_dense": (
            tokens,
            config.num_experts_per_tok,
            local_hidden,
            config.moe_intermediate_size,
        ),
        "local_bounded_gate_dense": (1, local_moe, config.hidden_size),
        "local_bounded_down_dense": (1, local_hidden, config.moe_intermediate_size),
    }
    result: dict[str, int] = {}
    for name, shape in shapes.items():
        dimensions = ",".join(str(size) for size in shape)
        for dtype in ("u8", "f8e4m3fn", "bf16", "f32"):
            result[f"{name}:{dtype}"] = len(re.findall(rf"{dtype}\[{dimensions}\]", hlo))
    return result


def _execution_gate(memory: Mapping[str, int | None]) -> dict[str, Any]:
    required = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    if any(not isinstance(memory.get(name), int) for name in required):
        raise ValueError("compiler memory analysis omitted a required device byte count")
    working_set = sum(int(memory[name]) for name in required)
    limit_with_margin = HBM_LIMIT_BYTES_PER_DEVICE - EXECUTION_SAFETY_MARGIN_BYTES_PER_DEVICE
    return {
        "compiler_working_set_upper_bound_bytes_per_device": working_set,
        "measured_hbm_limit_bytes_per_device": HBM_LIMIT_BYTES_PER_DEVICE,
        "required_safety_margin_bytes_per_device": EXECUTION_SAFETY_MARGIN_BYTES_PER_DEVICE,
        "headroom_before_safety_margin_bytes_per_device": HBM_LIMIT_BYTES_PER_DEVICE - working_set,
        "full_checkpoint_execution_authorized": working_set <= limit_with_margin,
    }


def _progress(event: Mapping[str, Any]) -> None:
    if event.get("event") == "header_ready" and event.get("prepared_shards") in {
        1,
        10,
        20,
        30,
        40,
        50,
        60,
        62,
    }:
        print(json.dumps(dict(event), sort_keys=True), flush=True)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    host_memory = {"after_distributed_init": _process_memory()}
    shm_before = _shm_usage()
    config = Glm53TextConfig.from_json(args.config)
    index = SafetensorsIndex.from_path(args.index)
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if config_sha256 != OFFICIAL_CHECKPOINT.config_sha256:
        raise ValueError("config SHA-256 does not match the pinned GLM-5.3 checkpoint")
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    if mesh.size != 16:
        raise ValueError("full GLM-5.3 LoRA backward compile requires a 16-chip model mesh")
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
    host_memory["after_abstract_tree"] = _process_memory()
    if set(network["requests_by_category"]) != {"header"}:
        raise ValueError("header-only abstract construction fetched checkpoint tensor payloads")

    lora_config = LoRAConfig(rank=args.rank, alpha=float(args.rank))
    adapters = _abstract_attention_adapters(params, config, mesh, rank=args.rank)
    placement = _adapter_placement(adapters)
    expected_count = attention_lora_parameter_count(config, rank=args.rank)
    if placement["global_parameter_count"] != expected_count:
        raise ValueError("abstract adapter count disagrees with the architecture contract")
    input_ids = jax.ShapeDtypeStruct(
        (BATCH_SIZE, SEQUENCE_LENGTH),
        jnp.int32,
        sharding=replicated,
    )
    loss_weights = jax.ShapeDtypeStruct(
        (BATCH_SIZE, SEQUENCE_LENGTH),
        jnp.float32,
        sharding=replicated,
    )
    adapter_shardings = jax.tree.map(lambda value: value.sharding, adapters)
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
    ).lower(params, adapters, input_ids, loss_weights)
    compiled = lowered.compile()
    compile_seconds = time.monotonic() - compile_started
    del lowered
    memory = _memory_analysis(compiled)
    gate = _execution_gate(memory)
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
        "test": "glm53_full_attention_lora_backward_header_only_compile_v4_probe",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "config_sha256": config_sha256,
            "index_sha256": index.sha256,
            "num_hidden_layers": config.num_hidden_layers,
            "attention_lora_target_count": placement["target_count"],
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "loss_token_count": 1,
            "rank": args.rank,
            "alpha": lora_config.alpha,
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
        "header_only_loader": {
            **network,
            "header_seconds": header_seconds,
            "checkpoint_payload_bytes_read": 0,
        },
        "adapter_placement": placement,
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
    parser.add_argument("--connections-per-shard", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if args.rank <= 0:
        raise ValueError("rank must be positive")
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
