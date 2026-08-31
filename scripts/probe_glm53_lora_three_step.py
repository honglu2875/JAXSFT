#!/usr/bin/env python3
"""Run three full-checkpoint GLM LoRA AdamW steps with a sharded restore boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxsft.lora import LoRAConfig
from jaxsft.models.glm5_3_flash import OFFICIAL_CHECKPOINT, Glm53TextConfig, SafetensorsIndex
from jaxsft.models.glm5_3_streaming import Glm53StreamingLoader
from jaxsft.optim import AdamWState, adamw_init
from jaxsft.sharded_checkpoint import (
    restore_local_sharded_pytree,
    save_local_sharded_pytree,
)
from scripts.probe_glm53_lora_backward import (
    EXPECTED_BASE_BYTES_PER_DEVICE,
    INITIALIZATION_STATISTIC_NAMES,
    INPUT_IDS,
    LOSS_WEIGHTS,
    _device_memory,
    _initialize_adapters,
    _parameter_placement,
    _progress,
    _require_execution_headroom,
    _tree_scalar_statistics,
)
from scripts.probe_glm53_lora_backward_compile import (
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    _abstract_attention_adapters,
    _adapter_placement,
    _initialize_distributed,
    _memory_analysis,
    _process_memory,
    _shape_mentions,
    _shm_usage,
)
from scripts.probe_glm53_lora_optimizer_compile import (
    DEFAULT_HYPERPARAMETERS,
    DEFAULT_LEARNING_RATE,
    _abstract_adamw_state,
    _loss_and_adamw_step,
    _optimizer_execution_gate,
    _optimizer_placement,
)


CHECKPOINT_STEP = 2
EXPECTED_CHECKPOINT_ROOT_KEYS = ("adapters", "optimizer")
EXPECTED_CHECKPOINT_LEAF_COUNT = 1_147
EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS = 30_867_457
EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES = 102_891_524
EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS = 2_866
EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES = 55_398_404
EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES = 174_100_496
TRAINING_STATISTIC_NAMES = (
    "all_finite",
    "adapter_a_l2_squared",
    "adapter_b_l2_squared",
    "adapter_a_l1",
    "adapter_b_l1",
    "adapter_a_max_abs",
    "adapter_b_max_abs",
    "adapter_a_nonzero_elements",
    "adapter_b_nonzero_elements",
    "adapter_a_nonzero_leaves",
    "adapter_b_nonzero_leaves",
    "first_moment_l2_squared",
    "first_moment_l1",
    "first_moment_max_abs",
    "first_moment_nonzero_elements",
    "first_moment_nonzero_leaves",
    "second_moment_l2_squared",
    "second_moment_l1",
    "second_moment_max_abs",
    "second_moment_nonzero_elements",
    "second_moment_nonzero_leaves",
    "optimizer_step",
)


def _load_unique_json(path: Path) -> tuple[dict[str, Any], str]:
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


def _validate_optimizer_compile_evidence(path: Path) -> tuple[dict[str, int], str]:
    value, digest = _load_unique_json(path)
    gate = value.get("gate", {})
    compilation = value.get("compilation", {})
    if (
        value.get("test") != "glm53_g6c0_full_attention_lora_optimizer_compile_acceptance"
        or gate.get("g6c0_full_attention_lora_optimizer_compile") != "passed"
        or gate.get("full_checkpoint_three_step_execution_authorized") is not True
        or compilation.get("full_checkpoint_optimizer_execution_authorized") is not True
        or compilation.get("no_assignment_wide_dense_weight_in_optimized_hlo") is not True
        or compilation.get("alias_bytes_not_subtracted_from_conservative_bound") is not True
        or compilation.get("required_safety_margin_bytes_per_device") != 1024**3
    ):
        raise ValueError("G6c0 evidence does not authorize a full-checkpoint optimizer probe")
    names = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    memory = {name: compilation.get(name) for name in names}
    if any(not isinstance(value, int) for value in memory.values()):
        raise ValueError("G6c0 evidence omits compiler memory fields")
    return memory, digest


def _initialize_optimizer(
    adapters: Mapping[str, Mapping[str, jax.Array]],
    abstract_optimizer: AdamWState,
) -> tuple[AdamWState, dict[str, int | None], float]:
    shardings = jax.tree.map(lambda value: value.sharding, abstract_optimizer)
    initializer = jax.jit(adamw_init, out_shardings=shardings)
    started = time.monotonic()
    lowered = initializer.lower(adapters)
    compiled = lowered.compile()
    memory = _memory_analysis(compiled)
    del lowered
    optimizer = compiled(adapters)
    jax.block_until_ready(optimizer)
    return optimizer, memory, time.monotonic() - started


def _moment_statistics(values: object) -> jax.Array:
    leaves = jax.tree.leaves(values)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])).astype(
        jnp.float32
    )
    squared = sum(
        (jnp.sum(jnp.square(value.astype(jnp.float32)), dtype=jnp.float32) for value in leaves),
        start=jnp.asarray(0.0, jnp.float32),
    )
    l1 = sum(
        (jnp.sum(jnp.abs(value.astype(jnp.float32)), dtype=jnp.float32) for value in leaves),
        start=jnp.asarray(0.0, jnp.float32),
    )
    maximum = jnp.max(
        jnp.stack([jnp.max(jnp.abs(value.astype(jnp.float32))) for value in leaves])
    )
    nonzero_elements = sum(
        (jnp.count_nonzero(value) for value in leaves),
        start=jnp.asarray(0, jnp.int32),
    ).astype(jnp.float32)
    nonzero_leaves = jnp.sum(
        jnp.stack([jnp.any(value != 0) for value in leaves]).astype(jnp.float32),
        dtype=jnp.float32,
    )
    return jnp.stack((finite, squared, l1, maximum, nonzero_elements, nonzero_leaves))


def _training_state_statistics(
    adapters: Mapping[str, Mapping[str, jax.Array]],
    optimizer: AdamWState,
) -> jax.Array:
    adapter = _tree_scalar_statistics(adapters, include_l1_and_leaf_counts=True)
    first = _moment_statistics(optimizer.first_moment)
    second = _moment_statistics(optimizer.second_moment)
    all_finite = adapter[0] * first[0] * second[0] * jnp.isfinite(optimizer.step).astype(
        jnp.float32
    )
    return jnp.concatenate(
        (
            all_finite[None],
            adapter[1:],
            first[1:],
            second[1:],
            optimizer.step.astype(jnp.float32)[None],
        )
    )


def _float32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def _require_checkpoint_contract(summary: Mapping[str, Any]) -> None:
    expected = {
        "root_keys": sorted(EXPECTED_CHECKPOINT_ROOT_KEYS),
        "leaf_count": EXPECTED_CHECKPOINT_LEAF_COUNT,
        "global_elements_including_replicas_once": EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
        "global_logical_bytes_including_replicas_once": (
            EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES
        ),
        "local_unique_shard_count": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
        "local_unique_tensor_bytes": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES,
        "local_device_resident_bytes": EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES,
    }
    if {name: summary.get(name) for name in expected} != expected:
        raise ValueError("adapter-only checkpoint inventory drifted")


def _checkpoint_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path("/tmp") or not resolved.is_relative_to(Path("/tmp")):
        raise ValueError("the bounded G6c1 checkpoint root must be a child of /tmp")
    return resolved


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    expected_compiler_memory, compile_evidence_sha256 = _validate_optimizer_compile_evidence(
        args.optimizer_compile_evidence
    )
    checkpoint_root = _checkpoint_root(args.checkpoint_root)
    checkpoint_directory = checkpoint_root / f"step-{CHECKPOINT_STEP:08d}"
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
        raise ValueError("full GLM-5.3 three-step LoRA probe requires a 16-chip model mesh")
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
            raise ValueError("G6c1 compile preflight fetched checkpoint tensor payloads")
        abstract_adapters = _abstract_attention_adapters(
            abstract_params,
            config,
            mesh,
            rank=args.rank,
        )
        abstract_optimizer = _abstract_adamw_state(abstract_adapters, replicated)
        adapter_shardings = jax.tree.map(lambda value: value.sharding, abstract_adapters)
        optimizer_shardings = jax.tree.map(lambda value: value.sharding, abstract_optimizer)
        abstract_input_ids = jax.ShapeDtypeStruct(
            (BATCH_SIZE, SEQUENCE_LENGTH), jnp.int32, sharding=replicated
        )
        abstract_loss_weights = jax.ShapeDtypeStruct(
            (BATCH_SIZE, SEQUENCE_LENGTH), jnp.float32, sharding=replicated
        )
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
        ).lower(
            abstract_params,
            abstract_adapters,
            abstract_optimizer,
            abstract_input_ids,
            abstract_loss_weights,
        )
        compiled = lowered.compile()
        compile_seconds = time.monotonic() - compile_started
        del lowered
        compiler_memory = _memory_analysis(compiled)
        if compiler_memory != expected_compiler_memory:
            raise ValueError("G6c1 compiler memory differs from the authorizing G6c0 gate")
        execution_gate = _optimizer_execution_gate(compiler_memory)
        if execution_gate["full_checkpoint_optimizer_execution_authorized"] is not True:
            raise ValueError("optimizer compiler safety gate rejected full-checkpoint execution")
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
        host_memory["after_optimizer_compile"] = _process_memory()
        device_memory["after_optimizer_compile"] = _device_memory()

        adapters, adapter_initializer_memory, adapter_initializer_seconds = _initialize_adapters(
            abstract_params,
            config,
            mesh,
            rank=args.rank,
            seed=args.seed,
        )
        optimizer, optimizer_initializer_memory, optimizer_initializer_seconds = (
            _initialize_optimizer(adapters, abstract_optimizer)
        )
        actual_adapter_placement = _adapter_placement(adapters)
        actual_optimizer_placement = _optimizer_placement(optimizer)
        statistics_function = jax.jit(
            _training_state_statistics,
            out_shardings=replicated,
        )
        initialization = np.asarray(
            statistics_function(adapters, optimizer).block_until_ready(),
            dtype=np.float32,
        )
        if (
            len(initialization) != len(TRAINING_STATISTIC_NAMES)
            or not np.isfinite(initialization).all()
            or initialization[0] != 1
            or initialization[1] <= 0
            or initialization[2] != 0
            or initialization[7] <= 0
            or initialization[8] != 0
            or np.any(initialization[11:21] != 0)
            or initialization[21] != 0
        ):
            raise ValueError("G6c1 initialization violated finite-A/zero-B/zero-Adam invariants")
        host_memory["after_training_state_initialization"] = _process_memory()
        device_memory["after_training_state_initialization"] = _device_memory()

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
    step_records: list[dict[str, Any]] = []

    def execute_step(step: int) -> tuple[dict[str, Any], np.ndarray]:
        nonlocal adapters, optimizer
        execute_started = time.monotonic()
        loss, adapters, optimizer, gradient_norm = compiled(
            params,
            adapters,
            optimizer,
            input_ids,
            loss_weights,
        )
        jax.block_until_ready((loss, adapters, optimizer, gradient_norm))
        execute_seconds = time.monotonic() - execute_started
        diagnostics_started = time.monotonic()
        statistics = np.asarray(
            statistics_function(adapters, optimizer).block_until_ready(),
            dtype=np.float32,
        )
        diagnostics_seconds = time.monotonic() - diagnostics_started
        loss_value = float(np.asarray(loss, dtype=np.float32))
        gradient_norm_value = float(np.asarray(gradient_norm, dtype=np.float32))
        if (
            not np.isfinite(loss_value)
            or loss_value <= 0
            or not np.isfinite(gradient_norm_value)
            or gradient_norm_value <= 0
            or len(statistics) != len(TRAINING_STATISTIC_NAMES)
            or not np.isfinite(statistics).all()
            or statistics[0] != 1
            or statistics[21] != step
            or statistics[2] <= 0
            or statistics[8] <= 0
            or statistics[11] <= 0
            or statistics[16] <= 0
        ):
            raise ValueError(f"G6c1 step {step} produced invalid loss, gradient, or state")
        record = {
            "step": step,
            "loss": loss_value,
            "loss_float32_sha256": _float32_sha256(np.asarray([loss_value])),
            "gradient_norm_before_clipping": gradient_norm_value,
            "gradient_norm_float32_sha256": _float32_sha256(
                np.asarray([gradient_norm_value])
            ),
            "training_statistics": statistics.tolist(),
            "training_statistics_float32_sha256": _float32_sha256(statistics),
            "execute_seconds": execute_seconds,
            "diagnostics_seconds": diagnostics_seconds,
        }
        host_memory[f"after_step_{step}"] = _process_memory()
        device_memory[f"after_step_{step}"] = _device_memory()
        return record, statistics

    record, _ = execute_step(1)
    step_records.append(record)
    record, pre_checkpoint_statistics = execute_step(2)
    step_records.append(record)

    checkpoint_identity = {
        "format_purpose": "GLM-5.3-Flash rank-4 attention-LoRA adapter-only AdamW",
        "model_repo_id": OFFICIAL_CHECKPOINT.repo_id,
        "model_revision": OFFICIAL_CHECKPOINT.revision,
        "source_revision": args.source_revision,
        "step": CHECKPOINT_STEP,
    }
    multihost_utils.sync_global_devices("glm53-g6c1-before-checkpoint-save")
    checkpoint_started = time.monotonic()
    checkpoint_summary = save_local_sharded_pytree(
        {"adapters": adapters, "optimizer": optimizer},
        checkpoint_directory,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        identity=checkpoint_identity,
        allowed_root_keys=EXPECTED_CHECKPOINT_ROOT_KEYS,
    )
    _require_checkpoint_contract(checkpoint_summary)
    multihost_utils.sync_global_devices("glm53-g6c1-after-checkpoint-save")
    checkpoint_save_seconds = time.monotonic() - checkpoint_started
    host_memory["after_checkpoint_save"] = _process_memory()
    device_memory["after_checkpoint_save"] = _device_memory()

    del adapters, optimizer
    gc.collect()
    host_memory["after_training_state_release"] = _process_memory()
    device_memory["after_training_state_release"] = _device_memory()
    multihost_utils.sync_global_devices("glm53-g6c1-before-checkpoint-restore")
    restore_started = time.monotonic()
    restored, restore_summary = restore_local_sharded_pytree(
        {"adapters": abstract_adapters, "optimizer": abstract_optimizer},
        checkpoint_directory,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        identity=checkpoint_identity,
        allowed_root_keys=EXPECTED_CHECKPOINT_ROOT_KEYS,
    )
    adapters = restored["adapters"]
    optimizer = restored["optimizer"]
    del restored
    restored_statistics = np.asarray(
        statistics_function(adapters, optimizer).block_until_ready(),
        dtype=np.float32,
    )
    if (
        restore_summary["all_local_shards_byte_exact"] is not True
        or restore_summary["manifest_sha256"] != checkpoint_summary["manifest_sha256"]
        or restore_summary["npz_sha256"] != checkpoint_summary["npz_sha256"]
        or restore_summary["local_payload_sha256"]
        != checkpoint_summary["local_payload_sha256"]
        or not np.array_equal(restored_statistics, pre_checkpoint_statistics)
    ):
        raise ValueError("G6c1 checkpoint restore did not reproduce the exact step-2 state")
    multihost_utils.sync_global_devices("glm53-g6c1-after-checkpoint-restore")
    checkpoint_restore_seconds = time.monotonic() - restore_started
    host_memory["after_checkpoint_restore"] = _process_memory()
    device_memory["after_checkpoint_restore"] = _device_memory()

    record, _ = execute_step(3)
    step_records.append(record)
    shm["after"] = _shm_usage()
    return {
        "schema_version": 1,
        "test": "glm53_complete_text_rank4_attention_lora_adamw_three_step_v4_probe",
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
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "beta1": DEFAULT_HYPERPARAMETERS.beta1,
            "beta2": DEFAULT_HYPERPARAMETERS.beta2,
            "epsilon": DEFAULT_HYPERPARAMETERS.epsilon,
            "weight_decay": DEFAULT_HYPERPARAMETERS.weight_decay,
            "max_grad_norm": DEFAULT_HYPERPARAMETERS.max_grad_norm,
            "adapter_dtype": "bfloat16",
            "moment_dtype": "float32",
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
        "compile_preflight": {
            "authorizing_evidence_sha256": compile_evidence_sha256,
            "header_network": header_network,
            "checkpoint_payload_bytes_read": 0,
            "compiler_memory": compiler_memory,
            "execution_gate": execution_gate,
            "optimized_hlo_sha256": hlo_sha256,
            "optimized_hlo_bytes": hlo_bytes,
            "optimized_hlo_shape_mentions": shape_mentions,
            "compile_seconds": compile_seconds,
        },
        "training_state": {
            "adapter_placement": actual_adapter_placement,
            "optimizer_placement": actual_optimizer_placement,
            "adapter_initializer_compiler_memory": adapter_initializer_memory,
            "optimizer_initializer_compiler_memory": optimizer_initializer_memory,
            "initialization_statistic_names": list(TRAINING_STATISTIC_NAMES),
            "initialization_statistics": initialization.tolist(),
            "initialization_statistics_float32_sha256": _float32_sha256(initialization),
        },
        "loader": {
            **loader_network,
            "load_seconds": load_seconds,
            "parameter_placement": parameter_placement,
        },
        "steps": step_records,
        "checkpoint": {
            **checkpoint_summary,
            "directory": str(checkpoint_directory),
            "checkpoint_step": CHECKPOINT_STEP,
            "frozen_base_included": False,
            "base_leaf_count": 0,
            "restore": restore_summary,
            "pre_save_statistics_float32_sha256": _float32_sha256(
                pre_checkpoint_statistics
            ),
            "restored_statistics_float32_sha256": _float32_sha256(restored_statistics),
            "pre_save_and_restored_statistics_equal": True,
        },
        "timing": {
            "header_seconds": header_seconds,
            "compile_seconds": compile_seconds,
            "adapter_initializer_seconds": adapter_initializer_seconds,
            "optimizer_initializer_seconds": optimizer_initializer_seconds,
            "load_seconds": load_seconds,
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "checkpoint_restore_seconds": checkpoint_restore_seconds,
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
    parser.add_argument("--optimizer-compile-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--connections-per-shard", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if args.rank <= 0 or args.seed < 0 or not 0 < args.learning_rate < 1:
        raise ValueError("rank/learning-rate must be positive and seed must be non-negative")
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
