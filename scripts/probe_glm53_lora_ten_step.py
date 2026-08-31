#!/usr/bin/env python3
"""Resume GLM-5.3 LoRA at step two and run through total step ten."""

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
from jaxsft.sharded_checkpoint import (
    restore_local_sharded_pytree,
    save_local_sharded_pytree,
)
from scripts.probe_glm53_lora_backward import (
    INPUT_IDS,
    LOSS_WEIGHTS,
    _device_memory,
    _parameter_placement,
    _progress,
    _require_execution_headroom,
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
from scripts.probe_glm53_lora_three_step import (
    EXPECTED_CHECKPOINT_ROOT_KEYS,
    TRAINING_STATISTIC_NAMES,
    _checkpoint_root,
    _float32_sha256,
    _load_unique_json,
    _require_checkpoint_contract,
    _training_state_statistics,
)
from scripts.summarize_glm53_lora_optimizer_compile import EXPECTED_COMPILER_MEMORY


RESUME_STEP = 2
FIRST_EXECUTED_STEP = 3
FINAL_STEP = 10
RESUME_SOURCE_REVISION = "543350c630da2c6e298194f8f10417e42ccbf4c0"
EXPECTED_THREE_STEP_EVIDENCE_SHA256 = "11a03caa159fe99b665fe80a7c2b73414b73e821354f19ed399513fdd94dd95d"
EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256 = "30f9cf81f7162b46157e5d7a5a6d18755464d0709972e1fd1fdb95ec5ccd655a"
EXPECTED_RESUME_STATISTICS_SHA256 = "82e4f02cd7c41aa8d8ebfa942dbac2cbde0c8c48e815d017c78c83acfff1f0f2"
EXPECTED_STEP_THREE = {
    "loss": 12.266050338745117,
    "loss_float32_sha256": ("275e08a1dd4a898594752ce0d3e559b36a814dd4ba209d09b7e36abcdc85e3cf"),
    "gradient_norm_before_clipping": 2.7583279609680176,
    "gradient_norm_float32_sha256": ("ea6a2241321d611c9500f729679b5eea799f0f3d6af3142630a38eafc4f30d36"),
    "training_statistics_float32_sha256": ("4a0efd4d20d4353c680e19cceee5d74f1be51710f6ef00aa2c9a0cea861e3d98"),
}


def _validate_three_step_evidence(path: Path) -> tuple[dict[str, Any], str]:
    evidence, digest = _load_unique_json(path)
    if digest != EXPECTED_THREE_STEP_EVIDENCE_SHA256:
        raise ValueError("G6c1 evidence SHA-256 identity drifted")
    gate = evidence.get("gate", {})
    checkpoint = evidence.get("checkpoint", {})
    execution = evidence.get("execution", {})
    trajectory = evidence.get("trajectory", {})
    if (
        evidence.get("schema_version") != 1
        or evidence.get("test") != "glm53_g6c1_three_step_optimizer_checkpoint_acceptance"
        or evidence.get("source_revision") != RESUME_SOURCE_REVISION
        or gate.get("g6c1_three_step_optimizer_checkpoint") != "passed"
        or gate.get("adapter_only_checkpoint_restore_proven") is not True
        or gate.get("ten_step_resume_probe_authorized") is not True
        or gate.get("fifty_step_probe_authorized") is not False
        or checkpoint.get("checkpoint_step") != RESUME_STEP
        or checkpoint.get("root_keys") != sorted(EXPECTED_CHECKPOINT_ROOT_KEYS)
        or checkpoint.get("frozen_base_included") is not False
        or checkpoint.get("global_payload_sha256") != EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256
        or execution.get("full_checkpoint_optimizer_execution_authorized") is not True
        or execution.get("no_assignment_wide_dense_weight_in_optimized_hlo") is not True
        or execution.get("alias_bytes_not_subtracted_from_conservative_bound") is not True
    ):
        raise ValueError("G6c1 evidence does not authorize the total-10-step resume probe")
    compiler_memory = {name: execution.get(name) for name in EXPECTED_COMPILER_MEMORY}
    if compiler_memory != EXPECTED_COMPILER_MEMORY:
        raise ValueError("G6c1 evidence compiler memory drifted")
    artifacts = checkpoint.get("artifact_sha256_by_process_index")
    if not isinstance(artifacts, dict) or set(artifacts) != {"0", "1", "2", "3"}:
        raise ValueError("G6c1 evidence omits the four rank-local checkpoint identities")
    for process_index, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) != {
            "local_payload_sha256",
            "manifest_sha256",
            "npz_sha256",
        }:
            raise ValueError(f"G6c1 rank {process_index} artifact identity is malformed")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in artifact.values()
        ):
            raise ValueError(f"G6c1 rank {process_index} artifact SHA-256 is malformed")
    steps = trajectory.get("steps")
    if (
        not isinstance(steps, list)
        or [record.get("step") for record in steps if isinstance(record, dict)] != [1, 2, 3]
        or len(steps) != 3
        or steps[1].get("training_statistics_float32_sha256") != EXPECTED_RESUME_STATISTICS_SHA256
        or {name: steps[2].get(name) for name in EXPECTED_STEP_THREE} != EXPECTED_STEP_THREE
    ):
        raise ValueError("G6c1 evidence resume state or step-three sentinel drifted")
    resume_statistics = np.asarray(steps[1].get("training_statistics"), dtype=np.float32)
    if (
        resume_statistics.shape != (len(TRAINING_STATISTIC_NAMES),)
        or _float32_sha256(resume_statistics) != EXPECTED_RESUME_STATISTICS_SHA256
        or resume_statistics[-1] != RESUME_STEP
    ):
        raise ValueError("G6c1 step-two statistics payload drifted")
    return evidence, digest


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_rank_artifact_files(
    directory: Path,
    *,
    process_index: int,
    artifact_identity: Mapping[str, str],
) -> dict[str, Any]:
    manifest = directory / f"rank-{process_index:03d}.json"
    payload = directory / f"rank-{process_index:03d}.npz"
    actual = {
        "manifest_sha256": _hash_file(manifest),
        "npz_sha256": _hash_file(payload),
    }
    if actual != {
        "manifest_sha256": artifact_identity.get("manifest_sha256"),
        "npz_sha256": artifact_identity.get("npz_sha256"),
    }:
        raise ValueError("rank-local G6c1 checkpoint file identity drifted")
    return {
        **actual,
        "manifest_bytes": manifest.stat().st_size,
        "npz_bytes": payload.stat().st_size,
    }


def _checkpoint_identity(*, source_revision: str, step: int) -> dict[str, Any]:
    identity = {
        "format_purpose": "GLM-5.3-Flash rank-4 attention-LoRA adapter-only AdamW",
        "model_repo_id": OFFICIAL_CHECKPOINT.repo_id,
        "model_revision": OFFICIAL_CHECKPOINT.revision,
        "source_revision": source_revision,
        "step": step,
    }
    if step == FINAL_STEP:
        identity["resumed_from"] = {
            "global_payload_sha256": EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
            "source_revision": RESUME_SOURCE_REVISION,
            "step": RESUME_STEP,
        }
    return identity


def _require_step_three_sentinel(record: Mapping[str, Any]) -> None:
    actual = {name: record.get(name) for name in EXPECTED_STEP_THREE}
    if actual != EXPECTED_STEP_THREE:
        raise ValueError("resumed step three differs from the G6c1 deterministic sentinel")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    evidence, evidence_sha256 = _validate_three_step_evidence(args.three_step_evidence)
    resume_root = _checkpoint_root(args.resume_checkpoint_root)
    output_root = _checkpoint_root(args.checkpoint_root)
    resume_directory = resume_root / f"step-{RESUME_STEP:08d}"
    output_directory = output_root / f"step-{FINAL_STEP:08d}"
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
        raise ValueError("full GLM-5.3 ten-step LoRA probe requires a 16-chip model mesh")
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
            raise ValueError("G6d compile preflight fetched checkpoint tensor payloads")
        abstract_adapters = _abstract_attention_adapters(
            abstract_params,
            config,
            mesh,
            rank=args.rank,
        )
        abstract_optimizer = _abstract_adamw_state(abstract_adapters, replicated)
        adapter_shardings = jax.tree.map(lambda value: value.sharding, abstract_adapters)
        optimizer_shardings = jax.tree.map(lambda value: value.sharding, abstract_optimizer)
        abstract_input_ids = jax.ShapeDtypeStruct((BATCH_SIZE, SEQUENCE_LENGTH), jnp.int32, sharding=replicated)
        abstract_loss_weights = jax.ShapeDtypeStruct((BATCH_SIZE, SEQUENCE_LENGTH), jnp.float32, sharding=replicated)
        compile_started = time.monotonic()
        lowered = jax.jit(
            lambda current_params, current_adapters, current_optimizer, tokens, weights: _loss_and_adamw_step(
                current_params,
                current_adapters,
                current_optimizer,
                tokens,
                weights,
                config=config,
                lora_config=lora_config,
                learning_rate=args.learning_rate,
                hyperparameters=DEFAULT_HYPERPARAMETERS,
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
        if compiler_memory != EXPECTED_COMPILER_MEMORY:
            raise ValueError("G6d compiler memory differs from the authorizing G6c1 gate")
        execution_gate = _optimizer_execution_gate(compiler_memory)
        if execution_gate["full_checkpoint_optimizer_execution_authorized"] is not True:
            raise ValueError("optimizer compiler safety gate rejected G6d execution")
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
            if any(shape_mentions[f"{prefix}:{dtype}"] for dtype in ("u8", "f8e4m3fn", "bf16", "f32")):
                raise ValueError(f"optimized HLO contains unbounded selected-weight shape {prefix}")
        del hlo
        gc.collect()
        host_memory["after_optimizer_compile"] = _process_memory()
        device_memory["after_optimizer_compile"] = _device_memory()

        process_index = jax.process_index()
        artifact_identity = evidence["checkpoint"]["artifact_sha256_by_process_index"][str(process_index)]
        artifact_files = _validate_rank_artifact_files(
            resume_directory,
            process_index=process_index,
            artifact_identity=artifact_identity,
        )
        statistics_function = jax.jit(
            _training_state_statistics,
            out_shardings=replicated,
        )
        multihost_utils.sync_global_devices("glm53-g6d-before-step-two-restore")
        restore_started = time.monotonic()
        restored, resume_restore = restore_local_sharded_pytree(
            {"adapters": abstract_adapters, "optimizer": abstract_optimizer},
            resume_directory,
            process_index=process_index,
            process_count=jax.process_count(),
            identity=_checkpoint_identity(
                source_revision=RESUME_SOURCE_REVISION,
                step=RESUME_STEP,
            ),
            allowed_root_keys=EXPECTED_CHECKPOINT_ROOT_KEYS,
        )
        adapters = restored["adapters"]
        optimizer = restored["optimizer"]
        del restored
        resume_statistics = np.asarray(
            statistics_function(adapters, optimizer).block_until_ready(),
            dtype=np.float32,
        )
        expected_resume_statistics = np.asarray(
            evidence["trajectory"]["steps"][1]["training_statistics"],
            dtype=np.float32,
        )
        if (
            resume_restore.get("all_local_shards_byte_exact") is not True
            or resume_restore.get("manifest_sha256") != artifact_identity["manifest_sha256"]
            or resume_restore.get("npz_sha256") != artifact_identity["npz_sha256"]
            or resume_restore.get("local_payload_sha256") != artifact_identity["local_payload_sha256"]
            or not np.array_equal(resume_statistics, expected_resume_statistics)
            or _float32_sha256(resume_statistics) != EXPECTED_RESUME_STATISTICS_SHA256
        ):
            raise ValueError("G6d did not restore the exact accepted G6c1 step-two state")
        multihost_utils.sync_global_devices("glm53-g6d-after-step-two-restore")
        resume_restore_seconds = time.monotonic() - restore_started
        host_memory["after_step_two_restore"] = _process_memory()
        device_memory["after_step_two_restore"] = _device_memory()

        actual_adapter_placement = _adapter_placement(adapters)
        actual_optimizer_placement = _optimizer_placement(optimizer)
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
            or statistics.shape != (len(TRAINING_STATISTIC_NAMES),)
            or not np.isfinite(statistics).all()
            or statistics[0] != 1
            or statistics[21] != step
            or statistics[2] <= 0
            or statistics[8] <= 0
            or statistics[11] <= 0
            or statistics[16] <= 0
        ):
            raise ValueError(f"G6d step {step} produced invalid loss, gradient, or state")
        record = {
            "step": step,
            "loss": loss_value,
            "loss_float32_sha256": _float32_sha256(np.asarray([loss_value])),
            "gradient_norm_before_clipping": gradient_norm_value,
            "gradient_norm_float32_sha256": _float32_sha256(np.asarray([gradient_norm_value])),
            "training_statistics": statistics.tolist(),
            "training_statistics_float32_sha256": _float32_sha256(statistics),
            "execute_seconds": execute_seconds,
            "diagnostics_seconds": diagnostics_seconds,
        }
        host_memory[f"after_step_{step}"] = _process_memory()
        device_memory[f"after_step_{step}"] = _device_memory()
        return record, statistics

    final_statistics: np.ndarray | None = None
    for step in range(FIRST_EXECUTED_STEP, FINAL_STEP + 1):
        record, final_statistics = execute_step(step)
        if step == FIRST_EXECUTED_STEP:
            _require_step_three_sentinel(record)
        step_records.append(record)
    if final_statistics is None:
        raise AssertionError("G6d step loop executed no steps")

    output_identity = _checkpoint_identity(
        source_revision=args.source_revision,
        step=FINAL_STEP,
    )
    multihost_utils.sync_global_devices("glm53-g6d-before-step-ten-save")
    checkpoint_started = time.monotonic()
    checkpoint_summary = save_local_sharded_pytree(
        {"adapters": adapters, "optimizer": optimizer},
        output_directory,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        identity=output_identity,
        allowed_root_keys=EXPECTED_CHECKPOINT_ROOT_KEYS,
    )
    _require_checkpoint_contract(checkpoint_summary)
    multihost_utils.sync_global_devices("glm53-g6d-after-step-ten-save")
    checkpoint_save_seconds = time.monotonic() - checkpoint_started
    host_memory["after_step_ten_checkpoint_save"] = _process_memory()
    device_memory["after_step_ten_checkpoint_save"] = _device_memory()

    del adapters, optimizer
    gc.collect()
    host_memory["after_step_ten_state_release"] = _process_memory()
    device_memory["after_step_ten_state_release"] = _device_memory()
    multihost_utils.sync_global_devices("glm53-g6d-before-step-ten-restore")
    final_restore_started = time.monotonic()
    restored, final_restore = restore_local_sharded_pytree(
        {"adapters": abstract_adapters, "optimizer": abstract_optimizer},
        output_directory,
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        identity=output_identity,
        allowed_root_keys=EXPECTED_CHECKPOINT_ROOT_KEYS,
    )
    adapters = restored["adapters"]
    optimizer = restored["optimizer"]
    del restored
    restored_final_statistics = np.asarray(
        statistics_function(adapters, optimizer).block_until_ready(),
        dtype=np.float32,
    )
    if (
        final_restore.get("all_local_shards_byte_exact") is not True
        or final_restore.get("manifest_sha256") != checkpoint_summary["manifest_sha256"]
        or final_restore.get("npz_sha256") != checkpoint_summary["npz_sha256"]
        or final_restore.get("local_payload_sha256") != checkpoint_summary["local_payload_sha256"]
        or not np.array_equal(restored_final_statistics, final_statistics)
    ):
        raise ValueError("G6d checkpoint restore did not reproduce the exact step-ten state")
    multihost_utils.sync_global_devices("glm53-g6d-after-step-ten-restore")
    final_restore_seconds = time.monotonic() - final_restore_started
    host_memory["after_step_ten_checkpoint_restore"] = _process_memory()
    device_memory["after_step_ten_checkpoint_restore"] = _device_memory()
    shm["after"] = _shm_usage()

    return {
        "schema_version": 1,
        "test": "glm53_complete_text_rank4_attention_lora_adamw_total_ten_step_v4_probe",
        "source_revision": args.source_revision,
        "authorization": {
            "three_step_evidence_sha256": evidence_sha256,
            "resume_source_revision": RESUME_SOURCE_REVISION,
            "resume_global_payload_sha256": EXPECTED_GLOBAL_RESUME_PAYLOAD_SHA256,
            "fifty_step_probe_authorized_before_this_run": False,
        },
        "model": {
            "repo_id": OFFICIAL_CHECKPOINT.repo_id,
            "revision": OFFICIAL_CHECKPOINT.revision,
            "config_sha256": config_sha256,
            "index_sha256": index.sha256,
            "num_hidden_layers": config.num_hidden_layers,
            "rank": args.rank,
            "alpha": lora_config.alpha,
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
            "header_network": header_network,
            "checkpoint_payload_bytes_read": 0,
            "compiler_memory": compiler_memory,
            "execution_gate": execution_gate,
            "authorizing_g6c1_hlo_sha256": evidence["execution"]["optimized_hlo_sha256"],
            "optimized_hlo_sha256": hlo_sha256,
            "optimized_hlo_bytes": hlo_bytes,
            "optimized_hlo_shape_mentions": shape_mentions,
            "compile_seconds": compile_seconds,
        },
        "resume": {
            "directory": str(resume_directory),
            "checkpoint_step": RESUME_STEP,
            "artifact_files": artifact_files,
            "restore": resume_restore,
            "statistics": resume_statistics.tolist(),
            "statistics_float32_sha256": _float32_sha256(resume_statistics),
            "restored_before_full_base_payload_load": True,
        },
        "training_state": {
            "adapter_placement": actual_adapter_placement,
            "optimizer_placement": actual_optimizer_placement,
            "statistic_names": list(TRAINING_STATISTIC_NAMES),
        },
        "loader": {
            **loader_network,
            "load_seconds": load_seconds,
            "parameter_placement": parameter_placement,
        },
        "steps": step_records,
        "checkpoint": {
            **checkpoint_summary,
            "directory": str(output_directory),
            "checkpoint_step": FINAL_STEP,
            "frozen_base_included": False,
            "base_leaf_count": 0,
            "restore": final_restore,
            "pre_save_statistics_float32_sha256": _float32_sha256(final_statistics),
            "restored_statistics_float32_sha256": _float32_sha256(restored_final_statistics),
            "pre_save_and_restored_statistics_equal": True,
        },
        "timing": {
            "header_seconds": header_seconds,
            "compile_seconds": compile_seconds,
            "resume_restore_seconds": resume_restore_seconds,
            "load_seconds": load_seconds,
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "checkpoint_restore_seconds": final_restore_seconds,
            "elapsed_seconds_before_shutdown": time.monotonic() - started,
        },
        "host_memory": host_memory,
        "device_memory": device_memory,
        "shm": {
            **shm,
            "used_delta_during_load_bytes": shm["after_load"]["used_bytes"] - shm["before"]["used_bytes"],
            "used_delta_total_bytes": shm["after"]["used_bytes"] - shm["before"]["used_bytes"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--three-step-evidence", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--connections-per-shard", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if len(args.source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_revision
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if args.rank != 4 or args.learning_rate != DEFAULT_LEARNING_RATE:
        raise ValueError("G6d requires rank 4 and the accepted 1e-5 learning rate")
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
