#!/usr/bin/env python3
"""Canonical, intentionally procedural JAXSFT training program."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxsft.config import Recipe, load_recipe
from jaxsft.loss import causal_loss_statistics
from jaxsft.optim import AdamWHyperparameters, AdamWState, adamw_init, adamw_update, cosine_learning_rate


CHECKPOINT_SCHEMA_VERSION = 3


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def emit(event: str, **fields: Any) -> None:
    payload = {"time": time.time(), "event": event, **fields}
    print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(temporary, path)


def git_identity(root: Path) -> dict[str, Any]:
    def run(*arguments: str, strip: bool = True) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
        )
        if result.returncode:
            return None
        return result.stdout.strip() if strip else result.stdout.rstrip("\n")

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", strip=False) or ""
    diff = run("diff", "--binary", "HEAD") if head else status
    material = hashlib.sha256((diff or "").encode())
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if untracked.returncode == 0:
        for raw_path in sorted(part for part in untracked.stdout.split(b"\0") if part):
            relative = Path(os.fsdecode(raw_path))
            unresolved = root / relative
            resolved = unresolved.resolve()
            if unresolved.is_symlink() or root not in resolved.parents or not resolved.is_file():
                raise RuntimeError(f"untracked source identity entry is unsafe: {relative}")
            material.update(b"\0untracked\0")
            material.update(raw_path)
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    material.update(block)
    return {
        "head": head,
        "dirty": bool(status),
        "status": status.splitlines(),
        "dirty_material_sha256": material.hexdigest(),
    }


def checkpoint_source_identity(root: Path) -> dict[str, Any]:
    capsule_sha256 = os.environ.get("JAXSFT_SOURCE_SHA256")
    if capsule_sha256:
        if len(capsule_sha256) != 64 or any(character not in "0123456789abcdef" for character in capsule_sha256):
            raise ValueError("JAXSFT_SOURCE_SHA256 must be a lowercase SHA-256 digest")
        return {"kind": "source_capsule", "sha256": capsule_sha256}
    identity = git_identity(root)
    if identity["head"] is None:
        raise RuntimeError("checkpointing requires Git metadata or JAXSFT_SOURCE_SHA256")
    return {"kind": "git_worktree", **identity}


def resolve_model_snapshot(recipe: Recipe) -> Path:
    if recipe.model.local_path:
        path = Path(recipe.model.local_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model.local_path does not exist: {path}")
        return path
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=recipe.model.repo_id,
        revision=recipe.model.revision,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    return Path(path)


def initialize_distributed() -> None:
    coordinator = os.environ.get("JAXSFT_COORDINATOR_ADDRESS")
    count = int(os.environ.get("JAXSFT_PROCESS_COUNT", "1"))
    process_id = int(os.environ.get("JAXSFT_PROCESS_ID", "0"))
    if coordinator:
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=count,
            process_id=process_id,
            initialization_timeout=1800,
            heartbeat_timeout_seconds=300,
        )
    elif os.environ.get("JAXSFT_AUTO_DISTRIBUTED") == "1":
        jax.distributed.initialize(initialization_timeout=1800, heartbeat_timeout_seconds=300)


def synthetic_batch(
    *,
    seed: int,
    process_index: int,
    local_devices: int,
    accumulation: int,
    per_device_batch: int,
    length: int,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed + 10_003 * process_index)
    shape = (local_devices, accumulation, per_device_batch, length)
    input_ids = generator.integers(0, vocab_size, shape, dtype=np.int32)
    attention_mask = np.ones(shape, dtype=np.bool_)
    loss_weights = np.zeros(shape, dtype=np.float32)
    loss_weights[..., max(1, length // 2) :] = 1.0
    loss_weights[..., 0] = 0.0
    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_weights": loss_weights}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    output_dir: Path,
    step: int,
    params: object,
    optimizer: object,
    *,
    recipe_identity_hash: str,
    source_identity: dict[str, Any],
    topology: dict[str, Any],
    data_state: dict[str, Any],
    rng_state: dict[str, Any],
) -> Path:
    """Atomically write a trusted, replicated smoke checkpoint and marker."""

    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    if int(topology.get("process_count", -1)) != 1:
        raise ValueError("replicated checkpoint format supports exactly one JAX process")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"step-{step:08d}.pkl"
    marker_path = target.with_suffix(".complete.json")
    if target.exists() or marker_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint or completion marker: {target}")
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    host_state = jax.device_get(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "recipe_identity_hash": recipe_identity_hash,
            "source_identity": source_identity,
            "topology": topology,
            "params": params,
            "optimizer": optimizer,
            "data_state": data_state,
            "rng_state": rng_state,
        }
    )
    try:
        with temporary.open("xb") as handle:
            pickle.dump(host_state, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    atomic_json(
        marker_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "file": target.name,
            "sha256": _file_sha256(target),
            "recipe_identity_hash": recipe_identity_hash,
            "source_identity": source_identity,
            "topology": topology,
        },
    )
    return target


def load_checkpoint(
    path: str | Path,
    *,
    recipe_identity_hash: str,
    source_identity: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Load a trusted local checkpoint after marker, hash, and identity checks."""

    path = Path(path).expanduser().resolve()
    marker_path = path.with_suffix(".complete.json")
    if not path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"checkpoint and completion marker are required: {path}")
    marker = json.loads(marker_path.read_text())
    if not isinstance(marker, dict) or marker.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint completion marker")
    if marker.get("file") != path.name:
        raise ValueError("checkpoint marker names a different payload")
    if marker.get("sha256") != _file_sha256(path):
        raise ValueError("checkpoint SHA-256 does not match its completion marker")
    if marker.get("recipe_identity_hash") != recipe_identity_hash:
        raise ValueError("checkpoint recipe identity differs from this recipe")
    if marker.get("source_identity") != source_identity:
        raise ValueError("checkpoint source identity differs from this run")
    if marker.get("topology") != topology:
        raise ValueError("checkpoint topology differs from this run")
    # Pickle is intentionally limited to checkpoints created and trusted by the
    # same experiment. Never load an untrusted checkpoint path.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint payload")
    if payload.get("recipe_identity_hash") != recipe_identity_hash:
        raise ValueError("checkpoint payload recipe identity differs from this recipe")
    if payload.get("source_identity") != source_identity:
        raise ValueError("checkpoint payload source identity differs from this run")
    if payload.get("topology") != topology:
        raise ValueError("checkpoint payload topology differs from this run")
    payload_step = int(payload.get("step", -1))
    if payload_step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if payload_step != int(marker.get("step", -2)):
        raise ValueError("checkpoint payload and marker steps differ")
    required = {"params", "optimizer", "data_state", "rng_state"}
    if not required.issubset(payload):
        raise ValueError(f"checkpoint is missing fields: {sorted(required - set(payload))}")
    return payload


def validate_optimizer_checkpoint(params: object, optimizer: object, *, expected_step: int) -> None:
    """Validate optimizer structure without allocating another set of slots."""

    if not isinstance(optimizer, AdamWState):
        raise ValueError("checkpoint optimizer is not an AdamWState")
    optimizer_step = np.asarray(jax.device_get(optimizer.step))
    if optimizer_step.shape or int(optimizer_step) != expected_step:
        raise ValueError("checkpoint optimizer step differs from checkpoint step")
    parameter_structure = jax.tree.structure(params)
    for name, moments in (
        ("first_moment", optimizer.first_moment),
        ("second_moment", optimizer.second_moment),
    ):
        if jax.tree.structure(moments) != parameter_structure:
            raise ValueError(f"checkpoint {name} structure differs from parameters")
        for index, (parameter, moment) in enumerate(zip(jax.tree.leaves(params), jax.tree.leaves(moments))):
            if tuple(parameter.shape) != tuple(moment.shape):
                raise ValueError(f"checkpoint {name} leaf {index} shape differs from parameters")


def validate_rng_checkpoint(raw: object, *, seed: int, expected_step: int) -> None:
    """Validate the explicit RNG cursor used by the currently deterministic trainer."""

    if not isinstance(raw, dict) or set(raw) != {"schema_version", "model_init_seed", "next_training_step"}:
        raise ValueError("checkpoint RNG state has unexpected fields")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint RNG state")
    if int(raw.get("model_init_seed", -1)) != seed:
        raise ValueError("checkpoint model initialization seed differs from this recipe")
    if int(raw.get("next_training_step", -1)) != expected_step:
        raise ValueError("checkpoint RNG cursor differs from checkpoint step")


def load_synthetic_cursor(raw: object, *, expected_step: int, length: int, vocab_size: int) -> int:
    expected_fields = {"schema_version", "kind", "batches_consumed", "length", "vocab_size"}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("synthetic data checkpoint has unexpected fields")
    if raw.get("schema_version") != 2 or raw.get("kind") != "synthetic":
        raise ValueError("unsupported synthetic data checkpoint")
    batches_consumed = int(raw.get("batches_consumed", -1))
    if batches_consumed != expected_step:
        raise ValueError("synthetic batch cursor differs from checkpoint step")
    if int(raw.get("length", -1)) != length or int(raw.get("vocab_size", -1)) != vocab_size:
        raise ValueError("synthetic data shape differs from checkpoint")
    return batches_consumed


def make_train_step(model_config, training, optimizer_hparams, model_forward):
    def train_step(params, optimizer, accumulated_batch):
        zero_gradients = jax.tree.map(jnp.zeros_like, params)
        initial = (
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            zero_gradients,
        )

        def accumulate(carry, microbatch):
            numerator, denominator, correct, input_tokens, gradients = carry

            def objective(current_params):
                logits = model_forward(
                    current_params,
                    model_config,
                    microbatch["input_ids"],
                    attention_mask=microbatch["attention_mask"],
                    remat=training.remat,
                )
                statistics = causal_loss_statistics(logits, microbatch["input_ids"], microbatch["loss_weights"])
                auxiliary = (
                    statistics.denominator,
                    statistics.correct_weight,
                    jnp.sum(microbatch["attention_mask"], dtype=jnp.float32),
                )
                return statistics.numerator, auxiliary

            (micro_numerator, auxiliary), micro_gradients = jax.value_and_grad(objective, has_aux=True)(params)
            micro_denominator, micro_correct, micro_input_tokens = auxiliary
            gradients = jax.tree.map(lambda total, value: total + value, gradients, micro_gradients)
            return (
                numerator + micro_numerator,
                denominator + micro_denominator,
                correct + micro_correct,
                input_tokens + micro_input_tokens,
                gradients,
            ), None

        (local_num, local_den, local_correct, local_input_tokens, gradients), _ = jax.lax.scan(
            accumulate, initial, accumulated_batch
        )
        global_num = jax.lax.psum(local_num, "data")
        global_den = jax.lax.psum(local_den, "data")
        global_correct = jax.lax.psum(local_correct, "data")
        global_input_tokens = jax.lax.psum(local_input_tokens, "data")
        gradients = jax.tree.map(lambda value: jax.lax.psum(value, "data") / global_den, gradients)
        learning_rate = cosine_learning_rate(
            optimizer.step,
            peak=training.peak_learning_rate,
            total_steps=training.steps,
            warmup_steps=training.warmup_steps,
            minimum_ratio=training.minimum_learning_rate_ratio,
        )
        params, optimizer, gradient_norm = adamw_update(
            params,
            gradients,
            optimizer,
            learning_rate=learning_rate,
            hyperparameters=optimizer_hparams,
        )
        metrics = {
            "loss": global_num / global_den,
            "loss_numerator": global_num,
            "loss_denominator": global_den,
            "selected_accuracy": global_correct / global_den,
            "input_tokens": global_input_tokens,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "step": optimizer.step,
        }
        return params, optimizer, metrics

    return train_step


def run(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.config)
    if args.dry_run:
        print(json.dumps(recipe.public_dict(), indent=2, sort_keys=True))
        return 0

    snapshot = None if args.synthetic else resolve_model_snapshot(recipe)
    initialize_distributed()
    process_index, process_count = jax.process_index(), jax.process_count()
    local_device_count, global_device_count = jax.local_device_count(), jax.device_count()
    if int(os.environ.get("JAXSFT_PROCESS_COUNT", process_count)) != process_count:
        raise RuntimeError("declared and initialized JAX process counts differ")
    topology = {
        "process_index": process_index,
        "process_count": process_count,
        "local_device_count": local_device_count,
        "global_device_count": global_device_count,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.local_devices()],
    }
    source_identity = checkpoint_source_identity(Path(__file__).resolve().parent)

    checkpoint_requested = bool(recipe.training.checkpoint_every or args.resume or args.stop_after_step is not None)
    if process_count > 1 and checkpoint_requested:
        raise RuntimeError(
            "checkpoint/resume currently supports one JAX process (including multiple local devices); "
            "portable multi-host checkpoints are not implemented"
        )

    restored = None
    start_step = 0
    if args.resume:
        restored = load_checkpoint(
            args.resume,
            recipe_identity_hash=recipe.identity_hash,
            source_identity=source_identity,
            topology=topology,
        )
        start_step = int(restored["step"])
        validate_rng_checkpoint(restored["rng_state"], seed=recipe.run.seed, expected_step=start_step)
    stop_step = recipe.training.steps if args.stop_after_step is None else args.stop_after_step
    if not start_step < stop_step <= recipe.training.steps:
        raise ValueError(
            f"training interval must satisfy start_step < stop_step <= {recipe.training.steps}; "
            f"got {start_step} < {stop_step}"
        )

    from jaxsft.models.registry import get_model_implementation

    model = get_model_implementation(recipe.model.architecture)

    dtype = jnp.bfloat16 if recipe.model.dtype == "bfloat16" else jnp.float32
    stream = None
    if args.synthetic:
        model_config = model.tiny_config(vocab_size=args.synthetic_vocab_size)
        params = (
            model.init_params(jax.random.key(recipe.run.seed), model_config, dtype=jnp.float32)
            if restored is None
            else restored["params"]
        )
    elif restored is not None:
        model_config = model.config_type.from_json(snapshot / "config.json")
        params = restored["params"]
    else:
        model_config, params = model.load_hf_checkpoint(snapshot, dtype=dtype)
    model.validate_params(params, model_config)

    tokenizer_snapshot = None
    encoder = None
    if not args.synthetic:
        from jaxsft.data.tokenize import TokenizerSnapshot

        tokenizer_snapshot, encoder = TokenizerSnapshot.load(
            snapshot,
            pad_token_id=getattr(model_config, "pad_token_id", None),
        )

    output_override = os.environ.get("JAXSFT_OUTPUT_DIR")
    output_dir = Path(output_override or recipe.run.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    emit(
        "initialized",
        rank=process_index,
        recipe=recipe.identity_hash,
        parameter_count=model.parameter_count(model_config),
        **topology,
    )
    atomic_json(output_dir / f"rank-{process_index:03d}-topology.json", topology)
    if process_index == 0:
        from jaxsft.data.render import renderer_identity

        packages = {}
        for package in ("jax", "jaxlib", "numpy", "safetensors", "tokenizers", "datasets", "huggingface-hub"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        manifest = {
            "schema_version": 3,
            "recipe": recipe.public_dict(),
            "model": {
                "architecture": recipe.model.architecture,
                "repo_id": recipe.model.repo_id,
                "revision": recipe.model.revision,
            },
            "data": {
                "repo_id": recipe.data.repo_id,
                "revision": recipe.data.revision,
                "loading_mode": recipe.data.loading_mode,
                "renderer": renderer_identity(recipe.data.renderer or recipe.model.architecture),
            },
            "tokenizer": None
            if tokenizer_snapshot is None
            else {
                "identity_sha256": tokenizer_snapshot.identity_hash,
                "pad_token_id": tokenizer_snapshot.pad_token_id,
            },
            "parameter_count": model.parameter_count(model_config),
            "topology": topology,
            "software": {"python": sys.version, "platform": platform.platform(), "packages": packages},
            "source": source_identity,
            "resume": None
            if args.resume is None
            else {"checkpoint": str(Path(args.resume).expanduser().resolve()), "start_step": start_step},
            "requested_stop_step": stop_step,
            "execution": {
                "synthetic": args.synthetic,
                "synthetic_length": args.synthetic_length if args.synthetic else None,
                "synthetic_vocab_size": args.synthetic_vocab_size if args.synthetic else None,
            },
        }
        atomic_json(output_dir / "run-manifest.json", manifest)

    if args.synthetic:
        synthetic_batches_consumed = (
            0
            if restored is None
            else load_synthetic_cursor(
                restored["data_state"],
                expected_step=start_step,
                length=min(recipe.training.max_length, args.synthetic_length),
                vocab_size=args.synthetic_vocab_size,
            )
        )

        def next_batch():
            nonlocal synthetic_batches_consumed
            batch = synthetic_batch(
                seed=recipe.run.seed + 1_000_003 * synthetic_batches_consumed,
                process_index=process_index,
                local_devices=local_device_count,
                accumulation=recipe.training.gradient_accumulation_steps,
                per_device_batch=recipe.training.per_device_batch_size,
                length=min(recipe.training.max_length, args.synthetic_length),
                vocab_size=model_config.vocab_size,
            )
            synthetic_batches_consumed += 1
            return batch

        def current_data_state():
            return {
                "schema_version": 2,
                "kind": "synthetic",
                "batches_consumed": synthetic_batches_consumed,
                "length": min(recipe.training.max_length, args.synthetic_length),
                "vocab_size": args.synthetic_vocab_size,
            }

        counters = None
    else:
        from jaxsft.data.stream import InstructionBatchStream
        from jaxsft.data.tokenize import LossPolicy

        assert tokenizer_snapshot is not None and encoder is not None
        stream = InstructionBatchStream(
            recipe.data,
            tokenizer_snapshot=tokenizer_snapshot,
            encoder=encoder,
            policy=LossPolicy.from_config(recipe.objective),
            process_index=process_index,
            process_count=process_count,
            local_device_count=local_device_count,
            per_device_batch_size=recipe.training.per_device_batch_size,
            accumulation_steps=recipe.training.gradient_accumulation_steps,
            max_length=recipe.training.max_length,
            truncation=recipe.training.truncation,
            truncation_min_context_tokens=recipe.training.truncation_min_context_tokens,
            renderer=recipe.data.renderer or recipe.model.architecture,
        )
        if restored is not None:
            stream.load_state_dict(restored["data_state"])
        next_batch = stream.next_batch
        counters = stream.counters

        def current_data_state():
            return stream.state_dict()

    optimizer_hparams = AdamWHyperparameters(
        beta1=recipe.training.beta1,
        beta2=recipe.training.beta2,
        epsilon=recipe.training.epsilon,
        weight_decay=recipe.training.weight_decay,
        max_grad_norm=recipe.training.max_grad_norm,
    )
    if restored is None:
        # Build optimizer slots independently on every replica; out_axes=None states
        # that the result is replicated rather than adding a fake leading axis.
        initialize_optimizer = jax.pmap(
            lambda _replica, model_params: adamw_init(model_params),
            in_axes=(0, None),
            out_axes=None,
            axis_name="data",
        )
        optimizer = initialize_optimizer(np.arange(local_device_count, dtype=np.int32), params)
    else:
        optimizer = restored["optimizer"]
        validate_optimizer_checkpoint(params, optimizer, expected_step=start_step)
    train_step = jax.pmap(
        make_train_step(model_config, recipe.training, optimizer_hparams, model.forward),
        axis_name="data",
        in_axes=(None, None, 0),
        out_axes=(None, None, None),
        donate_argnums=(0, 1),
    )

    from jax.experimental import multihost_utils

    started = time.monotonic()
    last_metrics = None
    try:
        first_batch = next_batch()
        multihost_utils.sync_global_devices("jaxsft-before-first-step")
        for step_index in range(start_step, stop_step):
            batch = first_batch if step_index == start_step else next_batch()
            step_started = time.monotonic()
            params, optimizer, metrics = train_step(params, optimizer, batch)
            metrics["loss"].block_until_ready()
            elapsed = time.monotonic() - step_started
            host_metrics = {key: np.asarray(jax.device_get(value)).item() for key, value in metrics.items()}
            host_metrics["seconds"] = elapsed
            host_metrics["tokens_per_second"] = host_metrics["input_tokens"] / elapsed
            last_metrics = host_metrics
            completed_step = step_index + 1
            if step_index == start_step or completed_step % recipe.training.log_every == 0:
                emit("train_step", rank=process_index, **host_metrics)
            cadence_checkpoint = bool(
                recipe.training.checkpoint_every
                and completed_step % recipe.training.checkpoint_every == 0
            )
            forced_stop_checkpoint = args.stop_after_step is not None and completed_step == stop_step
            if (cadence_checkpoint or forced_stop_checkpoint) and process_index == 0:
                checkpoint_path = save_checkpoint(
                    output_dir,
                    completed_step,
                    params,
                    optimizer,
                    recipe_identity_hash=recipe.identity_hash,
                    source_identity=source_identity,
                    topology=topology,
                    data_state=current_data_state(),
                    rng_state={
                        "schema_version": 1,
                        "model_init_seed": recipe.run.seed,
                        "next_training_step": completed_step,
                    },
                )
                emit("checkpoint", rank=process_index, step=completed_step, path=checkpoint_path)
    finally:
        if stream is not None:
            close_started = time.monotonic()
            stream.close()
            emit("data_stream_closed", rank=process_index, seconds=time.monotonic() - close_started)

    multihost_utils.sync_global_devices("jaxsft-finished")
    status = "complete" if stop_step == recipe.training.steps else "stopped"
    result = {
        "status": status,
        "rank": process_index,
        "steps": stop_step,
        "start_step": start_step,
        "completed_steps": stop_step,
        "target_steps": recipe.training.steps,
        "wall_seconds": time.monotonic() - started,
        "last_metrics": last_metrics,
        "stream_counters": None if counters is None else asdict(counters),
    }
    atomic_json(output_dir / f"rank-{process_index:03d}-result.json", result)
    emit(status, **result)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="strict YAML recipe")
    parser.add_argument("--dry-run", action="store_true", help="validate and resolve the recipe without side effects")
    parser.add_argument("--synthetic", action="store_true", help="run the recipe architecture's tiny model offline")
    parser.add_argument("--synthetic-length", type=int, default=32)
    parser.add_argument("--synthetic-vocab-size", type=int, default=128)
    parser.add_argument("--resume", help="trusted local step checkpoint to resume")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="stop at this absolute completed step and write a checkpoint",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    finally:
        if jax.distributed.is_initialized():
            started = time.monotonic()
            emit("distributed_shutdown_started", rank=jax.process_index())
            jax.distributed.shutdown()
            emit("distributed_shutdown_complete", seconds=time.monotonic() - started)


if __name__ == "__main__":
    raise SystemExit(main())
