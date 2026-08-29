#!/usr/bin/env python3
"""Canonical, intentionally procedural JAXSFT training program."""

from __future__ import annotations

import argparse
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
from jaxsft.optim import AdamWHyperparameters, adamw_init, adamw_update, cosine_learning_rate


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
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1") or ""
    diff = run("diff", "--binary", "HEAD") if head else status
    import hashlib

    return {
        "head": head,
        "dirty": bool(status),
        "status": status.splitlines(),
        "dirty_material_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
    }


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


def save_checkpoint(output_dir: Path, step: int, params: object, optimizer: object) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"step-{step:08d}.pkl"
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    host_state = jax.device_get({"schema_version": 1, "step": step, "params": params, "optimizer": optimizer})
    with temporary.open("wb") as handle:
        pickle.dump(host_state, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    atomic_json(target.with_suffix(".complete.json"), {"schema_version": 1, "step": step, "file": target.name})


def make_train_step(model_config, training, optimizer_hparams):
    from jaxsft.models.qwen3_5 import forward

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
                logits = forward(
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

    from jaxsft.models.qwen3_5 import init_params, load_hf_checkpoint, parameter_count, tiny_config

    dtype = jnp.bfloat16 if recipe.model.dtype == "bfloat16" else jnp.float32
    stream = None
    if args.synthetic:
        model_config = tiny_config(vocab_size=args.synthetic_vocab_size)
        params = init_params(jax.random.key(recipe.run.seed), model_config, dtype=jnp.float32)
    else:
        model_config, params = load_hf_checkpoint(snapshot, dtype=dtype)

    output_override = os.environ.get("JAXSFT_OUTPUT_DIR")
    output_dir = Path(output_override or recipe.run.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    topology = {
        "process_index": process_index,
        "process_count": process_count,
        "local_device_count": local_device_count,
        "global_device_count": global_device_count,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.local_devices()],
    }
    emit(
        "initialized",
        rank=process_index,
        recipe=recipe.identity_hash,
        parameter_count=parameter_count(model_config),
        **topology,
    )
    atomic_json(output_dir / f"rank-{process_index:03d}-topology.json", topology)
    if process_index == 0:
        packages = {}
        for package in ("jax", "jaxlib", "numpy", "safetensors", "tokenizers", "datasets", "huggingface-hub"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        source_identity = git_identity(Path(__file__).resolve().parent)
        capsule_sha256 = os.environ.get("JAXSFT_SOURCE_SHA256")
        if capsule_sha256:
            source_identity["capsule_sha256"] = capsule_sha256
        manifest = {
            "schema_version": 1,
            "recipe": recipe.public_dict(),
            "model": {"repo_id": recipe.model.repo_id, "revision": recipe.model.revision},
            "data": {"repo_id": recipe.data.repo_id, "revision": recipe.data.revision},
            "parameter_count": parameter_count(model_config),
            "topology": topology,
            "software": {"python": sys.version, "platform": platform.platform(), "packages": packages},
            "source": source_identity,
        }
        atomic_json(output_dir / "run-manifest.json", manifest)

    if args.synthetic:
        def next_batch():
            return synthetic_batch(
                seed=recipe.run.seed,
                process_index=process_index,
                local_devices=local_device_count,
                accumulation=recipe.training.gradient_accumulation_steps,
                per_device_batch=recipe.training.per_device_batch_size,
                length=min(recipe.training.max_length, args.synthetic_length),
                vocab_size=model_config.vocab_size,
            )

        counters = None
    else:
        from jaxsft.data.stream import InstructionBatchStream
        from jaxsft.data.tokenize import LossPolicy, TokenizerSnapshot

        tokenizer_snapshot, encoder = TokenizerSnapshot.load(snapshot)
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
        )
        next_batch = stream.next_batch
        counters = stream.counters

    optimizer_hparams = AdamWHyperparameters(
        beta1=recipe.training.beta1,
        beta2=recipe.training.beta2,
        epsilon=recipe.training.epsilon,
        weight_decay=recipe.training.weight_decay,
        max_grad_norm=recipe.training.max_grad_norm,
    )
    # Build optimizer slots independently on every replica; out_axes=None states
    # that the result is replicated rather than adding a fake leading axis.
    initialize_optimizer = jax.pmap(
        lambda _replica, model_params: adamw_init(model_params),
        in_axes=(0, None),
        out_axes=None,
        axis_name="data",
    )
    optimizer = initialize_optimizer(np.arange(local_device_count, dtype=np.int32), params)
    train_step = jax.pmap(
        make_train_step(model_config, recipe.training, optimizer_hparams),
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
        for step_index in range(recipe.training.steps):
            batch = first_batch if step_index == 0 else next_batch()
            step_started = time.monotonic()
            params, optimizer, metrics = train_step(params, optimizer, batch)
            metrics["loss"].block_until_ready()
            elapsed = time.monotonic() - step_started
            host_metrics = {key: np.asarray(jax.device_get(value)).item() for key, value in metrics.items()}
            host_metrics["seconds"] = elapsed
            host_metrics["tokens_per_second"] = host_metrics["input_tokens"] / elapsed
            last_metrics = host_metrics
            if step_index == 0 or (step_index + 1) % recipe.training.log_every == 0:
                emit("train_step", rank=process_index, **host_metrics)
            if (
                recipe.training.checkpoint_every
                and (step_index + 1) % recipe.training.checkpoint_every == 0
                and process_index == 0
            ):
                save_checkpoint(output_dir, step_index + 1, params, optimizer)
                emit("checkpoint", rank=process_index, step=step_index + 1)
    finally:
        if stream is not None:
            stream.close()

    multihost_utils.sync_global_devices("jaxsft-finished")
    result = {
        "status": "complete",
        "rank": process_index,
        "steps": recipe.training.steps,
        "wall_seconds": time.monotonic() - started,
        "last_metrics": last_metrics,
        "stream_counters": None if counters is None else asdict(counters),
    }
    atomic_json(output_dir / f"rank-{process_index:03d}-result.json", result)
    emit("complete", **result)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="strict YAML recipe")
    parser.add_argument("--dry-run", action="store_true", help="validate and resolve the recipe without side effects")
    parser.add_argument("--synthetic", action="store_true", help="run the tiny Qwen3.5 model without Hub access")
    parser.add_argument("--synthetic-length", type=int, default=32)
    parser.add_argument("--synthetic-vocab-size", type=int, default=128)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    finally:
        if jax.distributed.is_initialized():
            jax.distributed.shutdown()
