"""Strict, versioned recipe loading."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _strict(raw: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    repo_id: str
    revision: str
    dtype: str = "bfloat16"
    local_path: str | None = None


@dataclass(frozen=True)
class DataSpec:
    repo_id: str
    revision: str
    config: str
    split: str
    adapter: str
    shuffle_seed: int = 17
    shuffle_buffer_size: int = 10_000


@dataclass(frozen=True)
class TrainingSpec:
    max_length: int
    truncation: str
    per_device_batch_size: int
    gradient_accumulation_steps: int
    steps: int
    peak_learning_rate: float
    warmup_steps: int
    minimum_learning_rate_ratio: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float
    remat: bool
    log_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class RunSpec:
    seed: int
    output_dir: str


@dataclass(frozen=True)
class Recipe:
    schema_version: int
    model: ModelSpec
    data: DataSpec
    objective: dict[str, Any]
    training: TrainingSpec
    run: RunSpec
    source_path: str
    identity_hash: str

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        return result


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path)
    raw_text = path.read_text()
    raw = yaml.load(raw_text, Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a YAML mapping")
    _strict(raw, {"schema_version", "model", "data", "objective", "training", "run"}, "recipe")
    if raw.get("schema_version") != 1:
        raise ValueError("recipe schema_version must be 1")
    for section in ("model", "data", "objective", "training", "run"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"recipe.{section} must be a mapping")

    model = raw["model"]
    _strict(model, {"architecture", "repo_id", "revision", "dtype", "local_path"}, "model")
    model_spec = ModelSpec(
        architecture=str(model["architecture"]),
        repo_id=str(model["repo_id"]),
        revision=str(model["revision"]),
        dtype=str(model.get("dtype", "bfloat16")),
        local_path=model.get("local_path"),
    )
    if model_spec.architecture != "qwen3_5":
        raise ValueError("this executable baseline currently supports architecture: qwen3_5")
    if model_spec.dtype not in {"bfloat16", "float32"}:
        raise ValueError("model.dtype must be bfloat16 or float32")
    if not model_spec.repo_id or not model_spec.revision:
        raise ValueError("model.repo_id and model.revision must be pinned and non-empty")

    data = raw["data"]
    _strict(
        data,
        {"repo_id", "revision", "config", "split", "adapter", "shuffle_seed", "shuffle_buffer_size"},
        "data",
    )
    data_spec = DataSpec(
        repo_id=str(data["repo_id"]),
        revision=str(data["revision"]),
        config=str(data.get("config", "default")),
        split=str(data["split"]),
        adapter=str(data["adapter"]),
        shuffle_seed=int(data.get("shuffle_seed", 17)),
        shuffle_buffer_size=int(data.get("shuffle_buffer_size", 10_000)),
    )
    if data_spec.shuffle_buffer_size <= 0:
        raise ValueError("data.shuffle_buffer_size must be positive")

    training = raw["training"]
    allowed_training = {
        "max_length",
        "truncation",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "steps",
        "peak_learning_rate",
        "warmup_steps",
        "minimum_learning_rate_ratio",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "max_grad_norm",
        "remat",
        "log_every",
        "checkpoint_every",
    }
    _strict(training, allowed_training, "training")
    training_spec = TrainingSpec(
        max_length=int(training["max_length"]),
        truncation=str(training.get("truncation", "right")),
        per_device_batch_size=int(training.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        steps=int(training["steps"]),
        peak_learning_rate=float(training["peak_learning_rate"]),
        warmup_steps=int(training.get("warmup_steps", 0)),
        minimum_learning_rate_ratio=float(training.get("minimum_learning_rate_ratio", 0.1)),
        beta1=float(training.get("beta1", 0.9)),
        beta2=float(training.get("beta2", 0.95)),
        epsilon=float(training.get("epsilon", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.1)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        remat=bool(training.get("remat", True)),
        log_every=int(training.get("log_every", 1)),
        checkpoint_every=int(training.get("checkpoint_every", 0)),
    )
    for name in (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "steps",
        "log_every",
    ):
        if getattr(training_spec, name) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if training_spec.max_length < 2:
        raise ValueError("training.max_length must be at least 2 for causal SFT")
    if training_spec.truncation not in {"right", "left", "reject"}:
        raise ValueError("training.truncation must be right, left, or reject")
    if not 0 <= training_spec.warmup_steps < training_spec.steps:
        raise ValueError("training.warmup_steps must be smaller than steps")
    if training_spec.checkpoint_every < 0:
        raise ValueError("training.checkpoint_every must be non-negative")
    if not math.isfinite(training_spec.peak_learning_rate) or training_spec.peak_learning_rate <= 0:
        raise ValueError("training.peak_learning_rate must be finite and positive")
    if not 0.0 <= training_spec.minimum_learning_rate_ratio <= 1.0:
        raise ValueError("training.minimum_learning_rate_ratio must be in [0, 1]")
    if not 0.0 <= training_spec.beta1 < 1.0 or not 0.0 <= training_spec.beta2 < 1.0:
        raise ValueError("training beta values must be in [0, 1)")
    if not math.isfinite(training_spec.epsilon) or training_spec.epsilon <= 0:
        raise ValueError("training.epsilon must be finite and positive")
    if not math.isfinite(training_spec.weight_decay) or training_spec.weight_decay < 0:
        raise ValueError("training.weight_decay must be finite and non-negative")
    if not math.isfinite(training_spec.max_grad_norm) or training_spec.max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be finite and positive")

    objective = dict(raw["objective"])
    from .data.tokenize import LossPolicy

    LossPolicy.from_config(objective)

    run = raw["run"]
    _strict(run, {"seed", "output_dir"}, "run")
    run_spec = RunSpec(seed=int(run.get("seed", 17)), output_dir=str(run["output_dir"]))
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return Recipe(
        schema_version=1,
        model=model_spec,
        data=data_spec,
        objective=objective,
        training=training_spec,
        run=run_spec,
        source_path=str(path.resolve()),
        identity_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )
