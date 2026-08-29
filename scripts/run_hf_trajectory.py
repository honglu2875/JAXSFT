#!/usr/bin/env python3
"""Run an independent Transformers Trainer/Accelerate CPU trajectory oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


_ARRAY_SPECS = {
    "input_ids": ("input_ids.npy", np.dtype(np.int32)),
    "attention_mask": ("attention_mask.npy", np.dtype(np.bool_)),
    "loss_weights": ("loss_weights.npy", np.dtype(np.float32)),
}
_TAPE_FIELDS = {
    "schema_version",
    "kind",
    "recipe_identity_sha256",
    "model",
    "data",
    "tokenizer_identity_sha256",
    "pad_token_id",
    "shape",
    "arrays",
    "stream_counters_after_export",
    "identity_sha256",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")
    return dict(value)


@dataclass(frozen=True)
class OracleRecipe:
    """The independently parsed, fully resolved fields needed by the CPU oracle."""

    identity_hash: str
    model: Mapping[str, Any]
    data: Mapping[str, Any]
    training: Mapping[str, Any]
    run: Mapping[str, Any]


def load_oracle_recipe(path: str | Path) -> OracleRecipe:
    """Parse and hash schema 1 without importing any JAXSFT runtime code."""

    raw = yaml.safe_load(Path(path).expanduser().resolve().read_text())
    root = _mapping(
        raw,
        name="recipe",
        fields={"schema_version", "model", "data", "objective", "training", "run"},
    )
    if root["schema_version"] != 1:
        raise ValueError("recipe schema_version must be 1")

    model_raw = _mapping(
        root["model"],
        name="recipe.model",
        fields={"architecture", "repo_id", "revision", "dtype"},
    )
    model = {**model_raw, "local_path": None}
    if model["architecture"] != "olmo2" or model["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("this CPU trajectory oracle requires an OLMo 2 float32/bfloat16 recipe")

    data = _mapping(
        root["data"],
        name="recipe.data",
        fields={
            "repo_id",
            "revision",
            "config",
            "split",
            "adapter",
            "renderer",
            "loading_mode",
            "shuffle_seed",
            "shuffle_buffer_size",
        },
    )
    objective_raw = _mapping(
        root["objective"],
        name="recipe.objective",
        fields={"conflict_mode", "rules"},
    )
    if not isinstance(objective_raw["rules"], list) or not objective_raw["rules"]:
        raise ValueError("recipe.objective.rules must be a non-empty list")
    objective_rules = []
    for index, value in enumerate(objective_raw["rules"]):
        if not isinstance(value, Mapping) or not set(value) <= {
            "name",
            "select",
            "weight",
            "require_match",
        }:
            raise ValueError(f"recipe.objective.rules[{index}] has unexpected fields")
        if set(value) < {"name", "select", "weight"}:
            raise ValueError(f"recipe.objective.rules[{index}] is incomplete")
        objective_rules.append(
            {
                "name": str(value["name"]),
                "select": dict(value["select"]),
                "weight": float(value["weight"]),
                "require_match": bool(value.get("require_match", False)),
            }
        )
    objective = {"conflict_mode": str(objective_raw["conflict_mode"]), "rules": objective_rules}

    training = _mapping(
        root["training"],
        name="recipe.training",
        fields={
            "max_length",
            "truncation",
            "truncation_min_context_tokens",
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
        },
    )
    run = _mapping(root["run"], name="recipe.run", fields={"seed", "output_dir"})
    resolved = {
        "schema_version": 1,
        "model": model,
        "data": data,
        "objective": objective,
        "training": training,
        "run": run,
    }
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identity = hashlib.sha256(b"jaxsft-recipe-v1\0" + canonical.encode()).hexdigest()
    return OracleRecipe(identity_hash=identity, model=model, data=data, training=training, run=run)


@dataclass(frozen=True)
class OracleTape:
    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def identity_hash(self) -> str:
        return str(self.manifest["identity_sha256"])

    @property
    def steps(self) -> int:
        return int(self.manifest["shape"]["steps"])

    @property
    def batch_size(self) -> int:
        return int(self.manifest["shape"]["batch_size"])

    @property
    def length(self) -> int:
        return int(self.manifest["shape"]["length"])


def load_oracle_tape(path: str | Path, recipe: OracleRecipe) -> OracleTape:
    """Independently verify the content-addressed NumPy tape and its recipe pins."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"regular batch tape manifest is required below {root}")
    manifest = json.loads(manifest_path.read_text())
    _mapping(manifest, name="batch tape", fields=_TAPE_FIELDS)
    if manifest["schema_version"] != 1 or manifest["kind"] != "jaxsft_batch_tape":
        raise ValueError("unsupported batch tape schema")
    identity_payload = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    identity = hashlib.sha256(b"jaxsft-batch-tape-v1\0" + canonical.encode()).hexdigest()
    if manifest["identity_sha256"] != identity:
        raise ValueError("batch tape manifest identity differs from its contents")
    if manifest["recipe_identity_sha256"] != recipe.identity_hash:
        raise ValueError("batch tape recipe identity differs from the independently parsed recipe")
    expected_model = {key: recipe.model[key] for key in ("repo_id", "revision")}
    if manifest["model"] != expected_model:
        raise ValueError("batch tape model pins differ from the recipe")
    expected_data = {
        key: recipe.data[key]
        for key in ("repo_id", "revision", "config", "split", "adapter", "renderer", "loading_mode")
    }
    if manifest["data"] != expected_data:
        raise ValueError("batch tape data pins differ from the recipe")

    shape = _mapping(
        manifest["shape"],
        name="batch tape shape",
        fields={"steps", "batch_size", "length"},
    )
    dimensions = tuple(int(shape[name]) for name in ("steps", "batch_size", "length"))
    if any(dimension <= 0 for dimension in dimensions) or dimensions[-1] < 2:
        raise ValueError("batch tape dimensions are invalid")
    arrays_manifest = _mapping(
        manifest["arrays"],
        name="batch tape arrays",
        fields=set(_ARRAY_SPECS),
    )
    arrays = {}
    for name, (filename, dtype) in _ARRAY_SPECS.items():
        record = _mapping(
            arrays_manifest[name],
            name=f"batch tape {name}",
            fields={"file", "dtype", "sha256"},
        )
        array_path = root / filename
        if record["file"] != filename or record["dtype"] != dtype.name:
            raise ValueError(f"batch tape {name} declaration is invalid")
        if not array_path.is_file() or array_path.is_symlink() or record["sha256"] != _file_sha256(array_path):
            raise ValueError(f"batch tape {name} digest differs")
        values = np.load(array_path, allow_pickle=False, mmap_mode="r")
        if values.dtype != dtype or values.shape != dimensions:
            raise ValueError(f"batch tape {name} dtype or shape differs")
        arrays[name] = values
    if np.any(arrays["input_ids"] < 0):
        raise ValueError("batch tape input IDs must be non-negative")
    if np.any(np.diff(arrays["attention_mask"].astype(np.int8), axis=-1) > 0):
        raise ValueError("batch tape attention masks must be contiguous prefixes")
    weights = arrays["loss_weights"]
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("batch tape weights must be finite and non-negative")
    if np.any(weights[~arrays["attention_mask"]]) or np.any(weights[..., 0]):
        raise ValueError("batch tape weights must be zero on padding and first-token positions")
    if np.any(np.sum(weights[..., 1:], axis=(1, 2), dtype=np.float64) <= 0):
        raise ValueError("every batch tape step must contain selected target weight")
    return OracleTape(root=root, manifest=manifest, arrays=arrays)


def _snapshot_file_identity(snapshot: Path) -> dict[str, Any]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
    elif (snapshot / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError("model snapshot contains no safetensors checkpoint")
    records = []
    for name in names:
        target = snapshot / name
        records.append({"file": name, "bytes": target.stat().st_size, "sha256": _file_sha256(target)})
    return {"files": records}


def _oracle_source_identity() -> dict[str, str | None]:
    capsule = os.environ.get("JAXSFT_SOURCE_SHA256")
    if capsule is not None and (len(capsule) != 64 or any(value not in "0123456789abcdef" for value in capsule)):
        raise ValueError("JAXSFT_SOURCE_SHA256 must be a lowercase SHA-256 digest")
    return {
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "source_capsule_sha256": capsule,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-tape", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trainer-output", required=True)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=120)
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = load_oracle_recipe(args.config)
    training = recipe.training
    if int(training["gradient_accumulation_steps"]) != 1:
        raise ValueError("the CPU trajectory oracle requires gradient_accumulation_steps=1")
    tape = load_oracle_tape(args.batch_tape, recipe)
    steps = tape.steps if args.steps is None else args.steps
    if steps != int(training["steps"]) or tape.steps < steps:
        raise ValueError("the stock scheduler oracle must execute the recipe's complete trajectory")
    examples_per_virtual_device = int(training["per_device_batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    if tape.batch_size % examples_per_virtual_device or tape.length != int(training["max_length"]):
        raise ValueError("batch tape shape is incompatible with the TPU recipe")
    virtual_tpu_device_count = tape.batch_size // examples_per_virtual_device
    snapshot = Path(args.model_snapshot).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"model snapshot does not exist: {snapshot}")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite trajectory result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer_output = Path(args.trainer_output).expanduser().resolve()
    if trainer_output.exists():
        raise FileExistsError(f"refusing to reuse Trainer output directory: {trainer_output}")
    if not 1 <= args.threads <= (os.cpu_count() or 1):
        raise ValueError("--threads must be within the available logical CPU count")

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(4, args.threads))
    torch.manual_seed(int(recipe.run["seed"]))
    np.random.seed(int(recipe.run["seed"]))

    class TapeDataset(Dataset):
        def __len__(self):
            return steps * tape.batch_size

        def __getitem__(self, index):
            step, item = divmod(index, tape.batch_size)
            return {name: np.array(values[step, item], copy=True) for name, values in tape.arrays.items()}

    class WeightedSFTTrainer(Trainer):
        """Only the repository-specific weighted objective; Trainer owns optimization."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.model_accepts_loss_kwargs = False
            self.objective_metrics: list[dict[str, float | int]] = []

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            weights = inputs.pop("loss_weights").to(torch.float32)[:, 1:]
            inputs["input_ids"] = inputs["input_ids"].to(torch.int64)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            outputs = model(**inputs, use_cache=False)
            prediction = outputs.logits[:, :-1].to(torch.float32)
            targets = input_ids[:, 1:]
            token_losses = functional.cross_entropy(
                prediction.reshape(-1, prediction.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            numerator = torch.sum(token_losses * weights, dtype=torch.float32)
            denominator = torch.sum(weights, dtype=torch.float32)
            loss = numerator / torch.clamp_min(denominator, 1e-12)
            correct = torch.argmax(prediction, dim=-1) == targets
            correct_weight = torch.sum(correct.to(torch.float32) * weights, dtype=torch.float32)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.objective_metrics.append(
                {
                    "step": len(self.objective_metrics) + 1,
                    "loss": float(loss.detach()),
                    "loss_numerator": float(numerator.detach()),
                    "loss_denominator": float(denominator.detach()),
                    "selected_accuracy": float((correct_weight / denominator).detach()),
                    "input_tokens": int(attention_mask.sum().detach()),
                    "learning_rate": learning_rate,
                }
            )
            return (loss, outputs) if return_outputs else loss

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "eager"
    config.use_cache = False
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_load_seconds = time.monotonic() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 1_484_916_736:
        raise ValueError(f"unexpected OLMo 2 parameter count: {parameter_count}")

    training_args = TrainingArguments(
        output_dir=str(trainer_output),
        use_cpu=True,
        bf16=args.precision == "bfloat16",
        fp16=False,
        max_steps=steps,
        per_device_train_batch_size=tape.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=float(training["peak_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        adam_beta1=float(training["beta1"]),
        adam_beta2=float(training["beta2"]),
        adam_epsilon=float(training["epsilon"]),
        max_grad_norm=float(training["max_grad_norm"]),
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": float(training["minimum_learning_rate_ratio"])},
        warmup_steps=int(training["warmup_steps"]),
        gradient_checkpointing=bool(training["remat"]),
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        train_sampling_strategy="sequential",
        full_determinism=True,
        seed=int(recipe.run["seed"]),
        data_seed=int(recipe.data["shuffle_seed"]),
    )
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=TapeDataset(),
        data_collator=default_data_collator,
    )
    train_started = time.monotonic()
    train_result = trainer.train()
    train_wall_seconds = time.monotonic() - train_started
    if len(trainer.objective_metrics) != steps:
        raise RuntimeError(
            f"Trainer evaluated {len(trainer.objective_metrics)} objective batches for {steps} requested steps"
        )
    log_by_step = {
        int(record["step"]): record
        for record in trainer.state.log_history
        if "step" in record and "grad_norm" in record
    }
    trajectory = []
    for metric in trainer.objective_metrics:
        record = dict(metric)
        log = log_by_step.get(int(record["step"]), {})
        if "grad_norm" in log:
            record["gradient_norm"] = float(log["grad_norm"])
        trajectory.append(record)

    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    moment_dtypes = sorted(
        {
            str(value.dtype)
            for state in trainer.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
    )
    packages = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "accelerate", "numpy", "pyyaml")
    }
    payload = {
        "schema_version": 2,
        "status": "complete",
        "framework": "transformers.Trainer+accelerate",
        "oracle_source": _oracle_source_identity(),
        "recipe_identity_sha256": recipe.identity_hash,
        "batch_tape_identity_sha256": tape.identity_hash,
        "model": {
            "repo_id": recipe.model["repo_id"],
            "revision": recipe.model["revision"],
            "parameter_count": parameter_count,
            "parameter_dtypes": parameter_dtypes,
            "attention_implementation": "eager",
            "checkpoint": _snapshot_file_identity(snapshot),
        },
        "execution": {
            "device": "cpu",
            "precision": args.precision,
            "threads": args.threads,
            "steps": steps,
            "batch_size": tape.batch_size,
            "length": tape.length,
            "matched_virtual_tpu_device_count": virtual_tpu_device_count,
            "model_load_seconds": model_load_seconds,
            "train_wall_seconds": train_wall_seconds,
            "seconds_per_step": train_wall_seconds / steps,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "accelerator_class": type(trainer.accelerator).__name__,
        },
        "oracle_contract": {
            "jaxsft_runtime_imports": False,
            "model_api": "transformers.AutoModelForCausalLM",
            "training_api": "transformers.Trainer backed by accelerate.Accelerator",
            "objective_api": "torch.nn.functional.cross_entropy with external token weights",
            "optimizer_factory": "stock transformers.Trainer.create_optimizer",
            "scheduler_factory": "stock transformers.Trainer.create_scheduler",
        },
        "optimizer_contract": {
            "class": f"{type(trainer.optimizer).__module__}.{type(trainer.optimizer).__name__}",
            "parameter_dtypes": parameter_dtypes,
            "moment_dtypes": moment_dtypes,
            "weight_decay_by_group": [float(group["weight_decay"]) for group in trainer.optimizer.param_groups],
            "gradient_clipping": "Trainer/Accelerate global L2 norm",
            "scheduler_class": f"{type(trainer.lr_scheduler).__module__}.{type(trainer.lr_scheduler).__name__}",
            "scheduler": "transformers cosine_with_min_lr",
        },
        "trajectory": trajectory,
        "trainer_metrics": {key: float(value) for key, value in train_result.metrics.items()},
    }
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
