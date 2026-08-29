"""Content-addressed, framework-neutral batches for trajectory parity runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_ARRAY_FILES = {
    "input_ids": ("input_ids.npy", np.dtype(np.int32)),
    "attention_mask": ("attention_mask.npy", np.dtype(np.bool_)),
    "loss_weights": ("loss_weights.npy", np.dtype(np.float32)),
}
_MANIFEST_FIELDS = {
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


def _identity(payload: Mapping[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "identity_sha256"}
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(b"jaxsft-batch-tape-v1\0" + canonical.encode()).hexdigest()


def _strict_string_mapping(raw: object, *, name: str, fields: set[str]) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError(f"batch tape {name} fields must be exactly {sorted(fields)}")
    result = {str(key): value for key, value in raw.items()}
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError(f"batch tape {name} values must be non-empty strings")
    return result


@dataclass(frozen=True)
class BatchTape:
    """Validated arrays with step-major shape ``[steps, batch, length]``."""

    root: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def identity_hash(self) -> str:
        return str(self.manifest["identity_sha256"])

    @property
    def recipe_identity_hash(self) -> str:
        return str(self.manifest["recipe_identity_sha256"])

    @property
    def steps(self) -> int:
        return int(self.manifest["shape"]["steps"])

    @property
    def batch_size(self) -> int:
        return int(self.manifest["shape"]["batch_size"])

    @property
    def length(self) -> int:
        return int(self.manifest["shape"]["length"])

    def jax_batch(
        self,
        step: int,
        *,
        local_device_count: int,
        accumulation_steps: int,
        per_device_batch_size: int,
    ) -> dict[str, np.ndarray]:
        if not 0 <= step < self.steps:
            raise IndexError(f"batch tape step {step} is outside [0, {self.steps})")
        expected = local_device_count * accumulation_steps * per_device_batch_size
        if self.batch_size != expected:
            raise ValueError(
                f"batch tape has global batch {self.batch_size}, but topology/recipe require {expected}"
            )
        result = {}
        for name, values in self.arrays.items():
            flat = np.asarray(values[step])
            shaped = flat.reshape(
                accumulation_steps,
                local_device_count,
                per_device_batch_size,
                self.length,
            )
            result[name] = np.swapaxes(shaped, 0, 1)
        return result

    def state_dict(self, *, next_step: int) -> dict[str, Any]:
        if not 0 <= next_step <= self.steps:
            raise ValueError("batch tape cursor is outside its step range")
        return {
            "schema_version": 1,
            "kind": "batch_tape",
            "identity_sha256": self.identity_hash,
            "next_step": next_step,
        }

    def validate_state_dict(self, raw: object, *, expected_step: int) -> None:
        expected_fields = {"schema_version", "kind", "identity_sha256", "next_step"}
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ValueError("batch tape checkpoint has unexpected fields")
        if raw.get("schema_version") != 1 or raw.get("kind") != "batch_tape":
            raise ValueError("unsupported batch tape checkpoint state")
        if raw.get("identity_sha256") != self.identity_hash:
            raise ValueError("checkpoint batch tape differs from this run")
        if int(raw.get("next_step", -1)) != expected_step:
            raise ValueError("batch tape cursor differs from checkpoint step")

    @classmethod
    def load(cls, root: str | Path, *, expected_recipe_identity: str | None = None) -> BatchTape:
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not root.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError(f"batch tape directory and regular manifest are required: {root}")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS:
            raise ValueError("batch tape manifest has unexpected fields")
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "jaxsft_batch_tape":
            raise ValueError("unsupported batch tape manifest")
        identity = manifest.get("identity_sha256")
        if not isinstance(identity, str) or identity != _identity(manifest):
            raise ValueError("batch tape identity does not match its manifest")
        recipe_identity = manifest.get("recipe_identity_sha256")
        if not isinstance(recipe_identity, str) or len(recipe_identity) != 64:
            raise ValueError("batch tape recipe identity must be a SHA-256 digest")
        if expected_recipe_identity is not None and recipe_identity != expected_recipe_identity:
            raise ValueError("batch tape recipe identity differs from this run")
        _strict_string_mapping(manifest.get("model"), name="model", fields={"repo_id", "revision"})
        _strict_string_mapping(
            manifest.get("data"),
            name="data",
            fields={"repo_id", "revision", "config", "split", "adapter", "renderer", "loading_mode"},
        )
        tokenizer_identity = manifest.get("tokenizer_identity_sha256")
        if not isinstance(tokenizer_identity, str) or len(tokenizer_identity) != 64:
            raise ValueError("batch tape tokenizer identity must be a SHA-256 digest")
        pad_token_id = manifest.get("pad_token_id")
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("batch tape pad_token_id must be a non-negative integer")

        shape = manifest.get("shape")
        if not isinstance(shape, Mapping) or set(shape) != {"steps", "batch_size", "length"}:
            raise ValueError("batch tape shape fields are invalid")
        dimensions = tuple(int(shape[name]) for name in ("steps", "batch_size", "length"))
        if any(value <= 0 for value in dimensions) or dimensions[2] < 2:
            raise ValueError("batch tape dimensions must be positive and length must be at least two")

        arrays_manifest = manifest.get("arrays")
        if not isinstance(arrays_manifest, Mapping) or set(arrays_manifest) != set(_ARRAY_FILES):
            raise ValueError("batch tape array manifest is incomplete")
        arrays: dict[str, np.ndarray] = {}
        for name, (filename, dtype) in _ARRAY_FILES.items():
            record = arrays_manifest[name]
            if not isinstance(record, Mapping) or set(record) != {"file", "dtype", "sha256"}:
                raise ValueError(f"batch tape {name} record has unexpected fields")
            if record.get("file") != filename or record.get("dtype") != dtype.name:
                raise ValueError(f"batch tape {name} file or dtype declaration is invalid")
            path = root / filename
            if not path.is_file() or path.is_symlink() or record.get("sha256") != _file_sha256(path):
                raise ValueError(f"batch tape {name} file digest does not match")
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            if value.dtype != dtype or value.shape != dimensions:
                raise ValueError(f"batch tape {name} array shape or dtype differs from its contract")
            arrays[name] = value

        input_ids = np.asarray(arrays["input_ids"])
        attention_mask = np.asarray(arrays["attention_mask"])
        loss_weights = np.asarray(arrays["loss_weights"])
        if np.any(input_ids < 0):
            raise ValueError("batch tape input IDs must be non-negative")
        if np.any(np.diff(attention_mask.astype(np.int8), axis=-1) > 0):
            raise ValueError("batch tape attention masks must be contiguous prefixes")
        if not np.all(np.isfinite(loss_weights)) or np.any(loss_weights < 0):
            raise ValueError("batch tape loss weights must be finite and non-negative")
        if np.any(loss_weights[~attention_mask]) or np.any(loss_weights[..., 0]):
            raise ValueError("batch tape weights must be zero on padding and first-token positions")
        denominators = np.sum(loss_weights[..., 1:], axis=(1, 2), dtype=np.float64)
        if np.any(denominators <= 0):
            raise ValueError("every batch tape step must contain selected target weight")
        counters = manifest.get("stream_counters_after_export")
        if not isinstance(counters, Mapping):
            raise ValueError("batch tape stream counters must be a mapping")
        return cls(root=root, manifest=dict(manifest), arrays=arrays)


def write_batch_tape(
    root: str | Path,
    batches: Sequence[Mapping[str, np.ndarray]],
    *,
    recipe_identity_hash: str,
    model: Mapping[str, str],
    data: Mapping[str, str],
    tokenizer_identity_hash: str,
    pad_token_id: int,
    stream_counters: Mapping[str, Any],
) -> BatchTape:
    """Write a new tape and return it only after a strict read-back check."""

    if not batches:
        raise ValueError("cannot write an empty batch tape")
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    expected_shape = None
    for name, (_, dtype) in _ARRAY_FILES.items():
        try:
            value = np.stack([np.asarray(batch[name]) for batch in batches])
        except KeyError as error:
            raise ValueError(f"batch tape input is missing {name}") from error
        if value.ndim != 3:
            raise ValueError(f"batch tape {name} must stack to [steps, batch, length]")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError("batch tape arrays must have identical shapes")
        arrays[name] = value.astype(dtype, copy=False)
    assert expected_shape is not None

    array_records = {}
    for name, (filename, dtype) in _ARRAY_FILES.items():
        path = root / filename
        np.save(path, arrays[name], allow_pickle=False)
        array_records[name] = {"file": filename, "dtype": dtype.name, "sha256": _file_sha256(path)}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "jaxsft_batch_tape",
        "recipe_identity_sha256": recipe_identity_hash,
        "model": dict(model),
        "data": dict(data),
        "tokenizer_identity_sha256": tokenizer_identity_hash,
        "pad_token_id": int(pad_token_id),
        "shape": {
            "steps": int(expected_shape[0]),
            "batch_size": int(expected_shape[1]),
            "length": int(expected_shape[2]),
        },
        "arrays": array_records,
        "stream_counters_after_export": dict(stream_counters),
    }
    manifest["identity_sha256"] = _identity(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return BatchTape.load(root, expected_recipe_identity=recipe_identity_hash)
