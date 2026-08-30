"""Explicit, model-agnostic LoRA math over audited parameter paths.

Frozen base parameters and trainable adapters are deliberately separate
PyTrees.  A caller closes over the base tree and differentiates only with
respect to the returned adapter tree; this prevents accidental gradients or
Adam slots for a large frozen checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp


PathPart = str | int
ParameterPath = tuple[PathPart, ...]
AdapterTree = dict[str, dict[str, jax.Array]]
_PRECISION = jax.lax.Precision.HIGHEST


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float = 0.0
    initializer: str = "kaiming_uniform"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.initializer != "kaiming_uniform":
            raise ValueError("the initial LoRA path supports kaiming_uniform initialization only")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_parameter_path(path: ParameterPath) -> str:
    """Return one stable, readable key for a dict/list parameter path."""

    if not path:
        raise ValueError("parameter path must not be empty")
    pieces: list[str] = []
    for index, part in enumerate(path):
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise TypeError(f"path part {part!r} must be a string or integer")
        if isinstance(part, int):
            if part < 0:
                raise ValueError("parameter path indices must be non-negative")
            pieces.append(f"[{part}]")
        else:
            if not part or not part.isidentifier():
                raise ValueError(f"parameter path key must be a Python identifier: {part!r}")
            pieces.append(part if index == 0 else f".{part}")
    return "".join(pieces)


def parameter_at_path(tree: object, path: ParameterPath) -> Any:
    value = tree
    for part in path:
        if isinstance(part, str):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(f"parameter path {format_parameter_path(path)!r} does not exist")
            value = value[part]
        else:
            if not isinstance(value, (tuple, list)) or not 0 <= part < len(value):
                raise KeyError(f"parameter path {format_parameter_path(path)!r} does not exist")
            value = value[part]
    return value


def _replace_at_path(tree: object, path: ParameterPath, replacement: Any) -> object:
    if not path:
        return replacement
    part, *remainder = path
    rest = tuple(remainder)
    if isinstance(part, str):
        if not isinstance(tree, Mapping) or part not in tree:
            raise KeyError(f"parameter path {format_parameter_path(path)!r} does not exist")
        updated = dict(tree)
        updated[part] = _replace_at_path(tree[part], rest, replacement)
        return updated
    if not isinstance(tree, (tuple, list)) or not 0 <= part < len(tree):
        raise KeyError(f"parameter path {format_parameter_path(path)!r} does not exist")
    updated_sequence = list(tree)
    updated_sequence[part] = _replace_at_path(tree[part], rest, replacement)
    return tuple(updated_sequence) if isinstance(tree, tuple) else updated_sequence


def audit_lora_targets(
    base_params: object,
    requested_paths: Sequence[ParameterPath],
    *,
    eligible_paths: Sequence[ParameterPath],
) -> tuple[ParameterPath, ...]:
    """Validate an explicit target subset and return deterministic path order."""

    requested = tuple(tuple(path) for path in requested_paths)
    eligible = tuple(tuple(path) for path in eligible_paths)
    if not requested:
        raise ValueError("at least one LoRA target is required")
    if len(set(requested)) != len(requested):
        raise ValueError("LoRA target paths contain duplicates")
    if len(set(eligible)) != len(eligible):
        raise ValueError("model LoRA eligible paths contain duplicates")
    unknown = set(requested) - set(eligible)
    if unknown:
        names = sorted(format_parameter_path(path) for path in unknown)
        raise ValueError(f"LoRA targets are not declared eligible by the model: {names}")

    ordered = tuple(sorted(requested, key=format_parameter_path))
    for path in ordered:
        value = parameter_at_path(base_params, path)
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"LoRA target {format_parameter_path(path)!r} is not an array")
        if value.ndim != 2:
            raise ValueError(f"LoRA target {format_parameter_path(path)!r} must be a rank-2 kernel")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError(f"LoRA target {format_parameter_path(path)!r} must have a floating dtype")
    return ordered


def init_lora_adapters(
    key: jax.Array,
    base_params: object,
    target_paths: Sequence[ParameterPath],
    *,
    eligible_paths: Sequence[ParameterPath],
    config: LoRAConfig,
    dtype: Any = jnp.bfloat16,
) -> AdapterTree:
    """Initialize A with PEFT-compatible bounds and B to exact zeros."""

    targets = audit_lora_targets(base_params, target_paths, eligible_paths=eligible_paths)
    dtype = jnp.dtype(dtype)
    keys = jax.random.split(key, len(targets))
    adapters: AdapterTree = {}
    for target_key, path in zip(keys, targets, strict=True):
        kernel = parameter_at_path(base_params, path)
        input_size, output_size = (int(value) for value in kernel.shape)
        if config.rank > min(input_size, output_size):
            raise ValueError(
                f"LoRA rank {config.rank} exceeds a dimension of {format_parameter_path(path)!r} "
                f"with shape {tuple(kernel.shape)}"
            )
        # torch.nn.init.kaiming_uniform_(a=sqrt(5)) reduces to this bound.
        bound = 1.0 / math.sqrt(input_size)
        a = jax.random.uniform(
            target_key,
            (input_size, config.rank),
            dtype=jnp.float32,
            minval=-bound,
            maxval=bound,
        ).astype(dtype)
        b = jnp.zeros((config.rank, output_size), dtype=dtype)
        adapters[format_parameter_path(path)] = {"a": a, "b": b}
    validate_lora_adapters(base_params, adapters, targets, config=config)
    return adapters


def validate_lora_adapters(
    base_params: object,
    adapters: Mapping[str, Mapping[str, Any]],
    target_paths: Sequence[ParameterPath],
    *,
    config: LoRAConfig,
) -> None:
    targets = tuple(tuple(path) for path in target_paths)
    expected_keys = {format_parameter_path(path) for path in targets}
    if set(adapters) != expected_keys:
        missing = sorted(expected_keys - set(adapters))
        extra = sorted(set(adapters) - expected_keys)
        raise ValueError(f"LoRA adapter key audit failed: missing={missing}, unexpected={extra}")
    for path in targets:
        name = format_parameter_path(path)
        pair = adapters[name]
        if not isinstance(pair, Mapping) or set(pair) != {"a", "b"}:
            raise ValueError(f"LoRA adapter {name!r} must contain exactly a and b")
        kernel = parameter_at_path(base_params, path)
        expected_a = (int(kernel.shape[0]), config.rank)
        expected_b = (config.rank, int(kernel.shape[1]))
        if tuple(pair["a"].shape) != expected_a or tuple(pair["b"].shape) != expected_b:
            raise ValueError(
                f"LoRA adapter {name!r} shape mismatch: got {tuple(pair['a'].shape)}, "
                f"{tuple(pair['b'].shape)}; expected {expected_a}, {expected_b}"
            )
        if pair["a"].dtype != pair["b"].dtype or not jnp.issubdtype(pair["a"].dtype, jnp.inexact):
            raise TypeError(f"LoRA adapter {name!r} must have one floating dtype")


def adapter_for_path(adapters: Mapping[str, Mapping[str, jax.Array]], path: ParameterPath):
    try:
        return adapters[format_parameter_path(path)]
    except KeyError as error:
        raise KeyError(f"no LoRA adapter for {format_parameter_path(path)!r}") from error


def lora_linear(
    inputs: jax.Array,
    base_kernel: jax.Array,
    adapter: Mapping[str, jax.Array],
    *,
    config: LoRAConfig,
    training: bool = False,
    dropout_key: jax.Array | None = None,
    precision: jax.lax.Precision = _PRECISION,
) -> jax.Array:
    """Apply a dense base plus unmerged low-rank update."""

    if inputs.shape[-1] != base_kernel.shape[0]:
        raise ValueError("input width does not match base kernel")
    a, b = adapter["a"], adapter["b"]
    if tuple(a.shape) != (base_kernel.shape[0], config.rank) or tuple(b.shape) != (
        config.rank,
        base_kernel.shape[1],
    ):
        raise ValueError("LoRA adapter shapes do not match base kernel/config")
    adapter_inputs = inputs
    if training and config.dropout:
        if dropout_key is None:
            raise ValueError("LoRA dropout requires dropout_key while training")
        keep_probability = 1.0 - config.dropout
        mask = jax.random.bernoulli(dropout_key, keep_probability, inputs.shape)
        adapter_inputs = jnp.where(mask, inputs / keep_probability, jnp.zeros((), inputs.dtype))
    base_output = jnp.matmul(inputs, base_kernel, precision=precision)
    adapter_inputs = adapter_inputs.astype(a.dtype)
    hidden = jnp.matmul(adapter_inputs, a, precision=precision)
    delta = jnp.matmul(hidden, b, precision=precision) * jnp.asarray(config.scale, hidden.dtype)
    return base_output + delta.astype(base_output.dtype)


def merge_lora_adapters(
    base_params: object,
    adapters: Mapping[str, Mapping[str, jax.Array]],
    target_paths: Sequence[ParameterPath],
    *,
    config: LoRAConfig,
) -> object:
    """Return a new base tree with float32-computed adapter deltas merged."""

    targets = tuple(tuple(path) for path in target_paths)
    validate_lora_adapters(base_params, adapters, targets, config=config)
    merged = base_params
    for path in targets:
        kernel = parameter_at_path(merged, path)
        adapter = adapter_for_path(adapters, path)
        delta = jnp.matmul(
            adapter["a"].astype(jnp.float32),
            adapter["b"].astype(jnp.float32),
            precision=_PRECISION,
        ) * jnp.asarray(config.scale, jnp.float32)
        updated = (kernel.astype(jnp.float32) + delta).astype(kernel.dtype)
        merged = _replace_at_path(merged, path, updated)
    return merged


def lora_parameter_count(adapters: Mapping[str, Mapping[str, jax.Array]]) -> int:
    return sum(int(value.size) for value in jax.tree.leaves(adapters))


def flatten_lora_adapters(
    adapters: Mapping[str, Mapping[str, jax.Array]],
) -> dict[str, jax.Array]:
    """Flatten adapters into stable names suitable for an adapter-only file."""

    flat: dict[str, jax.Array] = {}
    for name, pair in sorted(adapters.items()):
        if not isinstance(pair, Mapping) or set(pair) != {"a", "b"}:
            raise ValueError(f"LoRA adapter {name!r} must contain exactly a and b")
        flat[f"{name}.lora_a"] = pair["a"]
        flat[f"{name}.lora_b"] = pair["b"]
    return flat


def unflatten_lora_adapters(
    flat: Mapping[str, jax.Array],
    base_params: object,
    target_paths: Sequence[ParameterPath],
    *,
    config: LoRAConfig,
) -> AdapterTree:
    expected: set[str] = set()
    adapters: AdapterTree = {}
    for path in target_paths:
        name = format_parameter_path(tuple(path))
        a_name, b_name = f"{name}.lora_a", f"{name}.lora_b"
        expected.update((a_name, b_name))
        if a_name not in flat or b_name not in flat:
            raise ValueError(f"adapter-only state is missing {name!r}")
        adapters[name] = {"a": flat[a_name], "b": flat[b_name]}
    unexpected = sorted(set(flat) - expected)
    if unexpected:
        raise ValueError(f"adapter-only state has unexpected tensors: {unexpected}")
    validate_lora_adapters(base_params, adapters, target_paths, config=config)
    return adapters


__all__ = [
    "AdapterTree",
    "LoRAConfig",
    "ParameterPath",
    "_PRECISION",
    "adapter_for_path",
    "audit_lora_targets",
    "flatten_lora_adapters",
    "format_parameter_path",
    "init_lora_adapters",
    "lora_linear",
    "lora_parameter_count",
    "merge_lora_adapters",
    "parameter_at_path",
    "unflatten_lora_adapters",
    "validate_lora_adapters",
]
