# Copyright 2026 JAXSFT contributors.
"""Single-file OLMo 2 causal language model implementation in pure JAX.

This module owns config normalization, parameter initialization, forward math,
Hugging Face safetensors conversion, validation, and parameter accounting. It
implements the dense text-only ``Olmo2ForCausalLM`` path used by AllenAI's OLMo
2 base checkpoints. Cache/generation kernels, attention dropout, non-default
RoPE scaling, and attention biases are deliberately outside this training path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ArrayTree = dict[str, Any]
_PRECISION = jax.lax.Precision.HIGHEST


@dataclass(frozen=True)
class Olmo2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500_000.0
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("OLMo 2 requires hidden_size == num_attention_heads * head_dim")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0 or self.initializer_range <= 0:
            raise ValueError("normalization, rotary, and initialization constants must be positive")
        if self.hidden_act != "silu":
            raise NotImplementedError("the OLMo 2 JAX path supports hidden_act='silu' only")
        if self.attention_dropout != 0.0:
            raise NotImplementedError("the deterministic OLMo 2 training path supports attention_dropout=0 only")
        if self.attention_bias:
            raise NotImplementedError("OLMo 2 attention bias is not implemented")
        for name in ("pad_token_id", "bos_token_id"):
            token_id = getattr(self, name)
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{name} must be within the vocabulary")
        eos_ids = () if self.eos_token_id is None else (
            self.eos_token_id if isinstance(self.eos_token_id, tuple) else (self.eos_token_id,)
        )
        if any(not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size for token_id in eos_ids):
            raise ValueError("eos_token_id must contain vocabulary indices")

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Olmo2Config":
        if raw.get("model_type") != "olmo2":
            raise ValueError(f"expected model_type='olmo2', got {raw.get('model_type')!r}")
        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"OLMo 2 config is missing fields: {missing}")
        heads = int(raw["num_attention_heads"])
        hidden = int(raw["hidden_size"])
        key_value_heads = raw.get("num_key_value_heads")
        rope_parameters = raw.get("rope_parameters")
        rope_scaling = raw.get("rope_scaling")
        if rope_parameters is not None and not isinstance(rope_parameters, Mapping):
            raise ValueError("rope_parameters must be a mapping when present")
        if rope_scaling not in (None, {}):
            raise NotImplementedError("OLMo 2 non-default rope_scaling is not implemented")
        rope = {} if rope_parameters is None else rope_parameters
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"OLMo 2 rope_type {rope_type!r} is not implemented")
        eos_token_id = raw.get("eos_token_id")
        if isinstance(eos_token_id, list):
            eos_token_id = tuple(int(value) for value in eos_token_id)
        elif eos_token_id is not None:
            eos_token_id = int(eos_token_id)
        return cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=heads if key_value_heads is None else int(key_value_heads),
            head_dim=int(raw.get("head_dim", hidden // heads)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 4096)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", 500_000.0))),
            initializer_range=float(raw.get("initializer_range", 0.02)),
            hidden_act=str(raw.get("hidden_act", "silu")),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            attention_bias=bool(raw.get("attention_bias", False)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            pad_token_id=None if raw.get("pad_token_id") is None else int(raw["pad_token_id"]),
            bos_token_id=None if raw.get("bos_token_id") is None else int(raw["bos_token_id"]),
            eos_token_id=eos_token_id,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Olmo2Config":
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, Mapping):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(raw)


def tiny_config(*, vocab_size: int = 128) -> Olmo2Config:
    """A small grouped-query configuration shared by CPU and HF parity tests."""

    return Olmo2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        pad_token_id=0,
        eos_token_id=2,
    )


def parameter_count(config: Olmo2Config) -> int:
    hidden = config.hidden_size
    query = config.query_width
    key_value = config.key_value_width
    embeddings_and_head = config.vocab_size * hidden
    if not config.tie_word_embeddings:
        embeddings_and_head += hidden * config.vocab_size
    attention = hidden * query + 2 * hidden * key_value + query * hidden + query + key_value
    mlp = 3 * hidden * config.intermediate_size
    block_norms = 2 * hidden
    return embeddings_and_head + hidden + config.num_hidden_layers * (attention + mlp + block_norms)


def _linear(x: jax.Array, kernel: jax.Array) -> jax.Array:
    return jnp.einsum("...i,io->...o", x, kernel, precision=_PRECISION)


def _rms_norm(x: jax.Array, weight: jax.Array, epsilon: float) -> jax.Array:
    dtype = x.dtype
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + epsilon)
    return (value * weight.astype(jnp.float32)).astype(dtype)


def _rotate_half(value: jax.Array) -> jax.Array:
    half = value.shape[-1] // 2
    return jnp.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def _apply_rope(
    query: jax.Array,
    key: jax.Array,
    position_ids: jax.Array,
    config: Olmo2Config,
) -> tuple[jax.Array, jax.Array]:
    inv_freq = 1.0 / (
        config.rope_theta
        ** (jnp.arange(0, config.head_dim, 2, dtype=jnp.float32) / config.head_dim)
    )
    frequencies = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None]
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)
    cos = jnp.cos(embedding)[:, :, None]
    sin = jnp.sin(embedding)[:, :, None]

    def rotate(value):
        return value * cos.astype(value.dtype) + _rotate_half(value) * sin.astype(value.dtype)

    return rotate(query), rotate(key)


def _attention(
    params: ArrayTree,
    config: Olmo2Config,
    hidden: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    batch, length, _ = hidden.shape
    query = _rms_norm(_linear(hidden, params["q_proj"]), params["q_norm"], config.rms_norm_eps)
    key = _rms_norm(_linear(hidden, params["k_proj"]), params["k_norm"], config.rms_norm_eps)
    value = _linear(hidden, params["v_proj"])
    query = query.reshape(batch, length, config.num_attention_heads, config.head_dim)
    key = key.reshape(batch, length, config.num_key_value_heads, config.head_dim)
    value = value.reshape(batch, length, config.num_key_value_heads, config.head_dim)
    query, key = _apply_rope(query, key, position_ids, config)
    repeats = config.num_attention_heads // config.num_key_value_heads
    key = jnp.repeat(key, repeats, axis=2)
    value = jnp.repeat(value, repeats, axis=2)
    scores = jnp.einsum("bshd,bthd->bhst", query, key, precision=_PRECISION).astype(jnp.float32)
    scores *= config.head_dim**-0.5
    causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
    allowed = causal[None, None] & attention_mask[:, None, None, :]
    scores = jnp.where(allowed, scores, -jnp.inf)
    all_masked = jnp.all(jnp.isneginf(scores), axis=-1, keepdims=True)
    weights = jax.nn.softmax(jnp.where(all_masked, 0.0, scores), axis=-1).astype(value.dtype)
    output = jnp.einsum("bhst,bthd->bshd", weights, value, precision=_PRECISION)
    output = output.reshape(batch, length, config.query_width)
    return _linear(output, params["o_proj"])


def _mlp(params: ArrayTree, hidden: jax.Array) -> jax.Array:
    gated = jax.nn.silu(_linear(hidden, params["gate_proj"])) * _linear(hidden, params["up_proj"])
    return _linear(gated, params["down_proj"])


def _decoder_layer(
    params: ArrayTree,
    config: Olmo2Config,
    hidden: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    residual = hidden
    hidden = _attention(params["self_attn"], config, hidden, attention_mask, position_ids)
    hidden = residual + _rms_norm(hidden, params["post_attention_layernorm"], config.rms_norm_eps)
    residual = hidden
    hidden = _mlp(params["mlp"], hidden)
    return residual + _rms_norm(hidden, params["post_feedforward_layernorm"], config.rms_norm_eps)


def forward(
    params: ArrayTree,
    config: Olmo2Config,
    input_ids: jax.Array,
    *,
    attention_mask: jax.Array | None = None,
    position_ids: jax.Array | None = None,
    remat: bool = False,
) -> jax.Array:
    """Return causal logits for a padded OLMo 2 text batch."""

    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    batch, length = input_ids.shape
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids, dtype=bool)
    else:
        attention_mask = jnp.asarray(attention_mask, dtype=bool)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if position_ids is None:
        position_ids = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    else:
        position_ids = jnp.asarray(position_ids, dtype=jnp.int32)
        if position_ids.shape == (1, length):
            position_ids = jnp.broadcast_to(position_ids, (batch, length))
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have shape [batch, length] or [1, length]")
    hidden = params["embed_tokens"][input_ids]
    layer_fn = _decoder_layer
    if remat:
        layer_fn = jax.checkpoint(layer_fn, static_argnums=(1,))
    for layer in params["layers"]:
        hidden = layer_fn(layer, config, hidden, attention_mask, position_ids)
    hidden = _rms_norm(hidden, params["norm"], config.rms_norm_eps)
    if config.tie_word_embeddings:
        return jnp.einsum("bsh,vh->bsv", hidden, params["embed_tokens"], precision=_PRECISION)
    return _linear(hidden, params["lm_head"])


def init_params(key: jax.Array, config: Olmo2Config, *, dtype: Any = jnp.float32) -> ArrayTree:
    """Initialize a readable parameter tree, primarily for tiny parity tests."""

    dtype = jnp.dtype(dtype)

    def normal(shape):
        nonlocal key
        key, subkey = jax.random.split(key)
        value = jax.random.normal(subkey, shape, jnp.float32) * config.initializer_range
        return value.astype(dtype)

    layers: list[ArrayTree] = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "self_attn": {
                    "q_proj": normal((config.hidden_size, config.query_width)),
                    "k_proj": normal((config.hidden_size, config.key_value_width)),
                    "v_proj": normal((config.hidden_size, config.key_value_width)),
                    "o_proj": normal((config.query_width, config.hidden_size)),
                    "q_norm": jnp.ones((config.query_width,), dtype),
                    "k_norm": jnp.ones((config.key_value_width,), dtype),
                },
                "mlp": {
                    "gate_proj": normal((config.hidden_size, config.intermediate_size)),
                    "up_proj": normal((config.hidden_size, config.intermediate_size)),
                    "down_proj": normal((config.intermediate_size, config.hidden_size)),
                },
                "post_attention_layernorm": jnp.ones((config.hidden_size,), dtype),
                "post_feedforward_layernorm": jnp.ones((config.hidden_size,), dtype),
            }
        )
    params: ArrayTree = {
        "embed_tokens": normal((config.vocab_size, config.hidden_size)),
        "layers": tuple(layers),
        "norm": jnp.ones((config.hidden_size,), dtype),
    }
    if not config.tie_word_embeddings:
        params["lm_head"] = normal((config.hidden_size, config.vocab_size))
    validate_params(params, config)
    return params


def _expected_shapes(config: Olmo2Config) -> ArrayTree:
    layers: list[ArrayTree] = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "self_attn": {
                    "q_proj": (config.hidden_size, config.query_width),
                    "k_proj": (config.hidden_size, config.key_value_width),
                    "v_proj": (config.hidden_size, config.key_value_width),
                    "o_proj": (config.query_width, config.hidden_size),
                    "q_norm": (config.query_width,),
                    "k_norm": (config.key_value_width,),
                },
                "mlp": {
                    "gate_proj": (config.hidden_size, config.intermediate_size),
                    "up_proj": (config.hidden_size, config.intermediate_size),
                    "down_proj": (config.intermediate_size, config.hidden_size),
                },
                "post_attention_layernorm": (config.hidden_size,),
                "post_feedforward_layernorm": (config.hidden_size,),
            }
        )
    shapes: ArrayTree = {
        "embed_tokens": (config.vocab_size, config.hidden_size),
        "layers": tuple(layers),
        "norm": (config.hidden_size,),
    }
    if not config.tie_word_embeddings:
        shapes["lm_head"] = (config.hidden_size, config.vocab_size)
    return shapes


def _shape_leaf(value: object) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, int) for item in value)


def validate_params(params: ArrayTree, config: Olmo2Config) -> None:
    expected = _expected_shapes(config)
    if jax.tree.structure(params) != jax.tree.structure(expected, is_leaf=_shape_leaf):
        raise ValueError("OLMo 2 parameter tree has unexpected keys or nesting")
    leaves = jax.tree.leaves(params)
    shapes = jax.tree.leaves(expected, is_leaf=_shape_leaf)
    for index, (value, shape) in enumerate(zip(leaves, shapes)):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"OLMo 2 parameter leaf {index} has shape {value.shape}, expected {shape}")
    if sum(int(np.prod(value.shape)) for value in leaves) != parameter_count(config):
        raise ValueError("OLMo 2 parameter count disagrees with the analytical count")


def _numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def convert_hf_state_dict(
    state: Mapping[str, Any],
    config: Olmo2Config,
    *,
    dtype: Any = jnp.bfloat16,
    strict: bool = True,
) -> ArrayTree:
    """Convert a Hugging Face ``Olmo2ForCausalLM`` state dictionary."""

    expected_keys: set[str] = set()

    def get(name: str) -> jax.Array:
        expected_keys.add(name)
        try:
            value = state[name]
        except KeyError as error:
            raise KeyError(f"OLMo 2 checkpoint is missing required tensor {name!r}") from error
        return jnp.asarray(_numpy_value(value), dtype)

    layers: list[ArrayTree] = []
    for index in range(config.num_hidden_layers):
        base = f"model.layers.{index}."
        attention = base + "self_attn."
        mlp = base + "mlp."
        layers.append(
            {
                "self_attn": {
                    "q_proj": get(attention + "q_proj.weight").T,
                    "k_proj": get(attention + "k_proj.weight").T,
                    "v_proj": get(attention + "v_proj.weight").T,
                    "o_proj": get(attention + "o_proj.weight").T,
                    "q_norm": get(attention + "q_norm.weight"),
                    "k_norm": get(attention + "k_norm.weight"),
                },
                "mlp": {
                    "gate_proj": get(mlp + "gate_proj.weight").T,
                    "up_proj": get(mlp + "up_proj.weight").T,
                    "down_proj": get(mlp + "down_proj.weight").T,
                },
                "post_attention_layernorm": get(base + "post_attention_layernorm.weight"),
                "post_feedforward_layernorm": get(base + "post_feedforward_layernorm.weight"),
            }
        )
    params: ArrayTree = {
        "embed_tokens": get("model.embed_tokens.weight"),
        "layers": tuple(layers),
        "norm": get("model.norm.weight"),
    }
    if config.tie_word_embeddings:
        if "lm_head.weight" in state:
            expected_keys.add("lm_head.weight")
            head = jnp.asarray(_numpy_value(state["lm_head.weight"]), dtype)
            if not np.array_equal(np.asarray(head), np.asarray(params["embed_tokens"])):
                raise ValueError("tied OLMo 2 lm_head and token embeddings differ")
    else:
        params["lm_head"] = get("lm_head.weight").T
    if strict:
        model_keys = {key for key in state if key.startswith("model.") or key == "lm_head.weight"}
        unexpected = model_keys - expected_keys
        if unexpected:
            raise ValueError(f"unexpected OLMo 2 tensors: {sorted(unexpected)}")
    validate_params(params, config)
    return params


class _SafeTensorMapping(Mapping[str, jax.Array]):
    def __init__(self, root: Path, weight_map: Mapping[str, str], stack: ExitStack):
        from safetensors import safe_open

        self.weight_map = dict(weight_map)
        self.handles = {
            filename: stack.enter_context(safe_open(str(root / filename), framework="flax"))
            for filename in sorted(set(weight_map.values()))
        }

    def __len__(self) -> int:
        return len(self.weight_map)

    def __iter__(self):
        return iter(self.weight_map)

    def __getitem__(self, key: str) -> jax.Array:
        filename = self.weight_map[key]
        try:
            return self.handles[filename].get_tensor(key)
        except KeyError as error:
            raise KeyError(f"indexed tensor {key!r} is missing from {filename!r}") from error


def load_hf_checkpoint(root: str | Path, *, dtype: Any = jnp.bfloat16) -> tuple[Olmo2Config, ArrayTree]:
    """Load OLMo 2 lazily from a local pinned Hugging Face snapshot."""

    root = Path(root)
    config = Olmo2Config.from_json(root / "config.json")
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_file():
        raw_index = json.loads(index_path.read_text())
        weight_map = raw_index.get("weight_map") if isinstance(raw_index, dict) else None
        if not isinstance(weight_map, Mapping):
            raise ValueError("safetensors index must contain a weight_map object")
    elif single_path.is_file():
        from safetensors import safe_open

        with safe_open(str(single_path), framework="flax") as handle:
            weight_map = {key: single_path.name for key in handle.keys()}
    else:
        raise FileNotFoundError(f"{root} has no safetensors checkpoint or index")
    with ExitStack() as stack:
        state = _SafeTensorMapping(root, weight_map, stack)
        params = convert_hf_state_dict(state, config, dtype=dtype, strict=True)
    return config, params


__all__ = [
    "Olmo2Config",
    "convert_hf_state_dict",
    "forward",
    "init_params",
    "load_hf_checkpoint",
    "parameter_count",
    "tiny_config",
    "validate_params",
]
