# Copyright 2026 JAXSFT contributors.
# Portions of the mathematical organization are derived from JAXML's
# Apache-2.0 Qwen3.5 implementation, Copyright 2026 Honglu Fan.
"""Single-file, text-only dense Qwen3.5 implementation in pure JAX.

This module owns config normalization, parameter initialization, forward math,
Hugging Face safetensors conversion, validation, and parameter accounting. The
released 0.8B checkpoint is wrapped in a multimodal outer model; vision and MTP
tensors are deliberately rejected from the text parameter tree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ArrayTree = dict[str, Any]
_PRECISION = jax.lax.Precision.HIGH


@dataclass(frozen=True)
class Qwen35Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_value_head_dim: int
    linear_num_value_heads: int
    max_position_embeddings: int = 262_144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    attention_dropout: float = 0.0
    attention_bias: bool = False
    attn_output_gate: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_value_head_dim",
            "linear_num_value_heads",
            "max_position_embeddings",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("layer_types supports only linear_attention and full_attention")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("head_dim * partial_rotary_factor must be a positive even integer")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")
        if self.attention_dropout != 0.0:
            raise NotImplementedError("the first JAXSFT Qwen3.5 path supports attention_dropout=0 only")
        if self.attention_bias:
            raise NotImplementedError("Qwen3.5 attention bias is not implemented")
        if not self.attn_output_gate:
            raise NotImplementedError("Qwen3.5 checkpoints require the full-attention output gate")
        if not self.tie_word_embeddings:
            raise NotImplementedError("the dense Qwen3.5 loader currently requires tied word embeddings")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_width(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_width(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_width(self) -> int:
        return 2 * self.linear_key_width + self.linear_value_width

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Qwen35Config":
        outer_type = raw.get("model_type")
        text = raw.get("text_config", raw)
        if not isinstance(text, Mapping):
            raise ValueError("Qwen3.5 text_config must be a mapping")
        model_type = text.get("model_type", outer_type)
        if model_type != "qwen3_5_text":
            raise ValueError(f"expected dense Qwen3.5 model_type='qwen3_5_text', got {model_type!r}")
        rope = text.get("rope_parameters", {})
        if not isinstance(rope, Mapping):
            raise ValueError("rope_parameters must be a mapping")
        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "layer_types",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_value_head_dim",
            "linear_num_value_heads",
        )
        missing = [name for name in required if name not in text]
        if missing:
            raise ValueError(f"Qwen3.5 config is missing fields: {missing}")
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            layer_types=tuple(text["layer_types"]),
            linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            max_position_embeddings=int(text.get("max_position_embeddings", 262_144)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10_000_000.0))),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))),
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            attention_bias=bool(text.get("attention_bias", False)),
            attn_output_gate=bool(text.get("attn_output_gate", True)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", raw.get("tie_word_embeddings", True))),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen35Config":
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, Mapping):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(raw)


def tiny_config(*, vocab_size: int = 128) -> Qwen35Config:
    """A two-layer configuration shared by CPU and Transformers parity tests."""

    return Qwen35Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=("linear_attention", "full_attention"),
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_num_key_heads=2,
        linear_value_head_dim=8,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        partial_rotary_factor=0.5,
    )


def parameter_count(config: Qwen35Config) -> int:
    hidden = config.hidden_size
    total = config.vocab_size * hidden + hidden
    mlp = 3 * hidden * config.intermediate_size
    full_attention = (
        hidden * config.num_attention_heads * 2 * config.head_dim
        + 2 * hidden * config.num_key_value_heads * config.head_dim
        + config.num_attention_heads * config.head_dim * hidden
        + 2 * config.head_dim
    )
    linear_attention = (
        2 * config.linear_num_value_heads
        + config.linear_conv_width * config.linear_conv_kernel_dim
        + hidden * config.linear_conv_width
        + hidden * config.linear_value_width
        + 2 * hidden * config.linear_num_value_heads
        + config.linear_value_width * hidden
        + config.linear_value_head_dim
    )
    for layer_type in config.layer_types:
        total += 2 * hidden + mlp
        total += linear_attention if layer_type == "linear_attention" else full_attention
    return total


def _linear(x: jax.Array, kernel: jax.Array) -> jax.Array:
    return jnp.einsum("...i,io->...o", x, kernel, precision=_PRECISION)


def _rms_norm(x: jax.Array, weight: jax.Array, epsilon: float, *, centered: bool) -> jax.Array:
    dtype = x.dtype
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + epsilon)
    scale = weight.astype(jnp.float32) + (1.0 if centered else 0.0)
    return (value * scale).astype(dtype)


def _causal_depthwise_conv(x: jax.Array, weight: jax.Array, token_mask: jax.Array) -> jax.Array:
    batch, _, channels = x.shape
    kernel_size = weight.shape[1]
    state = jnp.zeros((batch, channels, kernel_size), dtype=x.dtype)
    weight_f32 = weight.astype(jnp.float32)

    def step(history, inputs):
        token, valid = inputs
        candidate = jnp.concatenate((history[..., 1:], token[:, :, None]), axis=-1)
        output = jnp.einsum(
            "bck,ck->bc", candidate.astype(jnp.float32), weight_f32, precision=_PRECISION
        )
        output = jax.nn.silu(output).astype(x.dtype)
        history = jnp.where(valid[:, None, None], candidate, history)
        output = jnp.where(valid[:, None], output, jnp.zeros((), output.dtype))
        return history, output

    _, output = jax.lax.scan(step, state, (jnp.swapaxes(x, 0, 1), jnp.swapaxes(token_mask, 0, 1)))
    return jnp.swapaxes(output, 0, 1)


def gated_delta_recurrent_scan(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    decay_log: jax.Array,
    beta: jax.Array,
    token_mask: jax.Array,
) -> jax.Array:
    """Exact float32 recurrent Gated DeltaNet reference rule."""

    output_dtype = value.dtype
    batch, _, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    state = jnp.zeros((batch, heads, key_dim, value_dim), jnp.float32)
    query, key, value = query.astype(jnp.float32), key.astype(jnp.float32), value.astype(jnp.float32)
    query *= jax.lax.rsqrt(jnp.sum(jnp.square(query), axis=-1, keepdims=True) + 1e-6)
    key *= jax.lax.rsqrt(jnp.sum(jnp.square(key), axis=-1, keepdims=True) + 1e-6)
    query *= key_dim**-0.5

    def step(recurrent, inputs):
        q_t, k_t, v_t, g_t, beta_t, valid = inputs
        decayed = recurrent * jnp.exp(g_t.astype(jnp.float32))[..., None, None]
        predicted = jnp.einsum("bhk,bhkv->bhv", k_t, decayed, precision=_PRECISION)
        delta = (v_t - predicted) * beta_t.astype(jnp.float32)[..., None]
        candidate = decayed + jnp.einsum("bhk,bhv->bhkv", k_t, delta, precision=_PRECISION)
        output = jnp.einsum("bhk,bhkv->bhv", q_t, candidate, precision=_PRECISION)
        recurrent = jnp.where(valid[:, None, None, None], candidate, recurrent)
        output = jnp.where(valid[:, None, None], output, 0.0)
        return recurrent, output

    scan_inputs = tuple(
        jnp.swapaxes(item, 0, 1) for item in (query, key, value, decay_log, beta, token_mask)
    )
    _, output = jax.lax.scan(step, state, scan_inputs)
    return jnp.swapaxes(output, 0, 1).astype(output_dtype)


def _linear_attention(
    params: ArrayTree, config: Qwen35Config, hidden: jax.Array, attention_mask: jax.Array
) -> jax.Array:
    batch, length, _ = hidden.shape
    hidden = jnp.where(attention_mask[..., None], hidden, jnp.zeros((), hidden.dtype))
    mixed = _linear(hidden, params["in_proj_qkv"])
    mixed = _causal_depthwise_conv(mixed, params["conv1d"], attention_mask)
    key_width = config.linear_key_width
    query, key, value = jnp.split(mixed, (key_width, 2 * key_width), axis=-1)
    query = query.reshape(batch, length, config.linear_num_key_heads, config.linear_key_head_dim)
    key = key.reshape(batch, length, config.linear_num_key_heads, config.linear_key_head_dim)
    value = value.reshape(batch, length, config.linear_num_value_heads, config.linear_value_head_dim)
    repeats = config.linear_num_value_heads // config.linear_num_key_heads
    query, key = jnp.repeat(query, repeats, axis=2), jnp.repeat(key, repeats, axis=2)
    beta = jax.nn.sigmoid(_linear(hidden, params["in_proj_b"]).astype(jnp.float32))
    a = _linear(hidden, params["in_proj_a"]).astype(jnp.float32)
    decay_log = -jnp.exp(params["A_log"].astype(jnp.float32))[None, None] * jax.nn.softplus(
        a + params["dt_bias"].astype(jnp.float32)[None, None]
    )
    output = gated_delta_recurrent_scan(query, key, value, decay_log, beta, attention_mask)
    gate = _linear(hidden, params["in_proj_z"]).reshape(
        batch, length, config.linear_num_value_heads, config.linear_value_head_dim
    )
    output = _rms_norm(output, params["norm"], config.rms_norm_eps, centered=False)
    output = output * jax.nn.silu(gate.astype(jnp.float32)).astype(output.dtype)
    return _linear(output.reshape(batch, length, config.linear_value_width), params["out_proj"])


def _rotate_half(value: jax.Array) -> jax.Array:
    half = value.shape[-1] // 2
    return jnp.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def _apply_rope(
    query: jax.Array, key: jax.Array, position_ids: jax.Array, config: Qwen35Config
) -> tuple[jax.Array, jax.Array]:
    inv_freq = 1.0 / (
        config.rope_theta
        ** (jnp.arange(0, config.rotary_dim, 2, dtype=jnp.float32) / config.rotary_dim)
    )
    frequencies = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None]
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)
    cos, sin = jnp.cos(embedding)[:, :, None], jnp.sin(embedding)[:, :, None]

    def rotate(value):
        rotary, passthrough = value[..., : config.rotary_dim], value[..., config.rotary_dim :]
        rotary = rotary * cos.astype(value.dtype) + _rotate_half(rotary) * sin.astype(value.dtype)
        return jnp.concatenate((rotary, passthrough), axis=-1)

    return rotate(query), rotate(key)


def _full_attention(
    params: ArrayTree,
    config: Qwen35Config,
    hidden: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    batch, length, _ = hidden.shape
    projected = jnp.einsum("bsi,ihgd->bshgd", hidden, params["q_proj"], precision=_PRECISION)
    query, gate = projected[:, :, :, 0], projected[:, :, :, 1]
    key = jnp.einsum("bsi,ihd->bshd", hidden, params["k_proj"], precision=_PRECISION)
    value = jnp.einsum("bsi,ihd->bshd", hidden, params["v_proj"], precision=_PRECISION)
    query = _rms_norm(query, params["q_norm"], config.rms_norm_eps, centered=True)
    key = _rms_norm(key, params["k_norm"], config.rms_norm_eps, centered=True)
    query, key = _apply_rope(query, key, position_ids, config)
    repeats = config.num_attention_heads // config.num_key_value_heads
    key, value = jnp.repeat(key, repeats, axis=2), jnp.repeat(value, repeats, axis=2)
    scores = jnp.einsum("bshd,bthd->bhst", query, key, precision=_PRECISION).astype(jnp.float32)
    scores *= config.head_dim**-0.5
    causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
    allowed = causal[None, None] & attention_mask[:, None, None, :]
    scores = jnp.where(allowed, scores, -jnp.inf)
    all_masked = jnp.all(jnp.isneginf(scores), axis=-1, keepdims=True)
    weights = jax.nn.softmax(jnp.where(all_masked, 0.0, scores), axis=-1).astype(value.dtype)
    output = jnp.einsum("bhst,bthd->bshd", weights, value, precision=_PRECISION)
    output *= jax.nn.sigmoid(gate).astype(output.dtype)
    return _linear(output.reshape(batch, length, config.num_attention_heads * config.head_dim), params["o_proj"])


def _mlp(params: ArrayTree, hidden: jax.Array) -> jax.Array:
    return _linear(jax.nn.silu(_linear(hidden, params["gate_proj"])) * _linear(hidden, params["up_proj"]), params["down_proj"])


def _decoder_layer(
    params: ArrayTree,
    config: Qwen35Config,
    layer_type: str,
    hidden: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    normalized = _rms_norm(hidden, params["input_layernorm"], config.rms_norm_eps, centered=True)
    if layer_type == "linear_attention":
        mixed = _linear_attention(params["linear_attn"], config, normalized, attention_mask)
    else:
        mixed = _full_attention(params["self_attn"], config, normalized, attention_mask, position_ids)
    hidden = hidden + mixed
    normalized = _rms_norm(hidden, params["post_attention_layernorm"], config.rms_norm_eps, centered=True)
    return hidden + _mlp(params["mlp"], normalized)


def forward(
    params: ArrayTree,
    config: Qwen35Config,
    input_ids: jax.Array,
    *,
    attention_mask: jax.Array | None = None,
    position_ids: jax.Array | None = None,
    remat: bool = False,
) -> jax.Array:
    """Return tied-head causal logits for a right-padded text batch."""

    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids, dtype=bool)
    else:
        attention_mask = jnp.asarray(attention_mask, dtype=bool)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if position_ids is None:
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=-1, dtype=jnp.int32) - 1, 0)
    else:
        position_ids = jnp.asarray(position_ids, jnp.int32)
    hidden = params["embed_tokens"][input_ids]
    for layer_index, layer_type in enumerate(config.layer_types):
        layer_fn = _decoder_layer
        if remat:
            layer_fn = jax.checkpoint(layer_fn, static_argnums=(1, 2))
        hidden = layer_fn(
            params["layers"][layer_index], config, layer_type, hidden, attention_mask, position_ids
        )
    hidden = _rms_norm(hidden, params["norm"], config.rms_norm_eps, centered=True)
    return jnp.einsum("bsh,vh->bsv", hidden, params["embed_tokens"], precision=_PRECISION)


def init_params(key: jax.Array, config: Qwen35Config, *, dtype: Any = jnp.float32) -> ArrayTree:
    """Initialize a readable parameter tree, primarily for tiny parity tests."""

    dtype = jnp.dtype(dtype)

    def normal(shape, stddev=0.02):
        nonlocal key
        key, subkey = jax.random.split(key)
        return (jax.random.normal(subkey, shape, jnp.float32) * stddev).astype(dtype)

    layers: list[ArrayTree] = []
    for layer_type in config.layer_types:
        layer: ArrayTree = {
            "input_layernorm": jnp.zeros((config.hidden_size,), dtype),
            "post_attention_layernorm": jnp.zeros((config.hidden_size,), dtype),
            "mlp": {
                "gate_proj": normal((config.hidden_size, config.intermediate_size)),
                "up_proj": normal((config.hidden_size, config.intermediate_size)),
                "down_proj": normal((config.intermediate_size, config.hidden_size)),
            },
        }
        if layer_type == "linear_attention":
            key, a_key = jax.random.split(key)
            layer["linear_attn"] = {
                "dt_bias": jnp.ones((config.linear_num_value_heads,), jnp.float32),
                "A_log": jnp.log(
                    jax.random.uniform(a_key, (config.linear_num_value_heads,), jnp.float32, minval=0.01, maxval=16.0)
                ),
                "conv1d": normal((config.linear_conv_width, config.linear_conv_kernel_dim)),
                "norm": jnp.ones((config.linear_value_head_dim,), dtype),
                "in_proj_qkv": normal((config.hidden_size, config.linear_conv_width)),
                "in_proj_z": normal((config.hidden_size, config.linear_value_width)),
                "in_proj_b": normal((config.hidden_size, config.linear_num_value_heads)),
                "in_proj_a": normal((config.hidden_size, config.linear_num_value_heads)),
                "out_proj": normal((config.linear_value_width, config.hidden_size)),
            }
        else:
            layer["self_attn"] = {
                "q_proj": normal((config.hidden_size, config.num_attention_heads, 2, config.head_dim)),
                "k_proj": normal((config.hidden_size, config.num_key_value_heads, config.head_dim)),
                "v_proj": normal((config.hidden_size, config.num_key_value_heads, config.head_dim)),
                "o_proj": normal((config.num_attention_heads * config.head_dim, config.hidden_size)),
                "q_norm": jnp.zeros((config.head_dim,), dtype),
                "k_norm": jnp.zeros((config.head_dim,), dtype),
            }
        layers.append(layer)
    params: ArrayTree = {
        "embed_tokens": normal((config.vocab_size, config.hidden_size)),
        "layers": tuple(layers),
        "norm": jnp.zeros((config.hidden_size,), dtype),
    }
    validate_params(params, config)
    return params


def _expected_shapes(config: Qwen35Config) -> ArrayTree:
    layers: list[ArrayTree] = []
    for layer_type in config.layer_types:
        layer: ArrayTree = {
            "input_layernorm": (config.hidden_size,),
            "post_attention_layernorm": (config.hidden_size,),
            "mlp": {
                "gate_proj": (config.hidden_size, config.intermediate_size),
                "up_proj": (config.hidden_size, config.intermediate_size),
                "down_proj": (config.intermediate_size, config.hidden_size),
            },
        }
        if layer_type == "linear_attention":
            layer["linear_attn"] = {
                "dt_bias": (config.linear_num_value_heads,),
                "A_log": (config.linear_num_value_heads,),
                "conv1d": (config.linear_conv_width, config.linear_conv_kernel_dim),
                "norm": (config.linear_value_head_dim,),
                "in_proj_qkv": (config.hidden_size, config.linear_conv_width),
                "in_proj_z": (config.hidden_size, config.linear_value_width),
                "in_proj_b": (config.hidden_size, config.linear_num_value_heads),
                "in_proj_a": (config.hidden_size, config.linear_num_value_heads),
                "out_proj": (config.linear_value_width, config.hidden_size),
            }
        else:
            layer["self_attn"] = {
                "q_proj": (config.hidden_size, config.num_attention_heads, 2, config.head_dim),
                "k_proj": (config.hidden_size, config.num_key_value_heads, config.head_dim),
                "v_proj": (config.hidden_size, config.num_key_value_heads, config.head_dim),
                "o_proj": (config.num_attention_heads * config.head_dim, config.hidden_size),
                "q_norm": (config.head_dim,),
                "k_norm": (config.head_dim,),
            }
        layers.append(layer)
    return {
        "embed_tokens": (config.vocab_size, config.hidden_size),
        "layers": tuple(layers),
        "norm": (config.hidden_size,),
    }


def validate_params(params: ArrayTree, config: Qwen35Config) -> None:
    expected = _expected_shapes(config)
    params_structure = jax.tree.structure(params)
    expected_structure = jax.tree.structure(expected, is_leaf=lambda value: isinstance(value, tuple) and all(isinstance(x, int) for x in value))
    if params_structure != expected_structure:
        raise ValueError("Qwen3.5 parameter tree has unexpected keys or nesting")
    leaves = jax.tree.leaves(params)
    shapes = jax.tree.leaves(expected, is_leaf=lambda value: isinstance(value, tuple) and all(isinstance(x, int) for x in value))
    for index, (value, shape) in enumerate(zip(leaves, shapes)):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"Qwen3.5 parameter leaf {index} has shape {value.shape}, expected {shape}")
    if sum(int(np.prod(value.shape)) for value in leaves) != parameter_count(config):
        raise ValueError("Qwen3.5 parameter count disagrees with the analytical count")


def _numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def convert_hf_state_dict(
    state: Mapping[str, Any], config: Qwen35Config, *, dtype: Any = jnp.bfloat16, strict: bool = True
) -> ArrayTree:
    """Convert text-only or multimodal-prefixed Hugging Face tensors."""

    outer = "model.language_model.embed_tokens.weight"
    standalone = "model.embed_tokens.weight"
    if outer in state:
        prefix = "model.language_model."
    elif standalone in state:
        prefix = "model."
    else:
        raise KeyError(f"checkpoint contains neither {outer!r} nor {standalone!r}")
    expected_keys: set[str] = set()

    def get(name: str, *, target_dtype=dtype) -> jax.Array:
        key = prefix + name
        expected_keys.add(key)
        try:
            value = state[key]
        except KeyError as error:
            raise KeyError(f"Qwen3.5 checkpoint is missing required tensor {key!r}") from error
        return jnp.asarray(_numpy_value(value), target_dtype)

    layers: list[ArrayTree] = []
    for index, layer_type in enumerate(config.layer_types):
        base = f"layers.{index}."
        layer: ArrayTree = {
            "input_layernorm": get(base + "input_layernorm.weight"),
            "post_attention_layernorm": get(base + "post_attention_layernorm.weight"),
            "mlp": {
                "gate_proj": get(base + "mlp.gate_proj.weight").T,
                "up_proj": get(base + "mlp.up_proj.weight").T,
                "down_proj": get(base + "mlp.down_proj.weight").T,
            },
        }
        if layer_type == "linear_attention":
            attn = base + "linear_attn."
            layer["linear_attn"] = {
                "dt_bias": get(attn + "dt_bias", target_dtype=jnp.float32),
                "A_log": get(attn + "A_log", target_dtype=jnp.float32),
                "conv1d": jnp.squeeze(get(attn + "conv1d.weight"), axis=1),
                "norm": get(attn + "norm.weight"),
                "in_proj_qkv": get(attn + "in_proj_qkv.weight").T,
                "in_proj_z": get(attn + "in_proj_z.weight").T,
                "in_proj_b": get(attn + "in_proj_b.weight").T,
                "in_proj_a": get(attn + "in_proj_a.weight").T,
                "out_proj": get(attn + "out_proj.weight").T,
            }
        else:
            attn = base + "self_attn."
            layer["self_attn"] = {
                "q_proj": get(attn + "q_proj.weight").T.reshape(
                    config.hidden_size, config.num_attention_heads, 2, config.head_dim
                ),
                "k_proj": get(attn + "k_proj.weight").T.reshape(
                    config.hidden_size, config.num_key_value_heads, config.head_dim
                ),
                "v_proj": get(attn + "v_proj.weight").T.reshape(
                    config.hidden_size, config.num_key_value_heads, config.head_dim
                ),
                "o_proj": get(attn + "o_proj.weight").T,
                "q_norm": get(attn + "q_norm.weight"),
                "k_norm": get(attn + "k_norm.weight"),
            }
        layers.append(layer)
    params: ArrayTree = {
        "embed_tokens": get("embed_tokens.weight"),
        "layers": tuple(layers),
        "norm": get("norm.weight"),
    }
    if strict:
        text_keys = {
            key
            for key in state
            if key.startswith(prefix + "embed_tokens.")
            or key.startswith(prefix + "layers.")
            or key.startswith(prefix + "norm.")
        }
        unexpected = text_keys - expected_keys
        if unexpected:
            raise ValueError(f"unexpected Qwen3.5 text tensors: {sorted(unexpected)}")
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


def load_hf_checkpoint(root: str | Path, *, dtype: Any = jnp.bfloat16) -> tuple[Qwen35Config, ArrayTree]:
    """Load the language model lazily from a local pinned Hub snapshot."""

    root = Path(root)
    config = Qwen35Config.from_json(root / "config.json")
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
    "Qwen35Config",
    "convert_hf_state_dict",
    "forward",
    "gated_delta_recurrent_scan",
    "init_params",
    "load_hf_checkpoint",
    "parameter_count",
    "tiny_config",
    "validate_params",
]
