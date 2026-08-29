"""Numerically stable, additive causal-language-model objectives."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LossStatistics(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    correct_weight: jax.Array
    token_count: jax.Array


def causal_loss_statistics(logits: jax.Array, input_ids: jax.Array, loss_weights: jax.Array) -> LossStatistics:
    """Return additive statistics with weights stored beside target tokens."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, length, vocab], got {logits.shape}")
    if input_ids.shape != logits.shape[:2] or loss_weights.shape != input_ids.shape:
        raise ValueError("input_ids and loss_weights must match logits' batch/length axes")
    prediction_logits = logits[:, :-1].astype(jnp.float32)
    target_ids = input_ids[:, 1:]
    target_weights = loss_weights[:, 1:].astype(jnp.float32)
    target_logits = jnp.take_along_axis(prediction_logits, target_ids[..., None], axis=-1)[..., 0]
    nll = jax.nn.logsumexp(prediction_logits, axis=-1) - target_logits
    numerator = jnp.sum(nll * target_weights, dtype=jnp.float32)
    denominator = jnp.sum(target_weights, dtype=jnp.float32)
    correct = jnp.argmax(prediction_logits, axis=-1) == target_ids
    correct_weight = jnp.sum(correct.astype(jnp.float32) * target_weights, dtype=jnp.float32)
    return LossStatistics(numerator, denominator, correct_weight, jnp.asarray(target_ids.size, jnp.float32))


def normalized_loss(statistics: LossStatistics) -> jax.Array:
    return statistics.numerator / jnp.maximum(statistics.denominator, jnp.asarray(1e-12, jnp.float32))
