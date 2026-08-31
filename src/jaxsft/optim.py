"""A small visible AdamW implementation for the canonical trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AdamWState(NamedTuple):
    step: jax.Array
    first_moment: object
    second_moment: object


@dataclass(frozen=True)
class AdamWHyperparameters:
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if self.epsilon <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("AdamW epsilon/max_grad_norm must be positive and weight_decay non-negative")


def adamw_init(params: object) -> AdamWState:
    # Keep the slots physically distinct. Reusing one zero tree is numerically
    # harmless for a non-donated first update, but it presents the same buffers
    # twice when a compiled training step donates optimizer state.
    first = jax.tree.map(lambda value: jnp.zeros(value.shape, jnp.float32), params)
    second = jax.tree.map(lambda value: jnp.zeros(value.shape, jnp.float32), params)
    return AdamWState(jnp.asarray(0, jnp.int32), first, second)


def tree_global_norm(tree: object) -> jax.Array:
    squared = [jnp.sum(jnp.square(value.astype(jnp.float32)), dtype=jnp.float32) for value in jax.tree.leaves(tree)]
    return jnp.sqrt(sum(squared, jnp.asarray(0.0, jnp.float32)))


def clip_by_global_norm(tree: object, max_norm: float) -> tuple[object, jax.Array]:
    norm = tree_global_norm(tree)
    # Match torch.nn.utils.clip_grad_norm_: its 1e-6 denominator guard is
    # observable near the clipping boundary and therefore part of parity.
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, jnp.float32) / (norm + 1e-6))
    return jax.tree.map(lambda value: value * scale.astype(value.dtype), tree), norm


def adamw_update(
    params: object,
    gradients: object,
    state: AdamWState,
    *,
    learning_rate: jax.Array | float,
    hyperparameters: AdamWHyperparameters,
) -> tuple[object, AdamWState, jax.Array]:
    gradients, gradient_norm = clip_by_global_norm(gradients, hyperparameters.max_grad_norm)
    step = state.step + jnp.asarray(1, jnp.int32)
    beta1, beta2 = hyperparameters.beta1, hyperparameters.beta2
    first = jax.tree.map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad.astype(jnp.float32),
        state.first_moment,
        gradients,
    )
    second = jax.tree.map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad.astype(jnp.float32)),
        state.second_moment,
        gradients,
    )
    correction1 = 1.0 - beta1**step.astype(jnp.float32)
    correction2 = 1.0 - beta2**step.astype(jnp.float32)
    rate = jnp.asarray(learning_rate, jnp.float32)

    def update_parameter(param, moment1, moment2):
        direction = (moment1 / correction1) / (jnp.sqrt(moment2 / correction2) + hyperparameters.epsilon)
        # Matrix/tensor kernels decay; norm scales and scalar recurrence terms do not.
        if param.ndim >= 2:
            direction = direction + hyperparameters.weight_decay * param.astype(jnp.float32)
        return (param.astype(jnp.float32) - rate * direction).astype(param.dtype)

    params = jax.tree.map(update_parameter, params, first, second)
    return params, AdamWState(step, first, second), gradient_norm


def cosine_learning_rate(
    step: jax.Array | int,
    *,
    peak: float,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float = 0.1,
) -> jax.Array:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("schedule requires 0 <= warmup_steps < total_steps")
    step_f = jnp.asarray(step, jnp.float32)
    warmup = peak * step_f / max(warmup_steps, 1)
    progress = jnp.clip((step_f - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0)
    cosine = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + jnp.cos(math.pi * progress))
    return jnp.where(step_f < warmup_steps, warmup, peak * cosine)
