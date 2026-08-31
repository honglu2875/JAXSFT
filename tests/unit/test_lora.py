import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.lora import (
    _PRECISION,
    LoRAConfig,
    adapter_for_path,
    audit_lora_targets,
    flatten_lora_adapters,
    format_parameter_path,
    init_lora_adapters,
    lora_linear,
    lora_parameter_count,
    merge_lora_adapters,
    parameter_at_path,
    unflatten_lora_adapters,
)
from jaxsft.optim import adamw_init


Q_PATH = ("layers", 0, "attention", "q_proj")
O_PATH = ("layers", 0, "attention", "o_proj")


def _base_params():
    return {
        "embedding": jnp.ones((7, 4), jnp.float32),
        "layers": (
            {
                "attention": {
                    "q_proj": jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 17,
                    "o_proj": jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 19,
                },
                "norm": jnp.ones((4,), jnp.float32),
            },
        ),
    }


def test_paths_and_target_audit_are_explicit():
    base = _base_params()
    assert format_parameter_path(Q_PATH) == "layers[0].attention.q_proj"
    assert parameter_at_path(base, Q_PATH).shape == (4, 6)
    assert audit_lora_targets(base, (Q_PATH,), eligible_paths=(Q_PATH, O_PATH)) == (Q_PATH,)

    with pytest.raises(ValueError, match="not declared eligible"):
        audit_lora_targets(base, (("embedding",),), eligible_paths=(Q_PATH, O_PATH))
    with pytest.raises(ValueError, match="rank-2"):
        audit_lora_targets(
            base,
            (("layers", 0, "norm"),),
            eligible_paths=(("layers", 0, "norm"),),
        )


def test_zero_b_initialization_is_identity_and_optimizer_has_no_base_slots():
    base = _base_params()
    config = LoRAConfig(rank=2, alpha=4)
    adapters = init_lora_adapters(
        jax.random.key(3),
        base,
        (Q_PATH,),
        eligible_paths=(Q_PATH, O_PATH),
        config=config,
        dtype=jnp.float32,
    )
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 13
    kernel = parameter_at_path(base, Q_PATH)
    output = lora_linear(x, kernel, adapter_for_path(adapters, Q_PATH), config=config)
    assert np.array_equal(output, x @ kernel)

    def adapter_loss(trainable):
        value = lora_linear(x, kernel, adapter_for_path(trainable, Q_PATH), config=config)
        return jnp.mean(value**2)

    gradients = jax.grad(adapter_loss)(adapters)
    pair = adapter_for_path(gradients, Q_PATH)
    assert np.count_nonzero(np.asarray(pair["a"])) == 0
    assert np.linalg.norm(np.asarray(pair["b"])) > 0

    optimizer = adamw_init(adapters)
    assert jax.tree.structure(optimizer.first_moment) == jax.tree.structure(adapters)
    assert lora_parameter_count(optimizer.first_moment) == lora_parameter_count(adapters)
    assert lora_parameter_count(adapters) == 20
    assert sum(value.size for value in jax.tree.leaves(base)) == 80


def test_unmerged_and_merged_dense_results_and_gradients_match():
    base = _base_params()
    config = LoRAConfig(rank=2, alpha=3)
    adapters = init_lora_adapters(
        jax.random.key(7),
        base,
        (Q_PATH,),
        eligible_paths=(Q_PATH,),
        config=config,
        dtype=jnp.float32,
    )
    name = format_parameter_path(Q_PATH)
    adapters[name]["b"] = jnp.arange(12, dtype=jnp.float32).reshape(2, 6) / 23
    x = jnp.arange(20, dtype=jnp.float32).reshape(5, 4) / 11

    unmerged = lora_linear(
        x,
        parameter_at_path(base, Q_PATH),
        adapter_for_path(adapters, Q_PATH),
        config=config,
    )
    merged_base = merge_lora_adapters(base, adapters, (Q_PATH,), config=config)
    merged = x @ parameter_at_path(merged_base, Q_PATH)
    assert np.allclose(unmerged, merged, atol=2e-6, rtol=2e-6)
    unmerged_dx = jax.grad(
        lambda values: jnp.sum(
            lora_linear(
                values,
                parameter_at_path(base, Q_PATH),
                adapter_for_path(adapters, Q_PATH),
                config=config,
            )
        )
    )(x)
    merged_dx = jax.grad(lambda values: jnp.sum(values @ parameter_at_path(merged_base, Q_PATH)))(x)
    assert np.allclose(unmerged_dx, merged_dx, atol=2e-6, rtol=2e-6)
    # The functional merge must not mutate the frozen source tree.
    assert np.array_equal(parameter_at_path(base, Q_PATH), _base_params()["layers"][0]["attention"]["q_proj"])


def test_adapter_only_flatten_round_trip_is_strict():
    base = _base_params()
    config = LoRAConfig(rank=2, alpha=2)
    adapters = init_lora_adapters(
        jax.random.key(11),
        base,
        (Q_PATH, O_PATH),
        eligible_paths=(Q_PATH, O_PATH),
        config=config,
        dtype=jnp.float32,
    )
    flat = flatten_lora_adapters(adapters)
    restored = unflatten_lora_adapters(flat, base, (Q_PATH, O_PATH), config=config)
    for expected, actual in zip(jax.tree.leaves(adapters), jax.tree.leaves(restored), strict=True):
        assert np.array_equal(expected, actual)

    flat["unexpected.lora_a"] = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match="unexpected tensors"):
        unflatten_lora_adapters(flat, base, (Q_PATH, O_PATH), config=config)


def test_dropout_requires_key_and_highest_precision_is_default():
    assert _PRECISION is jax.lax.Precision.HIGHEST
    base = _base_params()
    config = LoRAConfig(rank=2, alpha=2, dropout=0.25)
    adapters = init_lora_adapters(
        jax.random.key(1),
        base,
        (Q_PATH,),
        eligible_paths=(Q_PATH,),
        config=config,
        dtype=jnp.float32,
    )
    with pytest.raises(ValueError, match="dropout_key"):
        lora_linear(
            jnp.ones((2, 4)),
            parameter_at_path(base, Q_PATH),
            adapter_for_path(adapters, Q_PATH),
            config=config,
            training=True,
        )
