import jax
import jax.numpy as jnp
import numpy as np

from jaxsft.models.qwen3_5 import (
    forward,
    gated_delta_recurrent_scan,
    init_params,
    parameter_count,
    tiny_config,
    validate_params,
)


def test_tiny_forward_backward_and_parameter_contract():
    config = tiny_config()
    params = init_params(jax.random.key(0), config)
    validate_params(params, config)
    assert parameter_count(config) == sum(value.size for value in jax.tree.leaves(params))
    ids = jnp.array([[1, 2, 3, 4], [4, 3, 0, 0]], jnp.int32)
    mask = jnp.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool)
    logits = forward(params, config, ids, attention_mask=mask)
    assert logits.shape == (2, 4, config.vocab_size)
    assert np.asarray(jnp.isfinite(logits).all())
    gradients = jax.grad(lambda tree: jnp.mean(forward(tree, config, ids, attention_mask=mask) ** 2))(params)
    assert all(np.asarray(jnp.isfinite(value).all()) for value in jax.tree.leaves(gradients))


def test_delta_scan_one_token_matches_hand_update():
    q = jnp.array([[[[2.0, 0.0]]]])
    k = jnp.array([[[[0.0, 3.0]]]])
    v = jnp.array([[[[4.0]]]])
    g = jnp.array([[[0.0]]])
    beta = jnp.array([[[0.5]]])
    output = gated_delta_recurrent_scan(q, k, v, g, beta, jnp.ones((1, 1), bool))
    # Orthogonal normalized q/k produce zero readout from the rank-one update.
    assert np.allclose(output, 0.0, atol=1e-7)


def test_right_padding_does_not_change_valid_prefix_logits():
    config = tiny_config()
    params = init_params(jax.random.key(4), config)
    short = forward(params, config, jnp.array([[2, 3, 4]]))
    padded = forward(
        params,
        config,
        jnp.array([[2, 3, 4, 0, 0]]),
        attention_mask=jnp.array([[1, 1, 1, 0, 0]], bool),
    )
    assert np.allclose(short, padded[:, :3], atol=2e-6, rtol=2e-6)
