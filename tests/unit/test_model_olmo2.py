import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.models.olmo2 import (
    Olmo2Config,
    forward,
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


def test_config_rejects_unsupported_or_inconsistent_architecture_options():
    raw = {
        "model_type": "olmo2",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    assert Olmo2Config.from_dict(raw).head_dim == 8
    with pytest.raises(NotImplementedError, match="rope_scaling"):
        Olmo2Config.from_dict({**raw, "rope_scaling": {"type": "linear", "factor": 2.0}})
    with pytest.raises(ValueError, match="hidden_size"):
        Olmo2Config.from_dict({**raw, "head_dim": 10})


def test_tied_head_has_one_parameter_source():
    untied = tiny_config(vocab_size=64)
    tied = Olmo2Config(**{**untied.__dict__, "tie_word_embeddings": True})
    params = init_params(jax.random.key(8), tied)
    assert "lm_head" not in params
    assert parameter_count(untied) - parameter_count(tied) == tied.hidden_size * tied.vocab_size
    logits = forward(params, tied, jnp.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, tied.vocab_size)
