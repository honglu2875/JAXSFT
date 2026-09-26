from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.models.qwen2 import Qwen2Config, forward, init_params, parameter_count, tiny_config
from jaxsft.models.registry import get_model_implementation


@pytest.mark.parametrize("tied", [False, True])
def test_qwen2_causal_padding_remat_and_parameter_contract(tied):
    config = replace(tiny_config(), tie_word_embeddings=tied)
    assert get_model_implementation("qwen2").config_type is Qwen2Config
    params = init_params(jax.random.key(7), config)
    assert sum(value.size for value in jax.tree.leaves(params)) == parameter_count(config)
    assert ("lm_head" not in params) == tied
    ids = jnp.array([[1, 5, 7, 9]], jnp.int32)
    baseline = forward(params, config, ids)
    changed = forward(params, config, ids.at[0, 3].set(10))
    np.testing.assert_allclose(baseline[:, :3], changed[:, :3], atol=1e-6)
    padded = forward(params, config, jnp.array([[1, 5, 7, 9, 0, 0]]),
                     attention_mask=jnp.array([[1, 1, 1, 1, 0, 0]], bool), remat=True)
    np.testing.assert_allclose(baseline, padded[:, :4], atol=2e-6, rtol=2e-6)
    gradients = jax.grad(lambda p: jnp.mean(forward(p, config, ids, remat=True) ** 2))(params)
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(gradients))


@pytest.mark.parametrize("unsupported", [
    {"rope_scaling": {"rope_type": "linear", "factor": 2}},
    {"rope_parameters": {"rope_type": "yarn", "factor": 2}},
    {"use_sliding_window": True},
    {"layer_types": ["full_attention", "sliding_attention"]},
    {"use_mrope": True},
    {"attention_dropout": 0.1},
    {"attention_bias": False},
    {"hidden_act": "gelu"},
])
def test_qwen2_rejects_unimplemented_numerics(unsupported):
    raw = {**tiny_config().__dict__, "model_type": "qwen2"}
    with pytest.raises(NotImplementedError):
        Qwen2Config.from_dict({**raw, **unsupported})


def test_qwen2_accepts_math_config_inactive_window_and_default_rope():
    raw = {**tiny_config().__dict__, "model_type": "qwen2", "sliding_window": 4096,
           "use_sliding_window": False, "layer_types": ["full_attention"] * 2,
           "rope_parameters": {"rope_type": "default", "rope_theta": 10000}}
    assert Qwen2Config.from_dict(raw) == tiny_config()
