import jax.numpy as jnp
import numpy as np

from jaxsft.optim import AdamWHyperparameters, adamw_init, adamw_update, cosine_learning_rate


def test_adamw_initial_moment_slots_do_not_alias_for_donation():
    state = adamw_init({"kernel": jnp.ones((2, 2))})
    assert state.first_moment["kernel"] is not state.second_moment["kernel"]


def test_adamw_updates_matrix_and_does_not_decay_norm_vector():
    params = {"kernel": jnp.ones((2, 2)), "norm": jnp.ones((2,))}
    gradients = {"kernel": jnp.zeros((2, 2)), "norm": jnp.zeros((2,))}
    updated, state, norm = adamw_update(
        params,
        gradients,
        adamw_init(params),
        learning_rate=0.1,
        hyperparameters=AdamWHyperparameters(weight_decay=0.2),
    )
    assert np.allclose(updated["kernel"], 0.98)
    assert np.allclose(updated["norm"], 1.0)
    assert int(state.step) == 1
    assert float(norm) == 0.0


def test_cosine_schedule_warms_up_then_decays():
    rates = [float(cosine_learning_rate(step, peak=1.0, total_steps=10, warmup_steps=2)) for step in range(10)]
    assert rates[0] == 0.0
    assert rates[2] == 1.0
    assert rates[-1] < rates[2]
