import jax.numpy as jnp
import numpy as np

from jaxsft.loss import causal_loss_statistics, normalized_loss


def test_causal_weights_are_stored_beside_predicted_token():
    # Position 0 strongly predicts token 2, position 1 strongly predicts token 0.
    logits = jnp.array([[[0.0, 0.0, 8.0], [8.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    input_ids = jnp.array([[1, 2, 0]])
    weights = jnp.array([[0.0, 1.0, 0.0]])
    stats = causal_loss_statistics(logits, input_ids, weights)
    assert np.isclose(float(normalized_loss(stats)), np.log(np.exp(8) + 2) - 8, atol=1e-6)
    assert float(stats.denominator) == 1.0
    assert float(stats.correct_weight) == 1.0


def test_fractional_weights_produce_additive_numerator_and_denominator():
    logits = jnp.zeros((1, 3, 4))
    ids = jnp.array([[0, 1, 2]])
    stats = causal_loss_statistics(logits, ids, jnp.array([[0.0, 0.5, 1.5]]))
    assert np.isclose(float(stats.numerator), 2.0 * np.log(4.0))
    assert float(stats.denominator) == 2.0
