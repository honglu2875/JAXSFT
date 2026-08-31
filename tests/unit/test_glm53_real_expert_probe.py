import jax.numpy as jnp
import numpy as np

from scripts.oracle_glm53_real_expert import EXPERT_INDICES, _SourceFingerprint
from scripts.probe_glm53_real_expert_loader import _selected_source_fingerprint


def _cpu_fingerprint(value):
    accumulator = _SourceFingerprint(value.shape[1:])
    for expert_index in EXPERT_INDICES:
        accumulator.update(value[expert_index])
    return accumulator.result()


def test_selected_source_fingerprint_matches_streaming_cpu_accumulator():
    bits = np.arange(288 * 2 * 3, dtype=np.uint8).reshape(288, 2, 3)
    scales = np.linspace(0.01, 1.0, 288 * 2 * 3, dtype=np.float32).reshape(288, 2, 3)
    assert np.asarray(_selected_source_fingerprint(jnp.asarray(bits))).tolist() == _cpu_fingerprint(
        bits
    )
    assert np.asarray(_selected_source_fingerprint(jnp.asarray(scales))).tolist() == _cpu_fingerprint(
        scales
    )
