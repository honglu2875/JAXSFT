import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from jaxsft.models.glm5_3_flash import attention_lora_parameter_count, init_params, tiny_config
from scripts.probe_glm53_lora_backward import (
    _initialize_adapters,
    _require_execution_headroom,
    _tree_scalar_statistics,
)
from scripts.probe_glm53_lora_backward_compile import HBM_LIMIT_BYTES_PER_DEVICE


def test_adapter_scalar_statistics_preserve_zero_b_initialization_and_gradient_roles():
    initialized = {
        "target": {
            "a": jnp.asarray([[1.0, 0.0]], jnp.bfloat16),
            "b": jnp.asarray([[0.0, 0.0]], jnp.bfloat16),
        }
    }
    assert np.array_equal(
        _tree_scalar_statistics(initialized, include_l1_and_leaf_counts=False),
        np.asarray([1, 1, 0, 1, 0, 1, 0], np.float32),
    )

    gradients = {
        "target": {
            "a": jnp.asarray([[0.0, 0.0]], jnp.bfloat16),
            "b": jnp.asarray([[3.0, -4.0]], jnp.bfloat16),
        }
    }
    assert np.array_equal(
        _tree_scalar_statistics(gradients, include_l1_and_leaf_counts=True),
        np.asarray([1, 0, 25, 0, 7, 0, 4, 0, 2, 0, 1], np.float32),
    )


def test_sharded_initializer_has_canonical_nonzero_a_and_exact_zero_b():
    config = tiny_config(vocab_size=32)
    params = init_params(jax.random.key(0), config, dtype=jnp.float32)
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    adapters, memory, elapsed = _initialize_adapters(params, config, mesh, rank=2, seed=7)
    statistics = np.asarray(
        _tree_scalar_statistics(adapters, include_l1_and_leaf_counts=False),
        dtype=np.float32,
    )
    assert statistics[0] == 1
    assert statistics[1] > 0
    assert statistics[2] == 0
    assert statistics[3] > 0
    assert statistics[4] == 0
    assert statistics[5] > 0
    assert statistics[6] == 0
    assert sum(value.size for value in jax.tree.leaves(adapters)) == attention_lora_parameter_count(
        config, rank=2
    )
    assert memory["argument_size_in_bytes"] == 0
    assert elapsed > 0


def test_runtime_headroom_gate_uses_compiler_temp_output_and_one_gib_margin():
    compiler = {"temp_size_in_bytes": 1_000_000_000, "output_size_in_bytes": 10_000_000}
    safe_free = 1_000_000_000 + 10_000_000 + 1024**3
    records = [
        {
            "stats": {
                "bytes_limit": HBM_LIMIT_BYTES_PER_DEVICE,
                "largest_free_block_bytes": safe_free,
            }
        }
        for _ in range(4)
    ]
    _require_execution_headroom(records, compiler)
    records[0]["stats"]["largest_free_block_bytes"] -= 1
    with pytest.raises(ValueError, match="safety margin"):
        _require_execution_headroom(records, compiler)
