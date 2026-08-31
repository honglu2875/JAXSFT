import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import scripts.probe_glm53_lora_optimizer_compile as probe
from jaxsft.optim import AdamWHyperparameters, adamw_init
from scripts.probe_glm53_lora_optimizer_compile import (
    _abstract_adamw_state,
    _optimizer_execution_gate,
    _optimizer_placement,
)


def test_abstract_adamw_state_is_fp32_sharded_and_has_distinct_slots():
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    output_sharded = NamedSharding(mesh, PartitionSpec(None, "model"))
    adapters = {
        "layer.q_proj": {
            "a": jax.ShapeDtypeStruct((6, 2), jnp.bfloat16, sharding=replicated),
            "b": jax.ShapeDtypeStruct((2, 8), jnp.bfloat16, sharding=output_sharded),
        }
    }
    state = _abstract_adamw_state(adapters, replicated)
    assert state.first_moment["layer.q_proj"]["a"].dtype == jnp.float32
    assert state.second_moment["layer.q_proj"]["b"].dtype == jnp.float32
    assert (
        state.first_moment["layer.q_proj"]["a"]
        is not state.second_moment["layer.q_proj"]["a"]
    )
    assert state.step.dtype == jnp.int32

    placement = _optimizer_placement(state)
    assert placement["moment_global_elements_per_slot"] == 28
    assert set(placement["first_moment_bytes_by_device"].values()) == {112}
    assert set(placement["second_moment_bytes_by_device"].values()) == {112}
    assert set(placement["step_bytes_by_device"].values()) == {4}
    assert set(placement["optimizer_state_bytes_by_device"].values()) == {228}


def test_optimizer_compile_gate_is_conservative_and_fail_closed():
    safe = _optimizer_execution_gate(
        {
            "argument_size_in_bytes": 20_000_000_000,
            "output_size_in_bytes": 50_000_000,
            "temp_size_in_bytes": 1_000_000_000,
            "alias_size_in_bytes": 50_000_000,
        }
    )
    assert safe["compiler_working_set_upper_bound_bytes_per_device"] == 21_050_000_000
    assert safe["alias_bytes_not_subtracted_from_conservative_bound"] is True
    assert safe["full_checkpoint_optimizer_execution_authorized"] is True

    unsafe = _optimizer_execution_gate(
        {
            "argument_size_in_bytes": 31_000_000_000,
            "output_size_in_bytes": 1_000_000_000,
            "temp_size_in_bytes": 1_000_000_000,
        }
    )
    assert unsafe["full_checkpoint_optimizer_execution_authorized"] is False


def test_donated_adamw_step_compiles_with_distinct_moment_buffers(monkeypatch):
    def fake_loss_and_gradients(
        params, adapters, input_ids, loss_weights, *, config, lora_config
    ):
        del params, input_ids, loss_weights, config, lora_config
        gradients = jax.tree.map(jnp.ones_like, adapters)
        return jnp.asarray(2.0, jnp.float32), gradients

    monkeypatch.setattr(probe, "_loss_and_adapter_gradients", fake_loss_and_gradients)
    adapters = {"layer": {"a": jnp.ones((2, 2)), "b": jnp.zeros((2, 2))}}
    optimizer = adamw_init(adapters)
    step = jax.jit(
        lambda current_adapters, current_optimizer: probe._loss_and_adamw_step(
            {},
            current_adapters,
            current_optimizer,
            jnp.asarray([[1, 2]], jnp.int32),
            jnp.asarray([[0.0, 1.0]], jnp.float32),
            config=None,
            lora_config=None,
            learning_rate=1e-3,
            hyperparameters=AdamWHyperparameters(),
        ),
        donate_argnums=(0, 1),
    )
    loss, updated, optimizer, gradient_norm = step(adapters, optimizer)
    jax.block_until_ready((loss, updated, optimizer, gradient_norm))
    assert float(loss) == 2.0
    assert int(optimizer.step) == 1
    assert np.isfinite(float(gradient_norm))
