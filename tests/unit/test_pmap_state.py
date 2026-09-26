import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train_sft import replicate_pmap_tree, unreplicate_pmap_tree


def test_state_placement_and_host_checkpoint_survive_donated_pmap_update():
    devices = list(jax.local_devices())
    host = {
        "params": {"bf16": np.asarray(jnp.arange(12, dtype=jnp.bfloat16)).reshape(3, 4).T},
        "optimizer": (np.arange(6, dtype=np.float32).reshape(2, 3), np.asarray(2, dtype=np.int32)),
    }
    state = replicate_pmap_tree(host, devices)
    for actual, expected in zip(jax.tree.leaves(state), jax.tree.leaves(host), strict=True):
        assert actual.shape == (len(devices), *expected.shape)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(jax.device_get(actual), np.broadcast_to(expected, actual.shape))

    # Distinct device values ensure extraction selects replica zero, rather
    # than accidentally aggregating copies or choosing a different replica.
    def update(tree):
        return jax.tree.map(lambda x: x + (jax.lax.axis_index("replica") + 1).astype(x.dtype), tree)

    updated = jax.pmap(update, axis_name="replica", devices=devices, donate_argnums=(0,))(state)
    checkpoint = unreplicate_pmap_tree(updated, local_device_count=len(devices))
    assert jax.tree.structure(checkpoint) == jax.tree.structure(host)
    for actual, expected in zip(jax.tree.leaves(checkpoint), jax.tree.leaves(host), strict=True):
        assert isinstance(actual, (np.ndarray, np.generic))
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected + np.asarray(1, dtype=expected.dtype))

    restored = replicate_pmap_tree(checkpoint, devices)
    round_trip = unreplicate_pmap_tree(restored, local_device_count=len(devices))
    for actual, expected in zip(jax.tree.leaves(round_trip), jax.tree.leaves(checkpoint), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_state_placement_rejects_bad_device_and_replica_layouts():
    devices = list(jax.local_devices())
    with pytest.raises(ValueError, match="at least one"):
        replicate_pmap_tree({"x": np.ones(2)}, [])
    with pytest.raises(ValueError, match="distinct local"):
        replicate_pmap_tree({"x": np.ones(2)}, [devices[0], devices[0]])
    with pytest.raises(ValueError, match="positive"):
        unreplicate_pmap_tree({}, local_device_count=0)
    with pytest.raises(ValueError, match="replica axis"):
        unreplicate_pmap_tree({"x": jnp.ones(())}, local_device_count=len(devices))
    if len(devices) > 1:
        # This has the right leading dimension but partitions the model axis.
        mesh = jax.sharding.Mesh(np.asarray(devices), ("model",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
        wrong = jax.device_put(np.ones((len(devices), len(devices))), sharding)
        with pytest.raises(ValueError, match="complete, addressable first replica"):
            unreplicate_pmap_tree({"x": wrong}, local_device_count=len(devices))
