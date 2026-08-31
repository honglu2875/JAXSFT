import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxsft.optim import adamw_init
from jaxsft.sharded_checkpoint import (
    restore_local_sharded_pytree,
    save_local_sharded_pytree,
)


def _state():
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("model",))
    replicated = NamedSharding(mesh, PartitionSpec())
    adapters = {
        "layer": {
            "a": jax.device_put(jnp.arange(6, dtype=jnp.bfloat16).reshape(3, 2), replicated),
            "b": jax.device_put(jnp.arange(8, dtype=jnp.bfloat16).reshape(2, 4), replicated),
        }
    }
    optimizer = jax.tree.map(
        lambda value: jax.device_put(value, replicated),
        adamw_init(adapters),
    )
    tree = {"adapters": adapters, "optimizer": optimizer}
    template = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(
            value.shape,
            value.dtype,
            sharding=value.sharding,
        ),
        tree,
    )
    return tree, template


def test_rank_local_sharded_checkpoint_round_trip_is_byte_exact(tmp_path):
    tree, template = _state()
    identity = {"model": "tiny", "step": 2}
    directory = tmp_path / "step-00000002"
    saved = save_local_sharded_pytree(
        tree,
        directory,
        process_index=0,
        process_count=1,
        identity=identity,
        allowed_root_keys=("adapters", "optimizer"),
    )
    assert saved["root_keys"] == ["adapters", "optimizer"]
    assert saved["leaf_count"] == 7
    assert saved["local_unique_shard_count"] == 7
    assert saved["local_unique_tensor_bytes"] < saved["npz_file_bytes"]

    restored, restore = restore_local_sharded_pytree(
        template,
        directory,
        process_index=0,
        process_count=1,
        identity=identity,
        allowed_root_keys=("adapters", "optimizer"),
    )
    assert restore["all_local_shards_byte_exact"] is True
    assert restore["manifest_sha256"] == saved["manifest_sha256"]
    for expected, actual in zip(
        jax.tree.leaves(tree),
        jax.tree.leaves(restored),
        strict=True,
    ):
        assert np.array_equal(np.asarray(expected), np.asarray(actual))


def test_sharded_checkpoint_rejects_base_roots_and_payload_tampering(tmp_path):
    tree, template = _state()
    with pytest.raises(ValueError, match="allowlist"):
        save_local_sharded_pytree(
            {**tree, "params": {}},
            tmp_path / "rejected",
            process_index=0,
            process_count=1,
            identity={"step": 2},
            allowed_root_keys=("adapters", "optimizer"),
        )

    directory = tmp_path / "tampered"
    save_local_sharded_pytree(
        tree,
        directory,
        process_index=0,
        process_count=1,
        identity={"step": 2},
        allowed_root_keys=("adapters", "optimizer"),
    )
    payload = directory.joinpath("rank-000.npz").read_bytes()
    directory.joinpath("rank-000.npz").write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        restore_local_sharded_pytree(
            template,
            directory,
            process_index=0,
            process_count=1,
            identity={"step": 2},
            allowed_root_keys=("adapters", "optimizer"),
        )
