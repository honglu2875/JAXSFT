import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.optim import adamw_init
from train_sft import (
    load_checkpoint,
    load_synthetic_cursor,
    save_checkpoint,
    validate_optimizer_checkpoint,
    validate_rng_checkpoint,
)


def _assert_trees_equal(left, right):
    assert len(jax.tree.leaves(left)) == len(jax.tree.leaves(right))
    for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def test_checkpoint_round_trip_has_content_marker_and_replay_cursors(tmp_path):
    params = {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "norm": jnp.ones(3)}
    optimizer = adamw_init(params)._replace(step=jnp.asarray(2, jnp.int32))
    identity = "a" * 64
    data_state = {"schema_version": 1, "kind": "synthetic", "batches_consumed": 2}
    rng_state = {"schema_version": 1, "model_init_seed": 17, "next_training_step": 2}

    path = save_checkpoint(
        tmp_path,
        2,
        params,
        optimizer,
        recipe_identity_hash=identity,
        process_count=1,
        data_state=data_state,
        rng_state=rng_state,
    )
    marker = json.loads(path.with_suffix(".complete.json").read_text())
    assert marker["file"] == path.name
    assert len(marker["sha256"]) == 64

    restored = load_checkpoint(path, recipe_identity_hash=identity, process_count=1)
    assert restored["step"] == 2
    _assert_trees_equal(restored["params"], params)
    _assert_trees_equal(restored["optimizer"], optimizer)
    validate_optimizer_checkpoint(restored["params"], restored["optimizer"], expected_step=2)
    validate_rng_checkpoint(restored["rng_state"], seed=17, expected_step=2)
    assert load_synthetic_cursor(restored["data_state"], expected_step=2) == 2


def test_checkpoint_rejects_corrupt_payload_before_unpickling(tmp_path):
    params = {"kernel": jnp.ones((2, 2), jnp.float32)}
    path = save_checkpoint(
        tmp_path,
        1,
        params,
        adamw_init(params)._replace(step=jnp.asarray(1, jnp.int32)),
        recipe_identity_hash="b" * 64,
        process_count=1,
        data_state={"schema_version": 1, "kind": "synthetic", "batches_consumed": 1},
        rng_state={"schema_version": 1, "model_init_seed": 3, "next_training_step": 1},
    )
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(path, recipe_identity_hash="b" * 64, process_count=1)
