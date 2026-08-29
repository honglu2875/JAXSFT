import json
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.optim import adamw_init
from train_sft import (
    environment_flag,
    git_identity,
    load_checkpoint,
    load_synthetic_cursor,
    save_checkpoint,
    validate_optimizer_checkpoint,
    validate_rng_checkpoint,
)


SOURCE = {"kind": "test", "sha256": "c" * 64}
TOPOLOGY = {
    "process_index": 0,
    "process_count": 1,
    "local_device_count": 4,
    "global_device_count": 4,
    "backend": "cpu",
    "devices": ["cpu:0", "cpu:1", "cpu:2", "cpu:3"],
}


def _assert_trees_equal(left, right):
    assert len(jax.tree.leaves(left)) == len(jax.tree.leaves(right))
    for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def test_environment_flag_is_explicit_and_rejects_ambiguous_values(monkeypatch):
    monkeypatch.delenv("JAXSFT_FORCE_FP32", raising=False)
    assert environment_flag("JAXSFT_FORCE_FP32") is False
    monkeypatch.setenv("JAXSFT_FORCE_FP32", "1")
    assert environment_flag("JAXSFT_FORCE_FP32") is True
    monkeypatch.setenv("JAXSFT_FORCE_FP32", "true")
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        environment_flag("JAXSFT_FORCE_FP32")


def test_checkpoint_round_trip_has_content_marker_and_replay_cursors(tmp_path):
    params = {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "norm": jnp.ones(3)}
    optimizer = adamw_init(params)._replace(step=jnp.asarray(2, jnp.int32))
    identity = "a" * 64
    data_state = {
        "schema_version": 2,
        "kind": "synthetic",
        "batches_consumed": 2,
        "length": 32,
        "vocab_size": 128,
    }
    rng_state = {"schema_version": 1, "model_init_seed": 17, "next_training_step": 2}

    path = save_checkpoint(
        tmp_path,
        2,
        params,
        optimizer,
        recipe_identity_hash=identity,
        source_identity=SOURCE,
        topology=TOPOLOGY,
        data_state=data_state,
        rng_state=rng_state,
    )
    marker = json.loads(path.with_suffix(".complete.json").read_text())
    assert marker["file"] == path.name
    assert len(marker["sha256"]) == 64

    restored = load_checkpoint(
        path,
        recipe_identity_hash=identity,
        source_identity=SOURCE,
        topology=TOPOLOGY,
    )
    assert restored["step"] == 2
    _assert_trees_equal(restored["params"], params)
    _assert_trees_equal(restored["optimizer"], optimizer)
    validate_optimizer_checkpoint(restored["params"], restored["optimizer"], expected_step=2)
    validate_rng_checkpoint(restored["rng_state"], seed=17, expected_step=2)
    assert load_synthetic_cursor(restored["data_state"], expected_step=2, length=32, vocab_size=128) == 2
    with pytest.raises(ValueError, match="source identity"):
        load_checkpoint(
            path,
            recipe_identity_hash=identity,
            source_identity={"kind": "test", "sha256": "d" * 64},
            topology=TOPOLOGY,
        )


def test_checkpoint_rejects_corrupt_payload_before_unpickling(tmp_path):
    params = {"kernel": jnp.ones((2, 2), jnp.float32)}
    path = save_checkpoint(
        tmp_path,
        1,
        params,
        adamw_init(params)._replace(step=jnp.asarray(1, jnp.int32)),
        recipe_identity_hash="b" * 64,
        source_identity=SOURCE,
        topology=TOPOLOGY,
        data_state={
            "schema_version": 2,
            "kind": "synthetic",
            "batches_consumed": 1,
            "length": 16,
            "vocab_size": 64,
        },
        rng_state={"schema_version": 1, "model_init_seed": 3, "next_training_step": 1},
    )
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(
            path,
            recipe_identity_hash="b" * 64,
            source_identity=SOURCE,
            topology=TOPOLOGY,
        )


def test_git_source_identity_hashes_untracked_file_contents(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=JAXSFT Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    clean = git_identity(tmp_path)
    untracked = tmp_path / "research.py"
    untracked.write_text("choice = 1\n")
    first = git_identity(tmp_path)
    untracked.write_text("choice = 2\n")
    second = git_identity(tmp_path)
    assert clean["dirty_material_sha256"] != first["dirty_material_sha256"]
    assert first["status"] == second["status"]
    assert first["dirty_material_sha256"] != second["dirty_material_sha256"]
