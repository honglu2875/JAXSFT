import ast
import json
from pathlib import Path

import numpy as np
import pytest

from jaxsft.batch_tape import BatchTape, write_batch_tape
from jaxsft.config import load_recipe
from scripts.run_hf_trajectory import OracleRecipe, load_oracle_recipe, load_oracle_tape


def _batches():
    result = []
    for step in range(3):
        input_ids = np.arange(step * 24, (step + 1) * 24, dtype=np.int32).reshape(4, 6)
        attention_mask = np.ones((4, 6), dtype=np.bool_)
        attention_mask[-1, -1] = False
        loss_weights = np.zeros((4, 6), dtype=np.float32)
        loss_weights[:, 3:5] = 1.0
        result.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_weights": loss_weights,
            }
        )
    return result


def _write(tmp_path):
    return write_batch_tape(
        tmp_path / "tape",
        _batches(),
        recipe_identity_hash="a" * 64,
        model={"repo_id": "org/model", "revision": "b" * 40},
        data={
            "repo_id": "org/data",
            "revision": "c" * 40,
            "config": "default",
            "split": "train",
            "adapter": "messages",
            "renderer": "olmo2_instruct",
            "loading_mode": "materialized",
        },
        tokenizer_identity_hash="d" * 64,
        pad_token_id=0,
        stream_counters={"rows_seen": 12, "rows_emitted": 12},
    )


def test_batch_tape_round_trip_and_jax_topology_reshape(tmp_path):
    tape = _write(tmp_path)
    assert tape.steps == 3
    assert tape.batch_size == 4
    assert tape.length == 6
    assert len(tape.identity_hash) == 64
    batch = tape.jax_batch(
        1,
        local_device_count=2,
        accumulation_steps=1,
        per_device_batch_size=2,
    )
    assert batch["input_ids"].shape == (2, 1, 2, 6)
    np.testing.assert_array_equal(batch["input_ids"].reshape(4, 6), _batches()[1]["input_ids"])

    state = tape.state_dict(next_step=2)
    tape.validate_state_dict(state, expected_step=2)
    with pytest.raises(ValueError, match="cursor"):
        tape.validate_state_dict(state, expected_step=1)


def test_batch_tape_rejects_tampered_array_and_manifest_identity(tmp_path):
    tape = _write(tmp_path)
    values = np.load(tape.root / "input_ids.npy", allow_pickle=False)
    values[0, 0, 0] += 1
    np.save(tape.root / "input_ids.npy", values, allow_pickle=False)
    with pytest.raises(ValueError, match="digest"):
        BatchTape.load(tape.root)

    clean = _write(tmp_path / "second")
    manifest_path = clean.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shape"]["steps"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        BatchTape.load(clean.root)


def test_hugging_face_oracle_independently_parses_recipe_and_tape(tmp_path):
    config_path = Path(__file__).parents[2] / "configs" / "recipes" / "olmo2_1b_ultrachat_trajectory_20.yaml"
    assert load_oracle_recipe(config_path).identity_hash == load_recipe(config_path).identity_hash

    tape = _write(tmp_path)
    oracle_recipe = OracleRecipe(
        identity_hash="a" * 64,
        model={"repo_id": "org/model", "revision": "b" * 40},
        data={
            "repo_id": "org/data",
            "revision": "c" * 40,
            "config": "default",
            "split": "train",
            "adapter": "messages",
            "renderer": "olmo2_instruct",
            "loading_mode": "materialized",
        },
        training={},
        run={},
    )
    independent = load_oracle_tape(tape.root, oracle_recipe)
    assert independent.identity_hash == tape.identity_hash
    np.testing.assert_array_equal(independent.arrays["input_ids"], tape.arrays["input_ids"])


def test_hugging_face_oracle_source_has_no_jaxsft_runtime_imports():
    source = Path(__file__).parents[2] / "scripts" / "run_hf_trajectory.py"
    tree = ast.parse(source.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name == "jaxsft" or name.startswith("jaxsft.")]
