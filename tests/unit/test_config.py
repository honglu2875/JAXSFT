from pathlib import Path

import pytest

from jaxsft.config import load_recipe


RECIPE = Path(__file__).parents[2] / "configs" / "recipes" / "qwen35_0_8b_ultrachat_smoke.yaml"


def test_smoke_recipe_is_pinned_and_strict():
    recipe = load_recipe(RECIPE)
    assert recipe.model.repo_id == "Qwen/Qwen3.5-0.8B-Base"
    assert len(recipe.model.revision) == 40
    assert recipe.data.adapter == "ultrachat_200k"
    assert len(recipe.data.revision) == 40
    assert len(recipe.identity_hash) == 64


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_recipe(path)


def test_invalid_optimizer_value_is_rejected_during_dry_run_validation(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(RECIPE.read_text().replace("peak_learning_rate: 2.0e-5", "peak_learning_rate: -1.0"))
    with pytest.raises(ValueError, match="peak_learning_rate"):
        load_recipe(path)


def test_unknown_objective_key_is_rejected_during_recipe_load(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(RECIPE.read_text().replace("objective:\n", "objective:\n  guessed_mask: true\n"))
    with pytest.raises(ValueError, match="unknown objective keys"):
        load_recipe(path)


def test_recipe_identity_hashes_resolved_defaults_not_yaml_spelling(tmp_path):
    original = load_recipe(RECIPE)
    explicit_path = tmp_path / "explicit-default.yaml"
    explicit_path.write_text(
        RECIPE.read_text().replace("  truncation: right\n", "  truncation: right\n  truncation_min_context_tokens: 0\n")
    )
    explicit = load_recipe(explicit_path)
    assert explicit.identity_hash == original.identity_hash
    assert explicit.public_dict() == original.public_dict()
