#!/usr/bin/env python3
"""Materialize pinned model and dataset inputs before a TPU launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from jaxsft.config import load_recipe


MODEL_ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    recipe = load_recipe(args.config)

    if recipe.model.local_path:
        model_path = Path(recipe.model.local_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"model.local_path does not exist: {model_path}")
    else:
        model_path = Path(
            snapshot_download(
                repo_id=recipe.model.repo_id,
                revision=recipe.model.revision,
                allow_patterns=MODEL_ALLOW_PATTERNS,
            )
        )
    model_files = sorted(path for path in model_path.rglob("*") if path.is_file())

    dataset_record = None
    if not args.skip_data:
        if recipe.data.loading_mode != "materialized":
            raise ValueError(
                "cluster staging requires data.loading_mode=materialized; "
                "streaming would retain network resources during TPU execution"
            )
        dataset = load_dataset(
            recipe.data.repo_id,
            name=recipe.data.config,
            split=recipe.data.split,
            revision=recipe.data.revision,
            streaming=False,
        )
        dataset_record = {
            "config": recipe.data.config,
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "repo_id": recipe.data.repo_id,
            "revision": recipe.data.revision,
            "rows": len(dataset),
            "split": recipe.data.split,
        }

    print(
        json.dumps(
            {
                "dataset": dataset_record,
                "model": {
                    "bytes": sum(path.stat().st_size for path in model_files),
                    "files": len(model_files),
                    "repo_id": recipe.model.repo_id,
                    "revision": recipe.model.revision,
                    "snapshot": str(model_path),
                },
                "recipe_identity_sha256": recipe.identity_hash,
                "schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
