#!/usr/bin/env python3
"""Export deterministic SFT batches for cross-framework trajectory checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jaxsft.batch_tape import write_batch_tape
from jaxsft.config import load_recipe
from jaxsft.data.stream import InstructionBatchStream
from jaxsft.data.tokenize import LossPolicy, TokenizerSnapshot


def resolve_snapshot(recipe, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model snapshot does not exist: {path}")
        return path
    if recipe.model.local_path:
        path = Path(recipe.model.local_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"model.local_path does not exist: {path}")
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=recipe.model.repo_id,
            revision=recipe.model.revision,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "config.json",
            ],
        )
    )


def flatten_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for name, value in batch.items():
        # [device, accumulation, per_device, length] -> canonical [batch, length].
        canonical = np.swapaxes(value, 0, 1)
        result[name] = canonical.reshape(-1, canonical.shape[-1])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model-snapshot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = load_recipe(args.config)
    steps = recipe.training.steps if args.steps is None else args.steps
    if not 0 < steps <= recipe.training.steps:
        raise ValueError("--steps must be in [1, recipe.training.steps]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if recipe.training.gradient_accumulation_steps != 1:
        raise ValueError("batch-tape export currently requires gradient_accumulation_steps=1")

    snapshot = resolve_snapshot(recipe, args.model_snapshot)
    tokenizer_snapshot, encoder = TokenizerSnapshot.load(snapshot)
    stream = InstructionBatchStream(
        recipe.data,
        tokenizer_snapshot=tokenizer_snapshot,
        encoder=encoder,
        policy=LossPolicy.from_config(recipe.objective),
        process_index=0,
        process_count=1,
        local_device_count=1,
        per_device_batch_size=args.batch_size,
        accumulation_steps=1,
        max_length=recipe.training.max_length,
        truncation=recipe.training.truncation,
        truncation_min_context_tokens=recipe.training.truncation_min_context_tokens,
        renderer=recipe.data.renderer or recipe.model.architecture,
    )
    try:
        batches = [flatten_batch(stream.next_batch()) for _ in range(steps)]
        tape = write_batch_tape(
            args.output,
            batches,
            recipe_identity_hash=recipe.identity_hash,
            model={"repo_id": recipe.model.repo_id, "revision": recipe.model.revision},
            data={
                "repo_id": recipe.data.repo_id,
                "revision": recipe.data.revision,
                "config": recipe.data.config,
                "split": recipe.data.split,
                "adapter": recipe.data.adapter,
                "renderer": recipe.data.renderer or recipe.model.architecture,
                "loading_mode": recipe.data.loading_mode,
            },
            tokenizer_identity_hash=tokenizer_snapshot.identity_hash,
            pad_token_id=tokenizer_snapshot.pad_token_id,
            stream_counters=asdict(stream.counters),
        )
    finally:
        stream.close()
    print(json.dumps(tape.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
