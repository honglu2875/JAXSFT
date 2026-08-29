#!/usr/bin/env python3
"""Load one local HF snapshot and emit a deterministic JAX forward audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxsft.data.tokenize import TokenizerSnapshot
from jaxsft.models.registry import get_model_implementation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=("qwen3_5", "olmo2"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    snapshot = Path(args.snapshot).expanduser().resolve()
    model = get_model_implementation(args.architecture)
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    config, params = model.load_hf_checkpoint(snapshot, dtype=dtype)
    tokenizer_snapshot, encoder = TokenizerSnapshot.load(
        snapshot,
        pad_token_id=getattr(config, "pad_token_id", None),
    )
    encoding = encoder.encode(args.prompt, add_special_tokens=False)
    if not encoding.ids:
        raise ValueError("prompt tokenized to an empty sequence")
    input_ids = jnp.asarray([encoding.ids], dtype=jnp.int32)
    logits = model.forward(params, config, input_ids)
    last = np.asarray(jax.device_get(logits[0, -1].astype(jnp.float32)))
    if not np.isfinite(last).all():
        raise FloatingPointError("public checkpoint produced non-finite logits")
    top_indices = np.argsort(last)[-5:][::-1]
    payload = {
        "architecture": args.architecture,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "snapshot": str(snapshot),
        "config_model_type": "olmo2" if args.architecture == "olmo2" else "qwen3_5_text",
        "parameter_count": model.parameter_count(config),
        "parameter_leaf_count": len(jax.tree.leaves(params)),
        "parameter_dtype_counts": {
            name: sum(value.size for value in jax.tree.leaves(params) if str(value.dtype) == name)
            for name in sorted({str(value.dtype) for value in jax.tree.leaves(params)})
        },
        "tokenizer_sha256": tokenizer_snapshot.identity_hash,
        "prompt": args.prompt,
        "input_ids": list(encoding.ids),
        "logits_shape": list(logits.shape),
        "last_logits_float32_sha256": hashlib.sha256(last.tobytes()).hexdigest(),
        "last_logits_first_8": last[:8].tolist(),
        "top_5_token_ids": top_indices.tolist(),
        "top_5_logits": last[top_indices].tolist(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
