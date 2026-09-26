# Qwen2 and Qwen2.5-Math

`src/jaxsft/models/qwen2.py` implements the dense, full-attention Qwen2
architecture in one independent JAX module. Select `model.architecture: qwen2`.
The integration checkpoint is
[`Qwen/Qwen2.5-Math-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/tree/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2),
revision `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`.

The implementation includes grouped-query causal attention, Q/K/V biases,
default RoPE, pre-attention and pre-MLP RMSNorm, SwiGLU, and tied or untied
language-model heads. Safetensors conversion supports single files and shard
indexes, checks every required tensor and shape, rejects unexpected tensors,
and checks duplicate tied-head values. Torch conversion copies storage so a
later in-place Torch update cannot mutate the JAX parameter tree.

The `qwen2_5_math` renderer matches this pinned checkpoint's text chat template,
including its default math system instruction when none is supplied. It keeps
content whitespace and semantic part ownership; assistant content and its
end marker receive the default objective weight. An explicit system message
overrides the default instruction. Tool messages, tool definitions, and
separate reasoning fields are rejected; reasoning can be ordinary content.
`qwen2` selects this Math renderer by default. Other Qwen2 checkpoint templates
are not asserted to share that dialect.

## Validation

Optional Transformers tests cover valid-position logits, fractionally weighted
causal loss, every parameter gradient, clipping and an AdamW update, tied and
untied heads, custom positions, padding, and single/sharded safetensors reloads.
The tokenizer oracle compares rendered bytes and token IDs for multiple turns,
both with and without system messages and generation prompts.

The public 1,543,714,304-parameter checkpoint passed a float32 comparison of all
759,680 logits from a five-token input against Transformers 5.16.1 / Torch
2.13.0+cpu. Maximum absolute error was `0.0001716614`; mean absolute error was
`0.0000103037` (`atol=rtol=3e-4`). This is an implementation comparison, not a
proof-generation quality measurement.

A v4-32 run completed three real BF16 updates across four processes and 16
global TPU devices. All reduced metrics agreed exactly across ranks, each rank
received a distinct first batch, and every process shut down cleanly. First-step
compilation/execution took about 78.2 seconds; subsequent steps took 0.155–0.187
seconds per host. The run did not write a full-size optimizer checkpoint. See
the [sanitized acceptance record](../results/qwen2_math_v4_32_smoke.json).

```bash
JAX_PLATFORMS=cpu uv run --extra parity pytest tests/parity/test_qwen2_hf.py
# Set JAXSFT_QWEN2_MATH_SNAPSHOT to the pinned local checkpoint for tokenizer tests.
# Set JAXSFT_QWEN2_PUBLIC_PARITY=1 as well to enable the full 1.5B CPU comparison.

uv run python train_sft.py \
  --config configs/recipes/qwen25_math_1_5b_ultrachat_smoke.yaml
```

The checked-in recipe is a three-update infrastructure smoke with assistant
loss on pinned UltraChat data. It is not a Math fine-tuning recommendation.
Checkpoint cadence is zero for this smoke; run/output paths remain ordinary
recipe or cluster-profile configuration.

## Scope

Nonzero attention dropout, sliding-window layers, non-default RoPE scaling,
multi-dimensional RoPE, and non-SiLU activations fail explicitly. This module
has no cached-generation, quantized-loading, or Hugging Face export path.
Training uses the existing replicated data-parallel trainer. Public checkpoints
other than the pinned Math 1.5B model require their own numerical validation.
