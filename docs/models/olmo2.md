# OLMo 2 dense text

## Covered surface

`src/jaxsft/models/olmo2.py` is a self-contained pure-JAX implementation of
the dense text-only Hugging Face `Olmo2ForCausalLM` architecture. It includes
config normalization, initialization, grouped-query attention, full RoPE,
q/k RMS normalization, post-attention/post-MLP RMS normalization, SwiGLU,
tied or untied heads, strict parameter validation, lazy safetensors loading,
and checkpoint conversion.

The public integration checkpoint is
`allenai/OLMo-2-0425-1B` at immutable revision
`a1847dff35000b4271fa70afc5db10fd29fedbdf`. Its audited tree contains 179
tensors and 1,484,916,736 parameters. The source checkpoint is float32; the
SFT smoke converts parameters directly to bfloat16.

The base tokenizer has no chat template. JAXSFT therefore names the formatting
choice explicitly: `olmo2_instruct`, copied semantically from the paired
`allenai/OLMo-2-0425-1B-Instruct` tokenizer config at revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`. The base and instruct
`tokenizer.json` files have the same SHA-256,
`73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca`.
The renderer preserves whitespace and matches Transformers rendered bytes and
token IDs for the fixture suite.

## Equivalence evidence

The seeded tiny float32 oracle compares complete valid-token logits,
fractionally weighted causal loss, a selected MLP gradient, global gradient
norm/clipping, and one AdamW kernel update against Transformers/PyTorch. The
declared tolerances are `rtol=atol=2e-4` for logits/loss and `5e-4` for the
selected gradient/update.

For the pinned public 1B checkpoint, a five-token prompt compared every one of
100,352 final-position logits between float32 JAX on TPU and float32
Transformers on CPU. It passed `rtol=atol=2e-4`; maximum absolute error was
0.0007782, mean absolute error was 0.0002363, and the ordered top-five token IDs
were identical. The bfloat16 loader/forward path separately produced finite
logits after the complete tensor audit.

A later full-model gate materialized 20 exact global batches (four examples by
256 tokens) and replayed them through three implementations. The CPU oracle
imports no JAXSFT runtime code: it independently parses/hashes the recipe and
batch tape, uses `AutoModelForCausalLM`, PyTorch cross-entropy, stock
`transformers.Trainer` AdamW parameter groups and cosine-with-minimum-LR
scheduler, and Accelerate on CPU. Recipe identity, tape identity, loss
denominator, input-token count, and update learning rate matched at every step.

Against that FP32 CPU run, production BF16 TPU loss error had an 8.52% maximum,
5.61% early post-update mean, 5.28% late mean, and 4.05% final value. Thus the
short trajectory was wider but did not progressively separate. A forced-FP32
TPU control reduced maximum relative loss error to 0.0214%; its least-squares
relative-error slope was 0.000308 percentage points per step. These comparisons
do not claim bit equality or isolate TPU tiling/reduction behavior from dtype
effects. Full evidence is in
[the 20-step trajectory record](../results/olmo2_1b_trajectory_parity_20.json).

## Training evidence

The same trainer and pinned UltraChat stream used by Qwen completed three
full-parameter updates on one v4-8 with four replicated local devices, sequence
length 256, per-device batch size one, rematerialized blocks, bfloat16
parameters, and FP32 Adam moments. All loss and gradient metrics were finite.
A tiny OLMo2 TPU run interrupted at step two and cold-resumed through step
three; its final checkpoint was byte-identical to the uninterrupted reference.

The later four-host v4-32 gate loaded the same 1,484,916,736-parameter revision
on four JAX processes/16 devices and completed three additional materialized
UltraChat updates. Runtime-rank first-batch hashes were all distinct, while
every globally reduced loss, gradient norm, token count, denominator, and
accuracy was identical across ranks. Loss moved from 1.93071 to 1.64470; the
first compiled update took 54.45 seconds and the next two took 0.14–0.23
seconds per host. All processes shut down cleanly. A same-trainer synthetic
OLMo configuration separately proved the four-rank schema-v4 cold-resume path;
resumed and uninterrupted step-three model/optimizer state hashes matched
exactly. See [the acceptance record](../results/v4_32_multihost_acceptance.json).

The 20-step BF16 trajectory subsequently completed under the same replicated
v4-8 topology with `jax_default_matmul_precision=highest` and explicit
`lax.Precision.HIGHEST`. Its first step, including compile, took 63.42 seconds;
steps 3–20 averaged 0.1166 seconds. The matched FP32 control fit as well, with a
105.83-second first step and 0.2480-second mean for steps 3–20.

Bfloat16 parameters occupy about 2.77 GiB and two FP32 Adam moment trees about
11.06 GiB per replica before gradients, activations, executable storage, and
allocator overhead. The smoke proves that this recipe fit the measured host; it
does not provide an allocator peak or promise that longer/larger variants fit.
See the [sanitized result](../results/olmo2_1b_v4_8_smoke.json).

## Deliberate exclusions

- attention dropout and training RNGs other than zero-dropout deterministic SFT;
- attention projection biases and non-default/scaled/dynamic RoPE;
- KV-cache generation, speculative decoding, and logits slicing;
- model-axis parameter sharding or sharded vocabulary loss;
- Hugging Face export and quantized checkpoints;
- vision/audio/MTP paths (none are part of the covered causal-LM checkpoint);
- tool calls, tool results, reasoning parts, and developer roles in the pinned
  OLMo Instruct template. The renderer rejects these instead of dropping them;
- a full 1.5B optimizer checkpoint/restore test. Model-independent checkpoint
  semantics were exercised with the tiny OLMo2 tree.

Other OLMo 2 checkpoints are not advertised merely because they share
`model_type=olmo2`; their configs must pass validation and a pinned weight audit.
