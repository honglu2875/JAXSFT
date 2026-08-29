# Single-file JAX model contract

## 1. Goal

A researcher should be able to open one architecture file and see every
model-specific choice that can change SFT numerics: configuration, tensor
layout, attention/recurrent/MoE blocks, normalization, positional encoding,
output head, checkpoint mapping, partition rules, auxiliary losses, and
capabilities.

“Single file” does not mean copying generic matrix multiplication or
cross-entropy code into every model. It means there is no inheritance maze,
central switchboard full of per-model cases, or conversion logic hidden in a
separate unrelated module.

## 2. Required contents of `models/<name>.py`

Each model file provides:

- immutable normalized config type and strict config validation;
- accepted Hugging Face `model_type`/architecture aliases;
- conversion from a pinned HF config to the normalized config;
- parameter specification and initialization;
- embeddings, all block variants, final normalization, and LM head;
- training forward path with dropout/rematerialization controls;
- optional inference/cache path only where it does not compromise clarity;
- declared auxiliary outputs such as router loss;
- direct safetensors key mapping, tensor transforms, and missing/unexpected-key
  validation;
- optional export mapping back to a supported HF checkpoint layout;
- logical axis/partition rules and topology constraints;
- capability record (training, cache, packing/position reset, MoE, quantized
  source, tied embeddings, supported checkpoint variants);
- tiny deterministic config constructor used by parity fixtures;
- a short limitations section in the module docstring.

Torch/Transformers imports occur inside compatibility functions so importing a
model for JAX training does not require them.

## 3. Allowed shared code

`models.common` may contain small, stable mathematical primitives such as:

- RMSNorm and standard activation functions;
- rotary application/table helpers with all variant parameters explicit;
- dense/embedding wrappers and logical-axis annotations;
- standard dense and grouped expert matmul primitives;
- ordinary causal attention interface;
- validation/tree helpers with no architecture dispatch.

A shared primitive must not inspect `model_type`. If two families differ in a
small but numerically important way, keep separate named functions or keep the
logic in each model file. Model files may not import one another.

Checkpoint name dispatch belongs to a small registry populated by each model's
declarative `ModelSpec`; it must not relocate conversion code into a giant
`hf_utils.py`.

## 4. Public model interface

The initial pure-PyTree implementation uses ordinary functions; every model must
satisfy an interface equivalent to:

```python
SPEC: ModelSpec

def config_from_hf(raw_config) -> Config: ...
def init(config, rng, *, dtype, param_dtype) -> Params: ...
def apply(params, batch, *, config, rngs, training) -> ModelOutput: ...
def load_hf(repo_or_path, revision, *, config, dtype, shardings) -> Params: ...
def export_hf(params, destination, *, config) -> Manifest: ...
def partition_rules(config) -> tuple[PartitionRule, ...]: ...
def estimate_memory(config, recipe, topology) -> MemoryEstimate: ...
```

`ModelOutput` contains logits or a chunked-loss-compatible hidden state/head,
plus named additive auxiliary losses/metrics. The model never chooses which
conversation tokens are supervised.

## 5. Parameter and checkpoint invariants

- Normalized config is sufficient to determine every parameter shape.
- Parameter-tree paths are stable within a model schema version.
- Tied embeddings have one source of truth and explicit export aliases.
- Every HF tensor is consumed exactly once or listed in an explicit ignored-key
  rule with rationale (for example, vision or MTP weights in a text-only path).
- Every JAX parameter is initialized or loaded; missing keys cannot retain a
  random default silently.
- Transpose, reshape, expert stacking, fused QKV/gate splitting, and dtype casts
  have individual fixture coverage.
- Quantized checkpoints require scale-aware conversion; stripping scale tensors
  and pretending weights are dense is forbidden.
- Loading is bounded-memory and reports peak host staging estimates.
- Export followed by supported HF reload preserves logits within tolerance.

## 6. Hugging Face equivalence ladder

Support is earned in layers.

### Tier 0 — Config and shape parity

- HF and JAX normalized values match for every numerically relevant field.
- Parameter counts and per-path shapes match.
- Aliases/outer text configs resolve explicitly.
- Unsupported features fail during config validation.

### Tier 1 — Component parity

Using seeded tiny float32 configs, compare embeddings, norm, rotary tables,
attention/recurrent mixer, MLP/MoE router and experts, one block, and head.
Include masks, padding, positions, and boundary lengths.

### Tier 2 — Full-forward parity

Convert a seeded tiny HF model and compare complete logits for multiple batches,
lengths, masks, and (where supported) train/eval modes. Tolerances are declared
per dtype/backend; loosening one requires a recorded reason.

### Tier 3 — Backward and objective parity

Use the same token IDs and fractional target weights to compare loss numerator,
denominator, selected parameter gradients, global gradient norm, clipping, and
one optimizer update. This catches correct logits paired with incorrect causal
shift or reduction.

### Tier 4 — Public checkpoint integration

At a pinned immutable Hub revision:

- direct safetensors load succeeds with a complete key audit;
- fixed prompt logits/hidden checksums or bounded slices match the HF reference;
- tokenizer/template fixtures match;
- JAX checkpoint save/restore preserves outputs;
- export/reload is tested when export is advertised.

### Tier 5 — Distributed training support

- declared partition rules cover every parameter;
- one update matches the unsharded reference within backend tolerance;
- model state checkpoints and restores across the target mesh;
- activation/parameter/optimizer memory fits the measured topology;
- a short real SFT run has finite stable metrics.

The compatibility table must distinguish “architecture parity,” “checkpoint
load,” and “trainable on this topology.”

## 7. Initial model order

| Priority | Family | Reason and first proof point |
|---|---|---|
| P0 | Qwen3.5 dense text | The 0.8B base checkpoint makes integration affordable while exercising hybrid Gated DeltaNet/full attention. |
| P0 | OLMo dense (OLMo 2 or 3 selected in M0) | Proves the interfaces are not tailored to Qwen and offers an open research baseline. |
| P1 | Qwen3 MoE | Introduces router/expert parameters and auxiliary-loss handling without hybrid recurrence. |
| P1 | Qwen3 dense and other Qwen3.5/3.6 variants | Broadens checkpoint coverage after the initial dense hybrid baseline. |
| P1 | GLM Flash text | Exercises MLA/fine-grained MoE through an independently tested model file. |
| P1/P2 | Kimi text family | High user priority, but exact variant and full-checkpoint feasibility need clarification. Tiny-config parity can precede large-weight training. |
| P2 | Llama/Gemma/Mistral | Broad ecosystem coverage after the differentiating data/loss path is stable. |

Where architectures share mathematics (for example, some Kimi/GLM and
DeepSeek-style MLA/MoE components), they may share explicit low-level primitives
but retain separate model files and config/checkpoint mappings.

## 8. Training-specific requirements

Every training-capable model declares and tests:

- dropout locations and deterministic RNG names;
- activation rematerialization boundaries;
- gradient dtype and parameter/optimizer-state dtype behavior;
- padding and attention-mask semantics;
- supported position-ID reset behavior for packing;
- tied/untied output head behavior;
- router auxiliary losses and whether they are token-weight aware;
- sharded-vocabulary loss interface;
- scan/unroll policy and compilation-shape constraints;
- behavior with zero selected target tokens in a local microbatch;
- serialization of trainable adapters and frozen parameters.

LoRA is expressed through explicit parameter transforms/filtering over model
paths and verified against a merged dense result. A model file declares eligible
paths; a generic adapter implementation performs the math.

## 9. Model review checklist

A pull request adding a model must answer:

1. Which exact upstream config classes and checkpoint revisions are covered?
2. Which text, vision, audio, MTP, cache, quantization, and custom-code paths are
   intentionally excluded?
3. What are the tiny parity tolerances and public-checkpoint proof?
4. Are all mapping exceptions explicit and tested?
5. What topology and memory were actually tested for training?
6. Does the model import only approved common primitives?
7. Can a reviewer understand the forward and conversion path from this file and
   its focused test file without searching a framework registry?
