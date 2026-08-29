# Reference review

This document records design lineage and implementation sources. The review was
performed on 2026-08-29 against the following snapshots:

- [honglu2875/rig](https://github.com/honglu2875/rig), commit
  `aed696d6b3ed9bbd15c14e0bc4c3161b8470fe12`.
- [honglu2875/jaxml](https://github.com/honglu2875/jaxml), commit
  `df150e3f2be0b4f8db3aa51c1dc4032b7fb66b85`.

Both upstream repositories are Apache-2.0 at the inspected revisions. JAXSFT's
controller is an independent implementation informed by RIG's operational
lessons. The mathematical organization in `models/qwen3_5.py` is adapted from
JAXML's Qwen3.5 implementation and checked against Hugging Face Transformers
commit `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`; the source header and `NOTICE`
preserve that attribution.

## 1. RIG: ideas to retain

RIG's valuable research-repository choices include:

- a short top-level command surface (`make check`, prepare, run, profile,
  report) backed by copyable commands;
- strict separation between versioned research constants and ignored
  machine-local cluster/cache/artifact settings;
- CPU-only tests as the everyday gate, with accelerator work explicitly
  separated;
- a clear result protocol and run artifacts that preserve source/config/data/
  topology identity;
- explicit single-slice versus multislice scope;
- ordinary `pdsh`/OpenSSH as process launch, with `rsync` for incremental or
  offline distribution;
- sequential SSH warm-up, safe non-interactive options, timeout handling, and
  retry of transport failures rather than application failures;
- `jax.distributed.initialize()` before device inspection;
- use of `jax.process_index()` rather than launcher rank variables or hostname
  suffixes;
- explicit artifact-host/controller identity because JAX rank order does not
  have to match `-w-N` order;
- rank-local data portions while arrays span a global JAX mesh;
- precise teardown matching a run instead of assuming killing local `pdsh`
  kills remote XLA processes;
- periodic artifact salvage from a remote controller;
- validation of cache mounts, free space, file sizes, and hashes before a run;
- capturing dirty/untracked experiment work rather than pretending only HEAD
  ran;
- named, immutable data manifests and no silent fallback to another corpus;
- structured smoke/development/official execution types and protocol tests.

The cluster behavior is concentrated in `rig/harness/cluster.py`; its focused
tests cover host expansion, transport retry, rsync filters, remote-controller
identity, cache ownership, launch construction, teardown, and artifact fetch.
That density of failure-mode tests is worth emulating even if JAXSFT's source
deployment mechanism differs.

## 2. RIG: choices to change for JAXSFT

- RIG deliberately favors clone-and-change algorithm directories. JAXSFT will
  use one normal SFT entry program plus strict recipes because the primary
  experimental dimensions are dataset/render/objective combinations. A copied
  entry remains an escape hatch for algorithm research.
- RIG mirrors a live checkout authoritatively to the same absolute path. JAXSFT
  proposes immutable run-specific source capsules, which better isolate
  concurrent long SFT runs and make dirty-state capture a first-class artifact.
- RIG's token data is a comparatively uniform pretraining format. JAXSFT
  needs a semantic canonical representation, model renderers, per-token
  metadata, and dataset-adapter fixture corpus.
- RIG can stage a known token corpus in protected RAM. JAXSFT must not assume
  instruction data, model weights, optimizer state, or multi-host checkpoints
  fit or persist in `/dev/shm`.
- RIG's shared utilities and single-file recipes target a fixed GPT family.
  JAXSFT needs self-contained files per external architecture and direct
  checkpoint conversion.

## 3. JAXML: ideas to retain

JAXML demonstrates that readable architecture modules can coexist with shared
neural-network primitives and a small runtime. Useful patterns include:

- one substantial model file per architecture/family;
- separate common attention, embedding, linear, normalization, and position
  primitives;
- normalized configs derived from Hugging Face configs;
- explicit checkpoint tensor conversion with shape/transposition tests;
- tiny seeded PyTorch/Transformers versus JAX logit tests;
- focused tests for RoPE variants, model heads, input masks, sharding, cache, and
  public API;
- lazy safetensors access for large Qwen/Kimi checkpoints so unrelated vision/
  MTP tensors and a full Torch model need not be materialized;
- an honest support table that calls out text-only paths, pending kernels,
  cache limitations, and memory feasibility;
- critical/milestone/TPU test cadences and frozen dependency checks.

At the inspected revision, JAXML contains text implementations relevant to the
requested direction: Llama/Mistral, Gemma3, DeepSeek/GLM/Kimi-compatible
MLA+MoE, Kimi Linear, and Qwen3.5/3.6 hybrid text. These are architectural
references; JAXSFT still needs independently gated training behavior,
gradients, checkpointing, and weighted-objective parity.

## 4. JAXML: choices to change for JAXSFT

- Keep each model's HF config/checkpoint mapping beside its model file instead
  of accumulating all families in one central compatibility module.
- Make training requirements first-class: dropout RNG, rematerialization,
  gradients, optimizer state, auxiliary losses, packing positions, and
  distributed checkpoint round trips.
- Require weighted loss and one-step optimizer parity, not logits alone.
- Separate “architecture implemented,” “public checkpoint loads,” and “model is
  trainable on the measured slice.”
- Prefer direct-to-final-dtype/sharding load as the model state grows; JAXML's
  README explicitly notes that its converted tree can still materialize on the
  host before sharding.

## 5. Hugging Face compatibility surface

The plan follows current official Hugging Face interfaces where they are useful,
while pinning revisions because those interfaces and model templates evolve.

- [Transformers chat template tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)
  exposes `return_assistant_tokens_mask` for templates containing generation
  blocks. JAXSFT treats it as a parity signal, not as enough metadata for
  reasoning/tool-part research.
- [Transformers tool-use guidance](https://huggingface.co/docs/transformers/chat_extras)
  recommends assistant `tool_calls` as a list of structured function calls and
  tool results as string content on `tool` messages. Adapters accept this as one
  family while retaining source-specific variants.
- [Writing chat templates](https://huggingface.co/docs/transformers/main/chat_templating_writing)
  stresses that tool formatting, special tokens, and whitespace are
  model-specific. This motivates the separate renderer and exact token fixtures.
- [TRL's SFT trainer](https://huggingface.co/docs/trl/sft_trainer) supports raw or
  conversational language modeling, prompt/completion data, pre-tokenized
  `input_ids`/labels, assistant-only loss, completion-only loss, and packing.
  JAXSFT should cover these baselines and then generalize from boolean masks
  to typed float-weighted semantic spans.
- [Datasets streaming guidance](https://huggingface.co/docs/datasets/stream)
  documents shard order, buffer shuffling, and iterable sharding behavior. Exact
  multi-host resume requires recording those details or preprocessing an
  immutable local artifact.

## 6. Pinned first integration and Hub catalog observations

The first executable integration pins
[`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) at
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` and
[`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
at `8049631c405ae6576f93f445c6b8166f76f5505a`. The public Hub was also queried
with the Hugging Face CLI on 2026-08-29 to avoid planning around stale or
ambiguous names.

- Official Qwen results include [Qwen3 dense](https://huggingface.co/Qwen/Qwen3-0.6B),
  [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B), and
  [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) families.
- Official AllenAI results include [OLMo 2](https://huggingface.co/allenai/OLMo-2-1124-7B)
  and [OLMo 3](https://huggingface.co/allenai/Olmo-3-1025-7B) base/instruction
  families.
- Official Moonshot results include [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5),
  Kimi-K2.6/K2.7, [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3), and
  [Kimi Linear](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base).
- Official Z.ai results include
  [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) and
  [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
- The literal search `Kimi 3.5 Flash` returned no result. It must be clarified
  before committing an implementation milestone; it may combine Qwen3.5, Kimi,
  and/or GLM Flash names.
- A tool-calling dataset search returned many incompatible schemas despite
  similar tags, reinforcing the adapter/fixture approach rather than dataset-ID
  conditionals in the trainer.

Catalog presence is not a compatibility promise. Before adding a model or
dataset, pin an immutable revision, inspect its license/card/config/files, and
record any required custom code or gated access.

## 7. Deliberate non-dependencies

JAXSFT should study mature systems such as MaxText/Tunix, TRL, Axolotl, and
LLaMA-Factory for recipes and edge cases, but the initial implementation should
not wrap one of them wholesale. The repository's purpose is to make JAX model
math, semantic data transforms, token objectives, and slice behavior unusually
visible. Dependencies are adopted when they preserve that boundary and reduce
undifferentiated work, not merely because they expose a training API.
