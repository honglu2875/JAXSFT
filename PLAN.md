# JAXSFT implementation plan

## Current implementation status — 2026-08-29

The repository spine and first executable vertical slice now exist. Completed:

- JAXSFT naming, Apache-2.0 licensing, packaging, frozen environment, strict
  recipe, structured run artifacts, and controller lifecycle;
- canonical messages/parts/tools, four source adapter families, exact Qwen3.5
  text rendering, whole-sequence offset alignment, selector-based weights, and
  token-window and complete-message/tool-atomic loss-aware truncation with an
  explicit context budget;
- explicit streaming/materialized Hub loading with loading-mode-bound replay
  state and conservative stale-libtpu-lock recovery in the controller;
- a single-file dense Qwen3.5 text model, direct safetensors map, weighted loss,
  AdamW, one-/four-device CPU smoke, deterministic single-process resume, and
  optional Hugging Face parity tests;
- a single-file dense OLMo 2 text model, explicit model/renderer dispatch,
  pinned plain-conversation template, full Hugging Face training-step parity,
  and public-checkpoint all-logit parity;
- pinned `Qwen/Qwen3.5-0.8B-Base` and UltraChat 200k smoke recipe;
- a single-host v4-8 smoke using one JAX process and four local TPU devices.

The Qwen renderer/tokenizer and tiny PyTorch valid-logit/loss/gradient/optimizer
parity gates pass. The pinned public checkpoint's 320 text tensors load with an
exact 752,393,024-parameter audit. A v4-8 completed five live pinned UltraChat
updates with finite loss/gradients; first backward compilation was about 101
seconds and a measured steady step was 0.403 seconds (2,538 input tokens/s).
Synthetic interruption at step 2 followed by a cold resume produced a
byte-identical step-4 checkpoint to the uninterrupted run. A full 7.52 GB
Qwen3.5 model/AdamW checkpoint also survived a fresh-process restore; its
replayed UltraChat steps 3–5 matched the uninterrupted reference exactly. The
loss-aware recipe then emitted 20 samples from 22 rows with no zero-objective
truncation drops, compared with 28 rows and six drops for the right-truncation
smoke. All 19 truncated emitted samples retained the configured 32-token
context budget. The original four-host slice was pre-empted, so multi-host
startup and portable checkpointing remain open. Packing, high-performance
chunkwise DeltaNet, model-axis sharding, and Kimi remain planned work rather
than advertised support.

The next model-breadth gate is also complete: pinned
`allenai/OLMo-2-0425-1B` loaded all 179 tensors (1,484,916,736 parameters), and
its complete 100,352-element final-logit vector matched float32 Transformers at
`rtol=atol=2e-4`. The same loss-aware UltraChat recipe completed three
full-parameter updates on a v4-8 with finite metrics. A tiny OLMo interrupted
run cold-resumed to a byte-identical final checkpoint. OLMo's pinned template
does not represent tools or reasoning, so those samples fail explicitly rather
than losing metadata.

The next data/objective gate is complete as well. `semantic_loss_aware` chooses
only complete-message windows, validates unique call/result links, keeps each
chained tool transaction plus final assistant answer indivisible, and retains a
tool-definition preamble whenever a tool exchange survives. A pinned,
materialized UltraChat run completed three OLMo 2 updates on four local v4
devices, emitted 12 semantically truncated samples with zero zero-objective
drops, recorded three oversized semantic-unit rejections and seven explicit
context relaxations, shut down JAX in 5 ms, and exited normally. A subsequent
synthetic launch proved that the controller can recover an unowned empty
libtpu lock without privileged or broad process cleanup. The hostname-free
record is `docs/results/olmo2_1b_v4_8_semantic_smoke.json`.
Direct remote parquet streaming remains experimental under the frozen data
stack: training/result emission succeeds, but repeated v4-8 checks reproduced a
PyArrow/Hugging Face HTTP finalizer wait after JAX shutdown. Materialized mode
is the validated lifecycle until that upstream boundary is replaced or fixed.

The full-model numerical trajectory gate is now complete for OLMo 2. A strict,
content-addressed tape captured 20 global batches after normal metadata-aware
rendering, semantic truncation, tokenization, and loss selection. An independent
CPU program imports no JAXSFT runtime code and uses stock Transformers Trainer,
Accelerate, PyTorch cross-entropy/AdamW, and the stock cosine scheduler. Against
the same tape, BF16 TPU relative loss error stayed bounded (5.61% early-half
mean, 5.28% late-half mean, 4.05% final), while an env-controlled FP32 JAX lane
reduced maximum relative error to 0.0214%. Both current TPU lanes request global
and explicit highest contraction precision. This is a 20-step stability gate,
not evidence for full-run convergence or four-host equivalence.

## 1. Outcome

Build a small, readable JAX SFT research repository in which a researcher can:

1. point at a pinned Hugging Face base-model revision and a set of instruction
   datasets;
2. adapt heterogeneous rows into a loss-aware canonical representation without
   rewriting the trainer;
3. inspect exactly which semantic spans and target tokens contribute to loss;
4. run the same single entry program on CPU, one accelerator host, or a
   four-host slice;
5. compare a JAX model and training step against a tiny Hugging Face reference;
6. resume from a content-addressed checkpoint with the same data order; and
7. publish enough provenance to reproduce or audit the run.

The repository should optimize for research modification and correctness before
maximal model breadth or peak throughput.

## 2. Definition of done for v0.1

The first release is complete only when all of the following are demonstrated:

- `Qwen/Qwen3.5-0.8B-Base` can complete
  a deterministic SFT smoke run and reduce held-out loss.
- An OLMo base checkpoint can run through the same trainer with only a model
  and recipe change.
- Tiny JAX and Transformers models match configs, parameter mapping, logits,
  weighted loss, gradients, and one optimizer update within declared tolerances.
- A fixture containing system/user/assistant turns, multiple tool calls, tool
  results, and reasoning/final-answer parts produces a human-readable token
  audit and the expected loss weights.
- Packed and unpacked versions of the same selected tokens report the same loss
  numerator and denominator when dropout is disabled.
- An interrupted run resumes with the same next sample IDs, optimizer state,
  RNG state, and loss trajectory as an uninterrupted reference.
- A four-host run initializes exactly four JAX processes, validates the global
  mesh, gives every process a disjoint deterministic input shard, writes a
  restorable checkpoint, and collects rank-specific logs.
- `make check` runs all CPU gates without requiring a Hub token or accelerator;
  network-, weight-, and TPU-dependent tests are explicit opt-in gates.
- Every completed run includes source, config, dependency, model, tokenizer,
  template, dataset, adapter, loss-policy, topology, and checkpoint identities.

## 3. Work streams

### A. Contracts and correctness

- Define typed, versioned contracts for canonical samples, rendered spans,
  tokenized samples, packed sequences, batches, model outputs, checkpoints, and
  run manifests.
- Write golden fixtures before the first real dataset adapter or model loader.
- Make every transform pure where practical: input plus explicit config yields
  output plus diagnostics.
- Add token-audit and sample-explain commands early; they are core research
  tools, not later UI work.

### B. Model implementations

- Keep each architecture in one file with its config normalization, JAX
  forward path, Hugging Face/safetensors mapping, partition rules, and declared
  capabilities.
- Begin with dense Qwen3.5 because a small public checkpoint makes end-to-end
  validation affordable and exercises both recurrent and full attention.
- Add OLMo next to prove that the contract is not Qwen-shaped.
- Add other dense and sparse families only after this baseline is stable:
  OLMo, Qwen3, Qwen MoE, GLM Flash, and Kimi variants.
- Stream safetensors into their final dtype/sharding where possible; do not
  require a second full PyTorch model in host memory outside parity tests.

### C. Data, tokenizer, and objective

- Normalize dataset syntax into typed conversation parts while retaining
  source provenance and otherwise-unmapped metadata.
- Render the canonical sample through a pinned, model-specific template into
  annotated spans.
- Tokenize once and attach token-level target weights plus compact metadata.
- Treat label shifting, BOS/EOS ownership, truncation, and pack boundaries as
  explicit contracts with adversarial tests.
- Support Python adapters first and a narrow declarative field mapper for common
  schemas; avoid a general transformation language.
- Require explicit adapters in reproducible runs. Schema auto-detection may
  suggest an adapter but must not silently select one.

### D. Trainer and recipes

- Use one canonical `train_sft.py`; keep the loss, gradient accumulation,
  optimizer update, metrics reduction, evaluation, and checkpoint calls visible.
- Use strict versioned recipe documents for normal variation. A research fork
  may point the harness at another entry file, whose hash is captured.
- Implement the full-update BF16 baseline first, then packing, mixing,
  activation rematerialization, LoRA, and specialized objectives.
- Precompile with synthetic data, then synchronize before measuring real work.

### E. Slice orchestration and provenance

- Use OpenSSH, `pdsh`, and `rsync`; do not introduce a resident cluster service.
- Probe access and environment before syncing data or launching JAX.
- Capture an immutable source capsule containing Git HEAD, the dirty patch,
  selected untracked files, and hashes. Sync that capsule to a run-specific
  remote directory so concurrent experiments cannot overwrite each other.
- Call `jax.distributed.initialize()` before any device query and derive rank
  identity from JAX, never `-w-N` host suffixes or launcher variables.
- Keep exact run PID files and use staged teardown. Never use broad process-name
  kills.
- Make checkpoint storage an explicit profile field and prove restore before a
  long run. Do not assume local disks are shared.

## 4. Milestones and exit gates

### M0 — Resolve the environment and freeze core choices

Deliverables:

- Read-only cluster audit for the configured private four-host slice: SSH reachability,
  hostnames, OS/architecture, Python/uv, JAX/libtpu, process/local/global device
  counts, accelerator kind, RAM, local disk, `/dev/shm`, and shared/object
  storage availability.
- A short JAX module-system spike comparing pure PyTrees with the current stable
  Flax API for initialization, scanning, rematerialization, and partitioning.
- A checkpoint spike that saves and restores a sharded toy state across all
  four hosts using the intended storage backend.
- Confirmation of project name, license, Python/JAX versions, and the intended
  meaning of “Kimi 3.5 Flash.”

Exit gate:

- One architecture style and one checkpoint/storage strategy are recorded as
  ADRs; the four-host topology is measured rather than inferred from its name.

### M1 — Repository spine and executable contracts

Deliverables:

- Packaging, frozen `uv` environment, `Makefile`, lint/type/test gates, and test
  cadence markers (`critical`, `parity`, `weights`, `tpu`).
- Typed config loader with unknown-key rejection and schema versions.
- Run manifest, atomic local artifact writer, structured logs, and run IDs.
- Canonical sample/span/token/batch types plus golden round-trip fixtures.
- `jaxsft data explain` showing source fields, rendered text, token IDs,
  decoded tokens, span ownership, shifted targets, and loss weights.

Exit gate:

- CPU CI can explain and validate all golden samples without network access.

### M2 — First exact model and single-host SFT

Deliverables:

- `models/qwen3_5.py` as the first single-file architecture.
- Direct safetensors loader and portable checkpoint export/import.
- Transformers parity for tiny dense config: config, weights, logits, loss,
  gradients, and one AdamW update.
- `train_sft.py` with BF16 full fine-tuning, FP32 reductions/optimizer moments,
  clipping, gradient accumulation, evaluation, and deterministic resume.
- Raw-text, prompt/completion, standard messages, and ShareGPT adapters.

Exit gate:

- CPU synthetic smoke, accelerator tiny smoke, and a short Qwen3.5 base-model SFT
  run all pass; a resumed run matches an uninterrupted run.

### M3 — Four-host slice path

Deliverables:

- `jaxsft cluster doctor|sync|run|status|stop|collect` with `--dry-run`.
- Immutable per-run source capsules and run-specific remote workspaces.
- Data-parallel mesh first, then model-axis partitioning required by the target
  checkpoint; host-local input sharding based on `jax.process_index()`.
- Rank-specific logs, global metric reductions, failure propagation, exact-job
  teardown, and periodic artifact salvage.
- Real-slice smoke and checkpoint-restore test on all four nodes.

Exit gate:

- A multi-host smoke run and resume complete with no duplicate examples, no
  cross-rank metric denominator error, and a restorable collected checkpoint.

### M4 — Metadata-first tool and multi-turn data

Deliverables:

- OpenAI/Hugging Face tool-call schema, tool-result, typed content-parts,
  Anthropic-style content-block, and action/observation adapters.
- Model template renderers with exact tokenization parity against pinned
  `apply_chat_template` fixtures.
- Selector-based loss policies over role, part kind, turn index, tool name,
  call ID, source, tags, and arbitrary numeric weights.
- Per-token and per-example/turn normalization; reporting by semantic slice.
- Deterministic truncation policies and block-diagonal sequence packing.
- Dataset mixing, source quotas, deterministic shuffle, quarantine output, and
  content-addressed preprocessing manifests.

Exit gate:

- A mixed multi-turn/tool fixture suite proves exact masks before and after
  truncation/packing, including adjacent tool calls and empty assistant turns.

### M5 — Model breadth

Status: the first item, OLMo 2 dense, has passed tiny parity, pinned public
checkpoint load/forward parity, shared-trainer SFT, 20-step full-model
BF16/FP32 trajectory comparison against a stock CPU oracle, and tiny checkpoint
resume on a measured single-host v4-8. Remaining items stay gated.

Deliverables, one gated model at a time:

1. OLMo dense (prefer a small base checkpoint for integration).
2. Qwen3 MoE.
3. Qwen3.5/3.6 text architectures, including hybrid recurrent/attention state.
4. GLM Flash through a shared MLA/MoE mathematical primitive where equivalent.
5. Kimi text variants (K2/K2.5/K3 or Kimi Linear after the requested target is
   clarified and feasibility is measured).

Each addition requires a model card in the repository that states supported
checkpoint types, training/inference capabilities, unimplemented paths,
memory estimates, reference revision, and parity tolerances.

Exit gate:

- No model is listed as supported until its tiny full-forward parity passes and
  at least one public checkpoint completes a load/forward test. Training support
  additionally requires gradient and checkpoint round trips.

### M6 — Major SFT recipes

Implement in this order, preserving the same batch/objective contracts:

1. assistant-only and completion-only baselines;
2. packed full-parameter SFT;
3. multi-source mixing and source/turn/part weighting;
4. reasoning-hidden/final-only, tool-call-only, and tool-result masking recipes;
5. LoRA with merge/export parity;
6. length-balanced batching and sequence-length curriculum;
7. NEFTune-style embedding noise and label smoothing;
8. long-context SFT with explicit position/reset policy;
9. DoRA and quantized adapters only after numerical/export contracts exist.

Exit gate:

- Every recipe is a strict config plus a focused invariant test and short smoke
  run. Recipe names alone never change implicit preprocessing or normalization.

### M7 — Research release quality

Deliverables:

- Public documentation, contribution rules, security boundaries, license,
  model/dataset compatibility table, and reproducible example reports.
- Offline Hub workflow, cache verification, resumable staging, and exact
  revision pinning.
- Regression dashboard for parity tolerances, compile time, memory, tokens/sec,
  selected-token throughput, and loss by semantic span.
- A clean-room four-host reproduction from a fresh clone.

Exit gate:

- A new contributor can add one dataset dialect and one loss policy without
  editing the trainer, or add one model by editing one model file plus tests.

## 5. Initial recipe set

The first checked-in recipes should answer distinct research questions rather
than enumerate every flag combination:

| Recipe | Objective | What it validates |
|---|---|---|
| `qwen35_full_assistant` | Score all assistant-generated spans. | End-to-end baseline. |
| `qwen35_final_only` | Ignore reasoning/tool calls; score final text. | Part-selective masks. |
| `qwen35_tools_weighted` | Weight tool calls, arguments, and final text separately. | Float token weights and semantic metrics. |
| `qwen35_mixed_sources` | Mix chat, prompt/completion, and tool trajectories. | Adapter isolation and deterministic mixing. |
| `qwen35_lora` | Same data/objective as baseline with LoRA. | Parameter filtering and export. |
| `olmo_full_assistant` | Match the baseline objective on OLMo. | Model independence. |

## 6. Test pyramid

- **Critical CPU:** schemas, adapters, renderers, token fixtures, mask shifting,
  loss normalization, packing, config validation, source capsules, command
  construction, and tiny model smoke.
- **Parity CPU:** tiny Transformers/JAX configs and weights; forward, backward,
  loss, and update equivalence.
- **Weight integration:** pinned small public checkpoints, direct load, forward,
  export/reload, and tokenizer-template snapshots.
- **Local multi-device:** forced CPU devices to test mesh and sharded reductions.
- **TPU single host:** compile, memory, checkpoint, and short training smoke.
- **TPU multi-host:** process discovery, disjoint input, failure handling,
  checkpoint collection, restore, and resume.

Tests that require network, gated weights, credentials, or accelerators must
self-identify and skip with a precise prerequisite. They must never make the
default CPU gate flaky.

## 7. Risk register

| Risk | Mitigation and proving test |
|---|---|
| Chat templates render subtly different whitespace/control tokens. | Pin tokenizer revision and template hash; compare complete token IDs with `apply_chat_template` golden fixtures. |
| A token straddles two differently weighted character spans. | Define ownership explicitly, detect ambiguity, and require template-specific token fixtures at every trainability boundary. |
| Label masks are shifted onto input rather than target positions. | Store target weights by predicted token and test a hand-computed two-token example. |
| Packing creates cross-example attention or targets. | Carry pack segment IDs, block attention, reset positions by policy, and prove packed/unpacked loss invariance. |
| Weighted loss is biased across hosts or accumulation microbatches. | Reduce numerator and denominator separately with `lax.psum`; never average local means. |
| Hub data changes behind a branch name. | Resolve and record immutable dataset/model/tokenizer commits plus file hashes before a run. |
| Streaming shuffle cannot resume exactly. | The baseline pins revisions/dependencies and replays the rank-local epoch prefix from recorded counters; test exact interruption and retain buffer-state checkpointing as the scalable follow-up. |
| Full PyTorch and JAX copies exceed host memory. | Use lazy safetensors conversion and preflight memory estimates; keep Torch construction in tiny parity tests. |
| Large Kimi/GLM/Qwen MoE models do not fit the measured slice. | Separate architecture parity from full-checkpoint training support and publish explicit feasibility status. |
| Multi-host checkpoint files land on non-shared local disks. | Make storage explicit; run save/collect/restore in M0; refuse long runs if restore is unproven. |
| Dirty research code is not reproducible. | Build and hash a source capsule containing HEAD, patch, and selected untracked files for every run. |
| One failed SSH session leaves accelerator workers alive. | Use run-specific PID files, signal escalation, bounded retries only for transport errors, and exact-job status checks. |
| Model-specific exceptions leak into generic code. | Keep capability declarations and exceptions in the model file; forbid imports from one model module into another. |

## 8. Decisions needed before implementation

1. **Exact target names.** No official Hugging Face result matched the literal
   “Kimi 3.5 Flash” on 2026-08-29. Plausible intended targets include Kimi-K3,
   Kimi-K2.5/K2.6, Qwen3.5, GLM-4.7-Flash, or GLM-5.3-Flash.
2. **Measured multi-host topology.** A single v4-8 is measured as one process
   with four local devices. Confirm accelerator generation, per-host device
   counts, aggregate HBM, and coordinator reachability on the next four-host
   slice.
3. **Checkpoint/artifact storage.** Choose a shared mount/object store or approve
   collection of host-local shards after the M0 restore spike.
4. **First optimization target.** Choose full-parameter SFT first (recommended
   for a correctness reference) or require LoRA in the first usable milestone.
5. **Base checkpoints.** SFT recipes should normally start from base models;
   instruct checkpoints may still be used for tokenizer/template and parity
   fixtures.
6. **Framework layer.** Select pure JAX PyTrees or the current stable Flax API
   only after the M0 implementation spike; avoid designing a home-grown module
   system by accident.
7. **Publication choices.** Confirm JAXSFT, Apache-2.0, remote namespace, and
   whether the concrete cluster hostname example belongs in the public tree.
