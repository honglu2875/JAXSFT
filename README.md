# JAXSFT

JAXSFT is a metadata-first research rig for supervised fine-tuning open-weight
models in JAX. It keeps message, reasoning, tool-call, tool-result, and template
ownership attached to tokens so loss policies can be changed without rewriting
the dataset or trainer.

The executable baselines target text-only `Qwen/Qwen3.5-0.8B-Base` and
`allenai/OLMo-2-0425-1B` on the pinned `HuggingFaceH4/ultrachat_200k`
`train_sft` split. Every Hub reference and model template choice is immutable
in its checked-in recipe or renderer identity.

## What works now

- Pure-JAX dense Qwen3.5 in one model file: Gated DeltaNet, gated full
  attention, dense SwiGLU, direct safetensors mapping, strict shape checks, and
  a tied language-model head.
- Pure-JAX OLMo 2 in one independent model file: grouped-query attention, full
  RoPE, q/k and post-sublayer RMS norms, SwiGLU, an untied head, and strict
  179-tensor public-checkpoint conversion.
- Standard messages, prompt/completion, ShareGPT, UltraChat, OpenAI tool calls,
  Anthropic-style typed blocks, reasoning, and tool-result adapters into one
  immutable conversation IR.
- Byte-exact Qwen3.5 and OLMo 2 Instruct rendering plus whole-document tokenizer
  offset alignment. Qwen's multi-turn reasoning/tool fixture and OLMo's plain
  conversation fixture match Hugging Face rendered text and token IDs exactly;
  unsupported OLMo tool/reasoning semantics are rejected.
- Ordered role/part/span/tool loss rules, fractional weights, explicit causal
  target shifting, and globally additive numerator/denominator reduction.
- Token-window and semantic loss-aware truncation that maximize retained
  configured objective weight and reserve an explicit prefix-context budget.
  The semantic policy keeps complete messages and tool-call/result/final-answer
  chains atomic while recording dropped messages, tokens, and objective weight.
- One visible `train_sft.py` loop with gradient accumulation, rematerialization,
  clipping, AdamW, cosine scheduling, rank-disjoint streaming or materialized
  Hub data, structured artifacts, replicated data parallelism, and strict
  checkpoint/resume.
- Content-addressed, framework-neutral batch tapes for replaying identical
  token IDs, masks, and metadata-derived loss weights through JAX and an
  independently parsed Hugging Face Trainer/Accelerate CPU oracle.
- Controller-only `cluster.py doctor|sync|run|status|stop|collect`, immutable
  source capsules, per-run remote directories, conservative stale-libtpu-lock
  recovery, and exact recorded-PID teardown.
- CPU tests, forced four-device CPU smoke, optional Transformers/PyTorch parity,
  and a frozen `uv.lock`.

The offline CPU suite, byte/token template fixtures, and tiny-model
valid-logit/loss/gradient/AdamW-step parity against Transformers and PyTorch
pass for both architectures. Qwen has an exact 320-tensor,
752,393,024-parameter public-checkpoint audit. A measured single-host v4-8 run
completed five live Qwen UltraChat updates across four TPU devices with finite
loss and gradients. The first backward compile took about 101 seconds; a
measured steady step took 0.403 seconds (2,538 input tokens/s). See the
[sanitized result](docs/results/qwen35_v4_8_smoke.json).

Checkpoint/resume was subsequently tested on the same v4-8. A synthetic
interrupted run produced a byte-identical final checkpoint to its uninterrupted
reference. A live-data Qwen3.5 checkpoint was 7,523,970,416 bytes; a fresh
process verified its SHA-256, restored without reconstructing weights from
safetensors, replayed the UltraChat cursor, and matched every step-3/4/5 metric
from a same-source uninterrupted run. See the
[resume result](docs/results/qwen35_v4_8_resume_smoke.json).

The 32-token-context loss-aware recipe was also run against the same pinned
stream. It emitted 20 samples after 22 rows with zero zero-objective drops,
versus 28 rows and six zero-objective drops under right truncation; all 19
truncated emitted samples met the context constraint. See the
[loss-aware result](docs/results/qwen35_v4_8_loss_aware_smoke.json).

OLMo 2 then exercised the same trainer without a Qwen-specific branch. Its
1,484,916,736-parameter base checkpoint passed a complete 179-tensor audit and
an all-100,352-logit float32 comparison against Transformers (`rtol=atol=2e-4`,
maximum absolute error 0.0007782). Three real loss-aware UltraChat updates fit
the same v4-8 with finite metrics, and a tiny cold-resume checkpoint was
byte-identical to an uninterrupted reference. See the
[model card](docs/models/olmo2.md) and
[sanitized result](docs/results/olmo2_1b_v4_8_smoke.json).

A four-host v4-32 acceptance run on 2026-08-30 then initialized four JAX
processes and 16 global TPU devices. Each runtime rank consumed a distinct
first batch while every globally reduced metric matched exactly across ranks.
The materialized, pinned UltraChat split drove three full-model updates for
both architectures:

- OLMo 2 1B moved from loss 1.93071 to 1.64470. Its first compiled update took
  54.45 seconds; subsequent updates took 0.14–0.23 seconds per host.
- Qwen3.5 0.8B moved from loss 1.86463 to 1.62334. Its first compiled update
  took 97.71 seconds; subsequent updates took 0.42–0.52 seconds per host.

Before either real-model launch, a synthetic run stopped at step two and wrote
four schema-v4 rank-local checkpoints. A fresh four-process job restored the
runtime-rank files and completed step three. Its model/optimizer state SHA-256
exactly matched a same-source uninterrupted run. This proves restart for the
current replicated data-parallel topology; it is not yet a model-axis-sharded
portable checkpoint format. The hostname-free evidence is in the
[v4-32 acceptance record](docs/results/v4_32_multihost_acceptance.json).

The next data/objective gate adds `semantic_loss_aware`: candidate windows must
align to complete messages, while linked tool-call/result/chained-call/final
answer transactions remain indivisible and retain any tool-definition preamble.
A three-step OLMo 2 run against a materialized copy of the same pinned UltraChat
split completed with 12/12 emitted samples semantically truncated, zero
zero-objective samples, explicit rejection/relaxation counters, and a clean
worker exit. See the
[semantic result](docs/results/olmo2_1b_v4_8_semantic_smoke.json).
With the frozen `datasets==4.8.5`/`huggingface-hub==1.29.0` stack, direct remote
parquet streaming completes training artifacts but can linger in PyArrow HTTP
finalization. It is therefore an experimental loading mode; use
`loading_mode: materialized` for a clean validated worker lifecycle.

A content-addressed 20-step tape then drove the same OLMo 2 batches through
production BF16 JAX, forced-FP32 JAX, and an independent FP32 CPU oracle using
stock Transformers Trainer, Accelerate, PyTorch AdamW, and the stock cosine
scheduler. BF16-to-CPU relative loss error showed no sustained half-to-half
widening (5.61% early-half mean, 5.28% late-half mean, 4.05% final). The FP32
control reduced maximum relative loss error to 0.0214%. Both JAX lanes record
global `highest` matmul precision and explicit `lax.Precision.HIGHEST`; the
FP32 compiler log also showed FP32 contractions with highest/highest operands.
See the [hostname-free trajectory evidence](docs/results/olmo2_1b_trajectory_parity_20.json).

## Quick start

Python 3.12 and `uv` are required.

```bash
uv sync --frozen --dev
make check

# Tiny hybrid Qwen3.5, synthetic data, CPU only.
make smoke

# Deterministic interruption/resume smoke; --stop-after-step is absolute.
uv run python train_sft.py \
  --config configs/recipes/qwen35_tiny_resume_smoke.yaml \
  --synthetic --stop-after-step 2
uv run python train_sft.py \
  --config configs/recipes/qwen35_tiny_resume_smoke.yaml \
  --synthetic \
  --resume artifacts/qwen35-tiny-resume-smoke/checkpoints/step-00000002.pkl

# Inspect the pinned, 32-token-context loss-aware recipe without downloading weights.
uv run python train_sft.py \
  --config configs/recipes/qwen35_0_8b_ultrachat_loss_aware_smoke.yaml \
  --dry-run

# Exercise the same trainer with the tiny OLMo 2 architecture, offline.
JAX_PLATFORMS=cpu uv run python train_sft.py \
  --config configs/recipes/olmo2_1b_ultrachat_loss_aware_smoke.yaml \
  --synthetic --synthetic-length 32

# Inspect the complete-message semantic policy and materialized-data mode.
uv run python train_sft.py \
  --config configs/recipes/olmo2_1b_ultrachat_semantic_smoke.yaml \
  --dry-run
```

Inspect one dataset row all the way through span ownership and token weights:

```bash
uv run jaxsft data explain \
  --row row.json \
  --adapter messages \
  --tokenizer /path/to/pinned/qwen3.5-snapshot \
  --recipe configs/recipes/qwen35_0_8b_ultrachat_loss_aware_smoke.yaml
```

## Four-host launch

Copy the example profile to an ignored local file and replace hostnames with
resolvable names or IPs when necessary:

```bash
cp configs/clusters/four-host-tpu.example.toml \
   configs/clusters/four-host-tpu.local.toml

uv run --no-project --python 3.12 python cluster.py doctor \
  --profile configs/clusters/four-host-tpu.local.toml
uv run --no-project --python 3.12 python cluster.py sync \
  --profile configs/clusters/four-host-tpu.local.toml
uv run --no-project --python 3.12 python cluster.py run \
  --profile configs/clusters/four-host-tpu.local.toml \
  --recipe configs/recipes/qwen35_0_8b_ultrachat_v4_32_smoke.yaml
uv run --no-project --python 3.12 python cluster.py status \
  --profile configs/clusters/four-host-tpu.local.toml
```

All workers install the frozen environment and cache Hub/JAX state under the
profile's dedicated roots. The public profile keeps run metadata/checkpoints
on persistent storage and puts only reproducible large caches in `/dev/shm`.
For RAM caches, passwordless `sudo` is used narrowly to seal the cache
root-owned between SSH sessions; this prevents `systemd-logind` with
`RemoveIPC=yes` from deleting it. Persistent cache paths do not require this
workaround. Model and materialized-dataset revisions are staged first, then TPU
workers launch with Hub and Datasets offline. A worker without `uv` receives
the controller's exact executable at a SHA-addressed cache path; nothing is
installed into system Python. Before launch, the controller refuses an active
TPU owner and only removes an unowned, regular, zero-byte libtpu lockfile. The
launcher never changes SSH keys, modifies host-wide packages, overwrites an
existing run directory, or kills by process name. Add `--synthetic` to
`cluster.py run` for a tiny architecture/topology smoke before resolving public
weights.

Multi-host interruption and resume use a new immutable destination run. Pass
`--stop-after-step N` to create one checkpoint file per runtime rank, then pass
the original `--resume-run-id` and `--resume-step N` when launching the fresh
run. The controller gives every process the shared step directory; the worker
selects its own `jax.process_index()` file.

For numerical controls, `cluster.py run --force-fp32` sets the strictly parsed
`JAXSFT_FORCE_FP32=1` override and records the recipe dtype, effective dtype,
global matmul setting, and explicit dot precision in every run artifact. It is
an equivalence lane, not the default production-memory configuration.

## Repository map

```text
train_sft.py                         canonical SFT experiment loop
cluster.py                           controller-side SSH lifecycle
configs/recipes/                     strict, immutable experiments
configs/clusters/                    public profile examples
src/jaxsft/data/                     IR, adapters, renderer, aligner, stream
src/jaxsft/batch_tape.py             strict framework-neutral replay batches
src/jaxsft/models/qwen3_5.py         complete dense Qwen3.5 text model
src/jaxsft/models/olmo2.py           complete dense OLMo 2 text model
src/jaxsft/models/registry.py        explicit trainer dispatch only
src/jaxsft/loss.py                   additive weighted causal objective
src/jaxsft/optim.py                  visible AdamW and schedule
tests/unit/                          offline CPU contracts
tests/parity/                        Hugging Face numerical/token oracles
scripts/run_hf_trajectory.py         independent stock Trainer CPU trajectory
scripts/compare_trajectories.py      loss-error and widening/stability gate
scripts/stage_recipe.py              pinned model/data preflight materialization
docs/MODELS.md                       evidence-based compatibility table
docs/                                architecture contracts, cards, results, roadmap
```

## Research contract

Dataset syntax and model-template syntax are separate. A row becomes typed
messages and parts, then ordered rendered spans, then one complete tokenization
with per-token metadata and weights. The loss never guesses assistant tokens
from token IDs. Every selected target token is aligned beside the token being
predicted; global loss is `sum(weight * nll) / sum(weight)` across devices.

The initial run deliberately uses a readable recurrent DeltaNet reference
kernel and replicated parameters. Schema-v4 checkpoints atomically capture the
model, optimizer, step, RNG cursor, and a deterministically replayable
rank-local data cursor. Each process writes a rank file and completion marker;
all ranks first prove an identical semantic model/optimizer hash. The loader
verifies file and state SHA-256 values plus recipe, source, topology, rank, and
cursor identities before unpickling trusted local state. Chunkwise TPU kernels,
model-axis sharding and its portable sharded checkpoint format, packing, LoRA,
Qwen MoE, and Kimi variants remain roadmap items. See [PLAN.md](PLAN.md) for their
exit gates and [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) for the data/loss
invariants.

## References and license

The orchestration design is inspired by
[RIG](https://github.com/honglu2875/rig), while the readable model/parity style
and Qwen3.5 math draw on [JAXML](https://github.com/honglu2875/jaxml) and the
official Hugging Face Transformers implementation. Specific retained and
changed ideas are recorded in [docs/REFERENCES.md](docs/REFERENCES.md).

JAXSFT is licensed under Apache-2.0; third-party attribution is in [NOTICE](NOTICE).
