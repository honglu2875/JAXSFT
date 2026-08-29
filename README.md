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
- Loss-aware truncation that maximizes retained configured objective weight,
  reserves an explicit prefix-context budget, and records every dropped input
  token, selected token, and unit of objective weight.
- One visible `train_sft.py` loop with gradient accumulation, rematerialization,
  clipping, AdamW, cosine scheduling, rank-disjoint streaming data, structured
  artifacts, replicated data parallelism, and strict checkpoint/resume.
- Controller-only `cluster.py doctor|sync|run|status|stop|collect`, immutable
  source capsules, per-run remote directories, and exact recorded-PID teardown.
- CPU tests, forced four-device CPU smoke, optional Transformers/PyTorch parity,
  and a frozen `uv.lock`.

The validation gates pass for 42 offline unit tests; byte/token template parity;
and tiny-model valid-logit/loss/gradient/AdamW-step parity against Transformers
and PyTorch for both architectures. Qwen has an exact 320-tensor,
752,393,024-parameter public-checkpoint audit. A measured single-host v4-8 run
completed five live Qwen UltraChat updates across four TPU devices with finite
loss and gradients. The first backward compile took about 101 seconds; a
measured steady step took 0.403 seconds (2,538 input tokens/s). See the
[sanitized result](docs/results/qwen35_v4_8_smoke.json). The original four-host
slice was pre-empted, so multi-host initialization and checkpoint portability
remain unproven.

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

python3.12 cluster.py doctor \
  --profile configs/clusters/four-host-tpu.local.toml
python3.12 cluster.py sync \
  --profile configs/clusters/four-host-tpu.local.toml
python3.12 cluster.py run \
  --profile configs/clusters/four-host-tpu.local.toml \
  --recipe configs/recipes/qwen35_0_8b_ultrachat_loss_aware_smoke.yaml
python3.12 cluster.py status \
  --profile configs/clusters/four-host-tpu.local.toml
```

All workers install the frozen environment and cache Hub/JAX state under the
profile's dedicated run/cache roots. A worker without `uv` receives the
controller's exact executable at a SHA-addressed cache path; nothing is
installed into system Python. The launcher never changes SSH keys, modifies
host-wide packages, overwrites an existing run directory, or kills by process
name. Add `--synthetic` to `cluster.py run` for a tiny architecture/topology
smoke before resolving public weights.

## Repository map

```text
train_sft.py                         canonical SFT experiment loop
cluster.py                           controller-side SSH lifecycle
configs/recipes/                     strict, immutable experiments
configs/clusters/                    public profile examples
src/jaxsft/data/                     IR, adapters, renderer, aligner, stream
src/jaxsft/models/qwen3_5.py         complete dense Qwen3.5 text model
src/jaxsft/models/olmo2.py           complete dense OLMo 2 text model
src/jaxsft/models/registry.py        explicit trainer dispatch only
src/jaxsft/loss.py                   additive weighted causal objective
src/jaxsft/optim.py                  visible AdamW and schedule
tests/unit/                          offline CPU contracts
tests/parity/                        Hugging Face numerical/token oracles
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
kernel and replicated parameters. Single-process checkpoints atomically capture
the model, optimizer, step, RNG cursor, and a deterministically replayable data
cursor; the loader verifies a completion marker, SHA-256, recipe identity, and
exact source/topology identity before unpickling trusted local state. Chunkwise
TPU kernels,
model-axis sharding, packing, portable multi-host checkpoints, LoRA, Qwen MoE,
and Kimi variants remain roadmap items. See [PLAN.md](PLAN.md) for their
exit gates and [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) for the data/loss
invariants.

## References and license

The orchestration design is inspired by
[RIG](https://github.com/honglu2875/rig), while the readable model/parity style
and Qwen3.5 math draw on [JAXML](https://github.com/honglu2875/jaxml) and the
official Hugging Face Transformers implementation. Specific retained and
changed ideas are recorded in [docs/REFERENCES.md](docs/REFERENCES.md).

JAXSFT is licensed under Apache-2.0; third-party attribution is in [NOTICE](NOTICE).
