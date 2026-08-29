# JAXSFT

JAXSFT is a metadata-first research rig for supervised fine-tuning open-weight
models in JAX. It keeps message, reasoning, tool-call, tool-result, and template
ownership attached to tokens so loss policies can be changed without rewriting
the dataset or trainer.

The executable baseline targets text-only `Qwen/Qwen3.5-0.8B-Base` on the
pinned `HuggingFaceH4/ultrachat_200k` `train_sft` split. Both Hub revisions are
immutable in the checked-in recipe.

## What works now

- Pure-JAX dense Qwen3.5 in one model file: Gated DeltaNet, gated full
  attention, dense SwiGLU, direct safetensors mapping, strict shape checks, and
  a tied language-model head.
- Standard messages, prompt/completion, ShareGPT, UltraChat, OpenAI tool calls,
  Anthropic-style typed blocks, reasoning, and tool-result adapters into one
  immutable conversation IR.
- Byte-exact Qwen3.5 rendering and whole-document tokenizer offset alignment.
  A multi-turn reasoning/tool fixture matches Hugging Face rendered text and
  token IDs exactly.
- Ordered role/part/span/tool loss rules, fractional weights, explicit causal
  target shifting, and globally additive numerator/denominator reduction.
- One visible `train_sft.py` loop with gradient accumulation, rematerialization,
  clipping, AdamW, cosine scheduling, rank-disjoint streaming data, structured
  artifacts, and replicated multi-host data parallelism.
- Controller-only `cluster.py doctor|sync|run|status|stop|collect`, immutable
  source capsules, per-run remote directories, and exact recorded-PID teardown.
- CPU tests, forced four-device CPU smoke, optional Transformers/PyTorch parity,
  and a frozen `uv.lock`.

The initial validation gates pass for 25 offline unit tests, byte/token template
parity, tiny-model valid-logit/loss/gradient parity against Transformers, a
320-tensor load of the pinned 752,393,024-parameter public text checkpoint, a
finite compiled public-weight forward pass, and one live pinned UltraChat batch.
The real TPU update remains gated on access to the intended four hosts.

## Quick start

Python 3.12 and `uv` are required.

```bash
uv sync --frozen --dev
make check

# Tiny hybrid Qwen3.5, synthetic data, CPU only.
make smoke

# Inspect the pinned production recipe without downloading weights.
uv run python train_sft.py \
  --config configs/recipes/qwen35_0_8b_ultrachat_smoke.yaml \
  --dry-run
```

Inspect one dataset row all the way through span ownership and token weights:

```bash
uv run jaxsft data explain \
  --row row.json \
  --adapter messages \
  --tokenizer /path/to/pinned/qwen3.5-snapshot \
  --recipe configs/recipes/qwen35_0_8b_ultrachat_smoke.yaml
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
  --recipe configs/recipes/qwen35_0_8b_ultrachat_smoke.yaml
python3.12 cluster.py status \
  --profile configs/clusters/four-host-tpu.local.toml
```

All workers install the frozen environment and cache Hub/JAX state under the
profile's dedicated run/cache roots. The launcher never changes SSH keys,
modifies host-wide packages, overwrites an existing run directory, or kills by
process name.

## Repository map

```text
train_sft.py                         canonical SFT experiment loop
cluster.py                           controller-side SSH lifecycle
configs/recipes/                     strict, immutable experiments
configs/clusters/                    public profile examples
src/jaxsft/data/                     IR, adapters, renderer, aligner, stream
src/jaxsft/models/qwen3_5.py         complete dense Qwen3.5 text model
src/jaxsft/loss.py                   additive weighted causal objective
src/jaxsft/optim.py                  visible AdamW and schedule
tests/unit/                          offline CPU contracts
tests/parity/                        Hugging Face numerical/token oracles
docs/                                architecture contracts and roadmap
```

## Research contract

Dataset syntax and model-template syntax are separate. A row becomes typed
messages and parts, then ordered rendered spans, then one complete tokenization
with per-token metadata and weights. The loss never guesses assistant tokens
from token IDs. Every selected target token is aligned beside the token being
predicted; global loss is `sum(weight * nll) / sum(weight)` across devices.

The initial run deliberately uses a readable recurrent DeltaNet reference
kernel and replicated parameters. Chunkwise TPU kernels, model-axis sharding,
packing, exact streaming resume, portable optimizer checkpoints, LoRA, OLMo,
Qwen MoE, and Kimi variants remain roadmap items. See [PLAN.md](PLAN.md) for
their exit gates and [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) for the
data/loss invariants.

## References and license

The orchestration design is inspired by
[RIG](https://github.com/honglu2875/rig), while the readable model/parity style
and Qwen3.5 math draw on [JAXML](https://github.com/honglu2875/jaxml) and the
official Hugging Face Transformers implementation. Specific retained and
changed ideas are recorded in [docs/REFERENCES.md](docs/REFERENCES.md).

JAXSFT is licensed under Apache-2.0; third-party attribution is in [NOTICE](NOTICE).
