# GLM-5.3-Flash LoRA on v4-32

Status: experimental, fail-closed, not registered with the trainer.

Branch: `exp/glm53-flash-lora`

## Scope and pinned source

The first target is text-only supervised fine-tuning of
`zai-org/GLM-5.3-Flash` at immutable revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. The experiment excludes vision,
the next-token-prediction (MTP) layer, inference caches, expert adapters, and
full-parameter updates.

Pinned metadata:

| Item | Value |
|---|---:|
| Logical parameters | 321,323,031,390 |
| E4M3 parameters | 314,396,639,232 |
| BF16 parameters | 6,926,096,640 |
| F32 parameters | 295,518 |
| Safetensors payload metadata | 328,326,771,576 bytes (305.778 GiB) |
| Safetensors tensors / files | 76,108 / 62 |
| `config.json` SHA-256 | `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` |
| index SHA-256 | `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05` |

The model has 45 hybrid text layers: 34 KDA linear-attention layers and 11
sparse-attention layers. Layers 0--2 use dense MLPs; the remaining layers use
288 routed experts with 8 selected per token. The checkpoint is block-scaled
E4M3 with 128x128 scales, while a large explicit allowlist remains BF16/F32.

## Why aggregate capacity is not a feasibility result

A v4-32 has 16 chips with nominal 32 GiB HBM each. Ideal 16-way placement of
the serialized checkpoint is 19.111 GiB per chip, leaving 12.889 GiB before
adapters, activations, executable/runtime state, collectives, and temporary
dequantization buffers. Expanding the floating tensors to BF16 requires
642,646,653,816 bytes (598.511 GiB) before any training state, so a persistent
BF16 copy is rejected.

TPU v4 does not make the checkpoint's FP8 serialization directly executable by
assumption. The only candidate is to retain block-FP8 frozen weights, perform a
scale-aware tiled conversion at each contraction, and demonstrate that XLA
does not materialize or retain a full BF16 copy. Until compiled HBM profiles
prove that property, the plan may report `static_fit: true` but must report
`runnable: false`.

The source files are serialization shards, not mesh/FSDP shards. Assigning one
file to one host is therefore incorrect: a final tensor partition can require
slices from most source files. The loader must either:

1. let every host range-read only the byte ranges for its local tensor slices;
2. have one owner read each tensor and scatter final slices to all hosts; or
3. perform a bounded one-file-at-a-time read per host, accepting duplicated
   network traffic as an initial correctness baseline.

The initial staging bound is one source file (conservatively 5.5 GB) per host,
not the entire 76.4 GiB ideal quarter-checkpoint. Header-only range reads have
already been validated and are roughly 90--175 KiB for sampled shards.

## Initial adapter surface

Rank-8 LoRA initially targets only attention matrix contractions:

- KDA layers: `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- sparse-attention layers: `q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, `kv_b_proj`, and `o_proj`.

That is 20,578,304 trainable parameters. Embeddings, LM head, mHC tensors,
normalization, short convolutions, indexer, dense/shared/expert MLPs, router,
vision tower, and MTP layer remain frozen. This small surface is deliberate: it
separates architecture/loader risk from expert-routing adapter research. The
generic LoRA math must be proven equivalent to a merged dense kernel before it
is used here.

## Staged gates

Each stage produces committed code/tests or a machine-readable evidence file.
A later stage must not be advertised if an earlier gate is red.

Current branch status:

| Gate | Status | Evidence |
|---|---|---|
| G0 metadata/static preflight | passed | commit `3966ee1`; real pinned index/config checked |
| G1 generic LoRA correctness | passed | commit `2ebff99`; separate adapter tree and merge/gradient tests |
| G2 reduced architecture parity | passed | commit `1ca6ccf`; [`glm53_reduced_hybrid_cpu_parity.json`](../results/glm53_reduced_hybrid_cpu_parity.json) |
| G3 block-FP8 primitive on v4 | pending | no executable/HBM evidence yet |
| G4 direct sharded loader | pending | no four-host checksum/RSS evidence yet |
| G5 full frozen forward | pending | blocked by G3/G4 |
| G6 bounded LoRA SFT | pending | blocked by G3--G5 |

### G0 — Metadata and static preflight

- Parse and hash the pinned config and index without downloading weights.
- Reject path traversal, malformed indexes, revision/hash drift, persistent
  BF16 execution, and full-model optimizer state.
- Report base, adapter, activation-reserve, runtime/dequant-reserve, and host
  staging bytes independently.

Run after fetching only `config.json` and `model.safetensors.index.json`:

```bash
PYTHONPATH=src uv run python scripts/plan_glm53_lora.py \
  --config /path/to/config.json \
  --index /path/to/model.safetensors.index.json \
  --rank 8
```

Expected result at this gate: `static_fit: true`, `runnable: false` for
`fp8_blockwise`; `static_fit: false` for `bfloat16`.

### G1 — Generic LoRA correctness

- Keep frozen base and trainable adapter PyTrees separate so Adam state is never
  allocated for the 321B base.
- Test zero-initialized-B identity, adapter gradients, merge equivalence,
  target-path auditing, checkpoint round-trip, and rank/shape validation.
- Keep all tiny equivalence tests in float32 with highest JAX contraction
  precision.

### G2 — Reduced GLM architecture parity

- Implement the text architecture in one model file: mHC, KDA, sparse
  attention/indexer, dense and routed/shared MoE, SwiGLU limit, router math,
  normalization, head, and explicit exclusions.
- Use a tiny dense checkpoint/config first, then a tiny MoE/hybrid config.
- Compare full logits, weighted causal loss, selected adapter gradients, and one
  Adam update against an independent Transformers/PyTorch reference.
- Record drift over 10--50 updates; do not judge parity from one step.

### G3 — Block-FP8 primitive on one v4 chip

- Range-read a small real weight plus its `weight_scale_inv` tensor.
- Verify dequantization against the Transformers reference in float32/BF16.
- Compile the exact tiled contraction on TPU v4 with highest supported JAX
  precision and capture HLO plus peak HBM.
- Fail if XLA retains a full BF16 weight, dequantization error is unexplained,
  or memory scales as source FP8 plus persistent BF16.

### G4 — Direct-to-final-shard loader

- Build final `NamedSharding` before reading payload bytes.
- Read bounded ranges, transpose/reshape/dequant metadata, place local slices,
  and release host buffers before advancing.
- Audit every text tensor exactly once and every excluded vision/MTP tensor by
  named rule; never silently discard scale tensors.
- On four hosts, verify local/global checksums, peak RSS, `/dev/shm` high-water
  mark, bytes downloaded, and that no full host or device replica exists.

### G5 — Full frozen forward and HBM measurement

- Load the text-only base without adapters and execute short-sequence forward
  passes at increasing lengths.
- Measure compiled executable size, per-chip HBM, dequant workspace, collective
  buffers, throughput, and numerical comparison on fixed public prompts.
- Replace the placeholder 8 GiB activation and 2 GiB runtime/dequant reserves
  with measured upper bounds plus explicit safety margin.

### G6 — Bounded LoRA SFT

- Start with rank 4/8, batch 1, short sequences, full rematerialization, and
  attention-only targets.
- Run 3 steps, restore from an adapter-only checkpoint, then run 10--50 steps if
  loss/gradient/HBM traces are stable.
- Compare the reduced configuration trajectory against canonical
  Transformers/Accelerate. For the full quantized model, track finite loss,
  gradient norm, router statistics, memory slope, and post-update drift; exact
  backend parity is not assumed.

## Stop conditions

Stop before downloading the full checkpoint if any of these remains true:

- a block-FP8 contraction cannot execute without persistent BF16 expansion;
- measured base plus conservative training reserves exceeds 32 GiB per chip;
- final tensor shardings require a full tensor or model replica on any host;
- the loader cannot prove complete key/scale coverage with bounded memory;
- reduced float32 forward/backward parity fails or divergence grows across the
  short trajectory;
- adapter-only optimizer/checkpoint state accidentally includes frozen base
  leaves.

Passing G0 is permission to investigate G1/G2, not permission to fetch 305.8
GiB of weights. Passing G3 and the loader dry run is the first point at which a
bounded real-weight download becomes justified.
