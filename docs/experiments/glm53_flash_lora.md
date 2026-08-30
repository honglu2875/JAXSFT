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
| Logical F32 parameters | 295,518 |
| F32 block-scale metadata | 19,189,248 elements |
| Serialized F32 elements | 19,484,766 |
| Safetensors payload | 328,326,771,576 bytes (305.778 GiB) |
| Safetensors tensors / files | 76,108 / 62 |
| `config.json` SHA-256 | `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` |
| index SHA-256 | `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05` |

The model has 45 hybrid text layers: 34 KDA linear-attention layers and 11
sparse-attention layers. Layers 0--2 use dense MLPs; the remaining layers use
288 routed experts with 8 selected per token. The checkpoint is block-scaled
E4M3 with 128x128 scales, while a large explicit allowlist remains BF16/F32.

## Why aggregate capacity is not a feasibility result

A v4-32 has 16 chips with nominal 32 GiB HBM each. Naively dividing the entire
serialized multimodal checkpoint gives 19.111 GiB per chip. The G4 header
audit provides a more precise text-only placement: 20,234,287,352 bytes
(18.845 GiB) per chip, including replicated small tensors and FP32 scale
grids. Expanding the floating tensors to BF16 requires 642,723,410,808 bytes
(598.583 GiB) before any training state, so a persistent BF16 copy is rejected.

TPU v4 does not make the checkpoint's FP8 serialization directly executable by
assumption. The G3 probe now demonstrates one real 1536x4096 block-FP8
contraction with scale-aware 128x128 conversion and no full BF16/F32 weight in
optimized HLO. This is kernel evidence, not a whole-model capacity result. The
plan may mark `executable_kernel_proven: true`. The G4 proof now marks
`direct_loader_proven: true`, but the plan remains `runnable: false` until the
complete frozen model passes a sharded forward with measured HBM.

The source files are serialization shards, not mesh/FSDP shards. Assigning one
file to one host is therefore incorrect: a final tensor partition can require
slices from most source files. The loader must either:

1. let every host range-read only the byte ranges for its local tensor slices;
2. have one owner read each tensor and scatter final slices to all hosts; or
3. perform a bounded one-file-at-a-time read per host, accepting duplicated
   network traffic as an initial correctness baseline.

The audited staging bound is one final device slice at a time, not one source
file and not the entire ideal quarter-checkpoint. The largest planned range is
the 79,298,560-byte BF16 LM-head slice. Loading every text tensor would stream
80,128,653,560 bytes (74.626 GiB) per host under the initial policy; large
payloads are downloaded once across the slice, while small replicated tensors
and scale grids are intentionally read by each host.

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
| G3 block-FP8 primitive on v4 | passed | commit `2bfda04`; [`glm53_fp8_v4_probe.json`](../results/glm53_fp8_v4_probe.json) |
| G4 direct sharded loader | passed | commit `fb08fcc`; [`glm53_direct_sharded_loader_v4.json`](../results/glm53_direct_sharded_loader_v4.json), [`glm53_checkpoint_header_audit.json`](../results/glm53_checkpoint_header_audit.json) |
| G5 full frozen forward | pending | next gate; whole-model path/HBM unproven |
| G6 bounded LoRA SFT | pending | blocked by G5 |

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
  --kernel-evidence docs/results/glm53_fp8_v4_probe.json \
  --loader-evidence docs/results/glm53_direct_sharded_loader_v4.json \
  --rank 8
```

Expected result after G4: `static_fit: true`, `runnable: false` for
`fp8_blockwise`, with 28.859 GiB provisionally used and 3.141 GiB free per
chip. The sole evidence blocker is the unmeasured full frozen forward.
`bfloat16` remains `static_fit: false`.

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

### G3 — Block-FP8 primitive on v4

- Range-read a small real weight plus its `weight_scale_inv` tensor.
- Verify dequantization against the Transformers reference in float32/BF16.
- Compile the exact tiled contraction on TPU v4 with highest supported JAX
  precision and capture HLO plus peak HBM.
- Fail if XLA retains a full BF16 weight, dequantization error is unexplained,
  or memory scales as source FP8 plus persistent BF16.

Measured result: exact HTTP ranges returned the pinned 6,291,456-byte
`q_a_proj` and 1,536-byte scale grid without downloading the 5.4 GB source
shard. Float32 JAX versus Transformers was `2.89e-7` relative L2. Four v4-32
processes produced the same BF16 output hash; TPU versus CPU BF16 was `1.18e-7`
relative L2. The tiled TPU executable reported 225,792 temporary bytes and no
full-size BF16/F32/FP8 weight in optimized HLO. This passes G3 only.

### G4 — Direct-to-final-shard loader

- Build final `NamedSharding` before reading payload bytes.
- Read bounded ranges, transpose/reshape/dequant metadata, place local slices,
  and release host buffers before advancing.
- Audit every text tensor exactly once and every excluded vision/MTP tensor by
  named rule; never silently discard scale tensors.
- On four hosts, verify local/global checksums, peak RSS, `/dev/shm` high-water
  mark, bytes downloaded, and that no full host or device replica exists.

Measured result: range-reading all 62 headers transferred 10,684,096 bytes and
proved exact coverage of 76,108 tensors. The explicit text scope contains
74,001 tensors; 1,760 MTP and 347 vision tensors are excluded by named rules.
Every one of 37,338 FP8 matrices has an exact F32 128x128 scale grid, and all
text tensors have a bounded 16-way placement.

For the four-host device proof, each local TPU received four contiguous
393,216-byte source ranges directly into final `NamedSharding`. The 16 ranges
cover the real 6,291,456-byte `q_a_proj` exactly once, and all processes
produced the same global fingerprint. Total network traffic was 7,008,800
bytes including four small header/scale reads; maximum process HWM was 4.477
GiB and the measured `/dev/shm` delta was zero. This passes bounded ordinary
RAM staging and direct device placement. It does not claim explicit host-page
pinning or a whole-model load.

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
GiB of weights. G3 and G4 now justify a bounded G5 load, but not an unbounded
download or a training run. G5 must materialize the text-only PyTree one range
at a time, stop on memory pressure, and demonstrate a complete frozen forward
before G6 allocates adapter optimizer state.
