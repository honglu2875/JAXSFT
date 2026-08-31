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
plan may mark `executable_kernel_proven: true`, and the G4 proof marks
`direct_loader_proven: true`. G5c2 now proves a complete frozen sharded
one-token forward. A `runnable: true` preflight therefore means only that the
next bounded G6 experiment is permitted; it does not mean that long-sequence
SFT has passed its memory or throughput gates.

The source files are serialization shards, not mesh/FSDP shards. Assigning one
file to one host is therefore incorrect: a final tensor partition can require
slices from most source files. The loader must either:

1. let every host range-read only the byte ranges for its local tensor slices;
2. have one owner read each tensor and scatter final slices to all hosts; or
3. perform a bounded one-file-at-a-time read per host, accepting duplicated
   network traffic as an initial correctness baseline.

The raw-tensor staging bound is one final device slice at a time, not one
source file and not the entire ideal quarter-checkpoint. The largest raw range
is the 79,298,560-byte BF16 LM-head slice. Executable expert packing raises the
actual peak device staging buffer to 150,994,944 bytes because 288 independently
named expert slices must form one local array. Loading every text tensor would
stream 80,128,653,560 bytes (74.626 GiB) per host under the initial policy;
large payloads are downloaded once across the slice, while small replicated
tensors and scale grids are intentionally read by each host.

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
| G5a executable tensor schema | passed | commit `5653518`; [`glm53_execution_schema_audit.json`](../results/glm53_execution_schema_audit.json) |
| G5b official-size expert kernel | passed | commit `d2eb6c1`; [`glm53_expert_fp8_v4_probe.json`](../results/glm53_expert_fp8_v4_probe.json) |
| G5c1 real checkpoint expert streaming | passed | commit `3869a9b`; [`glm53_real_expert_streaming_v4.json`](../results/glm53_real_expert_streaming_v4.json) |
| G5c2 full frozen forward | passed | run commit `da5c6a7`; [`glm53_full_forward_v4.json`](../results/glm53_full_forward_v4.json) |
| G6a bounded expert input gradient | passed | run commit `ed28e50`; [`glm53_bounded_expert_v4.json`](../results/glm53_bounded_expert_v4.json) |
| G6b0 full-model attention-LoRA backward compile | passed | run commit `56cd8b7`; [`glm53_lora_backward_compile_v4.json`](../results/glm53_lora_backward_compile_v4.json) |
| G6b1 full-checkpoint attention-LoRA backward execution | passed | run commit `ccd5416`; [`glm53_lora_backward_v4.json`](../results/glm53_lora_backward_v4.json) |
| G6c0 full-model adapter-only AdamW compile | passed | run commit `088c24a`; [`glm53_lora_optimizer_compile_v4.json`](../results/glm53_lora_optimizer_compile_v4.json) |
| G6c1 loss/update/checkpoint, 3 steps | passed | run commit `543350c`; [`glm53_lora_three_step_v4.json`](../results/glm53_lora_three_step_v4.json) |
| G6d total-10-step resume stability | passed | run commit `d47e00b`; [`glm53_lora_ten_step_v4.json`](../results/glm53_lora_ten_step_v4.json) |

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
  --execution-schema-evidence docs/results/glm53_execution_schema_audit.json \
  --expert-kernel-evidence docs/results/glm53_expert_fp8_v4_probe.json \
  --full-forward-evidence docs/results/glm53_full_forward_v4.json \
  --bounded-expert-evidence docs/results/glm53_bounded_expert_v4.json \
  --rank 8
```

Expected result with G5c2 evidence: `static_fit: true`, `runnable: true` for
the next bounded G6 probe only. The planner uses the measured
33,014,407,168-byte device limit rather than nominal 32 GiB; its deliberately
conservative 8 GiB activation and 2 GiB runtime reserves leave about 1.89 GiB
per chip. `bfloat16` remains `static_fit: false`.

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

### G5a — Executable tensor schema

- Derive source names and shapes from the architecture rather than trusting a
  permissive state-dict walk.
- Map every logical tensor and scale grid exactly once into a final executable
  target, including 288-way expert packs.
- Run the complete reduced hybrid model with source-oriented FP8 wrappers and
  compare logits, loss, and LoRA gradients with independently dequantized dense
  parameters.

Measured result: all 37,534 logical tensors and 36,467 scale grids map exactly
once to 1,372 executable targets. Of these, 305 are block-FP8 targets. The
36,288 expert matrices form exactly 126 packs: gate, up, and down for each of
42 sparse MLP layers. The full reduced quantized forward and adapter gradients
match the dequantized dense path within the test tolerance. G5a does not load
the full checkpoint or measure TPU memory.

### G5b/G5c — Expert kernel and full frozen forward

- Compile one official-size 288-expert layer first and reject a persistent BF16
  expansion of the full expert bank or an unbounded selected-weight temporary.
- Only after that probe passes, load the text-only base without adapters and
  execute a one-token frozen forward before increasing length.
- Measure compiled executable size, per-chip HBM, dequant workspace, collective
  buffers, throughput, and numerical comparison on fixed public prompts.
- Replace the placeholder 8 GiB activation and 2 GiB runtime/dequant reserves
  with measured upper bounds plus explicit safety margin.

Measured G5b result: one 288-expert gate/up/down bank contains 7,247,757,312
FP8 bytes globally and 452,984,832 bytes per chip. A one-token top-8 execution
reported 455,361,024 compiler argument bytes and 75,884,544 temporary bytes per
chip. Peak device bytes in use were 457,929,216; optimized local HLO retained
the sharded uint8 banks, contained no full local BF16/F32 expert-bank shape,
and materialized only selected BF16 shards. All four processes had identical
HLO and output hashes, zero `/dev/shm` delta, and clean shutdowns. This passes
G5b, but the selected-weight temporary grows with `tokens * top_k`; it is not
the long-sequence G6 dispatch design.

Measured G5c1 result: the loader range-read the real layer-3 gate, up, and down
expert banks from two pinned safetensors files. Each host fetched its disjoint
1,811,939,328-byte quarter, so the 7,247,757,312-byte source payload was
downloaded exactly once across the slice. Each chip held 455,357,952 bytes
after placement. Four ranks produced identical raw-bit/scale fingerprints,
HLO, and output statistics despite Cloud TPU process order differing from the
hostname suffix order. Independently range-read Transformers 5.16.1/PyTorch
2.10 CPU tensors matched all six selected source fingerprints exactly. Its
BF16 statistic comparison was 1.583% relative L2 and `1.54e-5` maximum
absolute error under explicit 2%/`2e-5` cross-backend bounds; most of the
relative error came from the near-zero logits sum. This validates real source
assembly and one real expert contraction, not the complete model.

Measured G5c2 result: all 1,372 executable targets (37,534 logical tensors and
36,467 scale tensors) were streamed from the pinned 62-shard checkpoint onto
the 16-chip mesh. Each host fetched 80,141,139,062 bytes through bounded HTTP
ranges; the largest request was 79,298,560 bytes, maximum expert staging was
603,979,776 bytes, maximum process HWM was 7,912,480,768 bytes, and measured
RAMFS payload growth was zero. Every chip held the header-audited
20,234,287,352-byte base.

The compiled forward reported 20,262,202,880 argument bytes and 225,031,168
temporary bytes per chip. Maximum observed device use was 20,303,898,624 of
33,014,407,168 bytes, leaving 12,710,508,544 bytes of raw headroom and a
12,558,333,440-byte largest free block. All ranks produced the same optimized
HLO and output hashes; two executions on every rank were bitwise identical and
finite. Maximum compilation time was 85.7 seconds. The steady one-token
forward still took 123.6 seconds, so this implementation is correctness and
capacity evidence, not a usable training kernel. The current selected-expert
temporary and reference execution must be replaced by capacity-bounded,
throughput-oriented dispatch before increasing token count.

### G6 — Bounded LoRA SFT

- Start with rank 4/8, batch 1, short sequences, full rematerialization, and
  attention-only targets.
- First compile a capacity-bounded dispatch/backward probe without the full
  checkpoint; reject any temporary proportional to all experts or unbounded
  `tokens * top_k` gathered weights.
- Run 3 steps, restore from an adapter-only checkpoint, then run 10--50 steps if
  loss/gradient/HBM traces are stable.
- Compare the reduced configuration trajectory against canonical
  Transformers/Accelerate. For the full quantized model, track finite loss,
  gradient norm, router statistics, memory slope, and post-update drift; exact
  backend parity is not assumed.

Measured G6a result: the correctness-first bounded primitive processes a
static one-matrix selected-weight chunk inside `lax.map` and rematerializes the
chunk for input-gradient computation. One official 288-expert gate/up/down
bank was held in its final 16-way sharding while compiling forward plus input
gradient at one token (8 assignments) and four tokens (32 assignments).
Compiler temporary memory was 774,144 and 741,888 bytes per chip respectively:
it did not grow with four times as many routed assignments. Optimized HLO
contained the local one-matrix chunk shapes and no assignment-wide dense
weight shape. Maximum device use was 460,946,432 bytes, maximum process HWM
was 6,720,667,648 bytes, and RAMFS growth was zero. Four-token forward plus
input gradient completed in at most 0.133 seconds across ranks. This passes the
expert input-gradient primitive only; it does not yet measure a complete-model
adapter gradient or optimizer update.

Measured G6b0 result: all four v4-32 ranks reconstructed the exact sharded
parameter signature from the 62 pinned headers and compiled a rank-4,
batch-1, sequence-2 weighted causal loss plus gradients for all 191 attention
LoRA targets. No checkpoint tensor or scale payload was read. LoRA `A` is
replicated and `B` is output-sharded; the 10,289,152 trainable parameters use
8,705,024 bytes per chip before gradients or optimizer state.

The executable reports 20,270,928,384 argument bytes, 8,718,336 output bytes,
and 1,248,113,152 temporary bytes per chip. Their conservative sum is
21,527,759,872 bytes, leaving 11,486,647,296 bytes against the measured HBM
limit before the explicit 1 GiB safety margin. All ranks emitted the same
optimized HLO hash. The HLO contains the bounded one-matrix local expert
shapes and no flattened or token/top-k assignment-wide dense expert weight.
This passes the compile-only G6b0 capacity gate and authorizes one smallest
full-checkpoint backward execution. It does not prove finite loss/gradients,
an optimizer update, or multi-step training.

Measured G6b1 result: the complete pinned checkpoint executed that rank-4
attention-LoRA backward on all four hosts. The weighted one-token loss was
`12.241244316101074` on every rank. All gradients were finite. Canonical
zero-`B` initialization produced exactly zero `A` gradients, while `B` had
4,821,488 nonzero gradient elements across 169 leaves with squared L2 norm
`7.319513320922852`. Loss and diagnostic hashes matched across ranks.

The maximum backward time was 248.7 seconds. Peak HBM was 20,512,925,696 of
33,014,407,168 bytes per chip, leaving 12,501,481,472 bytes of raw headroom;
the smallest observed post-execution free block was 11,769,517,568 bytes.
Each host streamed 80,141,139,062 checkpoint bytes with zero RAMFS payload
growth. Optimized HLO remained capacity-bounded and all distributed shutdowns
completed. This passes G6b1 and authorizes an adapter-only AdamW compile gate;
it does not yet prove an optimizer update, checkpoint restore, or multi-step
stability.

Measured G6c0 result: all ranks compiled the complete rank-4 attention-LoRA
loss/backward, global-norm clipping, BF16 adapter update, and two FP32 AdamW
moment slots using checkpoint headers only. The adapter uses 8,705,024 bytes
per chip and optimizer state uses 34,820,100 bytes per chip. XLA reported
43,581,952 donated alias bytes, but the authorization calculation
conservatively did not subtract them.

The executable reports 20,305,797,120 argument bytes, 43,587,584 output bytes,
and 1,331,296,768 temporary bytes per chip. Their conservative sum is
21,680,681,472 bytes, leaving 11,333,725,696 bytes against measured HBM before
the 1 GiB reserve. Every rank emitted the same HLO hash with bounded expert
shapes, read zero checkpoint payload bytes, and shut down cleanly. This passes
G6c0 and authorizes one three-step full-checkpoint execution with an
adapter-only save/restore boundary; it does not prove that execution yet.

Measured G6c1 result: the complete checkpoint executed three donated AdamW
steps with BF16 adapters and two FP32 moment slots. All ranks produced the same
losses (`12.2412443`, `12.2346916`, `12.2660503`), pre-clipping gradient norms
(`2.70546`, `2.76051`, `2.75833`), and state hashes. The loss range was 0.256%
of step-1 loss and step 3 was 0.203% above step 1, so the finite trajectory is
explicitly non-monotonic and is not an optimization-quality result. Each step
took at most 248.64 seconds.

After step 2, every host atomically wrote only the adapter and Adam state:
1,147 leaves, 2,866 unique local shards, and 55,398,404 raw tensor bytes. The
frozen base was excluded. Independent controller-side validation rehashed all
11,464 NPZ members, proved exact coverage of 9,742 global shards and
102,891,524 logical bytes, and reproduced global payload hash
`30f9cf81f7162b46157e5d7a5a6d18755464d0709972e1fd1fdb95ec5ccd655a`.
Every local shard and the complete step-2 statistics vector matched after
restore before step 3 executed.

Peak HBM was 20,601,684,480 of 33,014,407,168 bytes per chip, leaving
12,412,722,688 bytes of raw headroom. Peak growth from step 1 to step 3 was
only 86,016 bytes, the smallest free block was 11,615,084,032 bytes, and
RAMFS payload growth remained zero. This passes G6c1 and authorizes a single
resume-from-step-2 run through total step 10. It does not authorize 50 steps,
long sequences, or an instruction-dataset claim.

Measured G6d result: all four ranks restored the pinned step-2 adapter and Adam
shards before loading the frozen base, then reproduced step 3 exactly and ran
through total step 10. Losses remained finite but non-monotonic, ranging from
`12.1497936` to `12.2660503`; step-10 loss was `12.2053318`, 0.293% below
step 1. Every pre-clipping gradient norm was finite and above the 1.0 clipping
threshold, and all ranks produced identical per-step state hashes.

Peak HBM was 20,599,073,792 of 33,014,407,168 bytes per chip, leaving
12,415,333,376 bytes of raw headroom. The cumulative high-water mark did not
increase from step 3 through step 10, the smallest free block was
11,617,694,720 bytes, and RAMFS payload growth remained zero. Each step took at
most 248.76 seconds.

After step 10, every host again saved only adapters and Adam state. Independent
controller validation rehashed all 11,464 NPZ members, covered 9,742 global
shards and 102,891,524 logical bytes, and reproduced global payload hash
`bec4348e8a18d803ef0455867915bd2bc49ea64e834ebebb295472c8a108e8c2`.
The complete state statistics matched after byte-exact restore. G6d therefore
authorizes an optional total-50-step fixed-token resume probe. It does not
authorize a realistic instruction sequence: tokenizer/loss-mask parity and a
sequence-length-specific header-only compile/HBM gate remain required first.

## Stop conditions

Stop before downloading the full checkpoint if any of these remains true:

- a block-FP8 contraction cannot execute without persistent BF16 expansion;
- measured base plus conservative training reserves exceeds the reported
  33,014,407,168-byte device limit;
- final tensor shardings require a full tensor or model replica on any host;
- the loader cannot prove complete key/scale coverage with bounded memory;
- reduced float32 forward/backward parity fails or divergence grows across the
  short trajectory;
- adapter-only optimizer/checkpoint state accidentally includes frozen base
  leaves.

Passing G0 is permission to investigate G1/G2, not permission to fetch 305.8
GiB of weights. G3 and G4 justified the bounded G5 load, and G5c2 now permits
work on G6. It does not permit a long-sequence training run: G6 must first
bound expert dispatch, then measure adapter-only backward and optimizer HBM at
the smallest token count before increasing sequence length or step count.
