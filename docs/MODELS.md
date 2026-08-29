# Model compatibility

Support is reported by evidence tier rather than by architecture-name presence.

| Architecture | Tiny Transformers parity | Public checkpoint | Shared SFT trainer | Measured topology | Important limits |
|---|---|---|---|---|---|
| Qwen3.5 dense text | Logits, weighted loss, gradients, clipping, AdamW update, tokenizer/template | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe…`; exact 320-tensor/752,393,024-parameter audit | Five real UltraChat updates; deterministic tiny and full-state restore evidence | One v4-8, replicated | Text only; readable recurrent scan is not yet chunkwise; no model-axis sharding |
| OLMo 2 dense text | Logits, weighted loss, gradients, clipping, AdamW update, tokenizer/template | `allenai/OLMo-2-0425-1B` at `a1847d…`; 179 tensors/1,484,916,736 parameters; all-logit public parity | Three real UltraChat updates; byte-identical tiny interrupted/uninterrupted checkpoint | One v4-8, replicated | No cache/export, dropout, non-default RoPE scaling, tools in its pinned template, or model-axis sharding |

“Shared SFT trainer” means the same `train_sft.py` data, objective, optimizer,
checkpoint, and metric path, selected only by the recipe architecture and
renderer. It does not imply that every checkpoint variant in a family is
supported. Exact covered revisions and exclusions are in each model card and
the sanitized result records.

- [Dense Qwen3.5 result](results/qwen35_v4_8_loss_aware_smoke.json)
- [OLMo 2 model card](models/olmo2.md)
- [OLMo 2 result](results/olmo2_1b_v4_8_smoke.json)
