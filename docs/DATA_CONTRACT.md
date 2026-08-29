# Data, tokenizer, and loss contract

## 1. Why spans are the stable boundary

Instruction datasets vary in field names, role names, content representation,
tool-call encoding, argument serialization, reasoning visibility, and whether a
row represents a prompt, a completion, a whole conversation, or one event.
Model tokenizers add a separate dialect of control tokens and whitespace.

Trying to normalize both concerns directly into `messages` dictionaries loses
the exact ownership needed for research loss policies. JAXSFT therefore uses
three representations:

1. **Canonical sample:** model-independent semantic messages and typed parts.
2. **Rendered spans:** model-specific text/control sequences with semantic
   ownership retained.
3. **Tokenized sample:** exact token IDs, target weights, and compact metadata.

## 2. Canonical sample

The illustrative type shape is:

```python
Sample(
    id: str,
    source: SourceRef,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...],
    attributes: FrozenJSON,
)

Message(
    role: Literal["system", "developer", "user", "assistant", "tool"],
    parts: tuple[Part, ...],
    name: str | None,
    call_id: str | None,
    attributes: FrozenJSON,
)

Part(
    kind: Literal[
        "text", "reasoning", "code", "tool_call", "tool_result", "media"
    ],
    value: str | FrozenJSON | MediaRef,
    call_id: str | None,
    tags: frozenset[str],
    attributes: FrozenJSON,
)
```

This is a design sketch; the implementation may use dataclasses, immutable
mappings, or typed unions. The invariants matter:

- sample IDs are stable within a pinned source revision;
- message/part order is preserved exactly;
- tool calls and results have explicit IDs when the source supplies them;
- JSON arguments remain structured until the renderer chooses serialization;
- reasoning and final-answer text can remain distinct even if the source stores
  them in one string;
- source-specific fields needed for later selectors remain in namespaced
  attributes;
- unrecognized structures are rejected or quarantined, never stringified
  silently;
- empty parts and missing/empty content are distinct states;
- adapters do not manufacture assistant answers from prompts.

## 3. Adapter protocol

An adapter receives a row plus an immutable context containing dataset identity,
split, row/shard location, and adapter options. It returns samples and structured
diagnostics.

Required behavior:

- validate required fields and types;
- normalize role aliases through an explicit table;
- make JSON-string parsing opt-in per field;
- assign stable call IDs only when a deterministic source rule is declared;
- preserve row provenance without retaining secret fields;
- distinguish drop, quarantine, and hard failure;
- publish counters by reason;
- produce deterministic output for the same row/context/options;
- include an adapter schema version and source-code hash.

Reproducible runs name the adapter explicitly. `data inspect` may compare likely
adapters against a sample of rows, but it only reports suggestions.

### Initial source families

| Family | Typical shape | Canonical mapping |
|---|---|---|
| Raw language modeling | `text` | One explicitly tagged text part; full-sequence loss only when recipe requests it. |
| Prompt/completion | `prompt`, `completion` | User/prompt parts plus assistant completion, retaining conversational or string form. |
| Standard chat | `messages[{role, content}]` | Message roles and string/typed content parts. |
| ShareGPT | `conversations[{from, value}]` | Explicit role alias map and text parts. |
| OpenAI tool calls | assistant `tool_calls`, tool-role results | Structured call name/arguments/ID and result linkage. |
| Anthropic content blocks | typed `text`, `thinking`, `tool_use`, `tool_result` blocks | Corresponding typed parts without flattening. |
| Action/observation trajectories | alternating thought/action/observation fields | Reasoning, tool call, and tool result parts under a source-specific adapter. |
| Pre-tokenized | `input_ids` and optional labels/masks | Bypasses rendering only after tokenizer/template identity and alignment are declared. |

Dataset-ID-specific adapters are allowed where a source violates its nominal
family. They should be short, fixture-driven modules, not conditionals in the
generic loader.

## 4. Rendering contract

A renderer consumes the canonical sample, a tokenizer/template snapshot, and
render options. It emits ordered spans such as:

```python
RenderedSpan(
    payload: str | tuple[int, ...],
    semantic_ref: SemanticRef | None,
    kind: str,
    role: str | None,
    trainability: str,
    attributes: FrozenJSON,
)
```

Spans include both content and template-owned material: role headers,
separators, begin/end markers, tool-definition preambles, call wrappers, and
generation prompts. Template-owned spans have explicit ownership and default
loss behavior; they are not anonymous text.

Renderer requirements:

- normalize tool JSON with a declared stable serializer when the template does
  not own serialization;
- preserve whitespace exactly;
- state whether BOS/EOS/control tokens are text-rendered or tokenizer-inserted;
- declare behavior for developer/system roles and consecutive same-role turns;
- declare whether empty assistant/tool content is legal;
- distinguish training render from generation-prompt render;
- emit a normalized template hash and all template variables used.

For a pinned fixture suite, concatenated/rendered token IDs must equal the
tokenizer's authoritative Hugging Face `apply_chat_template` output. The test
includes ordinary chat, tools, multiple calls, results, reasoning/final parts,
empty content, Unicode, and adversarial whitespace.

## 5. Span-to-token alignment

The tokenizer encodes the complete rendered sequence so byte-pair/unigram merges
at span boundaries are not accidentally changed. A fast tokenizer's character
offset mapping is used where reliable. Added/control tokens require explicit
renderer alignment because they may have empty offsets.

Every output token receives:

- source sample ID (host-side or compact index);
- message and part index;
- role and part kind;
- call ID/tool name where relevant;
- template/content span class;
- semantic tags and metric group IDs;
- a float loss weight aligned to predicting this token.

If a token overlaps two spans with different loss semantics, the template
adapter must apply and document one of these policies:

1. assign the boundary token to an explicitly named side;
2. fuse the spans because they intentionally share identical semantics; or
3. reject the sample/template combination.

There is no global “nearest span” heuristic. Boundary cases live in golden
fixtures.

## 6. Loss policies

Loss policies are ordered selector rules over semantic metadata, for example:

```yaml
policies:
  - select: {role: assistant}
    weight: 1.0
  - select: {part_kind: reasoning}
    weight: 0.0
  - select: {part_kind: tool_call}
    weight: 1.5
  - select: {part_kind: tool_result}
    weight: 0.0
```

The actual selector grammar should remain narrow and typed. Initial selectors:

- source name/config;
- role;
- part kind;
- message/turn index or relative position;
- tool name and call ID presence;
- semantic tags;
- template/content class;
- numeric source attributes through explicit named fields.

Rules use a declared conflict mode (`last_match`, `multiply`, or error). The
resolved policy records match counts and selected token counts. A run fails if
its objective selects zero tokens globally or if a required selector matches
nothing.

Boolean masks are represented as weights 0/1. Float weights support tool-call,
reasoning, source-quality, curriculum, and importance experiments without a
second loss path. Negative and non-finite weights are forbidden in the base SFT
objective.

## 7. Causal target alignment

For token sequence `x[0:L]`, model logits at position `t-1` predict token `x[t]`.
JAXSFT stores `loss_weight[t]` beside the token being predicted:

```python
target_ids = input_ids[:, 1:]
target_weights = loss_weights[:, 1:]
token_nll = cross_entropy(logits[:, :-1, :], target_ids)
numerator = sum(token_nll * target_weights)
denominator = sum(target_weights)
```

`loss_weight[0]` is always zero because there is no preceding logit inside the
sample/packed segment. A pack boundary also forces the first token of the new
segment to zero unless an explicit external prefix supplies its context. Padding
weights are zero.

This convention avoids the common off-by-one error in which a user token's
weight is applied to the assistant token predicted after it.

## 8. Normalization

Every training/evaluation step reports additive numerator and denominator. The
base selected-token mean is:

```text
sum_over_all_hosts(weight * token_nll) / sum_over_all_hosts(weight)
```

Both quantities are summed across microbatches and processes before division.

Research options may additionally normalize per example, turn, source, or
semantic group. Those require explicit group IDs and a precise rule for empty
groups. The run manifest records the formula; logs report both the configured
objective and raw selected-token aggregates so experiments remain comparable.

Metrics include at least:

- all non-padding input tokens;
- selected target tokens and total target weight;
- numerator/denominator/loss by role, part kind, source, and named policy group;
- truncated/dropped/packed sample and token counts;
- sequence utilization and examples per packed sequence.

## 9. Truncation

Truncation occurs after rendering/tokenization because true length is
tokenizer-specific, but decisions can use semantic boundaries. Supported
policies should include:

- `reject`: quarantine overlength samples;
- `tail_tokens` or `head_tokens`: explicit token slicing for plain text only;
- `keep_last_turns`: drop complete early messages while retaining required
  system/tool definitions;
- `keep_completion`: reserve a target-token budget and truncate prompt context;
- `window`: produce multiple windows with stable derived sample IDs;
- `semantic`: user-provided policy operating on canonical messages before a
  final token-length check.

Every policy declares handling of dangling tool calls/results, incomplete
reasoning/final pairs, BOS/EOS, and minimum selected tokens. Statistics and the
pre/post sample IDs are recorded. Silent hard slicing is not a default.

The implemented token-window baseline exposes `reject`, `right`, `left`, and
`loss_aware`. `loss_aware` evaluates every fixed-length window after the loss
policy has assigned weights. It maximizes retained objective weight while
accounting for the fact that the first retained causal token cannot be scored;
ties prefer the later target chunk and then the largest available prefix
context. `training.truncation_min_context_tokens` can reserve a minimum number
of tokens before the first retained target. If no window can satisfy that
constraint, the run records a relaxation rather than silently pretending it
was met.

Every truncated sample carries its original/window lengths, original/retained
selected-token counts and objective weight, retained context, and constraint
status. Stream totals expose the same losses. This makes the policy suitable
for objective-aware experiments, but it can still cut through a semantic turn
or tool transaction. Complete-turn/tool-boundary policies remain a separate
contract, not an implicit behavior of `loss_aware`.

## 10. Packing and attention

Packing concatenates complete tokenized samples into fixed-length arrays while
retaining `segment_ids`. Invariants:

- causal attention never crosses segment IDs;
- the first token of every segment has zero target weight;
- padding is a separate non-attending segment with zero weight;
- position IDs either reset at each segment or remain monotonic according to an
  explicit model capability/recipe choice;
- EOS insertion and its loss ownership are declared once by the renderer/packer;
- sample order and best-fit/first-fit algorithm are deterministic and versioned;
- unpacked metadata can be reconstructed for audits and per-sample metrics.

With dropout disabled, summed loss numerator/denominator for a batch of samples
must match its packed representation within numerical tolerance.

## 11. Dataset mixing and distributed iteration

A data source is pinned by Hub/local identity, revision, config, split, and file
manifest. A mixer defines source weights/quotas, exhaustion behavior, shuffle
seeds/buffer sizes, and curriculum schedule.

Global sample order is logical and deterministic. Each JAX process receives a
disjoint rank-local slice based on `jax.process_index()` and process count.
Resume state includes each source cursor, shuffle-buffer state or reproducible
equivalent, mixer RNG/counters, packer buffer, and emitted sample IDs.

Changing process count on resume is rejected initially. Elastic resharding may
be added later only with an explicit order-preservation contract.

For streaming Hub datasets, immutable file/shard identities are resolved before
training. If the upstream API cannot make exact resume practical, preprocess to
content-addressed local Parquet/token shards rather than claiming deterministic
streaming resume.

## 12. Data artifact layers

Keep three independently cacheable artifacts:

1. **Canonical cache:** adapter output; reusable across model families.
2. **Token cache:** pinned renderer/tokenizer/objective output; reusable across
   optimization recipes with the same semantic policy.
3. **Packed epoch plan:** run seed, length, packing, mixing, and process-count
   specific; cheap to regenerate where possible.

Each cache has a schema version, content hash, input manifest, completion marker,
and validation command. Cache misses rebuild; identity mismatches never fall
back to a “close enough” artifact.

## 13. Required adversarial fixtures

- user text that contains strings identical to model control tokens;
- Unicode combining characters and multi-byte tool arguments;
- adjacent assistant tool calls with distinct IDs and out-of-order results;
- missing call ID, duplicate call ID, and orphan result;
- empty system/user/assistant/tool messages;
- reasoning-only and final-only assistant messages;
- multiple assistant turns with different weights;
- JSON arguments stored as a dict versus an escaped JSON string;
- a boundary at which the tokenizer would otherwise merge across spans;
- EOS already present in source content;
- exactly-at-limit and one-token-over-limit examples;
- one sample that fills a pack and a sample whose first token follows a pack
  boundary;
- all-zero weights, fractional weights, and very small total weight;
- identical data under one and multiple JAX processes.
