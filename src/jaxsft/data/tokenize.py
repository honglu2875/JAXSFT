"""Whole-document tokenization, span alignment, and loss-policy evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .render import RenderedDocument, RenderedSpan


class AlignmentError(ValueError):
    """A token cannot be assigned semantic ownership unambiguously."""


@dataclass(frozen=True)
class TokenMetadata:
    span_index: int
    span_class: str
    role: str | None
    part_kind: str | None
    message_index: int | None
    part_index: int | None
    call_id: str | None
    tool_name: str | None
    tags: frozenset[str]
    loss_rule_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class LossRule:
    select: dict[str, Any]
    weight: float
    name: str = ""
    require_match: bool = False

    def __post_init__(self) -> None:
        supported = {
            "span_class",
            "role",
            "part_kind",
            "message_index",
            "part_index",
            "call_id",
            "tool_name",
            "tags",
        }
        unknown = set(self.select) - supported
        if unknown:
            raise ValueError(f"unsupported loss selector fields: {sorted(unknown)}")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("loss-rule weights must be finite and non-negative")


@dataclass(frozen=True)
class LossPolicy:
    rules: tuple[LossRule, ...] = ()
    conflict_mode: str = "last_match"

    def __post_init__(self) -> None:
        if self.conflict_mode not in {"last_match", "multiply", "error"}:
            raise ValueError("conflict_mode must be last_match, multiply, or error")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LossPolicy":
        unknown = set(config) - {"conflict_mode", "rules"}
        if unknown:
            raise ValueError(f"unknown objective keys: {sorted(unknown)}")
        rules: list[LossRule] = []
        for index, raw in enumerate(config.get("rules", [])):
            if not isinstance(raw, dict):
                raise TypeError(f"objective.rules[{index}] must be a mapping")
            extra = set(raw) - {"name", "select", "weight", "require_match"}
            if extra:
                raise ValueError(f"unknown keys in objective.rules[{index}]: {sorted(extra)}")
            rules.append(
                LossRule(
                    name=str(raw.get("name", f"rule_{index}")),
                    select=dict(raw.get("select", {})),
                    weight=float(raw["weight"]),
                    require_match=bool(raw.get("require_match", False)),
                )
            )
        return cls(tuple(rules), str(config.get("conflict_mode", "last_match")))

    @staticmethod
    def _value_matches(actual: Any, expected: Any) -> bool:
        if isinstance(expected, (list, tuple, set, frozenset)):
            if isinstance(actual, frozenset):
                return bool(actual.intersection(expected))
            return actual in expected
        if isinstance(actual, frozenset):
            return expected in actual
        return actual == expected

    def evaluate(self, metadata: TokenMetadata, default: float) -> tuple[float, tuple[int, ...]]:
        weight = default
        matches: list[int] = []
        for index, rule in enumerate(self.rules):
            if all(self._value_matches(getattr(metadata, key), expected) for key, expected in rule.select.items()):
                matches.append(index)
                if self.conflict_mode == "last_match":
                    weight = rule.weight
                elif self.conflict_mode == "multiply":
                    weight *= rule.weight
                elif len(matches) > 1:
                    raise ValueError(f"loss token matched conflicting rules {matches}")
                else:
                    weight = rule.weight
        return weight, tuple(matches)


@dataclass(frozen=True)
class TokenizedSample:
    sample_id: str
    input_ids: tuple[int, ...]
    loss_weights: tuple[float, ...]
    metadata: tuple[TokenMetadata, ...]
    offsets: tuple[tuple[int, int], ...]
    tokenizer_hash: str
    truncated: bool = False
    truncation_record: "TruncationRecord | None" = None

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        if not (len(self.loss_weights) == len(self.metadata) == len(self.offsets) == size):
            raise ValueError("token arrays and metadata must have identical lengths")
        if size and self.loss_weights[0] != 0.0:
            raise ValueError("the first token in a causal sample must have zero loss weight")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.loss_weights):
            raise ValueError("token loss weights must be finite and non-negative")
        if self.truncated != (self.truncation_record is not None):
            raise ValueError("truncated flag and truncation record must agree")
        if self.truncation_record is not None and size != self.truncation_record.end - self.truncation_record.start:
            raise ValueError("tokenized length differs from its truncation window")
        if self.truncation_record is not None:
            if sum(weight > 0 for weight in self.loss_weights) != self.truncation_record.retained_selected_tokens:
                raise ValueError("retained selected-token count differs from truncation record")
            if not math.isclose(
                sum(self.loss_weights), self.truncation_record.retained_weight, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("retained objective weight differs from truncation record")

    @property
    def selected_tokens(self) -> int:
        return sum(weight > 0 for weight in self.loss_weights)

    @property
    def total_weight(self) -> float:
        return float(sum(self.loss_weights))


@dataclass(frozen=True)
class TruncationRecord:
    policy: str
    original_length: int
    start: int
    end: int
    original_selected_tokens: int
    retained_selected_tokens: int
    original_weight: float
    retained_weight: float
    minimum_context_tokens: int
    retained_context_tokens: int
    context_constraint_satisfied: bool

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end <= self.original_length:
            raise ValueError("invalid truncation window")
        if self.original_selected_tokens < self.retained_selected_tokens:
            raise ValueError("truncation cannot add selected tokens")
        if self.original_weight + 1e-12 < self.retained_weight:
            raise ValueError("truncation cannot add objective weight")
        if self.minimum_context_tokens < 0 or self.retained_context_tokens < 0:
            raise ValueError("truncation context counts must be non-negative")
        if self.context_constraint_satisfied and self.retained_context_tokens < self.minimum_context_tokens:
            raise ValueError("satisfied truncation record violates its context minimum")


class Encoded(Protocol):
    ids: list[int]
    offsets: list[tuple[int, int]]


class Encoder(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Encoded: ...


@dataclass(frozen=True)
class TokenizerSnapshot:
    path: Path
    identity_hash: str
    pad_token_id: int

    @classmethod
    def load(cls, path: str | Path) -> tuple["TokenizerSnapshot", Encoder]:
        from tokenizers import Tokenizer

        path = Path(path)
        tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
        payload = tokenizer_json.read_bytes()
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        pad_token_id = tokenizer.token_to_id("<|endoftext|>")
        if pad_token_id is None:
            raise ValueError("Qwen3.5 tokenizer has no <|endoftext|> padding token")
        return cls(tokenizer_json, hashlib.sha256(payload).hexdigest(), int(pad_token_id)), tokenizer


def _span_ranges(spans: tuple[RenderedSpan, ...]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for span in spans:
        ranges.append((cursor, cursor + len(span.text)))
        cursor += len(span.text)
    return tuple(ranges)


def _owner_for_offset(
    offset: tuple[int, int], spans: tuple[RenderedSpan, ...], ranges: tuple[tuple[int, int], ...]
) -> int:
    start, end = offset
    if start < 0 or end < start:
        raise AlignmentError(f"invalid tokenizer offset {offset}")
    if start == end:
        raise AlignmentError(
            f"tokenizer returned empty offset {offset}; control-token alignment must be implemented explicitly"
        )
    overlap = [index for index, (left, right) in enumerate(ranges) if start < right and end > left]
    if not overlap:
        raise AlignmentError(f"token offset {offset} lies outside rendered spans")
    if len(overlap) == 1:
        return overlap[0]
    boundaries = [spans[index].boundary_owner for index in overlap]
    if all(owner == "right" for owner in boundaries):
        return overlap[-1]
    if all(owner == "left" for owner in boundaries):
        return overlap[0]
    raise AlignmentError(f"token offset {offset} crosses spans {overlap} without one boundary policy")


def _loss_aware_window(
    weights: list[float], max_length: int, *, minimum_context_tokens: int
) -> tuple[int, int, int, bool]:
    """Choose the window retaining most objective weight and then latest targets.

    The first retained token cannot be predicted without external context, so
    its weight is excluded while scoring candidate windows. Ties first retain
    the later target chunk, then retain as much preceding context as possible.
    """

    size = len(weights)
    prefix = [0.0]
    last_positive: list[int] = []
    latest = -1
    for index, weight in enumerate(weights):
        prefix.append(prefix[-1] + weight)
        if weight > 0:
            latest = index
        last_positive.append(latest)
    next_positive = [size] * (size + 1)
    nearest = size
    for index in range(size - 1, -1, -1):
        if weights[index] > 0:
            nearest = index
        next_positive[index] = nearest
    if latest < 0:
        return 0, max_length, 0, minimum_context_tokens == 0

    def choose(*, enforce_context: bool) -> tuple[int, int] | None:
        best_start = 0
        best_weight = -1.0
        best_tie = (-1, 0)
        found = False
        for start in range(size - max_length + 1):
            end = start + max_length
            first_target = next_positive[start + 1]
            context_tokens = first_target - start if first_target < end else 0
            if enforce_context and context_tokens < minimum_context_tokens:
                continue
            retained_weight = prefix[end] - prefix[start + 1]
            last_target = last_positive[end - 1]
            if last_target <= start:
                last_target = -1
            tie = (last_target, -start)
            if retained_weight > best_weight and not math.isclose(
                retained_weight, best_weight, rel_tol=1e-12, abs_tol=1e-12
            ):
                best_weight = retained_weight
                best_tie = tie
                best_start = start
                found = True
            elif math.isclose(retained_weight, best_weight, rel_tol=1e-12, abs_tol=1e-12) and tie > best_tie:
                best_weight = retained_weight
                best_tie = tie
                best_start = start
                found = True
        return (best_start, best_start + max_length) if found else None

    window = choose(enforce_context=minimum_context_tokens > 0)
    constraint_satisfied = window is not None
    if window is None:
        window = choose(enforce_context=False)
    assert window is not None
    start, end = window
    first_target = next_positive[start + 1]
    retained_context_tokens = first_target - start if first_target < end else 0
    return start, end, retained_context_tokens, constraint_satisfied


def tokenize_document(
    document: RenderedDocument,
    encoder: Encoder,
    *,
    tokenizer_hash: str,
    policy: LossPolicy | None = None,
    max_length: int | None = None,
    truncation: str = "reject",
    truncation_min_context_tokens: int = 0,
) -> TokenizedSample:
    """Encode once, align offsets to semantic spans, and assign target weights."""

    policy = policy or LossPolicy()
    if truncation_min_context_tokens < 0:
        raise ValueError("truncation_min_context_tokens must be non-negative")
    if truncation != "loss_aware" and truncation_min_context_tokens:
        raise ValueError("truncation_min_context_tokens is only valid with loss_aware truncation")
    encoded = encoder.encode(document.text, add_special_tokens=False)
    ids = tuple(int(token_id) for token_id in encoded.ids)
    offsets = tuple((int(left), int(right)) for left, right in encoded.offsets)
    if len(ids) != len(offsets):
        raise AlignmentError("tokenizer returned a different number of IDs and offsets")
    ranges = _span_ranges(document.spans)
    metadata: list[TokenMetadata] = []
    weights: list[float] = []
    match_counts = [0] * len(policy.rules)
    for offset in offsets:
        span_index = _owner_for_offset(offset, document.spans, ranges)
        span = document.spans[span_index]
        ref = span.semantic_ref
        item = TokenMetadata(
            span_index=span_index,
            span_class=span.span_class,
            role=span.role,
            part_kind=span.part_kind,
            message_index=None if ref is None else ref.message_index,
            part_index=None if ref is None else ref.part_index,
            call_id=span.call_id,
            tool_name=span.tool_name,
            tags=span.tags,
        )
        weight, matches = policy.evaluate(item, span.default_weight)
        item = replace(item, loss_rule_indices=matches)
        metadata.append(item)
        weights.append(weight)
        for rule_index in matches:
            match_counts[rule_index] += 1
    if weights:
        weights[0] = 0.0
    missing = [
        policy.rules[index].name or str(index)
        for index, count in enumerate(match_counts)
        if policy.rules[index].require_match and count == 0
    ]
    if missing:
        raise ValueError(f"required loss rules matched no tokens: {missing}")

    truncation_record = None
    if max_length is not None and len(ids) > max_length:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if truncation == "reject":
            raise ValueError(f"sample has {len(ids)} tokens, exceeding max_length={max_length}")
        if truncation == "right":
            start, end = 0, max_length
        elif truncation == "left":
            start, end = len(ids) - max_length, len(ids)
        elif truncation == "loss_aware":
            if truncation_min_context_tokens >= max_length:
                raise ValueError("truncation_min_context_tokens must be smaller than max_length")
            start, end, retained_context_tokens, context_constraint_satisfied = _loss_aware_window(
                weights,
                max_length,
                minimum_context_tokens=truncation_min_context_tokens,
            )
        else:
            raise ValueError("truncation must be reject, right, left, or loss_aware")
        if truncation != "loss_aware":
            retained_context_tokens = 0
            context_constraint_satisfied = True
        original_length = len(ids)
        original_selected_tokens = sum(weight > 0 for weight in weights)
        original_weight = float(sum(weights))
        keep = slice(start, end)
        ids, offsets = ids[keep], offsets[keep]
        metadata, weights = metadata[keep], weights[keep]
        weights[0] = 0.0
        truncation_record = TruncationRecord(
            policy=truncation,
            original_length=original_length,
            start=start,
            end=end,
            original_selected_tokens=original_selected_tokens,
            retained_selected_tokens=sum(weight > 0 for weight in weights),
            original_weight=original_weight,
            retained_weight=float(sum(weights)),
            minimum_context_tokens=truncation_min_context_tokens,
            retained_context_tokens=retained_context_tokens,
            context_constraint_satisfied=context_constraint_satisfied,
        )

    return TokenizedSample(
        sample_id=document.sample_id,
        input_ids=ids,
        loss_weights=tuple(float(weight) for weight in weights),
        metadata=tuple(metadata),
        offsets=offsets,
        tokenizer_hash=tokenizer_hash,
        truncated=truncation_record is not None,
        truncation_record=truncation_record,
    )


def padded_arrays(sample: TokenizedSample, *, length: int, pad_token_id: int) -> dict[str, np.ndarray]:
    if len(sample.input_ids) > length:
        raise ValueError(f"sample length {len(sample.input_ids)} exceeds padded length {length}")
    input_ids = np.full((length,), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((length,), dtype=np.bool_)
    loss_weights = np.zeros((length,), dtype=np.float32)
    size = len(sample.input_ids)
    input_ids[:size] = sample.input_ids
    attention_mask[:size] = True
    loss_weights[:size] = sample.loss_weights
    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_weights": loss_weights}


def explain_tokens(document: RenderedDocument, tokenized: TokenizedSample, encoder: Encoder) -> str:
    rows = []
    token_strings = getattr(encoder.encode(document.text, add_special_tokens=False), "tokens", [])
    if tokenized.truncation_record is not None and token_strings:
        token_strings = token_strings[tokenized.truncation_record.start : tokenized.truncation_record.end]
    if not token_strings:
        token_strings = [""] * len(tokenized.input_ids)
    for index, (token_id, weight, meta, offset, token) in enumerate(
        zip(tokenized.input_ids, tokenized.loss_weights, tokenized.metadata, tokenized.offsets, token_strings)
    ):
        rows.append(
            {
                "index": index,
                "id": token_id,
                "token": token,
                "offset": offset,
                "weight": weight,
                "role": meta.role,
                "part_kind": meta.part_kind,
                "span_class": meta.span_class,
                "message": meta.message_index,
                "part": meta.part_index,
                "loss_rule_indices": meta.loss_rule_indices,
            }
        )
    return json.dumps(
        {
            "sample_id": document.sample_id,
            "text": document.text,
            "truncation": None
            if tokenized.truncation_record is None
            else asdict(tokenized.truncation_record),
            "tokens": rows,
        },
        ensure_ascii=False,
        indent=2,
    )
