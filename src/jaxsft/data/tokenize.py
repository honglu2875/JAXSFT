"""Whole-document tokenization, span alignment, and loss-policy evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        if not (len(self.loss_weights) == len(self.metadata) == len(self.offsets) == size):
            raise ValueError("token arrays and metadata must have identical lengths")
        if size and self.loss_weights[0] != 0.0:
            raise ValueError("the first token in a causal sample must have zero loss weight")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.loss_weights):
            raise ValueError("token loss weights must be finite and non-negative")

    @property
    def selected_tokens(self) -> int:
        return sum(weight > 0 for weight in self.loss_weights)

    @property
    def total_weight(self) -> float:
        return float(sum(self.loss_weights))


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


def tokenize_document(
    document: RenderedDocument,
    encoder: Encoder,
    *,
    tokenizer_hash: str,
    policy: LossPolicy | None = None,
    max_length: int | None = None,
    truncation: str = "reject",
) -> TokenizedSample:
    """Encode once, align offsets to semantic spans, and assign target weights."""

    policy = policy or LossPolicy()
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

    truncated = False
    if max_length is not None and len(ids) > max_length:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if truncation == "reject":
            raise ValueError(f"sample has {len(ids)} tokens, exceeding max_length={max_length}")
        if truncation == "right":
            keep = slice(0, max_length)
        elif truncation == "left":
            keep = slice(len(ids) - max_length, None)
        else:
            raise ValueError("truncation must be reject, right, or left")
        ids, offsets = ids[keep], offsets[keep]
        metadata, weights = metadata[keep], weights[keep]
        weights[0] = 0.0
        truncated = True

    return TokenizedSample(
        sample_id=document.sample_id,
        input_ids=ids,
        loss_weights=tuple(float(weight) for weight in weights),
        metadata=tuple(metadata),
        offsets=offsets,
        tokenizer_hash=tokenizer_hash,
        truncated=truncated,
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
    token_strings = getattr(encoder.encode(document.text, add_special_tokens=False), "tokens", [""] * len(tokenized.input_ids))
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
            }
        )
    return json.dumps({"sample_id": document.sample_id, "text": document.text, "tokens": rows}, ensure_ascii=False, indent=2)
