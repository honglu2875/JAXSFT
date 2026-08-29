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


class SemanticTruncationError(ValueError):
    """A sample cannot satisfy the declared semantic truncation contract."""


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
            if self.truncation_record.semantic_boundary_aligned:
                retained_messages = tuple(
                    sorted({item.message_index for item in self.metadata if item.message_index is not None})
                )
                if retained_messages != self.truncation_record.retained_message_indices:
                    raise ValueError("retained token metadata differs from semantic truncation record")

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
    semantic_boundary_aligned: bool = False
    original_message_indices: tuple[int, ...] = ()
    retained_message_indices: tuple[int, ...] = ()
    dropped_message_indices: tuple[int, ...] = ()
    original_atomic_units: int = 0
    retained_atomic_units: int = 0
    original_tool_atomic_units: int = 0
    retained_tool_atomic_units: int = 0

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
        if self.semantic_boundary_aligned:
            if self.policy != "semantic_loss_aware":
                raise ValueError("semantic boundary alignment requires semantic_loss_aware policy")
            original = set(self.original_message_indices)
            retained = set(self.retained_message_indices)
            dropped = set(self.dropped_message_indices)
            if retained & dropped or retained | dropped != original:
                raise ValueError("retained/dropped semantic messages must partition the original messages")
            if tuple(sorted(original)) != self.original_message_indices:
                raise ValueError("semantic message indices must be sorted and unique")
            if self.original_atomic_units < self.retained_atomic_units:
                raise ValueError("semantic truncation cannot add atomic units")
            if self.original_tool_atomic_units < self.retained_tool_atomic_units:
                raise ValueError("semantic truncation cannot add tool atomic units")
        elif any(
            (
                self.original_message_indices,
                self.retained_message_indices,
                self.dropped_message_indices,
                self.original_atomic_units,
                self.retained_atomic_units,
                self.original_tool_atomic_units,
                self.retained_tool_atomic_units,
            )
        ):
            raise ValueError("token-only truncation cannot claim semantic boundary metadata")


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
    def load(cls, path: str | Path, *, pad_token_id: int | None = None) -> tuple["TokenizerSnapshot", Encoder]:
        from tokenizers import Tokenizer

        path = Path(path)
        tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
        payload = tokenizer_json.read_bytes()
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        tokenizer_config = tokenizer_json.with_name("tokenizer_config.json")
        configured_pad = None
        config_payload = b""
        if tokenizer_config.is_file():
            config_payload = tokenizer_config.read_bytes()
            raw_config = json.loads(config_payload)
            if not isinstance(raw_config, dict):
                raise ValueError("tokenizer_config.json must contain an object")
            configured_pad = raw_config.get("pad_token")
            if isinstance(configured_pad, dict):
                configured_pad = configured_pad.get("content")
            if configured_pad is not None and not isinstance(configured_pad, str):
                raise ValueError("tokenizer pad_token must be a string or token object")
        if pad_token_id is None and configured_pad is not None:
            pad_token_id = tokenizer.token_to_id(configured_pad)
        if pad_token_id is None:
            for token in ("<|pad|>", "<|endoftext|>"):
                pad_token_id = tokenizer.token_to_id(token)
                if pad_token_id is not None:
                    break
        if pad_token_id is None or not 0 <= int(pad_token_id) < tokenizer.get_vocab_size():
            raise ValueError("tokenizer has no valid padding token")
        digest = hashlib.sha256(b"jaxsft-tokenizer-v1\0")
        digest.update(payload)
        digest.update(b"\0tokenizer-config\0")
        digest.update(config_payload)
        digest.update(b"\0pad-token-id\0")
        digest.update(str(int(pad_token_id)).encode())
        return cls(tokenizer_json, digest.hexdigest(), int(pad_token_id)), tokenizer


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


@dataclass(frozen=True)
class _SemanticUnit:
    start: int
    end: int
    message_indices: tuple[int, ...]
    tool_exchange: bool = False


@dataclass(frozen=True)
class _MessageBlock:
    start: int
    end: int
    message_index: int
    role: str
    tool_call_ids: frozenset[str] = frozenset()
    tool_result_ids: frozenset[str] = frozenset()
    missing_tool_call_id: bool = False
    duplicate_tool_call_id: bool = False

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_call_ids) or self.missing_tool_call_id


@dataclass(frozen=True)
class _SemanticWindow:
    start: int
    end: int
    retained_context_tokens: int
    context_constraint_satisfied: bool
    original_message_indices: tuple[int, ...]
    retained_message_indices: tuple[int, ...]
    original_atomic_units: int
    retained_atomic_units: int
    original_tool_atomic_units: int
    retained_tool_atomic_units: int


def _semantic_units(metadata: list[TokenMetadata]) -> tuple[_SemanticUnit, ...]:
    """Build contiguous message units, making each tool exchange indivisible."""

    message_indices: list[int] = []
    starts: list[int] = []
    seen: set[int] = set()
    previous: int | None = None
    for token_index, item in enumerate(metadata):
        message_index = item.message_index
        if message_index is None or message_index == previous:
            continue
        if message_index in seen or (message_indices and message_index <= message_indices[-1]):
            raise SemanticTruncationError(
                "semantic truncation requires each message to own one ordered token interval"
            )
        seen.add(message_index)
        message_indices.append(message_index)
        starts.append(token_index)
        previous = message_index
    if not message_indices:
        raise SemanticTruncationError("semantic truncation requires rendered message ownership metadata")
    # Leading BOS/tool-preamble tokens belong to the first rendered message;
    # unowned separators between messages remain with the message on their left.
    starts[0] = 0

    blocks: list[_MessageBlock] = []
    for block_index, (message_index, start) in enumerate(zip(message_indices, starts)):
        end = starts[block_index + 1] if block_index + 1 < len(starts) else len(metadata)
        owned = [item for item in metadata[start:end] if item.message_index == message_index]
        roles = {item.role for item in owned if item.role is not None}
        if len(roles) != 1:
            raise SemanticTruncationError(
                f"semantic truncation found ambiguous role ownership for message {message_index}"
            )
        tool_call_refs = {
            (item.part_index, item.call_id)
            for item in owned
            if item.part_kind == "tool_call"
        }
        nonmissing_call_refs = {item for item in tool_call_refs if item[1] is not None}
        blocks.append(
            _MessageBlock(
                start=start,
                end=end,
                message_index=message_index,
                role=next(iter(roles)),
                tool_call_ids=frozenset(item[1] for item in nonmissing_call_refs),
                tool_result_ids=frozenset(
                    item.call_id for item in owned if item.role == "tool" and item.call_id is not None
                ),
                missing_tool_call_id=any(item[1] is None for item in tool_call_refs),
                duplicate_tool_call_id=(
                    len(nonmissing_call_refs)
                    != len({item[1] for item in nonmissing_call_refs})
                ),
            )
        )

    seen_call_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    for block in blocks:
        if block.has_tool_call and block.role != "assistant":
            raise SemanticTruncationError(
                "semantic truncation requires tool calls to belong to an assistant message"
            )
        duplicate_calls = seen_call_ids & block.tool_call_ids
        if duplicate_calls:
            raise SemanticTruncationError("semantic truncation found a reused tool-call ID")
        seen_call_ids.update(block.tool_call_ids)
        if block.role == "tool":
            duplicate_results = seen_result_ids & block.tool_result_ids
            if duplicate_results:
                raise SemanticTruncationError("semantic truncation found duplicate results for one tool-call ID")
            seen_result_ids.update(block.tool_result_ids)

    units: list[_SemanticUnit] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.role == "tool":
            raise SemanticTruncationError(
                "semantic truncation found a tool result without its preceding tool-call message"
            )
        if not block.has_tool_call:
            units.append(
                _SemanticUnit(
                    start=block.start,
                    end=block.end,
                    message_indices=(block.message_index,),
                )
            )
            index += 1
            continue

        if block.missing_tool_call_id:
            raise SemanticTruncationError("semantic truncation requires every tool call to have a call ID")
        if block.duplicate_tool_call_id:
            raise SemanticTruncationError("semantic truncation found duplicate tool-call IDs in one message")

        # A tool-call assistant message, its consecutive result messages, any
        # chained calls, and the immediate final assistant answer form one unit.
        group = [block]
        index += 1
        pending_call_ids = set(block.tool_call_ids)
        observed_result_ids: set[str] = set()
        has_final_answer = False
        while index < len(blocks):
            candidate = blocks[index]
            if candidate.role == "tool":
                if len(candidate.tool_result_ids) != 1:
                    raise SemanticTruncationError(
                        "semantic truncation requires each tool result to have exactly one call ID"
                    )
                result_id = next(iter(candidate.tool_result_ids))
                if result_id not in pending_call_ids:
                    raise SemanticTruncationError(
                        "semantic truncation found a tool result linked to an unknown call ID"
                    )
                if result_id in observed_result_ids:
                    raise SemanticTruncationError(
                        "semantic truncation found duplicate results for one tool-call ID"
                    )
                observed_result_ids.add(result_id)
                group.append(candidate)
                index += 1
                continue
            if candidate.role == "assistant" and observed_result_ids == pending_call_ids:
                group.append(candidate)
                index += 1
                if candidate.has_tool_call:
                    if candidate.missing_tool_call_id:
                        raise SemanticTruncationError(
                            "semantic truncation requires every tool call to have a call ID"
                        )
                    if candidate.duplicate_tool_call_id:
                        raise SemanticTruncationError(
                            "semantic truncation found duplicate tool-call IDs in one message"
                        )
                    pending_call_ids = set(candidate.tool_call_ids)
                    observed_result_ids = set()
                    continue
                has_final_answer = True
                break
            break
        if observed_result_ids != pending_call_ids:
            raise SemanticTruncationError("semantic truncation found a tool call without all linked results")
        if not has_final_answer:
            raise SemanticTruncationError(
                "semantic truncation requires an immediate final assistant answer after tool results"
            )
        units.append(
            _SemanticUnit(
                start=group[0].start,
                end=group[-1].end,
                message_indices=tuple(item.message_index for item in group),
                tool_exchange=True,
            )
        )
    return tuple(units)


def _semantic_loss_aware_window(
    metadata: list[TokenMetadata],
    weights: list[float],
    max_length: int,
    *,
    minimum_context_tokens: int,
    units: tuple[_SemanticUnit, ...] | None = None,
) -> _SemanticWindow:
    """Optimize objective weight over complete message/tool-exchange units."""

    units = _semantic_units(metadata) if units is None else units
    tool_preamble_present = any(
        item.message_index is None and item.span_class in {"tool_preamble", "tool_definition"}
        for item in metadata
    )
    size = len(weights)
    prefix = [0.0]
    latest = -1
    last_positive: list[int] = []
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

    def choose(*, enforce_context: bool) -> tuple[int, int, int, int] | None:
        best: tuple[int, int, int, int] | None = None
        best_weight = -1.0
        best_tie = (-1, -1, -1, -1)
        for first_unit in range(len(units)):
            for final_unit in range(first_unit, len(units)):
                start, end = units[first_unit].start, units[final_unit].end
                if end - start > max_length:
                    break
                if (
                    tool_preamble_present
                    and first_unit > 0
                    and any(unit.tool_exchange for unit in units[first_unit : final_unit + 1])
                ):
                    continue
                first_target = next_positive[start + 1]
                retained_weight = prefix[end] - prefix[start + 1]
                if latest >= 0 and (first_target >= end or retained_weight <= 0):
                    continue
                context_tokens = first_target - start if first_target < end else 0
                if enforce_context and context_tokens < minimum_context_tokens:
                    continue
                last_target = last_positive[end - 1]
                if last_target <= start:
                    last_target = -1
                tie = (last_target, context_tokens, end - start, -start)
                if retained_weight > best_weight and not math.isclose(
                    retained_weight, best_weight, rel_tol=1e-12, abs_tol=1e-12
                ):
                    best_weight = retained_weight
                    best_tie = tie
                    best = (first_unit, final_unit, context_tokens, start)
                elif math.isclose(
                    retained_weight, best_weight, rel_tol=1e-12, abs_tol=1e-12
                ) and tie > best_tie:
                    best_weight = retained_weight
                    best_tie = tie
                    best = (first_unit, final_unit, context_tokens, start)
        return best

    selected = choose(enforce_context=minimum_context_tokens > 0)
    constraint_satisfied = selected is not None
    if selected is None:
        selected = choose(enforce_context=False)
    if selected is None:
        raise SemanticTruncationError(
            f"semantic truncation cannot retain a complete objective-bearing unit within max_length={max_length}"
        )
    first_unit, final_unit, context_tokens, _ = selected
    retained_units = units[first_unit : final_unit + 1]
    original_messages = tuple(index for unit in units for index in unit.message_indices)
    retained_messages = tuple(index for unit in retained_units for index in unit.message_indices)
    return _SemanticWindow(
        start=retained_units[0].start,
        end=retained_units[-1].end,
        retained_context_tokens=context_tokens,
        context_constraint_satisfied=constraint_satisfied,
        original_message_indices=original_messages,
        retained_message_indices=retained_messages,
        original_atomic_units=len(units),
        retained_atomic_units=len(retained_units),
        original_tool_atomic_units=sum(unit.tool_exchange for unit in units),
        retained_tool_atomic_units=sum(unit.tool_exchange for unit in retained_units),
    )


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
    loss_aware_policies = {"loss_aware", "semantic_loss_aware"}
    if truncation not in loss_aware_policies and truncation_min_context_tokens:
        raise ValueError("truncation_min_context_tokens is only valid with a loss-aware truncation policy")
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

    semantic_units = _semantic_units(metadata) if truncation == "semantic_loss_aware" else None

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
        elif truncation == "semantic_loss_aware":
            if truncation_min_context_tokens >= max_length:
                raise ValueError("truncation_min_context_tokens must be smaller than max_length")
            semantic = _semantic_loss_aware_window(
                metadata,
                weights,
                max_length,
                minimum_context_tokens=truncation_min_context_tokens,
                units=semantic_units,
            )
            start, end = semantic.start, semantic.end
            retained_context_tokens = semantic.retained_context_tokens
            context_constraint_satisfied = semantic.context_constraint_satisfied
        else:
            raise ValueError(
                "truncation must be reject, right, left, loss_aware, or semantic_loss_aware"
            )
        if truncation not in loss_aware_policies:
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
            semantic_boundary_aligned=truncation == "semantic_loss_aware",
            original_message_indices=()
            if truncation != "semantic_loss_aware"
            else semantic.original_message_indices,
            retained_message_indices=()
            if truncation != "semantic_loss_aware"
            else semantic.retained_message_indices,
            dropped_message_indices=()
            if truncation != "semantic_loss_aware"
            else tuple(
                index
                for index in semantic.original_message_indices
                if index not in set(semantic.retained_message_indices)
            ),
            original_atomic_units=0
            if truncation != "semantic_loss_aware"
            else semantic.original_atomic_units,
            retained_atomic_units=0
            if truncation != "semantic_loss_aware"
            else semantic.retained_atomic_units,
            original_tool_atomic_units=0
            if truncation != "semantic_loss_aware"
            else semantic.original_tool_atomic_units,
            retained_tool_atomic_units=0
            if truncation != "semantic_loss_aware"
            else semantic.retained_tool_atomic_units,
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
