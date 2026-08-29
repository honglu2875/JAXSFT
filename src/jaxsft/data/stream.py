"""Rank-disjoint streaming instruction batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np

from ..config import DataSpec
from .adapters import AdapterContext, AdapterError, get_adapter
from .render import get_renderer
from .tokenize import LossPolicy, TokenizerSnapshot, padded_arrays, tokenize_document


@dataclass
class StreamCounters:
    rows_seen: int = 0
    rows_seen_in_epoch: int = 0
    rows_emitted: int = 0
    adapter_errors: int = 0
    adapter_errors_by_reason: dict[str, int] = field(default_factory=dict)
    length_rejections: int = 0
    zero_objective: int = 0
    truncated_samples: int = 0
    tokens_truncated: int = 0
    selected_tokens_truncated: int = 0
    selected_weight_truncated: float = 0.0
    context_constraint_relaxations: int = 0
    epochs: int = 0


class InstructionBatchStream:
    """An endless, deterministic per-process stream of padded microbatches."""

    def __init__(
        self,
        spec: DataSpec,
        *,
        tokenizer_snapshot: TokenizerSnapshot,
        encoder: Any,
        policy: LossPolicy,
        process_index: int,
        process_count: int,
        local_device_count: int,
        per_device_batch_size: int,
        accumulation_steps: int,
        max_length: int,
        truncation: str,
        truncation_min_context_tokens: int = 0,
        renderer: str = "qwen3_5",
    ):
        self.spec = spec
        self.snapshot = tokenizer_snapshot
        self.encoder = encoder
        self.policy = policy
        self.process_index = process_index
        self.process_count = process_count
        self.local_device_count = local_device_count
        self.per_device_batch_size = per_device_batch_size
        self.accumulation_steps = accumulation_steps
        self.max_length = max_length
        self.truncation = truncation
        self.truncation_min_context_tokens = truncation_min_context_tokens
        self.adapter = get_adapter(spec.adapter)
        self.render = get_renderer(renderer)
        self.counters = StreamCounters()
        self._iterator = self._make_iterator(epoch=0)

    def close(self) -> None:
        """Release any open Arrow/HTTP resources owned by the iterable."""

        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

    def state_dict(self) -> dict[str, Any]:
        """Return a replayable rank-local cursor for the pinned iterable."""

        return {
            "schema_version": 2,
            "kind": "huggingface_stream_replay",
            "process_index": self.process_index,
            "process_count": self.process_count,
            "tokenizer_hash": self.snapshot.identity_hash,
            "counters": asdict(self.counters),
        }

    def load_state_dict(self, raw: Mapping[str, Any]) -> None:
        """Restore by deterministically replaying the current epoch's prefix."""

        allowed = {"schema_version", "kind", "process_index", "process_count", "tokenizer_hash", "counters"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown stream-state keys: {sorted(unknown)}")
        if raw.get("schema_version") != 2 or raw.get("kind") != "huggingface_stream_replay":
            raise ValueError("unsupported instruction stream state")
        if int(raw.get("process_index", -1)) != self.process_index:
            raise ValueError("stream checkpoint process_index differs from this process")
        if int(raw.get("process_count", -1)) != self.process_count:
            raise ValueError("stream checkpoint process_count differs from this run")
        if raw.get("tokenizer_hash") != self.snapshot.identity_hash:
            raise ValueError("stream checkpoint tokenizer differs from this run")
        counters_raw = raw.get("counters")
        if not isinstance(counters_raw, Mapping):
            raise ValueError("stream checkpoint counters must be a mapping")
        counter_fields = set(StreamCounters.__dataclass_fields__)
        if set(counters_raw) != counter_fields:
            raise ValueError("stream checkpoint counter fields do not match this version")
        reasons = counters_raw.get("adapter_errors_by_reason")
        if not isinstance(reasons, Mapping):
            raise ValueError("adapter_errors_by_reason must be a mapping")
        counters = StreamCounters(
            rows_seen=int(counters_raw["rows_seen"]),
            rows_seen_in_epoch=int(counters_raw["rows_seen_in_epoch"]),
            rows_emitted=int(counters_raw["rows_emitted"]),
            adapter_errors=int(counters_raw["adapter_errors"]),
            adapter_errors_by_reason={str(key): int(value) for key, value in reasons.items()},
            length_rejections=int(counters_raw["length_rejections"]),
            zero_objective=int(counters_raw["zero_objective"]),
            truncated_samples=int(counters_raw["truncated_samples"]),
            tokens_truncated=int(counters_raw["tokens_truncated"]),
            selected_tokens_truncated=int(counters_raw["selected_tokens_truncated"]),
            selected_weight_truncated=float(counters_raw["selected_weight_truncated"]),
            context_constraint_relaxations=int(counters_raw["context_constraint_relaxations"]),
            epochs=int(counters_raw["epochs"]),
        )
        numeric = (
            counters.rows_seen,
            counters.rows_seen_in_epoch,
            counters.rows_emitted,
            counters.adapter_errors,
            counters.length_rejections,
            counters.zero_objective,
            counters.truncated_samples,
            counters.tokens_truncated,
            counters.selected_tokens_truncated,
            counters.selected_weight_truncated,
            counters.context_constraint_relaxations,
            counters.epochs,
            *counters.adapter_errors_by_reason.values(),
        )
        if any(value < 0 for value in numeric):
            raise ValueError("stream checkpoint counters must be non-negative")
        if counters.rows_seen_in_epoch > counters.rows_seen:
            raise ValueError("rows_seen_in_epoch cannot exceed rows_seen")
        if counters.adapter_errors != sum(counters.adapter_errors_by_reason.values()):
            raise ValueError("adapter error total differs from per-reason counters")
        classified_rows = (
            counters.rows_emitted
            + counters.adapter_errors
            + counters.length_rejections
            + counters.zero_objective
        )
        if classified_rows != counters.rows_seen:
            raise ValueError("stream checkpoint row classifications do not sum to rows_seen")

        self.close()
        iterator = self._make_iterator(epoch=counters.epochs)
        for skipped in range(counters.rows_seen_in_epoch):
            try:
                next(iterator)
            except StopIteration as error:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
                raise ValueError(
                    f"dataset ended after {skipped} rows while replaying a {counters.rows_seen_in_epoch}-row cursor"
                ) from error
        self._iterator = iterator
        self.counters = counters

    def _make_iterator(self, *, epoch: int) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        dataset = load_dataset(
            self.spec.repo_id,
            name=self.spec.config,
            split=self.spec.split,
            revision=self.spec.revision,
            streaming=True,
        )
        dataset = dataset.shuffle(seed=self.spec.shuffle_seed + epoch, buffer_size=self.spec.shuffle_buffer_size)
        dataset = split_dataset_by_node(dataset, rank=self.process_index, world_size=self.process_count)
        return iter(dataset)

    def _next_row(self) -> tuple[dict[str, Any], int]:
        while True:
            try:
                row = next(self._iterator)
                row_index = self.counters.rows_seen
                self.counters.rows_seen += 1
                self.counters.rows_seen_in_epoch += 1
                return row, row_index
            except StopIteration:
                self.close()
                self.counters.epochs += 1
                self.counters.rows_seen_in_epoch = 0
                self._iterator = self._make_iterator(epoch=self.counters.epochs)

    def _next_sample_arrays(self) -> dict[str, np.ndarray]:
        while True:
            row, row_index = self._next_row()
            context = AdapterContext(
                repo_id=self.spec.repo_id,
                revision=self.spec.revision,
                config=self.spec.config,
                split=self.spec.split,
                row_index=row_index,
            )
            try:
                sample = self.adapter(row, context)
                document = self.render(sample)
                tokenized = tokenize_document(
                    document,
                    self.encoder,
                    tokenizer_hash=self.snapshot.identity_hash,
                    policy=self.policy,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    truncation_min_context_tokens=self.truncation_min_context_tokens,
                )
            except AdapterError as error:
                self.counters.adapter_errors += 1
                reason = f"{type(error).__name__}: {error}"[:240]
                self.counters.adapter_errors_by_reason[reason] = (
                    self.counters.adapter_errors_by_reason.get(reason, 0) + 1
                )
                continue
            except ValueError as error:
                if "exceeding max_length" in str(error):
                    self.counters.length_rejections += 1
                    continue
                raise
            truncation = tokenized.truncation_record
            if truncation is not None:
                self.counters.truncated_samples += 1
                self.counters.tokens_truncated += truncation.original_length - (truncation.end - truncation.start)
                self.counters.selected_tokens_truncated += (
                    truncation.original_selected_tokens - truncation.retained_selected_tokens
                )
                self.counters.selected_weight_truncated += truncation.original_weight - truncation.retained_weight
                if not truncation.context_constraint_satisfied:
                    self.counters.context_constraint_relaxations += 1
            if tokenized.selected_tokens == 0:
                self.counters.zero_objective += 1
                continue
            self.counters.rows_emitted += 1
            return padded_arrays(tokenized, length=self.max_length, pad_token_id=self.snapshot.pad_token_id)

    def next_batch(self) -> dict[str, np.ndarray]:
        microbatches: list[dict[str, np.ndarray]] = []
        local_batch = self.local_device_count * self.per_device_batch_size
        for _ in range(self.accumulation_steps):
            examples = [self._next_sample_arrays() for _ in range(local_batch)]
            microbatches.append({key: np.stack([item[key] for item in examples]) for key in examples[0]})
        result: dict[str, np.ndarray] = {}
        for key in microbatches[0]:
            # [accumulation, local_device, per_device_batch, length] -> pmap-local leading device axis.
            value = np.stack([microbatch[key] for microbatch in microbatches])
            value = value.reshape(
                self.accumulation_steps,
                self.local_device_count,
                self.per_device_batch_size,
                self.max_length,
            )
            result[key] = np.swapaxes(value, 0, 1)
        return result
