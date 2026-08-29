"""Rank-disjoint streaming instruction batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from ..config import DataSpec
from .adapters import AdapterContext, AdapterError, get_adapter
from .render import render_qwen3_5
from .tokenize import LossPolicy, TokenizerSnapshot, padded_arrays, tokenize_document


@dataclass
class StreamCounters:
    rows_seen: int = 0
    rows_emitted: int = 0
    adapter_errors: int = 0
    length_rejections: int = 0
    zero_objective: int = 0
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
        self.adapter = get_adapter(spec.adapter)
        self.counters = StreamCounters()
        self._iterator = self._make_iterator(epoch=0)

    def close(self) -> None:
        """Release any open Arrow/HTTP resources owned by the iterable."""

        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

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

    def _next_row(self) -> dict[str, Any]:
        while True:
            try:
                return next(self._iterator)
            except StopIteration:
                self.close()
                self.counters.epochs += 1
                self._iterator = self._make_iterator(epoch=self.counters.epochs)

    def _next_sample_arrays(self) -> dict[str, np.ndarray]:
        while True:
            row = self._next_row()
            row_index = self.counters.rows_seen
            self.counters.rows_seen += 1
            context = AdapterContext(
                repo_id=self.spec.repo_id,
                revision=self.spec.revision,
                config=self.spec.config,
                split=self.spec.split,
                row_index=row_index,
            )
            try:
                sample = self.adapter(row, context)
                document = render_qwen3_5(sample)
                tokenized = tokenize_document(
                    document,
                    self.encoder,
                    tokenizer_hash=self.snapshot.identity_hash,
                    policy=self.policy,
                    max_length=self.max_length,
                    truncation=self.truncation,
                )
            except AdapterError:
                self.counters.adapter_errors += 1
                continue
            except ValueError as error:
                if "exceeding max_length" in str(error):
                    self.counters.length_rejections += 1
                    continue
                raise
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
