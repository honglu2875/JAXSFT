#!/usr/bin/env python3
"""Benchmark bounded concurrent ranges for one official GLM-5.3 expert pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from jaxsft.checkpoint import StrictPooledHTTPRangeReader
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    SafetensorsIndex,
    SafetensorsShardHeader,
    SafetensorsTensorRange,
)


def _source_url(shard: str) -> str:
    return f"https://huggingface.co/{OFFICIAL_REPO_ID}/resolve/{OFFICIAL_REVISION}/{shard}"


def _read_header(reader: StrictPooledHTTPRangeReader) -> SafetensorsShardHeader:
    header_length = int.from_bytes(reader.read(0, 7), "little", signed=False)
    if header_length <= 0 or header_length > 100 * 1024 * 1024:
        raise ValueError(f"unsafe remote safetensors header length: {header_length}")
    payload = reader.read(8, 7 + header_length)
    return SafetensorsShardHeader.from_json_bytes(payload, header_length=header_length)


def _process_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, raw = line.partition(":")
        if name in {"VmRSS", "VmHWM"}:
            fields = raw.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"unexpected /proc/self/status memory line: {line!r}")
            values[name.lower() + "_bytes"] = int(fields[0]) * 1024
    if set(values) != {"vmrss_bytes", "vmhwm_bytes"}:
        raise ValueError("/proc/self/status did not expose VmRSS and VmHWM")
    return values


def _host_range(
    tensor: SafetensorsTensorRange,
    host_quarter: int,
) -> tuple[int, int, int]:
    if tensor.dtype != "F8_E4M3" or len(tensor.shape) != 2:
        raise ValueError(f"expert weight must be a rank-two block-FP8 matrix: {tensor}")
    rows, columns = tensor.shape
    if rows % 4:
        raise ValueError(f"expert output rows {rows} are not divisible across four hosts")
    rows_per_host = rows // 4
    row_start = host_quarter * rows_per_host
    row_stop = row_start + rows_per_host
    start = tensor.absolute_start + row_start * columns
    end = tensor.absolute_start + row_stop * columns - 1
    return start, end, rows_per_host * columns


def _run(args: argparse.Namespace) -> dict[str, Any]:
    index = SafetensorsIndex.from_path(args.index)
    index.verify(OFFICIAL_CHECKPOINT)
    weight_map = dict(index.tensor_files)
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts."
    selected: list[tuple[int, str, str]] = []
    for expert_index in range(args.expert_limit):
        name = prefix + f"{expert_index}.{args.projection}_proj.weight"
        try:
            shard = weight_map[name]
        except KeyError as error:
            raise ValueError(f"the pinned index has no expert tensor {name!r}") from error
        selected.append((expert_index, name, shard))
    shard_names = sorted({shard for _, _, shard in selected})

    started = time.monotonic()
    memory = {"before": _process_memory()}
    with ExitStack() as stack:
        readers = {
            shard: stack.enter_context(
                StrictPooledHTTPRangeReader(
                    _source_url(shard),
                    timeout_seconds=args.timeout_seconds,
                    maximum_request_bytes=args.maximum_range_bytes,
                    connections=args.connections,
                )
            )
            for shard in shard_names
        }
        headers: dict[str, SafetensorsShardHeader] = {}
        for shard in shard_names:
            header = _read_header(readers[shard])
            expected_names = {name for name, filename in index.tensor_files if filename == shard}
            actual_names = {tensor.name for tensor in header.tensors}
            if actual_names != expected_names:
                raise ValueError(f"remote header for {shard} does not exactly match the pinned index")
            headers[shard] = header
        memory["after_headers"] = _process_memory()

        work: list[tuple[int, str, tuple[int, int], int]] = []
        for expert_index, name, shard in selected:
            tensor = headers[shard].tensor(name)
            start, end, expected_bytes = _host_range(tensor, args.host_quarter)
            work.append((expert_index, shard, (start, end), expected_bytes))

        payload_started = time.monotonic()
        digests: dict[int, bytes] = {}
        payload_bytes = 0
        with ThreadPoolExecutor(max_workers=args.connections) as executor:
            futures = {
                executor.submit(readers[shard].read, *byte_range): (
                    expert_index,
                    expected_bytes,
                )
                for expert_index, shard, byte_range, expected_bytes in work
            }
            for future in as_completed(futures):
                expert_index, expected_bytes = futures.pop(future)
                payload = future.result()
                if len(payload) != expected_bytes:
                    raise ValueError(
                        f"expert {expert_index} returned {len(payload)} bytes, expected {expected_bytes}"
                    )
                payload_bytes += len(payload)
                digests[expert_index] = hashlib.sha256(payload).digest()
        payload_seconds = time.monotonic() - payload_started
        memory["after_payloads"] = _process_memory()

        combined = hashlib.sha256()
        for expert_index in range(args.expert_limit):
            combined.update(expert_index.to_bytes(4, "little"))
            combined.update(digests[expert_index])
        per_shard = {}
        for shard, reader in readers.items():
            records = reader.records
            per_shard[shard] = {
                "request_count_including_resolve_and_header": len(records),
                "bytes_read_including_resolve_and_header": reader.bytes_read,
                "largest_request_bytes": max(record.bytes_read for record in records),
                "resolved_once": reader.resolved,
            }

    return {
        "schema_version": 1,
        "test": "glm53_expert_pooled_range_benchmark",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "index_sha256": index.sha256,
        },
        "selection": {
            "layer": args.layer,
            "projection": args.projection,
            "expert_limit": args.expert_limit,
            "host_quarter": args.host_quarter,
            "source_shards": shard_names,
        },
        "network": {
            "connections": args.connections,
            "payload_request_count": len(work),
            "payload_bytes": payload_bytes,
            "payload_seconds": payload_seconds,
            "payload_bytes_per_second": payload_bytes / payload_seconds,
            "payload_sha256_of_expert_hashes": combined.hexdigest(),
            "per_shard": per_shard,
        },
        "host": {
            "hostname": socket.gethostname(),
            "memory": memory,
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--projection", choices=("gate", "up", "down"), default="gate")
    parser.add_argument("--expert-limit", type=int, default=32)
    parser.add_argument("--host-quarter", type=int, choices=range(4), default=0)
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--maximum-range-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if not 1 <= args.expert_limit <= 288:
        raise ValueError("expert-limit must be in [1, 288]")
    if args.connections <= 0:
        raise ValueError("connections must be positive")
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")

    result = _run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
