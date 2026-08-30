"""Bounded direct-to-TPU streaming for the pinned GLM-5.3 text checkpoint.

This module is deliberately model-specific and experimental.  It translates
the architecture-derived tensor schema in :mod:`glm5_3_flash` into final JAX
``NamedSharding`` arrays without creating a host or device copy of the full
checkpoint.  Public safetensors shards are serialization containers, not mesh
shards; every payload interval is therefore derived from the final device
index rather than from the source filename.
"""

from __future__ import annotations

import gc
import math
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from ..checkpoint import StrictPooledHTTPRangeReader
from .glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    BatchedBlockFP8LinearKernel,
    BlockFP8LinearKernel,
    Glm53CheckpointTensorSpec,
    Glm53TextConfig,
    SafetensorsIndex,
    SafetensorsShardHeader,
    SafetensorsTensorRange,
    checkpoint_text_tensor_specs,
)


_SMALL_REPLICATION_LIMIT = 1024 * 1024
_BLOCK_SHAPE = (128, 128)


def _scale_name(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".weight_scale_inv"
    return weight_name + "_scale_inv"


def _format_path(path: Sequence[str | int]) -> str:
    return ".".join(str(component) for component in path)


def _source_url(shard: str) -> str:
    return f"https://huggingface.co/{OFFICIAL_REPO_ID}/resolve/{OFFICIAL_REVISION}/{shard}"


def _normalize_slice(value: slice, size: int) -> tuple[int, int]:
    start = 0 if value.start is None else value.start
    stop = size if value.stop is None else value.stop
    if value.step not in (None, 1) or not (0 <= start < stop <= size):
        raise ValueError(f"streaming loader requires a positive contiguous slice, got {value!r}")
    return start, stop


def _normalize_index(index: Any, shape: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not isinstance(index, tuple) or len(index) != len(shape):
        raise ValueError(f"unexpected sharding index {index!r} for shape {tuple(shape)}")
    result = []
    for part, size in zip(index, shape, strict=True):
        if not isinstance(part, slice):
            raise ValueError(f"streaming loader does not support non-slice index {index!r}")
        result.append(_normalize_slice(part, size))
    return tuple(result)


def _source_numpy_dtype(dtype: str) -> np.dtype[Any]:
    if sys.byteorder != "little":
        raise RuntimeError("the experimental safetensors loader currently requires a little-endian host")
    try:
        return {
            "F8_E4M3": np.dtype(np.uint8),
            "BF16": np.dtype(jnp.bfloat16),
            "F32": np.dtype("<f4"),
        }[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported streamed safetensors dtype {dtype!r}") from error


def _array_from_payload(
    payload: bytes | bytearray | memoryview,
    tensor: SafetensorsTensorRange,
) -> np.ndarray[Any, Any]:
    if len(payload) != tensor.nbytes:
        raise ValueError(
            f"payload for {tensor.name!r} has {len(payload)} bytes, expected {tensor.nbytes}"
        )
    value = np.frombuffer(payload, dtype=_source_numpy_dtype(tensor.dtype))
    if value.size != math.prod(tensor.shape):
        raise ValueError(f"payload element count for {tensor.name!r} does not match its shape")
    return value.reshape(tensor.shape)


def _transform_source(value: np.ndarray[Any, Any], transform: str) -> np.ndarray[Any, Any]:
    if transform == "identity":
        return value
    if transform == "transpose":
        if value.ndim != 2:
            raise ValueError("checkpoint transpose requires a rank-two matrix")
        return np.swapaxes(value, 0, 1)
    if transform == "squeeze_conv":
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError("checkpoint convolution squeeze requires shape [channels,1,kernel]")
        return value[:, 0, :]
    raise ValueError(f"unsupported non-expert source transform {transform!r}")


def _target_shape(spec: Glm53CheckpointTensorSpec) -> tuple[int, ...]:
    if spec.transform == "identity":
        return spec.source_shape
    if spec.transform == "transpose":
        if len(spec.source_shape) != 2:
            raise ValueError(f"transpose spec {spec.source_name!r} is not rank two")
        return spec.source_shape[1], spec.source_shape[0]
    if spec.transform == "squeeze_conv":
        if len(spec.source_shape) != 3 or spec.source_shape[1] != 1:
            raise ValueError(f"convolution spec {spec.source_name!r} is malformed")
        return spec.source_shape[0], spec.source_shape[2]
    raise ValueError("packed experts do not have a single-source target shape")


def _is_axis0_sharded(tensor: SafetensorsTensorRange) -> bool:
    if tensor.nbytes <= _SMALL_REPLICATION_LIMIT or len(tensor.shape) < 2:
        return False
    if tensor.shape[0] % 16:
        raise ValueError(f"large text tensor {tensor.name!r} is not 16-way axis-0 shardable")
    return True


@dataclass(frozen=True)
class LoadedTarget:
    path: tuple[str | int, ...]
    value: Any
    logical_source_names: tuple[str, ...]
    scale_source_names: tuple[str, ...]


class Glm53StreamingLoader(AbstractContextManager["Glm53StreamingLoader"]):
    """Stream the pinned text checkpoint into its final 16-way model mesh."""

    def __init__(
        self,
        config: Glm53TextConfig,
        index: SafetensorsIndex,
        mesh: Mesh,
        *,
        connections_per_shard: int = 8,
        worker_threads: int = 16,
        timeout_seconds: float = 120.0,
        maximum_range_bytes: int = 384 * 1024 * 1024,
        maximum_scale_envelope_bytes: int = 16 * 1024 * 1024,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
        reader_factory: Callable[..., StrictPooledHTTPRangeReader] = StrictPooledHTTPRangeReader,
    ) -> None:
        index.verify(OFFICIAL_CHECKPOINT)
        if mesh.axis_names != ("model",) or mesh.size != 16:
            raise ValueError("GLM-5.3 streaming requires a 16-device mesh named 'model'")
        if connections_per_shard <= 0 or worker_threads <= 0:
            raise ValueError("streaming connection and worker counts must be positive")
        if maximum_scale_envelope_bytes <= 0:
            raise ValueError("maximum scale envelope must be positive")
        self.config = config
        self.index = index
        self.mesh = mesh
        self.connections_per_shard = connections_per_shard
        self.worker_threads = worker_threads
        self.timeout_seconds = timeout_seconds
        self.maximum_range_bytes = maximum_range_bytes
        self.maximum_scale_envelope_bytes = maximum_scale_envelope_bytes
        self.progress = progress
        self.reader_factory = reader_factory
        self._weight_map = dict(index.tensor_files)
        self._readers: dict[str, StrictPooledHTTPRangeReader] = {}
        self._shard_headers: dict[str, SafetensorsShardHeader] = {}
        self._headers: dict[str, tuple[str, SafetensorsTensorRange]] = {}
        self._scale_envelopes: dict[str, tuple[int, bytes]] = {}
        self._category_bytes: Counter[str] = Counter()
        self._category_requests: Counter[str] = Counter()
        self._stats_lock = threading.Lock()
        self._loaded_logical_names: set[str] = set()
        self._loaded_scale_names: set[str] = set()
        self._loaded_paths: set[tuple[str | int, ...]] = set()
        self._maximum_expert_host_buffer_bytes = 0
        self._closed = False

    def __enter__(self) -> Glm53StreamingLoader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for reader in self._readers.values():
            reader.close()
        self.release_host_cache()

    def _reader(self, shard: str) -> StrictPooledHTTPRangeReader:
        if self._closed:
            raise RuntimeError("GLM-5.3 streaming loader is closed")
        if shard not in self.index.shard_names:
            raise ValueError(f"unknown pinned checkpoint shard {shard!r}")
        if shard not in self._readers:
            self._readers[shard] = self.reader_factory(
                _source_url(shard),
                timeout_seconds=self.timeout_seconds,
                maximum_request_bytes=self.maximum_range_bytes,
                connections=self.connections_per_shard,
            )
        return self._readers[shard]

    def _read(self, shard: str, start: int, end: int, *, category: str) -> bytes:
        payload = self._reader(shard).read(start, end)
        with self._stats_lock:
            self._category_requests[category] += 1
            self._category_bytes[category] += len(payload)
        return payload

    def _prepare_shard(self, shard: str) -> None:
        if shard in self._shard_headers:
            return
        reader = self._reader(shard)
        header_length_bytes = self._read(shard, 0, 7, category="header")
        header_length = int.from_bytes(header_length_bytes, "little", signed=False)
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError(f"unsafe remote safetensors header length in {shard}: {header_length}")
        payload = self._read(shard, 8, 7 + header_length, category="header")
        header = SafetensorsShardHeader.from_json_bytes(payload, header_length=header_length)
        expected_names = {name for name, filename in self.index.tensor_files if filename == shard}
        actual_names = {tensor.name for tensor in header.tensors}
        if actual_names != expected_names:
            raise ValueError(f"remote header for {shard} does not exactly match the pinned index")
        expected_file_bytes = header.data_section_start + max(
            tensor.relative_end for tensor in header.tensors
        )
        if reader.total_size_bytes != expected_file_bytes:
            raise ValueError(
                f"remote size for {shard} is {reader.total_size_bytes}, expected {expected_file_bytes}"
            )
        for tensor in header.tensors:
            if tensor.name in self._headers or self._weight_map[tensor.name] != shard:
                raise ValueError(f"duplicate or mis-sharded tensor {tensor.name!r}")
            self._headers[tensor.name] = (shard, tensor)
        self._shard_headers[shard] = header
        if self.progress is not None:
            self.progress(
                {
                    "event": "header_ready",
                    "shard": shard,
                    "prepared_shards": len(self._shard_headers),
                }
            )

    def _prepare_scale_envelope(self, shard: str) -> None:
        if shard in self._scale_envelopes:
            return
        self._prepare_shard(shard)
        scales = [
            tensor
            for tensor in self._shard_headers[shard].tensors
            if tensor.name.endswith("weight_scale_inv")
        ]
        if not scales:
            self._scale_envelopes[shard] = (0, b"")
            return
        start = min(tensor.absolute_start for tensor in scales)
        end = max(tensor.absolute_end for tensor in scales) - 1
        envelope_bytes = end - start + 1
        if envelope_bytes > self.maximum_scale_envelope_bytes:
            raise ValueError(
                f"scale envelope for {shard} is {envelope_bytes} bytes, above "
                f"the {self.maximum_scale_envelope_bytes}-byte safety bound"
            )
        payload = self._read(shard, start, end, category="scale_envelope")
        self._scale_envelopes[shard] = (start, payload)

    def prepare(self, specs: Iterable[Glm53CheckpointTensorSpec]) -> None:
        shards = set()
        for spec in specs:
            try:
                shards.add(self._weight_map[spec.source_name])
            except KeyError as error:
                raise ValueError(f"pinned index is missing {spec.source_name!r}") from error
        for shard in sorted(shards):
            self._prepare_shard(shard)
        for shard in sorted(shards):
            self._prepare_scale_envelope(shard)

    def prepare_all(self) -> None:
        self.prepare(checkpoint_text_tensor_specs(self.config))
        if set(self._shard_headers) != set(self.index.shard_names):
            raise ValueError("complete text preparation did not cover all checkpoint shards")

    def _tensor(self, name: str) -> tuple[str, SafetensorsTensorRange]:
        try:
            shard = self._weight_map[name]
        except KeyError as error:
            raise ValueError(f"pinned index is missing tensor {name!r}") from error
        self._prepare_shard(shard)
        return self._headers[name]

    def _cached_payload(self, shard: str, tensor: SafetensorsTensorRange) -> memoryview | None:
        envelope = self._scale_envelopes.get(shard)
        if envelope is None:
            return None
        envelope_start, payload = envelope
        start = tensor.absolute_start - envelope_start
        end = tensor.absolute_end - envelope_start
        if 0 <= start < end <= len(payload):
            return memoryview(payload)[start:end]
        return None

    def _full_payload(self, shard: str, tensor: SafetensorsTensorRange) -> bytes | memoryview:
        cached = self._cached_payload(shard, tensor)
        if cached is not None:
            return cached
        return self._read(shard, *tensor.http_range, category="replicated_tensor")

    def _replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, PartitionSpec())

    def _replicate_host_array(self, value: np.ndarray[Any, Any]) -> jax.Array:
        value = np.ascontiguousarray(value)
        sharding = self._replicated_sharding()
        result = jax.make_array_from_callback(value.shape, sharding, lambda index: value[index])
        result.block_until_ready()
        return result

    def _load_replicated(
        self,
        shard: str,
        tensor: SafetensorsTensorRange,
        *,
        transform: str,
    ) -> jax.Array:
        payload = self._full_payload(shard, tensor)
        source = _array_from_payload(payload, tensor)
        value = _transform_source(source, transform)
        result = self._replicate_host_array(value)
        del value, source, payload
        return result

    def _final_sharding(
        self,
        spec: Glm53CheckpointTensorSpec,
        *,
        source_oriented: bool,
    ) -> NamedSharding:
        if source_oriented:
            return NamedSharding(
                self.mesh,
                PartitionSpec("model", *([None] * (len(spec.source_shape) - 1))),
            )
        if spec.transform == "transpose":
            return NamedSharding(self.mesh, PartitionSpec(None, "model"))
        shape = _target_shape(spec)
        return NamedSharding(
            self.mesh,
            PartitionSpec("model", *([None] * (len(shape) - 1))),
        )

    def _source_rows_for_final_index(
        self,
        spec: Glm53CheckpointTensorSpec,
        index: Any,
        *,
        source_oriented: bool,
    ) -> tuple[int, int]:
        shape = spec.source_shape if source_oriented else _target_shape(spec)
        normalized = _normalize_index(index, shape)
        if source_oriented or spec.transform in {"identity", "squeeze_conv"}:
            row_interval = normalized[0]
        elif spec.transform == "transpose":
            row_interval = normalized[1]
        else:
            raise ValueError(f"cannot map transform {spec.transform!r} to source rows")
        for axis, interval in enumerate(normalized):
            if axis == (0 if source_oriented or spec.transform != "transpose" else 1):
                continue
            if interval != (0, shape[axis]):
                raise ValueError(f"loader received an unsupported multi-axis shard index {index!r}")
        return row_interval

    def _load_axis0_sharded(
        self,
        spec: Glm53CheckpointTensorSpec,
        shard: str,
        tensor: SafetensorsTensorRange,
        *,
        source_oriented: bool,
    ) -> jax.Array:
        global_shape = spec.source_shape if source_oriented else _target_shape(spec)
        sharding = self._final_sharding(spec, source_oriented=source_oriented)
        index_items = list(sharding.addressable_devices_indices_map(global_shape).items())
        if len(index_items) != jax.local_device_count():
            raise ValueError("final sharding does not expose exactly one index per local TPU")
        row_bytes = tensor.nbytes // tensor.shape[0]
        if row_bytes * tensor.shape[0] != tensor.nbytes:
            raise ValueError(f"tensor {tensor.name!r} does not have integral source rows")

        local_arrays: list[jax.Array | None] = [None] * len(index_items)
        with ThreadPoolExecutor(max_workers=min(self.worker_threads, len(index_items))) as executor:
            futures = {}
            for position, (device, index) in enumerate(index_items):
                row_start, row_stop = self._source_rows_for_final_index(
                    spec,
                    index,
                    source_oriented=source_oriented,
                )
                start = tensor.absolute_start + row_start * row_bytes
                end = tensor.absolute_start + row_stop * row_bytes - 1
                future = executor.submit(
                    self._read,
                    shard,
                    start,
                    end,
                    category="axis0_tensor",
                )
                futures[future] = (position, device, row_start, row_stop)
            for future in as_completed(futures):
                position, device, row_start, row_stop = futures.pop(future)
                payload = future.result()
                local_source_shape = (row_stop - row_start, *tensor.shape[1:])
                expected_bytes = math.prod(local_source_shape) * _source_numpy_dtype(
                    tensor.dtype
                ).itemsize
                if len(payload) != expected_bytes:
                    raise ValueError(
                        f"local range for {tensor.name!r} has {len(payload)} bytes, "
                        f"expected {expected_bytes}"
                    )
                source = np.frombuffer(payload, dtype=_source_numpy_dtype(tensor.dtype)).reshape(
                    local_source_shape
                )
                value = source if source_oriented else _transform_source(source, spec.transform)
                value = np.ascontiguousarray(value)
                placed = jax.device_put(value, SingleDeviceSharding(device))
                placed.block_until_ready()
                local_arrays[position] = placed
                del placed, value, source, payload
        if any(value is None for value in local_arrays):
            raise AssertionError("not every addressable shard received a source payload")
        result = jax.make_array_from_single_device_arrays(
            global_shape,
            sharding,
            [value for value in local_arrays if value is not None],
        )
        result.block_until_ready()
        return result

    def _load_scale(self, weight: SafetensorsTensorRange) -> tuple[str, jax.Array]:
        name = _scale_name(weight.name)
        shard, scale = self._tensor(name)
        if scale.dtype != "F32" or len(weight.shape) != 2:
            raise ValueError(f"invalid block-FP8 scale contract for {weight.name!r}")
        expected_shape = tuple(size // block for size, block in zip(weight.shape, _BLOCK_SHAPE))
        if scale.shape != expected_shape:
            raise ValueError(
                f"scale {name!r} has shape {scale.shape}, expected {expected_shape}"
            )
        self._prepare_scale_envelope(shard)
        value = self._load_replicated(shard, scale, transform="identity")
        return name, value

    def _load_single_target(
        self,
        spec: Glm53CheckpointTensorSpec,
    ) -> LoadedTarget:
        shard, tensor = self._tensor(spec.source_name)
        if tensor.shape != spec.source_shape:
            raise ValueError(
                f"source shape for {spec.source_name!r} is {tensor.shape}, expected {spec.source_shape}"
            )
        if tensor.dtype == "F8_E4M3":
            if len(tensor.shape) != 2 or any(size % 128 for size in tensor.shape):
                raise ValueError(f"FP8 tensor {tensor.name!r} is not a 128x128-blocked matrix")
            if _is_axis0_sharded(tensor):
                bits = self._load_axis0_sharded(
                    spec,
                    shard,
                    tensor,
                    source_oriented=True,
                )
            else:
                bits = self._load_replicated(shard, tensor, transform="identity")
            scale_name, scale = self._load_scale(tensor)
            value = BlockFP8LinearKernel(bits, scale, compute_dtype=jnp.bfloat16)
            scales = (scale_name,)
        else:
            unexpected_scale = _scale_name(spec.source_name)
            if unexpected_scale in self._weight_map:
                raise ValueError(f"non-FP8 tensor {spec.source_name!r} has a scale companion")
            if _is_axis0_sharded(tensor):
                value = self._load_axis0_sharded(
                    spec,
                    shard,
                    tensor,
                    source_oriented=False,
                )
            else:
                value = self._load_replicated(shard, tensor, transform=spec.transform)
            scales = ()
        return LoadedTarget(spec.target_path, value, (spec.source_name,), scales)

    def _expert_host_layout(
        self,
        shape: tuple[int, int, int],
        sharding: NamedSharding,
    ) -> tuple[
        list[tuple[jax.Device, Any]],
        dict[jax.Device, tuple[int, int]],
        int,
        int,
    ]:
        index_items = list(sharding.addressable_devices_indices_map(shape).items())
        if len(index_items) != jax.local_device_count():
            raise ValueError("expert sharding does not expose exactly one index per local TPU")
        device_rows = {}
        intervals = []
        for device, index in index_items:
            normalized = _normalize_index(index, shape)
            if normalized[0] != (0, shape[0]) or normalized[2] != (0, shape[2]):
                raise ValueError(f"expert loader received unsupported shard index {index!r}")
            device_rows[device] = normalized[1]
            intervals.append(normalized[1])
        intervals.sort()
        host_start = intervals[0][0]
        host_stop = intervals[-1][1]
        expected = host_start
        for start, stop in intervals:
            if start != expected:
                raise ValueError("addressable expert output-row slices are not host-contiguous")
            expected = stop
        if expected != host_stop:
            raise AssertionError("invalid expert host interval")
        return index_items, device_rows, host_start, host_stop

    def _load_expert_target(
        self,
        specs: Sequence[Glm53CheckpointTensorSpec],
    ) -> LoadedTarget:
        ordered = sorted(specs, key=lambda spec: -1 if spec.pack_index is None else spec.pack_index)
        if [spec.pack_index for spec in ordered] != list(range(self.config.n_routed_experts)):
            raise ValueError(f"expert target {_format_path(specs[0].target_path)!r} is incomplete")
        if len({spec.source_shape for spec in ordered}) != 1:
            raise ValueError("expert target combines different source shapes")
        source_shape = ordered[0].source_shape
        if len(source_shape) != 2 or any(size % 128 for size in source_shape):
            raise ValueError("expert sources must be rank-two 128x128-blocked matrices")
        experts = len(ordered)
        rows, columns = source_shape
        global_shape = (experts, rows, columns)
        sharding = NamedSharding(self.mesh, PartitionSpec(None, "model", None))
        index_items, device_rows, host_start, host_stop = self._expert_host_layout(
            global_shape,
            sharding,
        )
        host_buffers = {
            device: np.empty((experts, stop - start, columns), dtype=np.uint8)
            for device, (start, stop) in device_rows.items()
        }
        host_buffer_bytes = sum(buffer.nbytes for buffer in host_buffers.values())
        self._maximum_expert_host_buffer_bytes = max(
            self._maximum_expert_host_buffer_bytes,
            host_buffer_bytes,
        )

        with ThreadPoolExecutor(max_workers=self.worker_threads) as executor:
            futures = {}
            for spec in ordered:
                assert spec.pack_index is not None
                shard, tensor = self._tensor(spec.source_name)
                if tensor.dtype != "F8_E4M3" or tensor.shape != source_shape:
                    raise ValueError(f"invalid expert source metadata for {spec.source_name!r}")
                start = tensor.absolute_start + host_start * columns
                end = tensor.absolute_start + host_stop * columns - 1
                future = executor.submit(
                    self._read,
                    shard,
                    start,
                    end,
                    category="expert_tensor",
                )
                futures[future] = (spec.pack_index, spec.source_name)
            for future in as_completed(futures):
                pack_index, source_name = futures.pop(future)
                payload = future.result()
                expected_bytes = (host_stop - host_start) * columns
                if len(payload) != expected_bytes:
                    raise ValueError(
                        f"expert range for {source_name!r} has {len(payload)} bytes, "
                        f"expected {expected_bytes}"
                    )
                host_matrix = np.frombuffer(payload, dtype=np.uint8).reshape(
                    host_stop - host_start,
                    columns,
                )
                for device, (start, stop) in device_rows.items():
                    host_buffers[device][pack_index] = host_matrix[
                        start - host_start : stop - host_start
                    ]
                del host_matrix, payload

        local_arrays = []
        for device, _ in index_items:
            host_buffer = host_buffers.pop(device)
            placed = jax.device_put(host_buffer, SingleDeviceSharding(device))
            placed.block_until_ready()
            local_arrays.append(placed)
            del host_buffer
        bits = jax.make_array_from_single_device_arrays(global_shape, sharding, local_arrays)
        bits.block_until_ready()
        del local_arrays, host_buffers

        scale_shape = (experts, rows // 128, columns // 128)
        host_scales = np.empty(scale_shape, dtype=np.float32)
        scale_names = []
        for spec in ordered:
            assert spec.pack_index is not None
            name = _scale_name(spec.source_name)
            shard, scale = self._tensor(name)
            if scale.dtype != "F32" or scale.shape != scale_shape[1:]:
                raise ValueError(f"invalid expert scale metadata for {name!r}")
            self._prepare_scale_envelope(shard)
            payload = self._full_payload(shard, scale)
            host_scales[spec.pack_index] = _array_from_payload(payload, scale)
            scale_names.append(name)
        scales = self._replicate_host_array(host_scales)
        del host_scales
        gc.collect()
        value = BatchedBlockFP8LinearKernel(bits, scales, compute_dtype=jnp.bfloat16)
        return LoadedTarget(
            ordered[0].target_path,
            value,
            tuple(spec.source_name for spec in ordered),
            tuple(scale_names),
        )

    def target_groups(
        self,
    ) -> tuple[tuple[tuple[str | int, ...], tuple[Glm53CheckpointTensorSpec, ...]], ...]:
        grouped: dict[tuple[str | int, ...], list[Glm53CheckpointTensorSpec]] = defaultdict(list)
        for spec in checkpoint_text_tensor_specs(self.config):
            grouped[spec.target_path].append(spec)
        return tuple(
            (path, tuple(members))
            for path, members in sorted(grouped.items(), key=lambda item: _format_path(item[0]))
        )

    def load_target(
        self,
        specs: Sequence[Glm53CheckpointTensorSpec],
    ) -> LoadedTarget:
        if not specs or len({spec.target_path for spec in specs}) != 1:
            raise ValueError("load_target requires one non-empty executable target group")
        path = specs[0].target_path
        if path in self._loaded_paths:
            raise ValueError(f"executable target {_format_path(path)!r} was already loaded")
        self.prepare(specs)
        if specs[0].transform == "expert_transpose":
            loaded = self._load_expert_target(specs)
        else:
            if len(specs) != 1:
                raise ValueError(f"non-expert target {_format_path(path)!r} has multiple sources")
            loaded = self._load_single_target(specs[0])
        duplicate_logical = self._loaded_logical_names.intersection(loaded.logical_source_names)
        duplicate_scales = self._loaded_scale_names.intersection(loaded.scale_source_names)
        if duplicate_logical or duplicate_scales:
            raise ValueError(
                f"streaming target repeats source tensors: {sorted(duplicate_logical | duplicate_scales)[:10]}"
            )
        self._loaded_paths.add(path)
        self._loaded_logical_names.update(loaded.logical_source_names)
        self._loaded_scale_names.update(loaded.scale_source_names)
        return loaded

    def _assemble_tree(self, values: Mapping[tuple[str | int, ...], Any]) -> dict[str, Any]:
        root: dict[str, Any] = {"layers": [{} for _ in range(self.config.num_hidden_layers)]}
        for path, value in values.items():
            if path[0] == "layers":
                if len(path) < 3 or not isinstance(path[1], int):
                    raise ValueError(f"invalid layer target path {path!r}")
                node = root["layers"][path[1]]
                remainder = path[2:]
            else:
                node = root
                remainder = path
            for component in remainder[:-1]:
                if not isinstance(component, str):
                    raise ValueError(f"unexpected integer target component in {path!r}")
                node = node.setdefault(component, {})
                if not isinstance(node, dict):
                    raise ValueError(f"target path collision at {path!r}")
            leaf = remainder[-1]
            if not isinstance(leaf, str) or leaf in node:
                raise ValueError(f"duplicate or malformed target leaf {path!r}")
            node[leaf] = value
        root["layers"] = tuple(root["layers"])
        return root

    def load_parameters(self) -> dict[str, Any]:
        groups = self.target_groups()
        self.prepare_all()
        values = {}
        for index, (path, specs) in enumerate(groups, start=1):
            loaded = self.load_target(specs)
            values[path] = loaded.value
            if self.progress is not None:
                self.progress(
                    {
                        "event": "target_ready",
                        "target": _format_path(path),
                        "target_index": index,
                        "target_count": len(groups),
                        "logical_source_count": len(self._loaded_logical_names),
                    }
                )
        expected_logical = {spec.source_name for spec in checkpoint_text_tensor_specs(self.config)}
        expected_scales = {
            _scale_name(name)
            for name in expected_logical
            if self._headers[name][1].dtype == "F8_E4M3"
        }
        if self._loaded_logical_names != expected_logical:
            raise ValueError("complete streaming load did not consume every logical text tensor once")
        if self._loaded_scale_names != expected_scales:
            raise ValueError("complete streaming load did not consume every text scale tensor once")
        if len(values) != 1372:
            raise ValueError(f"complete streaming tree has {len(values)} targets, expected 1372")
        return self._assemble_tree(values)

    def release_host_cache(self) -> None:
        self._scale_envelopes.clear()
        gc.collect()

    def network_summary(self) -> dict[str, Any]:
        records = [record for reader in self._readers.values() for record in reader.records]
        return {
            "prepared_shard_count": len(self._shard_headers),
            "request_count_including_resolves": len(records),
            "bytes_read_including_resolves": sum(record.bytes_read for record in records),
            "largest_request_bytes": max((record.bytes_read for record in records), default=0),
            "requests_by_category": dict(sorted(self._category_requests.items())),
            "bytes_by_category": dict(sorted(self._category_bytes.items())),
            "loaded_target_count": len(self._loaded_paths),
            "loaded_logical_tensor_count": len(self._loaded_logical_names),
            "loaded_scale_tensor_count": len(self._loaded_scale_names),
            "maximum_expert_host_buffer_bytes": self._maximum_expert_host_buffer_bytes,
        }


__all__ = ["Glm53StreamingLoader", "LoadedTarget"]
