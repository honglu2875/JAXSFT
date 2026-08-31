"""Atomic rank-local checkpoints for globally sharded JAX PyTrees.

The format stores each unique addressable index once, so replicated device
copies are not multiplied inside a host checkpoint. A manifest pins the global
shape, dtype, NamedSharding contract, local indexes, and SHA-256 of every
payload before reconstruction with :func:`jax.make_array_from_callback`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.sharding import NamedSharding


SCHEMA_VERSION = 1


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_unique_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _encode_partition_axis(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, tuple) and all(isinstance(axis, str) for axis in value):
        return list(value)
    raise TypeError(f"unsupported PartitionSpec axis {value!r}")


def _sharding_record(value: Any) -> dict[str, Any]:
    sharding = value.sharding
    if not isinstance(sharding, NamedSharding):
        raise TypeError("sharded checkpoints require NamedSharding on every leaf")
    mesh_devices = np.asarray(sharding.mesh.devices, dtype=object)
    return {
        "mesh_axis_names": list(sharding.mesh.axis_names),
        "mesh_shape": {name: int(size) for name, size in sharding.mesh.shape.items()},
        "mesh_device_ids": [int(device.id) for device in mesh_devices.reshape(-1)],
        "partition_spec": [_encode_partition_axis(axis) for axis in sharding.spec],
    }


def _encode_index(index: Any) -> list[dict[str, int | str | None]] | None:
    if index is None:
        return None
    encoded: list[dict[str, int | str | None]] = []
    for part in index:
        if isinstance(part, slice):
            if part.step not in (None, 1):
                raise ValueError("checkpoint indexes must be contiguous")
            encoded.append(
                {
                    "kind": "slice",
                    "start": part.start,
                    "stop": part.stop,
                }
            )
        elif isinstance(part, int):
            encoded.append({"kind": "integer", "value": part})
        else:
            raise TypeError(f"unsupported shard index component {part!r}")
    return encoded


def _index_key(index: Any) -> str:
    return json.dumps(_encode_index(index), sort_keys=True, separators=(",", ":"))


def _canonical_payload_hash(records: Sequence[tuple[str, str, str]]) -> str:
    payload = json.dumps(sorted(records), separators=(",", ":")).encode()
    return _sha256(payload)


def _copied_c_array(value: Any) -> np.ndarray:
    # np.ascontiguousarray promotes scalars to shape (1,), which breaks scalar
    # replicated state such as the Adam step counter.
    return np.array(np.asarray(value), copy=True, order="C", subok=False)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint file {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_local_sharded_pytree(
    tree: Mapping[str, Any],
    directory: str | Path,
    *,
    process_index: int,
    process_count: int,
    identity: Mapping[str, Any],
    allowed_root_keys: Sequence[str],
) -> dict[str, Any]:
    """Atomically save one process's unique addressable shards.

    The caller supplies an explicit root allowlist so a frozen base tree cannot
    silently enter an adapter-only checkpoint.
    """

    if process_count <= 0 or not 0 <= process_index < process_count:
        raise ValueError("checkpoint process topology is invalid")
    if set(tree) != set(allowed_root_keys):
        raise ValueError(
            f"checkpoint roots {sorted(tree)} do not match allowlist {sorted(allowed_root_keys)}"
        )
    # Round-trip through strict JSON now, before any device-to-host transfer.
    identity_value = json.loads(json.dumps(identity, sort_keys=True))
    if not isinstance(identity_value, dict):
        raise TypeError("checkpoint identity must be a JSON object")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    npz_path = directory / f"rank-{process_index:03d}.npz"
    manifest_path = directory / f"rank-{process_index:03d}.json"

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    arrays: dict[str, np.ndarray] = {}
    leaves: list[dict[str, Any]] = []
    local_records: list[tuple[str, str, str]] = []
    replicated_records: list[tuple[str, str, str]] = []
    sharded_records: list[tuple[str, str, str]] = []
    global_elements = 0
    global_logical_bytes = 0
    local_unique_bytes = 0
    local_device_resident_bytes = 0
    for leaf_index, (path, value) in enumerate(path_leaves):
        if not isinstance(value, jax.Array):
            raise TypeError("sharded checkpoint trees must contain only JAX array leaves")
        path_text = jax.tree_util.keystr(path)
        sharding = _sharding_record(value)
        global_elements += int(value.size)
        global_logical_bytes += int(value.size * value.dtype.itemsize)
        by_index: dict[str, dict[str, Any]] = {}
        global_index_count = len(
            {
                _index_key(index)
                for index in value.sharding.devices_indices_map(value.shape).values()
            }
        )
        for shard in value.addressable_shards:
            data = _copied_c_array(shard.data)
            local_device_resident_bytes += int(data.nbytes)
            index_key = _index_key(shard.index)
            existing = by_index.get(index_key)
            if existing is not None:
                stored_payload = arrays[existing["array_key"]]
                if (
                    existing["dtype"] != str(data.dtype)
                    or existing["shape"] != list(data.shape)
                    or stored_payload.tobytes(order="C") != data.tobytes(order="C")
                ):
                    raise ValueError(f"replicated local shards disagree for {path_text}")
                existing["addressable_device_ids"].append(int(shard.device.id))
                continue
            array_key = f"leaf_{leaf_index:04d}_shard_{len(by_index):03d}"
            raw_payload = data.tobytes(order="C")
            arrays[array_key] = np.frombuffer(raw_payload, dtype=np.uint8).copy()
            digest = _sha256(raw_payload)
            record = {
                "addressable_device_ids": [int(shard.device.id)],
                "array_key": array_key,
                "dtype": str(data.dtype),
                "index": _encode_index(shard.index),
                "nbytes": int(data.nbytes),
                "sha256": digest,
                "shape": list(data.shape),
            }
            by_index[index_key] = record
            canonical = (path_text, index_key, digest)
            local_records.append(canonical)
            if global_index_count == 1:
                replicated_records.append(canonical)
            else:
                sharded_records.append(canonical)
            local_unique_bytes += int(data.nbytes)
        if not by_index:
            raise ValueError(f"checkpoint leaf {path_text} has no addressable shards")
        for record in by_index.values():
            record["addressable_device_ids"].sort()
        leaves.append(
            {
                "dtype": str(value.dtype),
                "global_nbytes": int(value.size * value.dtype.itemsize),
                "global_shape": list(value.shape),
                "leaf_index": leaf_index,
                "local_shards": list(by_index.values()),
                "path": path_text,
                "sharding": sharding,
            }
        )

    temporary_npz = directory / f".{npz_path.name}.tmp-{os.getpid()}"
    if npz_path.exists() or temporary_npz.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint rank {process_index}")
    try:
        with temporary_npz.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_npz, npz_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
    npz_payload = npz_path.read_bytes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "jaxsft_rank_local_sharded_pytree_npz",
        "identity": identity_value,
        "process_index": process_index,
        "process_count": process_count,
        "root_keys": sorted(tree),
        "leaf_count": len(leaves),
        "global_elements_including_replicas_once": global_elements,
        "global_logical_bytes_including_replicas_once": global_logical_bytes,
        "local_unique_shard_count": len(arrays),
        "local_unique_tensor_bytes": local_unique_bytes,
        "local_device_resident_bytes": local_device_resident_bytes,
        "local_payload_sha256": _canonical_payload_hash(local_records),
        "replicated_payload_sha256": _canonical_payload_hash(replicated_records),
        "sharded_payload_sha256": _canonical_payload_hash(sharded_records),
        "npz_file": npz_path.name,
        "npz_file_bytes": len(npz_payload),
        "npz_sha256": _sha256(npz_payload),
        "leaves": leaves,
    }
    manifest_payload = _json_bytes(manifest)
    _atomic_write(manifest_path, manifest_payload)
    _fsync_directory(directory)
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "format",
            "identity",
            "process_index",
            "process_count",
            "root_keys",
            "leaf_count",
            "global_elements_including_replicas_once",
            "global_logical_bytes_including_replicas_once",
            "local_unique_shard_count",
            "local_unique_tensor_bytes",
            "local_device_resident_bytes",
            "local_payload_sha256",
            "replicated_payload_sha256",
            "sharded_payload_sha256",
            "npz_file_bytes",
            "npz_sha256",
        )
    } | {
        "manifest_file": manifest_path.name,
        "manifest_file_bytes": len(manifest_payload),
        "manifest_sha256": _sha256(manifest_payload),
    }


def restore_local_sharded_pytree(
    template: Mapping[str, Any],
    directory: str | Path,
    *,
    process_index: int,
    process_count: int,
    identity: Mapping[str, Any],
    allowed_root_keys: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore and byte-validate one process's addressable global-array shards."""

    if set(template) != set(allowed_root_keys):
        raise ValueError("restore template roots do not match the explicit allowlist")
    directory = Path(directory)
    manifest_path = directory / f"rank-{process_index:03d}.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = _load_unique_json(manifest_path)
    expected_identity = json.loads(json.dumps(identity, sort_keys=True))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("format") != "jaxsft_rank_local_sharded_pytree_npz"
        or manifest.get("identity") != expected_identity
        or manifest.get("process_index") != process_index
        or manifest.get("process_count") != process_count
        or manifest.get("root_keys") != sorted(template)
    ):
        raise ValueError("checkpoint manifest identity or topology drifted")
    npz_name = manifest.get("npz_file")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise ValueError("checkpoint manifest contains an unsafe NPZ path")
    npz_path = directory / npz_name
    npz_payload = npz_path.read_bytes()
    if (
        manifest.get("npz_file_bytes") != len(npz_payload)
        or manifest.get("npz_sha256") != _sha256(npz_payload)
    ):
        raise ValueError("checkpoint NPZ size or SHA-256 does not match its manifest")

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = manifest.get("leaves")
    if not isinstance(leaves, list) or len(leaves) != len(path_leaves):
        raise ValueError("checkpoint leaf count does not match the restore template")
    with np.load(npz_path, allow_pickle=False) as archive:
        expected_keys = {
            shard["array_key"] for leaf in leaves for shard in leaf.get("local_shards", [])
        }
        if set(archive.files) != expected_keys:
            raise ValueError("checkpoint NPZ members do not exactly match the manifest")
        host_payloads = {key: _copied_c_array(archive[key]) for key in expected_keys}

    restored_leaves = []
    local_records: list[tuple[str, str, str]] = []
    for expected_index, ((path, template_value), leaf) in enumerate(zip(path_leaves, leaves)):
        path_text = jax.tree_util.keystr(path)
        if (
            leaf.get("leaf_index") != expected_index
            or leaf.get("path") != path_text
            or leaf.get("global_shape") != list(template_value.shape)
            or leaf.get("dtype") != str(template_value.dtype)
            or leaf.get("global_nbytes")
            != int(template_value.size * template_value.dtype.itemsize)
            or leaf.get("sharding") != _sharding_record(template_value)
        ):
            raise ValueError(f"checkpoint leaf contract drifted at {path_text}")
        local_shards = leaf.get("local_shards")
        if not isinstance(local_shards, list) or not local_shards:
            raise ValueError(f"checkpoint leaf {path_text} has no local shards")
        by_index: dict[str, np.ndarray] = {}
        expected_devices: dict[str, list[int]] = {}
        for shard in local_shards:
            index_key = json.dumps(shard.get("index"), sort_keys=True, separators=(",", ":"))
            array_key = shard.get("array_key")
            if index_key in by_index or not isinstance(array_key, str):
                raise ValueError(f"checkpoint leaf {path_text} repeats a local index")
            payload = host_payloads[array_key]
            shape = shard.get("shape")
            nbytes = shard.get("nbytes")
            if (
                payload.dtype != np.uint8
                or payload.ndim != 1
                or not isinstance(shape, list)
                or not isinstance(nbytes, int)
                or payload.size != nbytes
            ):
                raise ValueError(f"checkpoint raw payload container drifted at {path_text}")
            raw_payload = payload.tobytes(order="C")
            try:
                data = np.frombuffer(raw_payload, dtype=np.dtype(template_value.dtype)).reshape(
                    tuple(shape)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"checkpoint payload cannot reconstruct {path_text}"
                ) from error
            data = _copied_c_array(data)
            digest = _sha256(raw_payload)
            if (
                shard.get("dtype") != str(data.dtype)
                or shard.get("shape") != list(data.shape)
                or shard.get("nbytes") != int(data.nbytes)
                or shard.get("sha256") != digest
            ):
                raise ValueError(f"checkpoint payload metadata drifted at {path_text}")
            by_index[index_key] = data
            expected_devices[index_key] = shard.get("addressable_device_ids")
            local_records.append((path_text, index_key, digest))

        def callback(index: Any, *, _by_index: Mapping[str, np.ndarray] = by_index) -> np.ndarray:
            try:
                return _by_index[_index_key(index)]
            except KeyError as error:
                raise ValueError(f"checkpoint has no local payload for index {index!r}") from error

        restored = jax.make_array_from_callback(
            template_value.shape,
            template_value.sharding,
            callback,
            dtype=template_value.dtype,
        )
        jax.block_until_ready(restored)
        actual_by_index: dict[str, tuple[np.ndarray, list[int]]] = {}
        for shard in restored.addressable_shards:
            key = _index_key(shard.index)
            data = _copied_c_array(shard.data)
            if key in actual_by_index:
                previous, device_ids = actual_by_index[key]
                if not np.array_equal(previous, data):
                    raise ValueError(f"restored replicated shards disagree at {path_text}")
                device_ids.append(int(shard.device.id))
            else:
                actual_by_index[key] = (data, [int(shard.device.id)])
        if set(actual_by_index) != set(by_index):
            raise ValueError(f"restored local indexes drifted at {path_text}")
        for key, (data, device_ids) in actual_by_index.items():
            if (
                sorted(device_ids) != expected_devices[key]
                or data.dtype != by_index[key].dtype
                or data.shape != by_index[key].shape
                or not np.array_equal(data, by_index[key])
            ):
                raise ValueError(f"restored local payload differs at {path_text}")
        restored_leaves.append(restored)

    if _canonical_payload_hash(local_records) != manifest.get("local_payload_sha256"):
        raise ValueError("checkpoint canonical local payload hash drifted")
    restored_tree = jax.tree_util.tree_unflatten(treedef, restored_leaves)
    return restored_tree, {
        "manifest_sha256": _sha256(manifest_payload),
        "npz_sha256": manifest["npz_sha256"],
        "local_payload_sha256": manifest["local_payload_sha256"],
        "leaf_count": len(restored_leaves),
        "local_unique_shard_count": manifest["local_unique_shard_count"],
        "all_local_shards_byte_exact": True,
    }


__all__ = ["restore_local_sharded_pytree", "save_local_sharded_pytree"]
