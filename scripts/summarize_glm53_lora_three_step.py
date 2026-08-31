#!/usr/bin/env python3
"""Validate G6c1 GLM LoRA AdamW execution and its complete sharded checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.probe_glm53_lora_three_step import (
    EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
    EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES,
    EXPECTED_CHECKPOINT_LEAF_COUNT,
    EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES,
    EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES,
    EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
    TRAINING_STATISTIC_NAMES,
)
from scripts.summarize_glm53_lora_backward import (
    EXPECTED_BASE_BYTES_PER_DEVICE,
    EXPECTED_INITIALIZER_MEMORY,
    EXPECTED_LOADER_BYTES_PER_HOST,
)
from scripts.summarize_glm53_lora_backward_compile import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHAPE_MENTIONS,
)
from scripts.summarize_glm53_lora_optimizer_compile import (
    EXPECTED_COMPILER_MEMORY,
    EXPECTED_EXECUTION_GATE,
)


GIB = 1024**3
SOURCE_TEST = "glm53_complete_text_rank4_attention_lora_adamw_three_step_v4_probe"
EXPECTED_AUTHORIZING_EVIDENCE_SHA256 = (
    "abb9bfcee8d96baad736932250eb1c8aad4bb51d9e66e761ce88f4d385d9067b"
)
EXPECTED_HLO_SHA256 = "d92954cee93966bd540b5a443b8dcd8a57fb2d12585e2c71248f43ccc6d73fe5"
EXPECTED_HLO_BYTES = 168_571_889
EXPECTED_OPTIMIZER_INITIALIZER_MEMORY = {
    "alias_size_in_bytes": 0,
    "argument_size_in_bytes": 0,
    "generated_code_size_in_bytes": 2_610_688,
    "host_argument_size_in_bytes": 0,
    "host_output_size_in_bytes": 0,
    "host_temp_size_in_bytes": 0,
    "output_size_in_bytes": 34_868_736,
    "temp_size_in_bytes": 0,
}
EXPECTED_INITIALIZATION_STATISTICS = [
    1.0,
    254.57638549804688,
    0.0,
    26_206.02734375,
    0.0,
    0.044189453125,
    0.0,
    3_956_735.0,
    0.0,
    191.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
EXPECTED_INITIALIZATION_SHA256 = (
    "5b231c53a8aff19c54bec5738231f75e8ab6b858400e31355392f896bd916873"
)
EXPECTED_STEP_RECORDS = [
    {
        "step": 1,
        "loss": 12.241244316101074,
        "loss_float32_sha256": (
            "ab4404e2ca8a94e73b62eef6c38d4a2e54f41f21d346d58230ba935c099c9101"
        ),
        "gradient_norm_before_clipping": 2.7054572105407715,
        "gradient_norm_float32_sha256": (
            "9ed4c6278fb6581e984405bf5c8352a23f67e3ed795396f82d7efbc4861bd742"
        ),
        "training_statistics": [
            1.0,
            254.57638549804688,
            0.00043387856567278504,
            26_206.02734375,
            44.85679244995117,
            0.044189453125,
            1.0013580322265625e-05,
            3_956_735.0,
            4_821_397.0,
            191.0,
            169.0,
            0.009973919950425625,
            54.15019226074219,
            0.005371284671127796,
            4_821_397.0,
            169.0,
            1.8073325236400706e-07,
            0.049869611859321594,
            0.00014425348490476608,
            4_819_796.0,
            169.0,
            1.0,
        ],
        "training_statistics_float32_sha256": (
            "b51c74bcd7e13090535352eb645837bbeedacbca88ab04b6b2bdce1fb94a3304"
        ),
    },
    {
        "step": 2,
        "loss": 12.234691619873047,
        "loss_float32_sha256": (
            "8bd81399a17588a44c6395cea3d8b22282cb9dff4cf64005d85344f992d59380"
        ),
        "gradient_norm_before_clipping": 2.7605135440826416,
        "gradient_norm_float32_sha256": (
            "144af77099418ca0ab982020c1568dcd2deeffd2f4a273cfca64f674a5d8ce11"
        ),
        "training_statistics": [
            1.0,
            254.5763702392578,
            0.0017276438884437084,
            26_206.021484375,
            89.41966247558594,
            0.044189453125,
            2.002716064453125e-05,
            3_956_736.0,
            4_821_678.0,
            191.0,
            169.0,
            0.03593651205301285,
            104.48735046386719,
            0.010268192738294601,
            8_530_605.0,
            338.0,
            6.862010764052684e-07,
            0.09712139517068863,
            0.0002846845891326666,
            8_529_017.0,
            338.0,
            2.0,
        ],
        "training_statistics_float32_sha256": (
            "82e4f02cd7c41aa8d8ebfa942dbac2cbde0c8c48e815d017c78c83acfff1f0f2"
        ),
    },
    {
        "step": 3,
        "loss": 12.266050338745117,
        "loss_float32_sha256": (
            "275e08a1dd4a898594752ce0d3e559b36a814dd4ba209d09b7e36abcdc85e3cf"
        ),
        "gradient_norm_before_clipping": 2.7583279609680176,
        "gradient_norm_float32_sha256": (
            "ea6a2241321d611c9500f729679b5eea799f0f3d6af3142630a38eafc4f30d36"
        ),
        "training_statistics": [
            1.0,
            254.57644653320312,
            0.0038795992732048035,
            26_206.048828125,
            133.9324493408203,
            0.044189453125,
            3.0040740966796875e-05,
            3_956_736.0,
            4_821_679.0,
            191.0,
            169.0,
            0.07326149940490723,
            151.81369018554688,
            0.014775736257433891,
            8_530_604.0,
            338.0,
            1.4909427363818395e-06,
            0.1424681693315506,
            0.000423596182372421,
            8_529_018.0,
            338.0,
            3.0,
        ],
        "training_statistics_float32_sha256": (
            "4a0efd4d20d4353c680e19cceee5d74f1be51710f6ef00aa2c9a0cea861e3d98"
        ),
    },
]
EXPECTED_REPLICATED_CHECKPOINT_SHA256 = (
    "47cb2e31c06172eeaf93d1f6bbc841b7126af0c71467673af5e3a4807723f4c0"
)
EXPECTED_GLOBAL_CHECKPOINT_SHA256 = (
    "30f9cf81f7162b46157e5d7a5a6d18755464d0709972e1fd1fdb95ec5ccd655a"
)
EXPECTED_CHECKPOINT_UNIQUE_GLOBAL_SHARDS = 9_742
EXPECTED_RESULT_SHA256 = {
    "c4007df084f4059419126e09b8103d6a4b0040d3809c3a5b887897045adb6723",
    "083fd01391e03a3ceac678bc8650c1fc861f78e1601bc98b23cdfa87ed0d241c",
    "ca81c774917abe2477edaefe040e432e8b3e63fe61c2ce246dcf0b87d620d08d",
    "e955d48fb9a587c7f8f88d1e331f92923b7effb5e7bb6669101c63edafad6eaf",
}
EXPECTED_CHECKPOINT_ARTIFACTS = {
    0: {
        "manifest_file_bytes": 2_569_484,
        "manifest_sha256": "23d0cbcdba88cdb333d7b58eee49ebbc5730d993c9eeec1ee2eace04fccd9f12",
        "npz_sha256": "5d1b1c0aa1ce1e689fc040b959630445ee9ed921fabbca6cfde2567f17db405e",
        "local_payload_sha256": "6bd761d749c541a2f2c2937f64b5b508a3a74cb3cd58670029b689a36d7f4107",
        "sharded_payload_sha256": "a78b203ab2b94f5b9299342aab6875ae7c9c6269d166bc765bc044a19816fd89",
    },
    1: {
        "manifest_file_bytes": 2_573_054,
        "manifest_sha256": "e01f8d1577b5c3a0ec648252b0fe40b92df731f27b06c3bab205dcad6a574e44",
        "npz_sha256": "b07b326976b85549b8c330e6cf4787dda844d5e349c66287b8f3f781afb707b6",
        "local_payload_sha256": "330c54098ee06239e4b8aaa54812fb0bbcb6719b4b836d756d45bb6961f8e4e9",
        "sharded_payload_sha256": "398f938d85aaf734120a169d29b1a9f53ed74da2194a72709f98e8c3453ef051",
    },
    2: {
        "manifest_file_bytes": 2_575_645,
        "manifest_sha256": "21e4b69ca0c4052e22a03e84d4a974e70209d85a8be19d0ccff3023c812125fa",
        "npz_sha256": "4df4fe939fab4e9a95708b6c76858ca6c760b5e26f829fe948c14ecbb7512df5",
        "local_payload_sha256": "c878a9ef5eec584e37912819f7dadaec13e9b85c65da49d4cc9e219fc381f8ad",
        "sharded_payload_sha256": "9475e9c363c797bb5a0116f1bf35e559240b079cbaf19d8032a06791cdff4c8c",
    },
    3: {
        "manifest_file_bytes": 2_578_203,
        "manifest_sha256": "df5e894893b651c9f2b42abe101bff51ff62ad06bf221db990b3a76f4365d955",
        "npz_sha256": "2d2408a28568b4c34ed99f32f23682529c452602efd93c8a23760345263ca576",
        "local_payload_sha256": "bff14c22342e4cca3081f3e7078b4a125189467c4ea55cbedba508165b49d033",
        "sharded_payload_sha256": "9aa18318915b9a3f3dc28008d07c4ba8c5c93ee2af1dd009b9fd79337412e807",
    },
}


def _load(path: Path) -> tuple[dict[str, Any], str, int]:
    payload = path.read_bytes()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest(), len(payload)


def _hash_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return digest, path.stat().st_size


def _index_key(index: Any) -> str:
    return json.dumps(index, sort_keys=True, separators=(",", ":"))


def _canonical_payload_hash(records: Sequence[tuple[str, str, str]]) -> str:
    payload = json.dumps(sorted(records), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _full_slice() -> dict[str, int | str | None]:
    return {"kind": "slice", "start": None, "stop": None}


def _expected_shard(
    global_shape: Sequence[int],
    partition_spec: Sequence[str | None],
    device_id: int,
) -> tuple[list[dict[str, int | str | None]], list[int]]:
    partition_spec = list(partition_spec)
    if partition_spec == []:
        return [_full_slice() for _ in global_shape], list(global_shape)
    if partition_spec != [None, "model"] or len(global_shape) != 2:
        raise ValueError("checkpoint contains an unsupported sharding contract")
    if global_shape[1] % 16:
        raise ValueError("checkpoint output dimension is not divisible by the 16-way mesh")
    width = global_shape[1] // 16
    return [
        _full_slice(),
        {"kind": "slice", "start": device_id * width, "stop": (device_id + 1) * width},
    ], [global_shape[0], width]


def _validate_checkpoint_artifacts(
    manifest_paths: Sequence[Path],
    payload_paths: Sequence[Path],
    *,
    source_revision: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if len(manifest_paths) != 4 or len(payload_paths) != 4:
        raise ValueError("exactly four checkpoint manifests and payloads are required")
    payloads_by_hash: dict[str, tuple[Path, int]] = {}
    for path in payload_paths:
        digest, size = _hash_file(path)
        if digest in payloads_by_hash:
            raise ValueError("checkpoint payload SHA-256 appears more than once")
        payloads_by_hash[digest] = (path, size)

    manifests: dict[int, dict[str, Any]] = {}
    contracts_by_rank: dict[int, list[dict[str, Any]]] = {}
    global_shards: dict[tuple[str, str], dict[str, Any]] = {}
    global_devices_by_shard: dict[tuple[str, str], set[int]] = defaultdict(set)
    total_manifest_bytes = 0
    total_npz_bytes = 0
    for manifest_path in manifest_paths:
        manifest, manifest_sha256, manifest_bytes = _load(manifest_path)
        rank = manifest.get("process_index")
        if not isinstance(rank, int) or rank not in range(4) or rank in manifests:
            raise ValueError("checkpoint manifests do not uniquely cover four process indexes")
        expected_artifact = EXPECTED_CHECKPOINT_ARTIFACTS[rank]
        if (
            manifest_sha256 != expected_artifact["manifest_sha256"]
            or manifest_bytes != expected_artifact["manifest_file_bytes"]
        ):
            raise ValueError("checkpoint manifest file identity drifted")
        expected_identity = {
            "format_purpose": "GLM-5.3-Flash rank-4 attention-LoRA adapter-only AdamW",
            "model_repo_id": "zai-org/GLM-5.3-Flash",
            "model_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
            "source_revision": source_revision,
            "step": 2,
        }
        expected_common = {
            "schema_version": 1,
            "format": "jaxsft_rank_local_sharded_pytree_npz",
            "identity": expected_identity,
            "process_index": rank,
            "process_count": 4,
            "root_keys": ["adapters", "optimizer"],
            "leaf_count": EXPECTED_CHECKPOINT_LEAF_COUNT,
            "global_elements_including_replicas_once": EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
            "global_logical_bytes_including_replicas_once": (
                EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES
            ),
            "local_unique_shard_count": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
            "local_unique_tensor_bytes": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES,
            "local_device_resident_bytes": EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES,
            "replicated_payload_sha256": EXPECTED_REPLICATED_CHECKPOINT_SHA256,
            "npz_file": f"rank-{rank:03d}.npz",
            "npz_file_bytes": 56_172_246,
        }
        if {key: manifest.get(key) for key in expected_common} != expected_common:
            raise ValueError("checkpoint manifest topology or inventory drifted")
        for name in ("local_payload_sha256", "sharded_payload_sha256", "npz_sha256"):
            if manifest.get(name) != expected_artifact[name]:
                raise ValueError(f"checkpoint manifest {name} drifted")
        try:
            payload_path, payload_size = payloads_by_hash[manifest["npz_sha256"]]
        except KeyError as error:
            raise ValueError("checkpoint manifest has no independently supplied NPZ payload") from error
        if payload_size != manifest["npz_file_bytes"]:
            raise ValueError("checkpoint NPZ file size drifted")

        leaves = manifest.get("leaves")
        if not isinstance(leaves, list) or len(leaves) != EXPECTED_CHECKPOINT_LEAF_COUNT:
            raise ValueError("checkpoint manifest leaf inventory drifted")
        rank_devices = set(range(rank * 4, rank * 4 + 4))
        local_records: list[tuple[str, str, str]] = []
        replicated_records: list[tuple[str, str, str]] = []
        sharded_records: list[tuple[str, str, str]] = []
        local_device_ids: set[int] = set()
        local_array_keys: set[str] = set()
        local_tensor_bytes = 0
        local_device_resident_bytes = 0
        contracts: list[dict[str, Any]] = []
        expected_archive_members: dict[str, dict[str, Any]] = {}
        dtype_leaf_counts = Counter()
        root_leaf_counts = Counter()
        for leaf_index, leaf in enumerate(leaves):
            path = leaf.get("path")
            shape = leaf.get("global_shape")
            dtype = leaf.get("dtype")
            sharding = leaf.get("sharding")
            if (
                leaf.get("leaf_index") != leaf_index
                or not isinstance(path, str)
                or not isinstance(shape, list)
                or any(not isinstance(size, int) or size <= 0 for size in shape)
                or dtype not in {"bfloat16", "float32", "int32"}
                or not isinstance(sharding, dict)
            ):
                raise ValueError("checkpoint leaf identity, shape, or dtype is malformed")
            if path.startswith("['adapters']"):
                expected_dtype = "bfloat16"
                root_leaf_counts["adapters"] += 1
            elif path == "['optimizer'].step":
                expected_dtype = "int32"
                root_leaf_counts["optimizer_step"] += 1
            elif path.startswith("['optimizer'].first_moment"):
                expected_dtype = "float32"
                root_leaf_counts["first_moment"] += 1
            elif path.startswith("['optimizer'].second_moment"):
                expected_dtype = "float32"
                root_leaf_counts["second_moment"] += 1
            else:
                raise ValueError("checkpoint contains a leaf outside adapter-only roots")
            if dtype != expected_dtype:
                raise ValueError("checkpoint leaf dtype disagrees with adapter/Adam contract")
            expected_partition = [None, "model"] if path.endswith("['b']") else []
            expected_sharding = {
                "mesh_axis_names": ["model"],
                "mesh_shape": {"model": 16},
                "mesh_device_ids": list(range(16)),
                "partition_spec": expected_partition,
            }
            if sharding != expected_sharding:
                raise ValueError("checkpoint NamedSharding contract drifted")
            itemsize = {"bfloat16": 2, "float32": 4, "int32": 4}[dtype]
            global_nbytes = math.prod(shape) * itemsize
            if leaf.get("global_nbytes") != global_nbytes:
                raise ValueError("checkpoint global leaf bytes drifted")
            local_shards = leaf.get("local_shards")
            expected_local_count = 1 if expected_partition == [] else 4
            if not isinstance(local_shards, list) or len(local_shards) != expected_local_count:
                raise ValueError("checkpoint local shard multiplicity drifted")
            leaf_devices: set[int] = set()
            for shard in local_shards:
                device_ids = shard.get("addressable_device_ids")
                expected_device_count = 4 if expected_partition == [] else 1
                if (
                    not isinstance(device_ids, list)
                    or len(device_ids) != expected_device_count
                    or len(set(device_ids)) != expected_device_count
                    or not set(device_ids) <= rank_devices
                ):
                    raise ValueError("checkpoint addressable device coverage drifted")
                leaf_devices.update(device_ids)
                representative = min(device_ids)
                expected_index, expected_shape = _expected_shard(
                    shape, expected_partition, representative
                )
                array_key = shard.get("array_key")
                digest = shard.get("sha256")
                if (
                    shard.get("index") != expected_index
                    or shard.get("shape") != expected_shape
                    or shard.get("dtype") != dtype
                    or shard.get("nbytes") != math.prod(expected_shape) * itemsize
                    or not isinstance(array_key, str)
                    or array_key in local_array_keys
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError("checkpoint local shard metadata drifted")
                local_array_keys.add(array_key)
                local_device_ids.update(device_ids)
                local_tensor_bytes += shard["nbytes"]
                local_device_resident_bytes += shard["nbytes"] * len(device_ids)
                index_key = _index_key(shard["index"])
                canonical = (path, index_key, digest)
                local_records.append(canonical)
                if expected_partition == []:
                    replicated_records.append(canonical)
                else:
                    sharded_records.append(canonical)
                global_key = (path, index_key)
                global_record = {
                    "sha256": digest,
                    "nbytes": shard["nbytes"],
                    "dtype": dtype,
                    "replicated": expected_partition == [],
                }
                existing = global_shards.setdefault(
                    global_key,
                    global_record,
                )
                if existing != global_record:
                    raise ValueError("replicated checkpoint payloads disagree across ranks")
                global_devices_by_shard[global_key].update(device_ids)
                expected_archive_members[array_key] = shard
            if leaf_devices != rank_devices:
                raise ValueError("checkpoint leaf does not cover every local device")
            dtype_leaf_counts[dtype] += 1
            contracts.append({key: leaf[key] for key in leaf if key != "local_shards"})

        if (
            root_leaf_counts
            != Counter(
                {
                    "adapters": 382,
                    "first_moment": 382,
                    "second_moment": 382,
                    "optimizer_step": 1,
                }
            )
            or dtype_leaf_counts != Counter({"float32": 764, "bfloat16": 382, "int32": 1})
            or local_device_ids != rank_devices
            or len(local_array_keys) != EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS
            or local_tensor_bytes != EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES
            or local_device_resident_bytes != EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES
            or _canonical_payload_hash(local_records) != manifest["local_payload_sha256"]
            or _canonical_payload_hash(replicated_records)
            != manifest["replicated_payload_sha256"]
            or _canonical_payload_hash(sharded_records) != manifest["sharded_payload_sha256"]
        ):
            raise ValueError("checkpoint local payload inventory or canonical hash drifted")
        with np.load(payload_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_archive_members):
                raise ValueError("checkpoint NPZ members do not exactly match the manifest")
            for array_key, shard in expected_archive_members.items():
                payload = np.asarray(archive[array_key])
                if (
                    payload.dtype != np.uint8
                    or payload.ndim != 1
                    or payload.size != shard["nbytes"]
                    or hashlib.sha256(payload.tobytes()).hexdigest() != shard["sha256"]
                ):
                    raise ValueError("checkpoint NPZ member bytes disagree with the manifest")
        manifests[rank] = manifest
        contracts_by_rank[rank] = contracts
        total_manifest_bytes += manifest_bytes
        total_npz_bytes += payload_size

    if set(manifests) != set(range(4)) or len(payloads_by_hash) != 4:
        raise ValueError("checkpoint artifacts do not cover four unique ranks")
    reference_contract = contracts_by_rank[0]
    if any(contracts_by_rank[rank] != reference_contract for rank in range(1, 4)):
        raise ValueError("checkpoint global leaf contract differs between ranks")
    global_devices_by_leaf: dict[str, set[int]] = defaultdict(set)
    for (path, _), devices in global_devices_by_shard.items():
        global_devices_by_leaf[path].update(devices)
    if (
        len(global_shards) != EXPECTED_CHECKPOINT_UNIQUE_GLOBAL_SHARDS
        or sum(record["nbytes"] for record in global_shards.values())
        != EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES
        or any(devices != set(range(16)) for devices in global_devices_by_leaf.values())
        or any(
            devices != (set(range(16)) if global_shards[key]["replicated"] else {next(iter(devices))})
            for key, devices in global_devices_by_shard.items()
        )
    ):
        raise ValueError("checkpoint artifacts do not exactly cover all 16 global device shards")
    global_records = [
        (path, index, record["sha256"])
        for (path, index), record in global_shards.items()
    ]
    global_sha256 = _canonical_payload_hash(global_records)
    if global_sha256 != EXPECTED_GLOBAL_CHECKPOINT_SHA256:
        raise ValueError("global adapter-only checkpoint payload hash drifted")
    return (
        {
            "format": "jaxsft_rank_local_sharded_pytree_npz",
            "checkpoint_step": 2,
            "root_keys": ["adapters", "optimizer"],
            "frozen_base_included": False,
            "leaf_count": EXPECTED_CHECKPOINT_LEAF_COUNT,
            "global_elements": EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
            "global_logical_tensor_bytes": EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES,
            "globally_unique_shard_count": EXPECTED_CHECKPOINT_UNIQUE_GLOBAL_SHARDS,
            "global_payload_sha256": global_sha256,
            "replicated_payload_sha256": EXPECTED_REPLICATED_CHECKPOINT_SHA256,
            "npz_bytes_across_hosts": total_npz_bytes,
            "manifest_bytes_across_hosts": total_manifest_bytes,
            "all_npz_member_hashes_verified": True,
            "all_global_device_shards_covered": True,
            "all_replicated_payloads_equal": True,
        },
        manifests,
    )


def summarize(
    rank_paths: Sequence[Path],
    manifest_paths: Sequence[Path],
    payload_paths: Sequence[Path],
    *,
    source_revision: str,
) -> dict[str, Any]:
    if len(rank_paths) != 4:
        raise ValueError("exactly four G6c1 rank results are required")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source_revision must be a full lowercase Git hash")
    checkpoint_evidence, manifests = _validate_checkpoint_artifacts(
        manifest_paths,
        payload_paths,
        source_revision=source_revision,
    )
    values_with_hashes = [_load(path) for path in rank_paths]
    if {digest for _, digest, _ in values_with_hashes} != EXPECTED_RESULT_SHA256:
        raise ValueError("G6c1 raw rank-result SHA-256 identity drifted")

    expected_model = {
        "alpha": 4.0,
        "attention_lora_target_count": 191,
        "batch_size": 1,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "index_sha256": EXPECTED_INDEX_SHA256,
        "input_ids": [[1, 2]],
        "loss_token_count": 1,
        "loss_weights": [[0.0, 1.0]],
        "num_hidden_layers": 45,
        "rank": 4,
        "rematerialize_each_decoder_layer": True,
        "repo_id": "zai-org/GLM-5.3-Flash",
        "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "seed": 0,
        "sequence_length": 2,
    }
    expected_optimizer = {
        "adapter_dtype": "bfloat16",
        "beta1": 0.9,
        "beta2": 0.95,
        "donated_argument_numbers": [1, 2],
        "epsilon": 1e-8,
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,
        "moment_dtype": "float32",
        "name": "AdamW",
        "weight_decay": 0.1,
    }
    expected_header_network = {
        "bytes_by_category": {"header": 10_684_096},
        "bytes_read_including_resolves": 10_684_158,
        "largest_request_bytes": 180_384,
        "loaded_logical_tensor_count": 0,
        "loaded_scale_tensor_count": 0,
        "loaded_target_count": 0,
        "maximum_expert_host_buffer_bytes": 0,
        "prepared_shard_count": 62,
        "request_count_including_resolves": 186,
        "requests_by_category": {"header": 124},
    }
    expected_adapter = {
        "a_partition_spec": [],
        "b_partition_spec": [None, "model"],
        "global_parameter_count": 10_289_152,
        "global_parameter_count_by_factor": {"a": 3_956_736, "b": 6_332_416},
        "parameter_bytes_by_device": {str(device): 8_705_024 for device in range(16)},
        "target_count": 191,
    }
    expected_optimizer_placement = {
        "first_moment_bytes_by_device": {
            str(device): 17_410_048 for device in range(16)
        },
        "moment_dtype": "float32",
        "moment_global_elements_per_slot": 10_289_152,
        "moment_slot_count": 2,
        "optimizer_state_bytes_by_device": {
            str(device): 34_820_100 for device in range(16)
        },
        "second_moment_bytes_by_device": {
            str(device): 17_410_048 for device in range(16)
        },
        "step_bytes_by_device": {str(device): 4 for device in range(16)},
        "step_dtype": "int32",
        "step_global_elements": 1,
    }
    expected_loader = {
        "bytes_by_category": {
            "axis0_tensor": 3_757_703_168,
            "expert_tensor": 76_101_451_776,
            "header": 10_684_096,
            "replicated_tensor": 193_428_312,
            "scale_envelope": 77_871_648,
        },
        "bytes_read_including_resolves": EXPECTED_LOADER_BYTES_PER_HOST,
        "largest_request_bytes": 79_298_560,
        "loaded_logical_tensor_count": 37_534,
        "loaded_scale_tensor_count": 36_467,
        "loaded_target_count": 1_372,
        "maximum_expert_host_buffer_bytes": 603_979_776,
        "prepared_shard_count": 62,
        "request_count_including_resolves": 38_847,
        "requests_by_category": {
            "axis0_tensor": 1_796,
            "expert_tensor": 36_288,
            "header": 124,
            "replicated_tensor": 516,
            "scale_envelope": 61,
        },
    }
    expected_base_placement = {
        "all_local_devices_match_header_audit": True,
        "array_leaf_count": 1_677,
        "expected_base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
        "global_leaf_elements_by_dtype": {
            "bfloat16": 6_303_463_936,
            "float32": 19_034_430,
            "uint8": 307_023_052_800,
        },
        "global_leaf_elements_including_scale_metadata": 313_345_551_166,
    }
    expected_checkpoint_identity = {
        "format_purpose": "GLM-5.3-Flash rank-4 attention-LoRA adapter-only AdamW",
        "model_repo_id": "zai-org/GLM-5.3-Flash",
        "model_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "source_revision": source_revision,
        "step": 2,
    }

    hostnames: set[str] = set()
    process_indexes: set[int] = set()
    device_ids: set[int] = set()
    maximum_process_vmhwm = 0
    maximum_device_peak = 0
    minimum_free_block = 33_014_407_168
    maximum_phase_peak: Counter[str] = Counter()
    maxima = {
        "header_seconds": 0.0,
        "compile_seconds": 0.0,
        "adapter_initializer_seconds": 0.0,
        "optimizer_initializer_seconds": 0.0,
        "load_seconds": 0.0,
        "checkpoint_save_seconds": 0.0,
        "checkpoint_restore_seconds": 0.0,
        "elapsed_seconds_before_shutdown": 0.0,
        "step_execute_seconds": 0.0,
        "step_diagnostics_seconds": 0.0,
    }
    required_memory_phases = {
        "after_checkpoint_restore",
        "after_checkpoint_save",
        "after_distributed_init",
        "after_full_base_placement",
        "after_optimizer_compile",
        "after_step_1",
        "after_step_2",
        "after_step_3",
        "after_training_state_initialization",
        "after_training_state_release",
    }
    result_hash_by_rank: dict[str, str] = {}
    for value, result_sha256, _ in values_with_hashes:
        if (
            value.get("schema_version") != 1
            or value.get("test") != SOURCE_TEST
            or value.get("source_revision") != source_revision
            or value.get("model") != expected_model
            or value.get("optimizer") != expected_optimizer
        ):
            raise ValueError("G6c1 schema, source, model, or optimizer recipe drifted")
        runtime = value.get("runtime", {})
        if (
            runtime.get("jax_version") != "0.11.0"
            or runtime.get("backend") != "tpu"
            or runtime.get("device_kinds") != ["TPU v4"]
            or runtime.get("process_count") != 4
            or runtime.get("local_device_count") != 4
            or runtime.get("global_device_count") != 16
            or runtime.get("mesh_shape") != {"model": 16}
            or runtime.get("precision") != "HIGHEST"
            or runtime.get("distributed_initialized") is not True
            or runtime.get("distributed_shutdown_complete") is not True
        ):
            raise ValueError("G6c1 topology, precision, or lifecycle drifted")
        hostname = runtime.get("hostname")
        rank = runtime.get("process_index")
        if not isinstance(hostname, str) or not isinstance(rank, int):
            raise ValueError("G6c1 runtime identity is malformed")
        hostnames.add(hostname)
        process_indexes.add(rank)
        result_hash_by_rank[str(rank)] = result_sha256

        preflight = value.get("compile_preflight", {})
        if (
            preflight.get("authorizing_evidence_sha256")
            != EXPECTED_AUTHORIZING_EVIDENCE_SHA256
            or preflight.get("header_network") != expected_header_network
            or preflight.get("checkpoint_payload_bytes_read") != 0
            or preflight.get("compiler_memory") != EXPECTED_COMPILER_MEMORY
            or preflight.get("execution_gate") != EXPECTED_EXECUTION_GATE
            or preflight.get("optimized_hlo_sha256") != EXPECTED_HLO_SHA256
            or preflight.get("optimized_hlo_bytes") != EXPECTED_HLO_BYTES
            or preflight.get("optimized_hlo_shape_mentions") != EXPECTED_SHAPE_MENTIONS
        ):
            raise ValueError("G6c1 compile preflight or bounded HLO drifted")
        training_state = value.get("training_state", {})
        if (
            training_state.get("adapter_placement") != expected_adapter
            or training_state.get("optimizer_placement") != expected_optimizer_placement
            or training_state.get("adapter_initializer_compiler_memory")
            != EXPECTED_INITIALIZER_MEMORY
            or training_state.get("optimizer_initializer_compiler_memory")
            != EXPECTED_OPTIMIZER_INITIALIZER_MEMORY
            or training_state.get("initialization_statistic_names")
            != list(TRAINING_STATISTIC_NAMES)
            or training_state.get("initialization_statistics")
            != EXPECTED_INITIALIZATION_STATISTICS
            or training_state.get("initialization_statistics_float32_sha256")
            != EXPECTED_INITIALIZATION_SHA256
        ):
            raise ValueError("G6c1 adapter or optimizer initialization drifted")
        loader = value.get("loader", {})
        if {key: loader.get(key) for key in expected_loader} != expected_loader:
            raise ValueError("G6c1 loader accounting drifted")
        placement = loader.get("parameter_placement", {})
        if (
            {key: placement.get(key) for key in expected_base_placement}
            != expected_base_placement
        ):
            raise ValueError("G6c1 base placement drifted")
        local_base_bytes = placement.get("local_bytes_by_device")
        if (
            not isinstance(local_base_bytes, dict)
            or len(local_base_bytes) != 4
            or set(local_base_bytes.values()) != {EXPECTED_BASE_BYTES_PER_DEVICE}
        ):
            raise ValueError("G6c1 local base bytes drifted")
        rank_devices = {int(device_id) for device_id in local_base_bytes}
        if rank_devices != set(range(rank * 4, rank * 4 + 4)):
            raise ValueError("G6c1 rank-to-device placement drifted")

        steps = value.get("steps")
        if not isinstance(steps, list) or len(steps) != 3:
            raise ValueError("G6c1 must contain exactly three optimizer steps")
        for actual, expected in zip(steps, EXPECTED_STEP_RECORDS, strict=True):
            if {key: actual.get(key) for key in expected} != expected:
                raise ValueError("G6c1 loss, gradient norm, or state trajectory drifted")
            execute_seconds = actual.get("execute_seconds")
            diagnostics_seconds = actual.get("diagnostics_seconds")
            if (
                not isinstance(execute_seconds, (int, float))
                or not 0 < execute_seconds <= 1_200
                or not isinstance(diagnostics_seconds, (int, float))
                or not 0 < diagnostics_seconds <= 120
            ):
                raise ValueError("G6c1 step timing is invalid")
            maxima["step_execute_seconds"] = max(
                maxima["step_execute_seconds"], execute_seconds
            )
            maxima["step_diagnostics_seconds"] = max(
                maxima["step_diagnostics_seconds"], diagnostics_seconds
            )

        checkpoint = value.get("checkpoint", {})
        expected_artifact = EXPECTED_CHECKPOINT_ARTIFACTS[rank]
        expected_checkpoint_common = {
            "schema_version": 1,
            "format": "jaxsft_rank_local_sharded_pytree_npz",
            "identity": expected_checkpoint_identity,
            "process_index": rank,
            "process_count": 4,
            "root_keys": ["adapters", "optimizer"],
            "leaf_count": EXPECTED_CHECKPOINT_LEAF_COUNT,
            "global_elements_including_replicas_once": EXPECTED_CHECKPOINT_GLOBAL_ELEMENTS,
            "global_logical_bytes_including_replicas_once": (
                EXPECTED_CHECKPOINT_GLOBAL_LOGICAL_BYTES
            ),
            "local_unique_shard_count": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
            "local_unique_tensor_bytes": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_BYTES,
            "local_device_resident_bytes": EXPECTED_CHECKPOINT_LOCAL_DEVICE_RESIDENT_BYTES,
            "replicated_payload_sha256": EXPECTED_REPLICATED_CHECKPOINT_SHA256,
            "npz_file_bytes": 56_172_246,
            "manifest_file": f"rank-{rank:03d}.json",
            "directory": "/tmp/jaxsft-glm53-g6c1-543350c/step-00000002",
            "checkpoint_step": 2,
            "frozen_base_included": False,
            "base_leaf_count": 0,
            "pre_save_statistics_float32_sha256": EXPECTED_STEP_RECORDS[1][
                "training_statistics_float32_sha256"
            ],
            "restored_statistics_float32_sha256": EXPECTED_STEP_RECORDS[1][
                "training_statistics_float32_sha256"
            ],
            "pre_save_and_restored_statistics_equal": True,
        }
        if (
            {key: checkpoint.get(key) for key in expected_checkpoint_common}
            != expected_checkpoint_common
        ):
            raise ValueError("G6c1 checkpoint identity or adapter-only inventory drifted")
        for name in (
            "manifest_file_bytes",
            "manifest_sha256",
            "npz_sha256",
            "local_payload_sha256",
            "sharded_payload_sha256",
        ):
            if checkpoint.get(name) != expected_artifact[name]:
                raise ValueError(f"G6c1 checkpoint {name} drifted")
        restore = checkpoint.get("restore", {})
        if restore != {
            "all_local_shards_byte_exact": True,
            "leaf_count": EXPECTED_CHECKPOINT_LEAF_COUNT,
            "local_payload_sha256": expected_artifact["local_payload_sha256"],
            "local_unique_shard_count": EXPECTED_CHECKPOINT_LOCAL_UNIQUE_SHARDS,
            "manifest_sha256": expected_artifact["manifest_sha256"],
            "npz_sha256": expected_artifact["npz_sha256"],
        }:
            raise ValueError("G6c1 byte-exact checkpoint restore evidence drifted")
        manifest = manifests[rank]
        if (
            manifest["npz_sha256"] != checkpoint["npz_sha256"]
            or manifest["local_payload_sha256"] != checkpoint["local_payload_sha256"]
        ):
            raise ValueError("G6c1 run report does not match independent checkpoint artifacts")

        memory = value.get("device_memory", {})
        if set(memory) != required_memory_phases:
            raise ValueError("G6c1 device-memory phase inventory drifted")
        for phase, records in memory.items():
            if (
                not isinstance(records, list)
                or len(records) != 4
                or {record.get("device_id") for record in records} != rank_devices
            ):
                raise ValueError("G6c1 device-memory coverage drifted")
            for record in records:
                stats = record.get("stats", {})
                if stats.get("bytes_limit") != 33_014_407_168:
                    raise ValueError("G6c1 TPU HBM limit drifted")
                peak = stats.get("peak_bytes_in_use")
                free_block = stats.get("largest_free_block_bytes")
                if not isinstance(peak, int) or not isinstance(free_block, int):
                    raise ValueError("G6c1 device-memory record is malformed")
                maximum_device_peak = max(maximum_device_peak, peak)
                minimum_free_block = min(minimum_free_block, free_block)
                maximum_phase_peak[phase] = max(maximum_phase_peak[phase], peak)
                if phase == "after_full_base_placement" and (
                    stats.get("bytes_in_use") != 20_380_723_712
                    or stats.get("largest_alloc_size") != 150_994_944
                ):
                    raise ValueError("G6c1 placed-base HBM drifted")
                if phase.startswith("after_step_") or phase.startswith("after_checkpoint_"):
                    if (
                        peak > 21 * GIB
                        or free_block < 10 * GIB
                        or stats.get("largest_alloc_size") != 220_750_848
                    ):
                        raise ValueError("G6c1 optimizer execution exceeded HBM bounds")
        if max(
            record["stats"]["bytes_in_use"]
            for record in memory["after_training_state_release"]
        ) >= min(
            record["stats"]["bytes_in_use"] for record in memory["after_checkpoint_save"]
        ):
            raise ValueError("G6c1 did not release donated training state before restore")
        for device_id in rank_devices:
            if device_id in device_ids:
                raise ValueError("G6c1 device appears in more than one rank result")
            device_ids.add(device_id)

        if (
            value.get("shm", {}).get("used_delta_during_load_bytes") != 0
            or value.get("shm", {}).get("used_delta_total_bytes") != 0
        ):
            raise ValueError("G6c1 unexpectedly consumed RAMFS payload space")
        process_hwm = max(
            phase.get("vmhwm_bytes", 0) for phase in value.get("host_memory", {}).values()
        )
        if not isinstance(process_hwm, int) or not 0 < process_hwm <= 20 * GIB:
            raise ValueError("G6c1 process HWM exceeds 20 GiB")
        maximum_process_vmhwm = max(maximum_process_vmhwm, process_hwm)
        timing = value.get("timing", {})
        bounds = {
            "header_seconds": 120,
            "compile_seconds": 1_200,
            "adapter_initializer_seconds": 120,
            "optimizer_initializer_seconds": 120,
            "load_seconds": 3_600,
            "checkpoint_save_seconds": 60,
            "checkpoint_restore_seconds": 60,
            "elapsed_seconds_before_shutdown": 5_000,
        }
        for name, bound in bounds.items():
            item = timing.get(name)
            if not isinstance(item, (int, float)) or not 0 < item <= bound:
                raise ValueError(f"G6c1 {name} is invalid")
            maxima[name] = max(maxima[name], item)

    if (
        len(hostnames) != 4
        or process_indexes != set(range(4))
        or device_ids != set(range(16))
    ):
        raise ValueError("G6c1 does not cover four hosts, ranks, and 16 devices")
    expected_phase_peaks = {
        "after_checkpoint_restore": 20_601_651_712,
        "after_checkpoint_save": 20_601_651_712,
        "after_distributed_init": 13_824,
        "after_full_base_placement": 20_380_723_712,
        "after_optimizer_compile": 13_824,
        "after_step_1": 20_601_598_464,
        "after_step_2": 20_601_598_464,
        "after_step_3": 20_601_684_480,
        "after_training_state_initialization": 118_521_856,
        "after_training_state_release": 20_601_651_712,
    }
    if dict(maximum_phase_peak) != expected_phase_peaks:
        raise ValueError("G6c1 cross-rank phase peak HBM drifted")
    if maximum_device_peak != 20_601_684_480 or minimum_free_block != 11_615_084_032:
        raise ValueError("G6c1 aggregate device-memory envelope drifted")

    losses = [record["loss"] for record in EXPECTED_STEP_RECORDS]
    step_one_to_three_relative_change = (losses[2] - losses[0]) / losses[0]
    loss_range_relative_to_step_one = (max(losses) - min(losses)) / losses[0]
    peak_slope = expected_phase_peaks["after_step_3"] - expected_phase_peaks["after_step_1"]
    return {
        "schema_version": 1,
        "test": "glm53_g6c1_three_step_optimizer_checkpoint_acceptance",
        "source_revision": source_revision,
        "topology": {
            "accelerator_type": "v4-32",
            "host_count": 4,
            "process_count": 4,
            "global_device_count": 16,
            "physical_hostnames": sorted(hostnames),
            "physical_hostname_order_independent_of_process_index": True,
        },
        "streaming": {
            "network_bytes_per_host": EXPECTED_LOADER_BYTES_PER_HOST,
            "network_bytes_across_hosts": 4 * EXPECTED_LOADER_BYTES_PER_HOST,
            "loaded_target_count": 1_372,
            "loaded_logical_tensor_count": 37_534,
            "loaded_scale_tensor_count": 36_467,
            "base_bytes_per_device": EXPECTED_BASE_BYTES_PER_DEVICE,
            "maximum_load_seconds": maxima["load_seconds"],
            "maximum_shm_used_delta_bytes": 0,
        },
        "training_state": {
            "attention_lora_target_count": 191,
            "adapter_global_parameter_count": 10_289_152,
            "adapter_bytes_per_device": 8_705_024,
            "optimizer_state_bytes_per_device": 34_820_100,
            "optimizer_moment_dtype": "float32",
            "adapter_dtype": "bfloat16",
            "initialization_statistics": EXPECTED_INITIALIZATION_STATISTICS,
            "initialization_statistics_float32_sha256": (
                EXPECTED_INITIALIZATION_SHA256
            ),
        },
        "trajectory": {
            "steps": EXPECTED_STEP_RECORDS,
            "all_rank_trajectories_equal": True,
            "all_losses_gradients_and_states_finite": True,
            "all_gradient_norms_exceeded_clip_threshold": True,
            "loss_step_one_to_two_change": losses[1] - losses[0],
            "loss_step_one_to_three_change": losses[2] - losses[0],
            "loss_step_one_to_three_relative_change": step_one_to_three_relative_change,
            "loss_range_relative_to_step_one": loss_range_relative_to_step_one,
            "loss_monotonic_decrease": False,
            "maximum_step_execute_seconds": maxima["step_execute_seconds"],
            "maximum_step_diagnostics_seconds": maxima["step_diagnostics_seconds"],
        },
        "checkpoint": checkpoint_evidence,
        "memory": {
            "maximum_phase_peak_bytes_in_use": expected_phase_peaks,
            "maximum_device_peak_bytes_in_use": maximum_device_peak,
            "headroom_after_peak_bytes_per_device": 33_014_407_168 - maximum_device_peak,
            "minimum_largest_free_block_bytes": minimum_free_block,
            "step_one_to_step_three_peak_slope_bytes": peak_slope,
            "maximum_process_vmhwm_bytes": maximum_process_vmhwm,
        },
        "timing": {f"maximum_{name}": value for name, value in maxima.items()},
        "execution": {
            **EXPECTED_COMPILER_MEMORY,
            **EXPECTED_EXECUTION_GATE,
            "optimized_hlo_sha256": EXPECTED_HLO_SHA256,
            "optimized_hlo_bytes": EXPECTED_HLO_BYTES,
            "all_rank_hlo_equal": True,
            "no_assignment_wide_dense_weight_in_optimized_hlo": True,
            "all_distributed_shutdowns_complete": True,
        },
        "rank_result_sha256_by_process_index": dict(
            sorted(result_hash_by_rank.items(), key=lambda item: int(item[0]))
        ),
        "gate": {
            "g6c1_three_step_optimizer_checkpoint": "passed",
            "full_model_optimizer_update_proven": True,
            "adapter_only_checkpoint_restore_proven": True,
            "ten_step_resume_probe_authorized": True,
            "fifty_step_probe_authorized": False,
            "long_run_stability_proven": False,
            "remaining_blockers": [
                "The three-step fixed-token trajectory is finite but non-monotonic.",
                "Total-10-step resume determinism and memory slope remain unmeasured.",
                "Instruction-data/tokenizer throughput and long-sequence capacity remain unmeasured.",
                "A 50-step run is not authorized until the total-10-step gate passes.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-results", type=Path, nargs=4, required=True)
    parser.add_argument("--checkpoint-manifests", type=Path, nargs=4, required=True)
    parser.add_argument("--checkpoint-payloads", type=Path, nargs=4, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.rank_results,
        args.checkpoint_manifests,
        args.checkpoint_payloads,
        source_revision=args.source_revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
