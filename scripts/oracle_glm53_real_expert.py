#!/usr/bin/env python3
"""Check the real GLM expert probe with Transformers' CPU FP8 dequantizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from jaxsft.checkpoint import StrictPooledHTTPRangeReader
from jaxsft.models.glm5_3_flash import (
    OFFICIAL_CHECKPOINT,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    SafetensorsIndex,
    SafetensorsShardHeader,
    SafetensorsTensorRange,
)


EXPERT_INDICES = (0, 17, 63, 95, 127, 191, 255, 287)
TPU_STATISTICS_SHA256 = "046a54bf22934b0271d5dd30e8cfd5349190147d5252f7d5310e1ae0a0b2bbf7"


def _source_url(shard: str) -> str:
    return f"https://huggingface.co/{OFFICIAL_REPO_ID}/resolve/{OFFICIAL_REVISION}/{shard}"


def _read_header(reader: StrictPooledHTTPRangeReader) -> SafetensorsShardHeader:
    header_length = int.from_bytes(reader.read(0, 7), "little", signed=False)
    if header_length <= 0 or header_length > 100 * 1024 * 1024:
        raise ValueError(f"unsafe remote safetensors header length: {header_length}")
    payload = reader.read(8, 7 + header_length)
    return SafetensorsShardHeader.from_json_bytes(payload, header_length=header_length)


def _scale_name(name: str) -> str:
    return name[: -len(".weight")] + ".weight_scale_inv"


def _numpy_tensor(payload: bytes, tensor: SafetensorsTensorRange) -> np.ndarray[Any, Any]:
    dtype = {"F8_E4M3": np.uint8, "F32": np.dtype("<f4")}[tensor.dtype]
    value = np.frombuffer(payload, dtype=dtype)
    if value.size != int(np.prod(tensor.shape)):
        raise ValueError(f"payload for {tensor.name!r} does not match {tensor.shape}")
    return value.reshape(tensor.shape)


class _SourceFingerprint:
    def __init__(self, tensor_shape: tuple[int, int]) -> None:
        self.tensor_shape = tensor_shape
        self.tensor_size = math.prod(tensor_shape)
        self.tensor_count = 0
        self.total = 0
        self.xor = 0
        self.square = 0
        self.weighted = 0

    def update(self, value: np.ndarray[Any, Any]) -> None:
        if value.shape != self.tensor_shape:
            raise ValueError(f"fingerprint shape {value.shape} does not match {self.tensor_shape}")
        if value.dtype == np.float32:
            words = value.view(np.uint32).reshape(-1)
        else:
            words = value.astype(np.uint32, copy=False).reshape(-1)
        raw = words.astype(np.uint64)
        positions = np.arange(
            self.tensor_count * self.tensor_size + 1,
            (self.tensor_count + 1) * self.tensor_size + 1,
            dtype=np.uint64,
        )
        self.total = (self.total + int(np.sum(raw, dtype=np.uint64))) & 0xFFFFFFFF
        self.xor ^= int(np.bitwise_xor.reduce(words, dtype=np.uint32))
        self.square = (self.square + int(np.sum(raw * raw, dtype=np.uint64))) & 0xFFFFFFFF
        self.weighted = (
            self.weighted + int(np.sum(raw * positions, dtype=np.uint64))
        ) & 0xFFFFFFFF
        self.tensor_count += 1

    def result(self) -> list[int]:
        if self.tensor_count != len(EXPERT_INDICES):
            raise ValueError(f"fingerprint consumed {self.tensor_count} tensors")
        return [self.total, self.xor, self.square, self.weighted]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    started = time.monotonic()
    torch.set_num_threads(args.torch_threads)
    index = SafetensorsIndex.from_path(args.index)
    index.verify(OFFICIAL_CHECKPOINT)
    weight_map = dict(index.tensor_files)
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts."
    names = [
        prefix + f"{expert_index}.{projection}_proj.weight"
        for expert_index in EXPERT_INDICES
        for projection in ("gate", "up", "down")
    ]
    required_names = names + [_scale_name(name) for name in names]
    shard_names = sorted({weight_map[name] for name in required_names})

    with ExitStack() as stack:
        readers = {
            shard: stack.enter_context(
                StrictPooledHTTPRangeReader(
                    _source_url(shard),
                    timeout_seconds=args.timeout_seconds,
                    maximum_request_bytes=16 * 1024 * 1024,
                    connections=4,
                )
            )
            for shard in shard_names
        }
        headers = {shard: _read_header(readers[shard]) for shard in shard_names}
        tensors = {
            name: (weight_map[name], headers[weight_map[name]].tensor(name))
            for name in required_names
        }

        dequantizer = Fp8Dequantize(None)

        weight_fingerprints = {
            "gate": _SourceFingerprint((2048, 4096)),
            "up": _SourceFingerprint((2048, 4096)),
            "down": _SourceFingerprint((4096, 2048)),
        }
        scale_fingerprints = {
            "gate": _SourceFingerprint((16, 32)),
            "up": _SourceFingerprint((16, 32)),
            "down": _SourceFingerprint((32, 16)),
        }

        def dense(name: str, projection: str):
            shard, weight = tensors[name]
            scale_shard, scale = tensors[_scale_name(name)]
            weight_payload = readers[shard].read(*weight.http_range)
            scale_payload = readers[scale_shard].read(*scale.http_range)
            bits = _numpy_tensor(weight_payload, weight)
            scales = _numpy_tensor(scale_payload, scale)
            weight_fingerprints[projection].update(bits)
            scale_fingerprints[projection].update(scales)
            quantized = torch.from_numpy(np.array(bits, copy=True)).view(torch.float8_e4m3fn)
            torch_scales = torch.from_numpy(np.array(scales, copy=True))
            return dequantizer._dequantize_one(
                quantized,
                torch_scales,
                output_dtype=torch.bfloat16,
            )

        inputs = torch.full((1, 4096), 0.01, dtype=torch.bfloat16)
        routed = []
        for expert_index in EXPERT_INDICES:
            base = prefix + f"{expert_index}."
            gate_weight = dense(base + "gate_proj.weight", "gate")
            gate = (inputs.float() @ gate_weight.float().T).to(torch.bfloat16)
            del gate_weight
            up_weight = dense(base + "up_proj.weight", "up")
            up = (inputs.float() @ up_weight.float().T).to(torch.bfloat16)
            del up_weight
            activated = torch.nn.functional.silu(gate.float()) * up.float()
            down_inputs = activated.to(torch.bfloat16)
            down_weight = dense(base + "down_proj.weight", "down")
            routed.append(down_inputs.float() @ down_weight.float().T)
            del down_weight
        routed_tensor = torch.stack(routed, dim=1)
        weights = torch.arange(1, 9, dtype=torch.float32)
        weights /= weights.sum()
        output = torch.sum(routed_tensor * weights[None, :, None], dim=1)
        statistics = torch.stack(
            (
                output.sum(dtype=torch.float32),
                output.square().sum(dtype=torch.float32),
                output.max(),
                output.min(),
            )
        ).numpy(force=True)
        network_bytes = sum(reader.bytes_read for reader in readers.values())
        network_requests = sum(len(reader.records) for reader in readers.values())

    tpu_payload = json.loads(args.tpu_result.read_text())
    tpu_statistics = np.asarray(tpu_payload["output"]["statistics"], dtype=np.float32)
    if tpu_payload["output"]["statistics_float32_sha256"] != TPU_STATISTICS_SHA256:
        raise ValueError("TPU result does not carry the accepted real-expert statistic hash")
    source_fingerprints = {
        name: {
            "weight_bits_uint32": weight_fingerprints[name].result(),
            "weight_scale_inv_bits_uint32": scale_fingerprints[name].result(),
        }
        for name in ("gate", "up", "down")
    }
    if tpu_payload.get("selected_source_fingerprints") != source_fingerprints:
        raise ValueError("TPU selected-source fingerprints do not match the CPU range payloads")
    difference = statistics.astype(np.float64) - tpu_statistics.astype(np.float64)
    relative_l2 = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(tpu_statistics.astype(np.float64)), np.finfo(np.float64).tiny)
    )
    maximum_absolute = float(np.max(np.abs(difference)))
    passed = relative_l2 <= args.maximum_relative_l2 and maximum_absolute <= args.maximum_absolute
    if not passed:
        raise ValueError(
            f"Transformers CPU oracle drifted from TPU: rel_l2={relative_l2}, "
            f"max_abs={maximum_absolute}"
        )
    return {
        "schema_version": 1,
        "test": "glm53_real_expert_transformers_cpu_oracle",
        "source_revision": args.source_revision,
        "model": {
            "repo_id": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "index_sha256": index.sha256,
            "layer": args.layer,
            "expert_indices": list(EXPERT_INDICES),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "network": {
            "source_shards": shard_names,
            "request_count_including_resolves_and_headers": network_requests,
            "bytes_read_including_resolves_and_headers": network_bytes,
        },
        "selected_source_fingerprints": source_fingerprints,
        "comparison": {
            "tpu_statistics": tpu_statistics.tolist(),
            "transformers_cpu_statistics": statistics.tolist(),
            "maximum_absolute": maximum_absolute,
            "relative_l2": relative_l2,
            "maximum_absolute_tolerance": args.maximum_absolute,
            "maximum_relative_l2_tolerance": args.maximum_relative_l2,
            "passed": passed,
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tpu-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.02)
    parser.add_argument("--maximum-absolute", type=float, default=2e-5)
    args = parser.parse_args()
    if (
        len(args.source_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.source_revision)
    ):
        raise ValueError("source-revision must be a full lowercase Git hash")
    if args.torch_threads <= 0 or args.maximum_relative_l2 <= 0 or args.maximum_absolute <= 0:
        raise ValueError("thread and comparison bounds must be positive")
    result = _run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
