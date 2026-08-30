"""Experimental GLM-5.3-Flash text configuration and feasibility gates.

This module intentionally stops short of advertising a trainable model.  It
pins and validates the upstream configuration/checkpoint contract, describes
the initial attention-only LoRA target set, and performs conservative memory
preflight.  Registration in :mod:`jaxsft.models.registry` is gated on reduced
float32 Hugging Face parity and a proven block-FP8 execution path on TPU v4.

The first experimental path excludes vision inputs, the MTP prediction layer,
expert LoRA, inference caches, and full-parameter training.  Those exclusions
are explicit because the official checkpoint is multimodal, contains 321B
logical parameters, and stores most weights in block-scaled FP8.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping


GIB = 1 << 30

OFFICIAL_REPO_ID = "zai-org/GLM-5.3-Flash"
OFFICIAL_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
OFFICIAL_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
OFFICIAL_INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"


@dataclass(frozen=True)
class CheckpointContract:
    """Immutable facts obtained from the pinned Hub revision."""

    repo_id: str
    revision: str
    config_sha256: str
    index_sha256: str
    total_size_bytes: int
    tensor_count: int
    shard_count: int
    parameter_counts_by_dtype: tuple[tuple[str, int], ...]
    maximum_source_shard_bytes: int

    @property
    def logical_parameter_count(self) -> int:
        return sum(count for _, count in self.parameter_counts_by_dtype)

    def expanded_parameter_bytes(self, dtype: str) -> int:
        """Return resident bytes after expanding floating checkpoint tensors.

        Scale tensors that are already float32 remain float32.  This is the
        relevant lower bound when asking whether the checkpoint can simply be
        converted to BF16 before execution.
        """

        if dtype != "bfloat16":
            raise ValueError("only bfloat16 expansion is defined for the pinned checkpoint")
        widths = {"F8_E4M3": 2, "BF16": 2, "F32": 4}
        unknown = {name for name, _ in self.parameter_counts_by_dtype} - set(widths)
        if unknown:
            raise ValueError(f"no expansion rule for checkpoint dtypes: {sorted(unknown)}")
        return sum(widths[name] * count for name, count in self.parameter_counts_by_dtype)


OFFICIAL_CHECKPOINT = CheckpointContract(
    repo_id=OFFICIAL_REPO_ID,
    revision=OFFICIAL_REVISION,
    config_sha256=OFFICIAL_CONFIG_SHA256,
    index_sha256=OFFICIAL_INDEX_SHA256,
    total_size_bytes=328_326_771_576,
    tensor_count=76_108,
    shard_count=62,
    parameter_counts_by_dtype=(
        ("F8_E4M3", 314_396_639_232),
        ("BF16", 6_926_096_640),
        ("F32", 295_518),
    ),
    # The Hub dry-run currently reports ordinary shards as 5.3--5.4 GB and
    # the final shard as 1.3 GB.  Use a decimal 5.5 GB ceiling until the
    # header-only manifest records exact Content-Length values.
    maximum_source_shard_bytes=5_500_000_000,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class SafetensorsIndex:
    """Validated, immutable view of a sharded safetensors index."""

    total_size_bytes: int
    tensor_files: tuple[tuple[str, str], ...]
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "SafetensorsIndex":
        payload = Path(path).read_bytes()
        raw = json.loads(payload, object_pairs_hook=_unique_json_object)
        if not isinstance(raw, dict) or set(raw) != {"metadata", "weight_map"}:
            raise ValueError("safetensors index must contain exactly metadata and weight_map")
        metadata = raw["metadata"]
        weight_map = raw["weight_map"]
        if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
            raise ValueError("safetensors index metadata must contain exactly total_size")
        total_size = metadata["total_size"]
        if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size <= 0:
            raise ValueError("safetensors index total_size must be a positive integer")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index weight_map must be a non-empty object")

        entries: list[tuple[str, str]] = []
        for tensor_name, filename in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError("safetensors tensor names must be non-empty strings")
            if not isinstance(filename, str) or not filename.endswith(".safetensors"):
                raise ValueError(f"invalid safetensors filename for {tensor_name!r}: {filename!r}")
            # Index files are manifests, not a vehicle for path traversal.
            if PurePath(filename).name != filename or ".." in PurePath(filename).parts:
                raise ValueError(f"safetensors filename must be a basename: {filename!r}")
            entries.append((tensor_name, filename))
        return cls(
            total_size_bytes=total_size,
            tensor_files=tuple(sorted(entries)),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    @property
    def tensor_count(self) -> int:
        return len(self.tensor_files)

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(sorted({filename for _, filename in self.tensor_files}))

    @property
    def shard_count(self) -> int:
        return len(self.shard_names)

    def verify(self, contract: CheckpointContract, *, require_hash: bool = True) -> None:
        mismatches: list[str] = []
        if self.total_size_bytes != contract.total_size_bytes:
            mismatches.append(
                f"total_size={self.total_size_bytes}, expected {contract.total_size_bytes}"
            )
        if self.tensor_count != contract.tensor_count:
            mismatches.append(f"tensor_count={self.tensor_count}, expected {contract.tensor_count}")
        if self.shard_count != contract.shard_count:
            mismatches.append(f"shard_count={self.shard_count}, expected {contract.shard_count}")
        if require_hash and self.sha256 != contract.index_sha256:
            mismatches.append(f"sha256={self.sha256}, expected {contract.index_sha256}")
        if mismatches:
            raise ValueError("checkpoint index does not match its pinned contract: " + "; ".join(mismatches))


@dataclass(frozen=True)
class LinearAttentionConfig:
    num_heads: int
    head_dim: int
    short_conv_kernel_size: int
    gate_lower_bound: float
    kda_layers: tuple[int, ...]
    full_attn_layers: tuple[int, ...]


@dataclass(frozen=True)
class Glm53TextConfig:
    """Numerically relevant text configuration for the first GLM-5.3 path."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    linear_attention: LinearAttentionConfig
    max_position_embeddings: int
    rms_norm_eps: float
    initializer_range: float
    hidden_act: str
    attention_dropout: float
    attention_bias: bool
    tie_word_embeddings: bool
    first_k_dense_replace: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    mla_use_nope: bool
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    scoring_func: str
    topk_method: str
    routed_scaling_factor: float
    router_aux_loss_coef: float
    moe_router_dtype: str
    swiglu_limit: float
    mhc: bool
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool
    indexer_rope_interleave: bool
    num_nextn_predict_layers: int
    source_quant_method: str | None
    source_quant_format: str | None
    source_quant_block_shape: tuple[int, int] | None

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"GLM-5.3 config fields must be positive: {invalid}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types length must equal num_hidden_layers")
        if set(self.layer_types) - {"linear_attention", "deepseek_sparse_attention"}:
            raise ValueError("unsupported GLM-5.3 attention layer type")
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("unsupported GLM-5.3 MLP layer type")
        kda = set(self.linear_attention.kda_layers)
        sparse = set(self.linear_attention.full_attn_layers)
        expected = set(range(self.num_hidden_layers))
        if kda & sparse or kda | sparse != expected:
            raise ValueError("linear/full attention layer lists must form an exact partition")
        if any(self.layer_types[index] != "linear_attention" for index in kda):
            raise ValueError("kda_layers disagrees with layer_types")
        if any(self.layer_types[index] != "deepseek_sparse_attention" for index in sparse):
            raise ValueError("full_attn_layers disagrees with layer_types")
        if self.first_k_dense_replace != sum(kind == "dense" for kind in self.mlp_layer_types):
            raise ValueError("first_k_dense_replace disagrees with mlp_layer_types")
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim:
            raise ValueError("qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
        if self.n_shared_experts < 0 or self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("invalid shared/routed expert counts")
        if self.rms_norm_eps <= 0 or self.hc_eps <= 0 or self.initializer_range <= 0:
            raise ValueError("normalization epsilons and initializer_range must be positive")
        if self.hidden_act != "silu" or self.attention_dropout != 0 or self.attention_bias:
            raise NotImplementedError("the initial GLM-5.3 path requires SiLU and bias/dropout-free attention")
        if self.tie_word_embeddings:
            raise NotImplementedError("the official GLM-5.3 checkpoint has an untied LM head")
        if self.source_quant_method not in {None, "fp8"}:
            raise NotImplementedError(f"unsupported source quantization {self.source_quant_method!r}")
        if self.source_quant_method == "fp8" and (
            self.source_quant_format != "e4m3" or self.source_quant_block_shape != (128, 128)
        ):
            raise NotImplementedError("only the official 128x128 block-scaled E4M3 checkpoint is covered")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Glm53TextConfig":
        outer_type = raw.get("model_type")
        text = raw.get("text_config", raw)
        if not isinstance(text, Mapping):
            raise ValueError("GLM-5.3 text_config must be a mapping")
        model_type = text.get("model_type", outer_type)
        if model_type != "glm5_next_text":
            raise ValueError(f"expected model_type='glm5_next_text', got {model_type!r}")
        if "text_config" in raw and outer_type != "glm5_next":
            raise ValueError(f"expected outer model_type='glm5_next', got {outer_type!r}")

        linear = text.get("linear_attn_config")
        if not isinstance(linear, Mapping):
            raise ValueError("linear_attn_config must be a mapping")
        quant = raw.get("quantization_config")
        if quant is not None and not isinstance(quant, Mapping):
            raise ValueError("quantization_config must be a mapping")
        quant = quant or {}
        block = quant.get("weight_block_size")
        block_shape = None if block is None else tuple(int(value) for value in block)
        if block_shape is not None and len(block_shape) != 2:
            raise ValueError("weight_block_size must have exactly two dimensions")

        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "layer_types",
            "mlp_layer_types",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "n_routed_experts",
            "num_experts_per_tok",
        )
        missing = [name for name in required if name not in text]
        if missing:
            raise ValueError(f"GLM-5.3 config is missing fields: {missing}")

        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            layer_types=tuple(str(value) for value in text["layer_types"]),
            mlp_layer_types=tuple(str(value) for value in text["mlp_layer_types"]),
            linear_attention=LinearAttentionConfig(
                num_heads=int(linear["num_heads"]),
                head_dim=int(linear["head_dim"]),
                short_conv_kernel_size=int(linear["short_conv_kernel_size"]),
                gate_lower_bound=float(linear.get("gate_lower_bound", -5.0)),
                kda_layers=tuple(int(value) for value in linear["kda_layers"]),
                full_attn_layers=tuple(int(value) for value in linear["full_attn_layers"]),
            ),
            max_position_embeddings=int(text.get("max_position_embeddings", 1_048_576)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-5)),
            initializer_range=float(text.get("initializer_range", 0.02)),
            hidden_act=str(text.get("hidden_act", "silu")),
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            attention_bias=bool(text.get("attention_bias", False)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))),
            first_k_dense_replace=int(text.get("first_k_dense_replace", 0)),
            q_lora_rank=int(text["q_lora_rank"]),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_head_dim=int(text["qk_head_dim"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            mla_use_nope=bool(text.get("mla_use_nope", True)),
            n_routed_experts=int(text["n_routed_experts"]),
            n_shared_experts=int(text.get("n_shared_experts", 0)),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            n_group=int(text.get("n_group", 1)),
            topk_group=int(text.get("topk_group", 1)),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            scoring_func=str(text.get("scoring_func", "sigmoid")),
            topk_method=str(text.get("topk_method", "noaux_tc")),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            router_aux_loss_coef=float(text.get("router_aux_loss_coef", 0.0)),
            moe_router_dtype=str(text.get("moe_router_dtype", "float32")),
            swiglu_limit=float(text.get("swiglu_limit", math.inf)),
            mhc=bool(text.get("mhc", False)),
            hc_mult=int(text.get("hc_mult", 1)),
            hc_eps=float(text.get("hc_eps", 1e-6)),
            hc_sinkhorn_iters=int(text.get("hc_sinkhorn_iters", 20)),
            index_n_heads=int(text.get("index_n_heads", 0)),
            index_head_dim=int(text.get("index_head_dim", 0)),
            index_topk=int(text.get("index_topk", 0)),
            index_kpool=int(text.get("index_kpool", 1)),
            index_kpool_compress=bool(text.get("index_kpool_compress", False)),
            index_kpool_always_select_tail=bool(text.get("index_kpool_always_select_tail", False)),
            indexer_rope_interleave=bool(text.get("indexer_rope_interleave", False)),
            num_nextn_predict_layers=int(text.get("num_nextn_predict_layers", 0)),
            source_quant_method=None if quant.get("quant_method") is None else str(quant["quant_method"]),
            source_quant_format=None if quant.get("fmt") is None else str(quant["fmt"]),
            source_quant_block_shape=block_shape,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Glm53TextConfig":
        raw = json.loads(Path(path).read_text(), object_pairs_hook=_unique_json_object)
        if not isinstance(raw, Mapping):
            raise ValueError("config.json must contain an object")
        return cls.from_dict(raw)


def attention_lora_parameter_count(
    config: Glm53TextConfig,
    *,
    rank: int,
    include_sparse_indexer: bool = False,
) -> int:
    """Count the initial LoRA target set from exact upstream matrix shapes.

    The default targets q/k/v/o in KDA layers and q_a/q_b/kv_a/kv_b/o in
    sparse-attention layers.  Expert, shared-expert, dense-MLP, convolution,
    mHC, normalization, embedding, head, vision, and MTP tensors stay frozen.
    """

    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    hidden = config.hidden_size
    linear_width = config.linear_attention.num_heads * config.linear_attention.head_dim
    linear_per_layer = 4 * rank * (hidden + linear_width)

    query_width = config.num_attention_heads * config.qk_head_dim
    value_width = config.num_attention_heads * config.v_head_dim
    kv_a_width = config.kv_lora_rank + config.qk_rope_head_dim
    sparse_dimensions = (
        (hidden, config.q_lora_rank),
        (config.q_lora_rank, query_width),
        (hidden, kv_a_width),
        (config.kv_lora_rank, config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)),
        (value_width, hidden),
    )
    if include_sparse_indexer:
        sparse_dimensions += (
            (hidden, config.index_n_heads),
            (hidden, config.index_head_dim),
            (config.q_lora_rank, config.index_n_heads * config.index_head_dim),
        )
    sparse_per_layer = rank * sum(input_size + output_size for input_size, output_size in sparse_dimensions)
    return (
        len(config.linear_attention.kda_layers) * linear_per_layer
        + len(config.linear_attention.full_attn_layers) * sparse_per_layer
    )


@dataclass(frozen=True)
class MemoryLine:
    name: str
    aggregate_bytes: int
    per_device_bytes: int
    assumption: str


@dataclass(frozen=True)
class V432LoRAPreflight:
    """Static lower-bound budget plus explicit evidence gates for v4-32."""

    execution_weight_format: str
    hosts: int
    devices: int
    hbm_per_device_bytes: int
    adapter_parameter_count: int
    memory: tuple[MemoryLine, ...]
    staging_per_host_bytes: int
    static_fit: bool
    executable_kernel_proven: bool
    direct_loader_proven: bool
    runnable: bool
    blockers: tuple[str, ...]

    @property
    def used_per_device_bytes(self) -> int:
        return sum(line.per_device_bytes for line in self.memory)

    @property
    def free_per_device_bytes(self) -> int:
        return self.hbm_per_device_bytes - self.used_per_device_bytes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["used_per_device_bytes"] = self.used_per_device_bytes
        result["free_per_device_bytes"] = self.free_per_device_bytes
        return result


def v4_32_lora_preflight(
    config: Glm53TextConfig,
    index: SafetensorsIndex,
    *,
    rank: int = 8,
    execution_weight_format: str = "fp8_blockwise",
    hosts: int = 4,
    devices: int = 16,
    hbm_per_device_bytes: int = 32 * GIB,
    model_shards: int = 16,
    adapter_shards: int = 16,
    activation_reserve_per_device_bytes: int = 8 * GIB,
    runtime_reserve_per_device_bytes: int = 2 * GIB,
    adapter_state_bytes_per_parameter: int = 12,
    executable_kernel_proven: bool = False,
    direct_loader_proven: bool = False,
) -> V432LoRAPreflight:
    """Build a fail-closed memory/evidence plan for attention-only LoRA."""

    if hosts <= 0 or devices <= 0 or devices % hosts:
        raise ValueError("devices must be positive and evenly divisible across hosts")
    if model_shards <= 0 or devices % model_shards:
        raise ValueError("model_shards must be a positive divisor of devices")
    if adapter_shards <= 0 or devices % adapter_shards:
        raise ValueError("adapter_shards must be a positive divisor of devices")
    if hbm_per_device_bytes <= 0 or adapter_state_bytes_per_parameter <= 0:
        raise ValueError("memory sizes must be positive")
    if execution_weight_format == "fp8_blockwise":
        base_bytes = index.total_size_bytes
        base_assumption = (
            "checkpoint-sized lower bound; requires scale-aware tiled dequantization without a persistent BF16 copy"
        )
    elif execution_weight_format == "bfloat16":
        base_bytes = OFFICIAL_CHECKPOINT.expanded_parameter_bytes("bfloat16")
        base_assumption = "all floating checkpoint tensors expanded to BF16; F32 scale tensors remain F32"
    else:
        raise ValueError("execution_weight_format must be fp8_blockwise or bfloat16")

    adapter_count = attention_lora_parameter_count(config, rank=rank)
    adapter_bytes = adapter_count * adapter_state_bytes_per_parameter
    memory = (
        MemoryLine(
            "frozen_base_weights",
            base_bytes,
            math.ceil(base_bytes / model_shards),
            base_assumption,
        ),
        MemoryLine(
            "lora_parameters_gradients_adam",
            adapter_bytes,
            math.ceil(adapter_bytes / adapter_shards),
            "BF16 parameters/gradients + FP32 Adam moments; excludes transient update buffers",
        ),
        MemoryLine(
            "activation_reserve",
            activation_reserve_per_device_bytes * devices,
            activation_reserve_per_device_bytes,
            "placeholder until measured with rematerialization at the selected sequence length",
        ),
        MemoryLine(
            "runtime_and_dequant_reserve",
            runtime_reserve_per_device_bytes * devices,
            runtime_reserve_per_device_bytes,
            "XLA runtime, collectives, executable, and one-layer dequantization workspace",
        ),
    )
    used_per_device = sum(line.per_device_bytes for line in memory)
    static_fit = used_per_device <= hbm_per_device_bytes
    blockers: list[str] = []
    if not static_fit:
        blockers.append(
            f"static lower bound exceeds HBM by {used_per_device - hbm_per_device_bytes} bytes per device"
        )
    if execution_weight_format != "fp8_blockwise":
        blockers.append("v4-32 cannot hold a persistent BF16 expansion of the official checkpoint")
    if not executable_kernel_proven:
        blockers.append("block-FP8 storage/dequantization has not been compiled and memory-profiled on TPU v4")
    if not direct_loader_proven:
        blockers.append("direct-to-final-shard loading has not passed a four-host checksum/peak-RSS test")
    runnable = static_fit and execution_weight_format == "fp8_blockwise" and not blockers
    return V432LoRAPreflight(
        execution_weight_format=execution_weight_format,
        hosts=hosts,
        devices=devices,
        hbm_per_device_bytes=hbm_per_device_bytes,
        adapter_parameter_count=adapter_count,
        memory=memory,
        staging_per_host_bytes=OFFICIAL_CHECKPOINT.maximum_source_shard_bytes,
        static_fit=static_fit,
        executable_kernel_proven=executable_kernel_proven,
        direct_loader_proven=direct_loader_proven,
        runnable=runnable,
        blockers=tuple(blockers),
    )


__all__ = [
    "GIB",
    "OFFICIAL_CHECKPOINT",
    "OFFICIAL_CONFIG_SHA256",
    "OFFICIAL_INDEX_SHA256",
    "OFFICIAL_REPO_ID",
    "OFFICIAL_REVISION",
    "CheckpointContract",
    "Glm53TextConfig",
    "LinearAttentionConfig",
    "MemoryLine",
    "SafetensorsIndex",
    "V432LoRAPreflight",
    "attention_lora_parameter_count",
    "v4_32_lora_preflight",
]
