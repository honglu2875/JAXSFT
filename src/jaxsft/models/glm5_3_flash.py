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

import jax
import jax.numpy as jnp
import numpy as np

from ..lora import LoRAConfig, adapter_for_path, format_parameter_path, lora_linear


GIB = 1 << 30
ArrayTree = dict[str, Any]
_PRECISION = jax.lax.Precision.HIGHEST

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
    indexer_types: tuple[str, ...]
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
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError("indexer_types length must equal num_hidden_layers")
        if set(self.layer_types) - {"linear_attention", "deepseek_sparse_attention"}:
            raise ValueError("unsupported GLM-5.3 attention layer type")
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("unsupported GLM-5.3 MLP layer type")
        if set(self.indexer_types) != {"full"}:
            raise NotImplementedError("the initial GLM-5.3 training path supports full indexers only")
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
        if self.qk_rope_head_dim:
            raise NotImplementedError("the GLM-5.3 DSA path is NoPE and requires qk_rope_head_dim=0")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("num_key_value_heads must equal num_attention_heads")
        if self.n_shared_experts < 0 or self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("invalid shared/routed expert counts")
        if self.n_group <= 0 or self.n_routed_experts % self.n_group:
            raise ValueError("n_group must be positive and divide n_routed_experts")
        if self.n_routed_experts // self.n_group < 2:
            raise ValueError("each expert group must contain at least two experts")
        if not 0 < self.topk_group <= self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            raise NotImplementedError("the initial GLM-5.3 router requires sigmoid/noaux_tc")
        if self.moe_router_dtype != "float32":
            raise NotImplementedError("the initial GLM-5.3 router requires float32 logits")
        if self.rms_norm_eps <= 0 or self.hc_eps <= 0 or self.initializer_range <= 0:
            raise ValueError("normalization epsilons and initializer_range must be positive")
        if self.hc_mult <= 0 or self.hc_sinkhorn_iters <= 0 or not self.mhc:
            raise ValueError("GLM-5.3 requires positive mHC dimensions/iterations")
        if (
            self.linear_attention.num_heads <= 0
            or self.linear_attention.head_dim <= 0
            or self.linear_attention.short_conv_kernel_size <= 0
        ):
            raise ValueError("linear-attention dimensions must be positive")
        if self.index_kpool <= 0 or self.index_topk <= 0 or self.index_topk % self.index_kpool:
            raise ValueError("index_topk must be positive and divisible by index_kpool")
        if self.index_n_heads <= 0 or self.index_head_dim <= 0:
            raise ValueError("index_n_heads and index_head_dim must be positive")
        if not self.index_kpool_compress:
            raise NotImplementedError("the initial GLM-5.3 path requires compressed DSA k-pools")
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
            indexer_types=tuple(str(value) for value in text.get("indexer_types", ("full",) * int(text["num_hidden_layers"]))),
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


def attention_lora_target_paths(
    config: Glm53TextConfig,
    *,
    include_sparse_indexer: bool = False,
) -> tuple[tuple[str | int, ...], ...]:
    """Declare the exact JAX parameter paths eligible for initial GLM LoRA."""

    paths: list[tuple[str | int, ...]] = []
    for layer_index, layer_type in enumerate(config.layer_types):
        prefix: tuple[str | int, ...] = ("layers", layer_index, "self_attn")
        if layer_type == "linear_attention":
            names = ("q_proj", "k_proj", "v_proj", "o_proj")
        else:
            names = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
        paths.extend(prefix + (name,) for name in names)
        if layer_type == "deepseek_sparse_attention" and include_sparse_indexer:
            paths.extend(prefix + ("indexer", name) for name in ("weights_proj", "wk", "wq_b"))
    return tuple(paths)


def tiny_config(*, vocab_size: int = 128) -> Glm53TextConfig:
    """Two-layer hybrid config for deterministic JAX/Transformers parity."""

    return Glm53TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        layer_types=("linear_attention", "deepseek_sparse_attention"),
        mlp_layer_types=("dense", "sparse"),
        indexer_types=("full", "full"),
        linear_attention=LinearAttentionConfig(
            num_heads=4,
            head_dim=8,
            short_conv_kernel_size=2,
            gate_lower_bound=-5.0,
            kda_layers=(0,),
            full_attn_layers=(1,),
        ),
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        hidden_act="silu",
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        first_k_dense_replace=1,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_head_dim=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        mla_use_nope=True,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        routed_scaling_factor=2.5,
        router_aux_loss_coef=0.001,
        moe_router_dtype="float32",
        swiglu_limit=10.0,
        mhc=True,
        hc_mult=2,
        hc_eps=1e-6,
        hc_sinkhorn_iters=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        index_kpool=2,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        indexer_rope_interleave=True,
        num_nextn_predict_layers=0,
        source_quant_method=None,
        source_quant_format=None,
        source_quant_block_shape=None,
    )


def _hyper_connection_shapes(config: Glm53TextConfig) -> ArrayTree:
    mix = (2 + config.hc_mult) * config.hc_mult
    return {
        "fn": (config.hc_mult * config.hidden_size, mix),
        "base": (mix,),
        "scale": (3,),
    }


def _expected_shapes(config: Glm53TextConfig) -> ArrayTree:
    layers: list[ArrayTree] = []
    linear_width = config.linear_attention.num_heads * config.linear_attention.head_dim
    for layer_index, (attention_type, mlp_type) in enumerate(
        zip(config.layer_types, config.mlp_layer_types, strict=True)
    ):
        layer: ArrayTree = {
            "input_layernorm": (config.hidden_size,),
            "post_attention_layernorm": (config.hidden_size,),
            "attn_hc": _hyper_connection_shapes(config),
            "ffn_hc": _hyper_connection_shapes(config),
        }
        if attention_type == "linear_attention":
            layer["self_attn"] = {
                "q_proj": (config.hidden_size, linear_width),
                "k_proj": (config.hidden_size, linear_width),
                "v_proj": (config.hidden_size, linear_width),
                "conv1d": (3 * linear_width, config.linear_attention.short_conv_kernel_size),
                "f_a_proj": (config.hidden_size, config.linear_attention.head_dim),
                "f_b_proj": (config.linear_attention.head_dim, linear_width),
                "dt_bias": (linear_width,),
                "A_log": (config.linear_attention.num_heads,),
                "b_proj": (config.hidden_size, config.linear_attention.num_heads),
                "g_a_proj": (config.hidden_size, config.linear_attention.head_dim),
                "g_b_proj": (config.linear_attention.head_dim, linear_width),
                "o_norm": (config.linear_attention.head_dim,),
                "o_proj": (linear_width, config.hidden_size),
            }
        else:
            indexer = {
                "wq_b": (config.q_lora_rank, config.index_n_heads * config.index_head_dim),
                "wk": (config.hidden_size, config.index_head_dim),
                "k_norm_weight": (config.index_head_dim,),
                "k_norm_bias": (config.index_head_dim,),
                "weights_proj": (config.hidden_size, config.index_n_heads),
                "index_kpool_compress_ape": (config.index_kpool, config.index_head_dim),
                "index_kpool_compress_gate": (config.hidden_size, config.index_head_dim),
            }
            layer["self_attn"] = {
                "q_a_proj": (config.hidden_size, config.q_lora_rank),
                "q_a_layernorm": (config.q_lora_rank,),
                "q_b_proj": (config.q_lora_rank, config.num_attention_heads * config.qk_head_dim),
                "kv_a_proj_with_mqa": (
                    config.hidden_size,
                    config.kv_lora_rank + config.qk_rope_head_dim,
                ),
                "kv_a_layernorm": (config.kv_lora_rank,),
                "kv_b_proj": (
                    config.kv_lora_rank,
                    config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                ),
                "o_proj": (config.num_attention_heads * config.v_head_dim, config.hidden_size),
                "indexer": indexer,
            }
        if mlp_type == "dense":
            layer["mlp"] = {
                "gate_proj": (config.hidden_size, config.intermediate_size),
                "up_proj": (config.hidden_size, config.intermediate_size),
                "down_proj": (config.intermediate_size, config.hidden_size),
            }
        else:
            shared_width = config.moe_intermediate_size * config.n_shared_experts
            layer["mlp"] = {
                "router": (config.hidden_size, config.n_routed_experts),
                "router_correction_bias": (config.n_routed_experts,),
                "experts_gate_up": (
                    config.n_routed_experts,
                    config.hidden_size,
                    2 * config.moe_intermediate_size,
                ),
                "experts_down": (
                    config.n_routed_experts,
                    config.moe_intermediate_size,
                    config.hidden_size,
                ),
                "shared_gate_proj": (config.hidden_size, shared_width),
                "shared_up_proj": (config.hidden_size, shared_width),
                "shared_down_proj": (shared_width, config.hidden_size),
            }
        layers.append(layer)
    return {
        "embed_tokens": (config.vocab_size, config.hidden_size),
        "layers": tuple(layers),
        "norm": (config.hidden_size,),
        "lm_head": (config.hidden_size, config.vocab_size),
    }


def parameter_count(config: Glm53TextConfig) -> int:
    shapes = jax.tree.leaves(
        _expected_shapes(config),
        is_leaf=lambda value: isinstance(value, tuple) and all(isinstance(item, int) for item in value),
    )
    return sum(int(np.prod(shape)) for shape in shapes)


def init_params(key: jax.Array, config: Glm53TextConfig, *, dtype: Any = jnp.float32) -> ArrayTree:
    """Initialize the reduced text model; public FP8 loading is a later gate."""

    dtype = jnp.dtype(dtype)

    def normal(shape, *, result_dtype=dtype, stddev=config.initializer_range):
        nonlocal key
        key, subkey = jax.random.split(key)
        return (jax.random.normal(subkey, shape, jnp.float32) * stddev).astype(result_dtype)

    def hyper_connection():
        mix = (2 + config.hc_mult) * config.hc_mult
        return {
            "fn": normal((config.hc_mult * config.hidden_size, mix), stddev=0.02),
            "base": jnp.zeros((mix,), dtype),
            "scale": jnp.ones((3,), dtype),
        }

    linear_width = config.linear_attention.num_heads * config.linear_attention.head_dim
    layers: list[ArrayTree] = []
    for attention_type, mlp_type in zip(config.layer_types, config.mlp_layer_types, strict=True):
        layer: ArrayTree = {
            "input_layernorm": jnp.ones((config.hidden_size,), dtype),
            "post_attention_layernorm": jnp.ones((config.hidden_size,), dtype),
            "attn_hc": hyper_connection(),
            "ffn_hc": hyper_connection(),
        }
        if attention_type == "linear_attention":
            nonlocal_dt_key = None
            key, nonlocal_dt_key = jax.random.split(key)
            raw_dt = jax.random.uniform(
                nonlocal_dt_key,
                (linear_width,),
                jnp.float32,
                minval=math.log(1e-3),
                maxval=math.log(1e-1),
            )
            dt = jnp.maximum(jnp.exp(raw_dt), 1e-4)
            dt_bias = dt + jnp.log(-jnp.expm1(-dt))
            layer["self_attn"] = {
                "q_proj": normal((config.hidden_size, linear_width)),
                "k_proj": normal((config.hidden_size, linear_width)),
                "v_proj": normal((config.hidden_size, linear_width)),
                "conv1d": normal(
                    (3 * linear_width, config.linear_attention.short_conv_kernel_size)
                ),
                "f_a_proj": normal((config.hidden_size, config.linear_attention.head_dim)),
                "f_b_proj": normal((config.linear_attention.head_dim, linear_width)),
                "dt_bias": dt_bias,
                "A_log": jnp.zeros((config.linear_attention.num_heads,), jnp.float32),
                "b_proj": normal((config.hidden_size, config.linear_attention.num_heads)),
                "g_a_proj": normal((config.hidden_size, config.linear_attention.head_dim)),
                "g_b_proj": normal((config.linear_attention.head_dim, linear_width)),
                "o_norm": jnp.ones((config.linear_attention.head_dim,), dtype),
                "o_proj": normal((linear_width, config.hidden_size)),
            }
        else:
            layer["self_attn"] = {
                "q_a_proj": normal((config.hidden_size, config.q_lora_rank)),
                "q_a_layernorm": jnp.ones((config.q_lora_rank,), dtype),
                "q_b_proj": normal((config.q_lora_rank, config.num_attention_heads * config.qk_head_dim)),
                "kv_a_proj_with_mqa": normal(
                    (config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim)
                ),
                "kv_a_layernorm": jnp.ones((config.kv_lora_rank,), dtype),
                "kv_b_proj": normal(
                    (
                        config.kv_lora_rank,
                        config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                    )
                ),
                "o_proj": normal((config.num_attention_heads * config.v_head_dim, config.hidden_size)),
                "indexer": {
                    "wq_b": normal((config.q_lora_rank, config.index_n_heads * config.index_head_dim)),
                    "wk": normal((config.hidden_size, config.index_head_dim)),
                    "k_norm_weight": jnp.ones((config.index_head_dim,), dtype),
                    "k_norm_bias": jnp.zeros((config.index_head_dim,), dtype),
                    "weights_proj": normal((config.hidden_size, config.index_n_heads)),
                    "index_kpool_compress_ape": jnp.zeros(
                        (config.index_kpool, config.index_head_dim), dtype
                    ),
                    "index_kpool_compress_gate": jnp.ones(
                        (config.hidden_size, config.index_head_dim), dtype
                    ),
                },
            }
        if mlp_type == "dense":
            layer["mlp"] = {
                "gate_proj": normal((config.hidden_size, config.intermediate_size)),
                "up_proj": normal((config.hidden_size, config.intermediate_size)),
                "down_proj": normal((config.intermediate_size, config.hidden_size)),
            }
        else:
            shared_width = config.moe_intermediate_size * config.n_shared_experts
            layer["mlp"] = {
                "router": normal((config.hidden_size, config.n_routed_experts)),
                "router_correction_bias": jnp.zeros((config.n_routed_experts,), jnp.float32),
                "experts_gate_up": normal(
                    (config.n_routed_experts, config.hidden_size, 2 * config.moe_intermediate_size)
                ),
                "experts_down": normal(
                    (config.n_routed_experts, config.moe_intermediate_size, config.hidden_size)
                ),
                "shared_gate_proj": normal((config.hidden_size, shared_width)),
                "shared_up_proj": normal((config.hidden_size, shared_width)),
                "shared_down_proj": normal((shared_width, config.hidden_size)),
            }
        layers.append(layer)
    params: ArrayTree = {
        "embed_tokens": normal((config.vocab_size, config.hidden_size)),
        "layers": tuple(layers),
        "norm": jnp.ones((config.hidden_size,), dtype),
        "lm_head": normal((config.hidden_size, config.vocab_size)),
    }
    validate_params(params, config)
    return params


def validate_params(params: ArrayTree, config: Glm53TextConfig) -> None:
    expected = _expected_shapes(config)
    shape_leaf = lambda value: isinstance(value, tuple) and all(isinstance(item, int) for item in value)
    if jax.tree.structure(params) != jax.tree.structure(expected, is_leaf=shape_leaf):
        raise ValueError("GLM-5.3 parameter tree has unexpected keys or nesting")
    leaves = jax.tree.leaves(params)
    shapes = jax.tree.leaves(expected, is_leaf=shape_leaf)
    for index, (value, shape) in enumerate(zip(leaves, shapes, strict=True)):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"GLM-5.3 parameter leaf {index} has shape {value.shape}, expected {shape}")
    if sum(int(value.size) for value in leaves) != parameter_count(config):
        raise ValueError("GLM-5.3 parameter count disagrees with its shape contract")


def _rms_norm(x: jax.Array, weight: jax.Array, epsilon: float) -> jax.Array:
    output_dtype = x.dtype
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + epsilon)
    return (value * weight.astype(jnp.float32)).astype(output_dtype)


def _unweighted_rms_norm(x: jax.Array, epsilon: float) -> jax.Array:
    value = x.astype(jnp.float32)
    return value * jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + epsilon)


def _layer_norm(x: jax.Array, weight: jax.Array, bias: jax.Array, epsilon: float) -> jax.Array:
    output_dtype = x.dtype
    value = x.astype(jnp.float32)
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    value = (value - mean) * jax.lax.rsqrt(variance + epsilon)
    return (value * weight.astype(jnp.float32) + bias.astype(jnp.float32)).astype(output_dtype)


def _linear(
    x: jax.Array,
    kernel: jax.Array,
    path: tuple[str | int, ...],
    adapters: Mapping[str, Mapping[str, jax.Array]] | None,
    lora_config: LoRAConfig | None,
) -> jax.Array:
    if adapters is not None and format_parameter_path(path) in adapters:
        if lora_config is None:
            raise ValueError("LoRA adapters require a LoRAConfig")
        return lora_linear(x, kernel, adapter_for_path(adapters, path), config=lora_config)
    return jnp.einsum("...i,io->...o", x, kernel, precision=_PRECISION)


def _hyper_connection(
    hidden_streams: jax.Array,
    params: ArrayTree,
    config: Glm53TextConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    hc = config.hc_mult
    flat = hidden_streams.reshape(*hidden_streams.shape[:2], hc * config.hidden_size)
    flat = _unweighted_rms_norm(flat, config.rms_norm_eps)
    logits = jnp.einsum(
        "...i,io->...o",
        flat,
        params["fn"].astype(jnp.float32),
        precision=_PRECISION,
    )
    pre_w, post_w, comb_w = jnp.split(logits, (hc, 2 * hc), axis=-1)
    pre_b, post_b, comb_b = jnp.split(params["base"].astype(jnp.float32), (hc, 2 * hc))
    scales = params["scale"].astype(jnp.float32)
    pre = jax.nn.sigmoid(pre_w * scales[0] + pre_b) + config.hc_eps
    post = 2.0 * jax.nn.sigmoid(post_w * scales[1] + post_b)
    comb_logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * scales[2] + comb_b.reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, axis=-1) + config.hc_eps
    comb /= jnp.sum(comb, axis=-2, keepdims=True) + config.hc_eps
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb /= jnp.sum(comb, axis=-1, keepdims=True) + config.hc_eps
        comb /= jnp.sum(comb, axis=-2, keepdims=True) + config.hc_eps
    collapsed = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2)
    return post, comb, collapsed.astype(hidden_streams.dtype)


def _apply_hyper_residual(
    residual: jax.Array,
    sublayer_output: jax.Array,
    post: jax.Array,
    comb: jax.Array,
) -> jax.Array:
    dtype = residual.dtype
    mixed = jnp.einsum(
        "...ij,...jd->...id",
        jnp.swapaxes(comb, -1, -2).astype(dtype),
        residual,
        precision=_PRECISION,
    )
    return post.astype(dtype)[..., None] * sublayer_output[..., None, :] + mixed


def _causal_depthwise_conv(x: jax.Array, weight: jax.Array) -> jax.Array:
    batch, _, channels = x.shape
    kernel_size = weight.shape[1]
    state = jnp.zeros((batch, channels, kernel_size), x.dtype)
    weight_f32 = weight.astype(jnp.float32)

    def step(history, token):
        candidate = jnp.concatenate((history[..., 1:], token[..., None]), axis=-1)
        output = jnp.einsum(
            "bck,ck->bc", candidate.astype(jnp.float32), weight_f32, precision=_PRECISION
        )
        return candidate, jax.nn.silu(output).astype(x.dtype)

    _, output = jax.lax.scan(step, state, jnp.swapaxes(x, 0, 1))
    return jnp.swapaxes(output, 0, 1)


def recurrent_kimi_delta_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    decay_log: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Exact float32 recurrent KDA reference used for reduced parity."""

    output_dtype = query.dtype
    batch, _, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    query, key, value, decay_log, beta = (
        query.astype(jnp.float32),
        key.astype(jnp.float32),
        value.astype(jnp.float32),
        decay_log.astype(jnp.float32),
        beta.astype(jnp.float32),
    )
    query /= jnp.sqrt(jnp.sum(jnp.square(query), axis=-1, keepdims=True) + 1e-6)
    key /= jnp.sqrt(jnp.sum(jnp.square(key), axis=-1, keepdims=True) + 1e-6)
    query *= key_dim**-0.5
    state = jnp.zeros((batch, heads, key_dim, value_dim), jnp.float32)

    def step(recurrent, inputs):
        q_t, k_t, v_t, g_t, beta_t = inputs
        recurrent *= jnp.exp(g_t)[..., None]
        prediction = jnp.einsum("bhkv,bhk->bhv", recurrent, k_t, precision=_PRECISION)
        delta = (v_t - prediction) * beta_t[..., None]
        recurrent += jnp.einsum("bhk,bhv->bhkv", k_t, delta, precision=_PRECISION)
        output = jnp.einsum("bhkv,bhk->bhv", recurrent, q_t, precision=_PRECISION)
        return recurrent, output

    inputs = tuple(jnp.swapaxes(value, 0, 1) for value in (query, key, value, decay_log, beta))
    _, output = jax.lax.scan(step, state, inputs)
    return jnp.swapaxes(output, 0, 1).astype(output_dtype)


def _linear_attention(
    params: ArrayTree,
    config: Glm53TextConfig,
    hidden: jax.Array,
    attention_mask: jax.Array,
    *,
    layer_index: int,
    adapters: Mapping[str, Mapping[str, jax.Array]] | None,
    lora_config: LoRAConfig | None,
) -> jax.Array:
    batch, length, _ = hidden.shape
    prefix: tuple[str | int, ...] = ("layers", layer_index, "self_attn")
    hidden = jnp.where(attention_mask[..., None], hidden, jnp.zeros((), hidden.dtype))
    projected = [
        _linear(hidden, params[name], prefix + (name,), adapters, lora_config)
        for name in ("q_proj", "k_proj", "v_proj")
    ]
    mixed = _causal_depthwise_conv(jnp.concatenate(projected, axis=-1), params["conv1d"])
    query, key, value = jnp.split(mixed, 3, axis=-1)
    heads = config.linear_attention.num_heads
    head_dim = config.linear_attention.head_dim
    query, key, value = [item.reshape(batch, length, heads, head_dim) for item in (query, key, value)]

    forget = _linear(hidden, params["f_a_proj"], prefix + ("f_a_proj",), None, None)
    forget = _linear(forget, params["f_b_proj"], prefix + ("f_b_proj",), None, None)
    forget = (forget.astype(jnp.float32) + params["dt_bias"].astype(jnp.float32)).reshape(
        batch, length, heads, head_dim
    )
    decay_rate = jnp.exp(params["A_log"].astype(jnp.float32))[None, None, :, None]
    decay_log = config.linear_attention.gate_lower_bound * jax.nn.sigmoid(decay_rate * forget)
    beta = jax.nn.sigmoid(
        _linear(hidden, params["b_proj"], prefix + ("b_proj",), None, None).astype(jnp.float32)
    )
    output = recurrent_kimi_delta_attention(query, key, value, decay_log, beta)

    gate = _linear(hidden, params["g_a_proj"], prefix + ("g_a_proj",), None, None)
    gate = _linear(gate, params["g_b_proj"], prefix + ("g_b_proj",), None, None)
    gate = gate.reshape(batch, length, heads, head_dim)
    output_dtype = output.dtype
    normalized = output.astype(jnp.float32)
    normalized *= jax.lax.rsqrt(
        jnp.mean(jnp.square(normalized), axis=-1, keepdims=True) + config.rms_norm_eps
    )
    normalized *= params["o_norm"].astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
    output = normalized.astype(output_dtype).reshape(batch, length, heads * head_dim)
    return _linear(output, params["o_proj"], prefix + ("o_proj",), adapters, lora_config)


def _gather_sequence(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather `[batch, ...indices, trailing...]` from a `[batch, length, ...]` array."""

    batch_indices = jnp.arange(values.shape[0]).reshape(
        values.shape[0], *([1] * (indices.ndim - 1))
    )
    return values[batch_indices, indices]


def _sparse_topk_indices(
    params: ArrayTree,
    config: Glm53TextConfig,
    hidden: jax.Array,
    q_residual: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """No-cache DSA k-pool indexer with static shapes for SFT."""

    batch, length, _ = hidden.shape
    q = jnp.einsum("...i,io->...o", q_residual, params["wq_b"], precision=_PRECISION)
    q = q.reshape(batch, length, config.index_n_heads, config.index_head_dim)
    key = jnp.einsum("...i,io->...o", hidden, params["wk"], precision=_PRECISION)
    key = _layer_norm(key, params["k_norm_weight"], params["k_norm_bias"], 1e-6)
    gate_scores = jnp.einsum(
        "...i,io->...o", hidden, params["index_kpool_compress_gate"], precision=_PRECISION
    )
    valid_keys = attention_mask.astype(bool)
    query_positions = jnp.arange(length)[None, :, None]
    key_positions = jnp.arange(length)[None, None, :]
    visible = (key_positions <= query_positions) & valid_keys[:, None, :]

    pool_size = config.index_kpool
    pool_count = (length + pool_size - 1) // pool_size
    has_key = jnp.any(valid_keys, axis=-1)
    first_key = jnp.where(has_key, jnp.argmax(valid_keys, axis=-1), length)
    offsets = jnp.arange(pool_count * pool_size).reshape(1, pool_count, pool_size)
    pool_indices = first_key[:, None, None] + offsets
    safe_pool_indices = jnp.clip(pool_indices, 0, length - 1)
    grouped_keys = _gather_sequence(key, safe_pool_indices)
    grouped_gate_scores = _gather_sequence(gate_scores, safe_pool_indices)
    grouped_valid = _gather_sequence(valid_keys, safe_pool_indices) & (pool_indices < length)
    pool_valid = jnp.all(grouped_valid, axis=-1)
    pool_indices = jnp.where(grouped_valid, pool_indices, -1)

    logits = grouped_gate_scores.astype(jnp.float32) + params[
        "index_kpool_compress_ape"
    ].astype(jnp.float32)[None, None]
    logits = jnp.where(grouped_valid[..., None], logits, -jnp.inf)
    probabilities = jnp.nan_to_num(jax.nn.softmax(logits, axis=2)).astype(grouped_keys.dtype)
    pool_keys = jnp.sum(probabilities * grouped_keys, axis=2)

    scores = jnp.einsum(
        "bshd,bpd->bshp",
        q.astype(jnp.float32),
        pool_keys.astype(jnp.float32),
        precision=_PRECISION,
    )
    scores = jax.nn.relu(scores * (config.index_head_dim**-0.5))
    weights = jnp.einsum(
        "...i,io->...o", hidden, params["weights_proj"], precision=_PRECISION
    ).astype(jnp.float32) * (config.index_n_heads**-0.5)
    index_scores = jnp.einsum("bsh,bshp->bsp", weights, scores, precision=_PRECISION)

    pool_end = jnp.clip(pool_indices[..., -1], 0, length - 1)
    pool_visible = jnp.take_along_axis(
        visible,
        jnp.broadcast_to(pool_end[:, None, :], (batch, length, pool_count)),
        axis=-1,
    )
    valid_candidates = pool_visible & pool_valid[:, None, :]
    index_scores = jnp.where(valid_candidates, index_scores, jnp.finfo(index_scores.dtype).min)
    select_k = min(config.index_topk // pool_size, pool_count)
    _, selected = jax.lax.top_k(index_scores, select_k)
    selected_valid = jnp.take_along_axis(valid_candidates, selected, axis=-1)
    expanded_pools = jnp.broadcast_to(pool_indices[:, None], (batch, length, pool_count, pool_size))
    selected_indices = jnp.take_along_axis(expanded_pools, selected[..., None], axis=2)
    selected_indices = jnp.where(selected_valid[..., None], selected_indices, -1)
    topk_indices = selected_indices.reshape(batch, length, select_k * pool_size)

    output_width = config.index_topk
    if config.index_kpool_always_select_tail:
        max_tail_width = pool_size - 1
        output_width += max_tail_width
        if max_tail_width:
            visible_count = jnp.sum(visible, axis=-1, dtype=jnp.int32)
            tail_count = jnp.mod(visible_count, pool_size)
            tail_offsets = jnp.arange(max_tail_width)
            tail_start = first_key[:, None] + visible_count - tail_count
            tail_indices = tail_start[..., None] + tail_offsets
            tail_valid = (tail_offsets < tail_count[..., None]) & (tail_indices < length)
            safe_tail = jnp.clip(tail_indices, 0, length - 1)
            tail_visible = jnp.take_along_axis(visible, safe_tail, axis=-1)
            tail_indices = jnp.where(tail_valid & tail_visible, tail_indices, -1)
            topk_indices = jnp.concatenate((topk_indices, tail_indices), axis=-1)
    if topk_indices.shape[-1] < output_width:
        topk_indices = jnp.pad(
            topk_indices,
            ((0, 0), (0, 0), (0, output_width - topk_indices.shape[-1])),
            constant_values=-1,
        )
    topk_indices = topk_indices[..., :output_width]
    topk_indices = jnp.where(attention_mask[..., None], topk_indices, -1)
    return jax.lax.stop_gradient(topk_indices.astype(jnp.int32))


def _sparse_attention(
    params: ArrayTree,
    config: Glm53TextConfig,
    hidden: jax.Array,
    attention_mask: jax.Array,
    *,
    layer_index: int,
    adapters: Mapping[str, Mapping[str, jax.Array]] | None,
    lora_config: LoRAConfig | None,
) -> jax.Array:
    batch, length, _ = hidden.shape
    prefix: tuple[str | int, ...] = ("layers", layer_index, "self_attn")
    q_residual = _linear(hidden, params["q_a_proj"], prefix + ("q_a_proj",), adapters, lora_config)
    q_residual = _rms_norm(q_residual, params["q_a_layernorm"], config.rms_norm_eps)
    query = _linear(q_residual, params["q_b_proj"], prefix + ("q_b_proj",), adapters, lora_config)
    query = query.reshape(batch, length, config.num_attention_heads, config.qk_head_dim)
    query = jnp.transpose(query, (0, 2, 1, 3))

    compressed_kv = _linear(
        hidden,
        params["kv_a_proj_with_mqa"],
        prefix + ("kv_a_proj_with_mqa",),
        adapters,
        lora_config,
    )
    kv_pass = compressed_kv[..., : config.kv_lora_rank]
    kv_pass = _rms_norm(kv_pass, params["kv_a_layernorm"], config.rms_norm_eps)
    expanded_kv = _linear(
        kv_pass,
        params["kv_b_proj"],
        prefix + ("kv_b_proj",),
        adapters,
        lora_config,
    )
    expanded_kv = expanded_kv.reshape(
        batch,
        length,
        config.num_attention_heads,
        config.qk_nope_head_dim + config.v_head_dim,
    )
    key = jnp.transpose(expanded_kv[..., : config.qk_nope_head_dim], (0, 2, 1, 3))
    value = jnp.transpose(expanded_kv[..., config.qk_nope_head_dim :], (0, 2, 1, 3))
    topk_indices = _sparse_topk_indices(
        params["indexer"], config, hidden, q_residual, attention_mask
    )

    safe_indices = jnp.clip(topk_indices, 0, length - 1)
    key_by_token = jnp.transpose(key, (0, 2, 1, 3))
    value_by_token = jnp.transpose(value, (0, 2, 1, 3))
    selected_key = _gather_sequence(key_by_token, safe_indices)
    selected_value = _gather_sequence(value_by_token, safe_indices)
    selected_key = jnp.transpose(selected_key, (0, 3, 1, 2, 4))
    selected_value = jnp.transpose(selected_value, (0, 3, 1, 2, 4))
    scores = jnp.einsum(
        "bhsd,bhskd->bhsk",
        query,
        selected_key,
        precision=_PRECISION,
    ) * (config.qk_head_dim**-0.5)
    valid = topk_indices[:, None] >= 0
    scores = jnp.where(valid, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
    output = jnp.einsum("bhsk,bhskd->bhsd", weights, selected_value, precision=_PRECISION)
    output = jnp.transpose(output, (0, 2, 1, 3)).reshape(
        batch, length, config.num_attention_heads * config.v_head_dim
    )
    return _linear(output, params["o_proj"], prefix + ("o_proj",), adapters, lora_config)


def _swiglu(gate: jax.Array, up: jax.Array, limit: float) -> jax.Array:
    gate = jnp.minimum(gate, jnp.asarray(limit, gate.dtype))
    up = jnp.clip(up, -limit, limit)
    return jax.nn.silu(gate) * up


def _dense_mlp(params: ArrayTree, config: Glm53TextConfig, hidden: jax.Array) -> jax.Array:
    gate = jnp.einsum("...i,io->...o", hidden, params["gate_proj"], precision=_PRECISION)
    up = jnp.einsum("...i,io->...o", hidden, params["up_proj"], precision=_PRECISION)
    activated = _swiglu(gate, up, config.swiglu_limit)
    return jnp.einsum("...i,io->...o", activated, params["down_proj"], precision=_PRECISION)


def _moe(params: ArrayTree, config: Glm53TextConfig, hidden: jax.Array) -> jax.Array:
    original_shape = hidden.shape
    tokens = hidden.reshape(-1, config.hidden_size)
    router_logits = jnp.einsum(
        "ti,ie->te",
        tokens.astype(jnp.float32),
        params["router"].astype(jnp.float32),
        precision=_PRECISION,
    )
    scores = jax.nn.sigmoid(router_logits)
    choice_scores = scores + params["router_correction_bias"].astype(jnp.float32)
    experts_per_group = config.n_routed_experts // config.n_group
    grouped = choice_scores.reshape(tokens.shape[0], config.n_group, experts_per_group)
    group_scores = jnp.sum(jax.lax.top_k(grouped, 2)[0], axis=-1)
    _, selected_groups = jax.lax.top_k(group_scores, config.topk_group)
    group_mask = jnp.sum(jax.nn.one_hot(selected_groups, config.n_group, dtype=jnp.int32), axis=1) > 0
    expert_mask = jnp.repeat(group_mask, experts_per_group, axis=-1)
    choice_scores = jnp.where(expert_mask, choice_scores, -jnp.inf)
    _, topk_indices = jax.lax.top_k(choice_scores, config.num_experts_per_tok)
    topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)
    if config.norm_topk_prob:
        topk_weights /= jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
    topk_weights *= config.routed_scaling_factor

    selected_gate_up = params["experts_gate_up"][topk_indices]
    gate_up = jnp.einsum(
        "th,tkhi->tki", tokens, selected_gate_up, precision=_PRECISION
    )
    gate, up = jnp.split(gate_up, 2, axis=-1)
    activated = _swiglu(gate, up, config.swiglu_limit)
    selected_down = params["experts_down"][topk_indices]
    routed = jnp.einsum("tki,tkih->tkh", activated, selected_down, precision=_PRECISION)
    routed = jnp.sum(routed * topk_weights.astype(routed.dtype)[..., None], axis=1)

    shared_gate = jnp.einsum(
        "...i,io->...o", hidden, params["shared_gate_proj"], precision=_PRECISION
    )
    shared_up = jnp.einsum(
        "...i,io->...o", hidden, params["shared_up_proj"], precision=_PRECISION
    )
    shared = _swiglu(shared_gate, shared_up, config.swiglu_limit)
    shared = jnp.einsum(
        "...i,io->...o", shared, params["shared_down_proj"], precision=_PRECISION
    )
    return routed.reshape(original_shape) + shared


def _decoder_layer(
    hidden_streams: jax.Array,
    params: ArrayTree,
    config: Glm53TextConfig,
    attention_mask: jax.Array,
    *,
    layer_index: int,
    adapters: Mapping[str, Mapping[str, jax.Array]] | None,
    lora_config: LoRAConfig | None,
) -> jax.Array:
    residual = hidden_streams
    post, comb, collapsed = _hyper_connection(hidden_streams, params["attn_hc"], config)
    collapsed = _rms_norm(collapsed, params["input_layernorm"], config.rms_norm_eps)
    if config.layer_types[layer_index] == "linear_attention":
        output = _linear_attention(
            params["self_attn"],
            config,
            collapsed,
            attention_mask,
            layer_index=layer_index,
            adapters=adapters,
            lora_config=lora_config,
        )
    else:
        output = _sparse_attention(
            params["self_attn"],
            config,
            collapsed,
            attention_mask,
            layer_index=layer_index,
            adapters=adapters,
            lora_config=lora_config,
        )
    hidden_streams = _apply_hyper_residual(residual, output, post, comb)

    residual = hidden_streams
    post, comb, collapsed = _hyper_connection(hidden_streams, params["ffn_hc"], config)
    collapsed = _rms_norm(collapsed, params["post_attention_layernorm"], config.rms_norm_eps)
    if config.mlp_layer_types[layer_index] == "dense":
        output = _dense_mlp(params["mlp"], config, collapsed)
    else:
        output = _moe(params["mlp"], config, collapsed)
    return _apply_hyper_residual(residual, output, post, comb)


def forward(
    params: ArrayTree,
    config: Glm53TextConfig,
    input_ids: jax.Array,
    *,
    attention_mask: jax.Array | None = None,
    adapters: Mapping[str, Mapping[str, jax.Array]] | None = None,
    lora_config: LoRAConfig | None = None,
    remat: bool = False,
) -> jax.Array:
    """Text-only training forward; cache, vision, and MTP are intentionally absent."""

    input_ids = jnp.asarray(input_ids, jnp.int32)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids, dtype=bool)
    else:
        attention_mask = jnp.asarray(attention_mask, dtype=bool)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if (adapters is None) != (lora_config is None):
        raise ValueError("adapters and lora_config must be provided together")
    if lora_config is not None and lora_config.dropout:
        raise NotImplementedError("GLM LoRA dropout RNG plumbing is not part of the reduced parity gate")

    hidden = params["embed_tokens"][input_ids]
    hidden_streams = jnp.broadcast_to(
        hidden[:, :, None, :],
        (*hidden.shape[:2], config.hc_mult, config.hidden_size),
    )
    for layer_index, layer_params in enumerate(params["layers"]):
        layer_fn = lambda streams, current_params: _decoder_layer(
            streams,
            current_params,
            config,
            attention_mask,
            layer_index=layer_index,
            adapters=adapters,
            lora_config=lora_config,
        )
        if remat:
            layer_fn = jax.checkpoint(layer_fn)
        hidden_streams = layer_fn(hidden_streams, layer_params)
    hidden = _rms_norm(jnp.mean(hidden_streams, axis=2), params["norm"], config.rms_norm_eps)
    return jnp.einsum("...i,iv->...v", hidden, params["lm_head"], precision=_PRECISION)


def _numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def convert_hf_state_dict(
    state: Mapping[str, Any],
    config: Glm53TextConfig,
    *,
    dtype: Any = jnp.bfloat16,
    strict: bool = True,
) -> ArrayTree:
    """Convert an unquantized Transformers text state for reduced parity.

    The public block-FP8 checkpoint is intentionally rejected here.  Its
    `weight_scale_inv` tensors require the later scale-aware, sharded loader;
    passing the F8 values through this dense converter would be incorrect.
    """

    scale_keys = [name for name in state if name.endswith("weight_scale_inv")]
    if scale_keys:
        raise NotImplementedError(
            "block-FP8 checkpoints require the scale-aware direct loader; "
            f"found {len(scale_keys)} weight_scale_inv tensors"
        )
    dtype = jnp.dtype(dtype)
    prefixes = ("model.language_model.", "language_model.", "")
    try:
        prefix = next(candidate for candidate in prefixes if candidate + "embed_tokens.weight" in state)
    except StopIteration as error:
        raise KeyError("checkpoint has no GLM-5.3 text embedding tensor") from error
    consumed: set[str] = set()

    def get(
        *suffixes: str,
        target_dtype: Any = dtype,
        transpose: bool = False,
    ) -> jax.Array:
        candidates = [prefix + suffix for suffix in suffixes]
        for name in candidates:
            if name in state:
                consumed.add(name)
                value = _numpy_value(state[name])
                if transpose:
                    value = np.swapaxes(value, -1, -2)
                return jnp.asarray(value, target_dtype)
        raise KeyError(f"GLM-5.3 checkpoint is missing required tensor; tried {candidates}")

    def get_exact(name: str, *, transpose: bool = False) -> jax.Array:
        candidates = (name, prefix + name)
        for candidate in candidates:
            if candidate in state:
                consumed.add(candidate)
                value = _numpy_value(state[candidate])
                if transpose:
                    value = np.swapaxes(value, -1, -2)
                return jnp.asarray(value, dtype)
        raise KeyError(f"GLM-5.3 checkpoint is missing required tensor; tried {list(candidates)}")

    def hyper(layer_prefix: str, site: str) -> ArrayTree:
        checkpoint_name = "attn" if site == "attn_hc" else "ffn"
        return {
            "fn": get(
                layer_prefix + f"{site}.fn",
                layer_prefix + f"hc_{checkpoint_name}_fn",
                transpose=True,
            ),
            "base": get(
                layer_prefix + f"{site}.base",
                layer_prefix + f"hc_{checkpoint_name}_base",
            ),
            "scale": get(
                layer_prefix + f"{site}.scale",
                layer_prefix + f"hc_{checkpoint_name}_scale",
            ),
        }

    layers: list[ArrayTree] = []
    linear_width = config.linear_attention.num_heads * config.linear_attention.head_dim
    for layer_index, (attention_type, mlp_type) in enumerate(
        zip(config.layer_types, config.mlp_layer_types, strict=True)
    ):
        layer_prefix = f"layers.{layer_index}."
        attention_prefix = layer_prefix + "self_attn."
        layer: ArrayTree = {
            "input_layernorm": get(layer_prefix + "input_layernorm.weight"),
            "post_attention_layernorm": get(layer_prefix + "post_attention_layernorm.weight"),
            "attn_hc": hyper(layer_prefix, "attn_hc"),
            "ffn_hc": hyper(layer_prefix, "ffn_hc"),
        }
        if attention_type == "linear_attention":
            conv = get(attention_prefix + "conv1d.weight")
            if conv.ndim == 3 and conv.shape[1] == 1:
                conv = conv[:, 0, :]
            if conv.shape != (3 * linear_width, config.linear_attention.short_conv_kernel_size):
                raise ValueError(f"unexpected GLM-5.3 conv1d shape {conv.shape}")
            layer["self_attn"] = {
                "q_proj": get(attention_prefix + "q_proj.weight", transpose=True),
                "k_proj": get(attention_prefix + "k_proj.weight", transpose=True),
                "v_proj": get(attention_prefix + "v_proj.weight", transpose=True),
                "conv1d": conv,
                "f_a_proj": get(attention_prefix + "forget_gate.f_a_proj.weight", attention_prefix + "f_a_proj.weight", transpose=True),
                "f_b_proj": get(attention_prefix + "forget_gate.f_b_proj.weight", attention_prefix + "f_b_proj.weight", transpose=True),
                "dt_bias": get(
                    attention_prefix + "forget_gate.dt_bias",
                    attention_prefix + "dt_bias",
                    target_dtype=jnp.float32,
                ),
                "A_log": get(
                    attention_prefix + "forget_gate.A_log",
                    attention_prefix + "A_log",
                    target_dtype=jnp.float32,
                ),
                "b_proj": get(attention_prefix + "b_proj.weight", transpose=True),
                "g_a_proj": get(attention_prefix + "g_a_proj.weight", transpose=True),
                "g_b_proj": get(attention_prefix + "g_b_proj.weight", transpose=True),
                "o_norm": get(attention_prefix + "o_norm.weight"),
                "o_proj": get(attention_prefix + "o_proj.weight", transpose=True),
            }
        else:
            indexer_prefix = attention_prefix + "indexer."
            layer["self_attn"] = {
                "q_a_proj": get(attention_prefix + "q_a_proj.weight", transpose=True),
                "q_a_layernorm": get(attention_prefix + "q_a_layernorm.weight"),
                "q_b_proj": get(attention_prefix + "q_b_proj.weight", transpose=True),
                "kv_a_proj_with_mqa": get(
                    attention_prefix + "kv_a_proj_with_mqa.weight", transpose=True
                ),
                "kv_a_layernorm": get(attention_prefix + "kv_a_layernorm.weight"),
                "kv_b_proj": get(attention_prefix + "kv_b_proj.weight", transpose=True),
                "o_proj": get(attention_prefix + "o_proj.weight", transpose=True),
                "indexer": {
                    "wq_b": get(indexer_prefix + "wq_b.weight", transpose=True),
                    "wk": get(indexer_prefix + "wk.weight", transpose=True),
                    "k_norm_weight": get(indexer_prefix + "k_norm.weight"),
                    "k_norm_bias": get(indexer_prefix + "k_norm.bias"),
                    "weights_proj": get(indexer_prefix + "weights_proj.weight", transpose=True),
                    "index_kpool_compress_ape": get(indexer_prefix + "index_kpool_compress_ape"),
                    "index_kpool_compress_gate": get(
                        indexer_prefix + "index_kpool_compress_gate", transpose=True
                    ),
                },
            }

        mlp_prefix = layer_prefix + "mlp."
        if mlp_type == "dense":
            layer["mlp"] = {
                "gate_proj": get(mlp_prefix + "gate_proj.weight", transpose=True),
                "up_proj": get(mlp_prefix + "up_proj.weight", transpose=True),
                "down_proj": get(mlp_prefix + "down_proj.weight", transpose=True),
            }
        else:
            packed_gate_up = prefix + mlp_prefix + "experts.gate_up_proj"
            packed_down = prefix + mlp_prefix + "experts.down_proj"
            if packed_gate_up in state and packed_down in state:
                consumed.update((packed_gate_up, packed_down))
                experts_gate_up = jnp.asarray(
                    np.transpose(_numpy_value(state[packed_gate_up]), (0, 2, 1)), dtype
                )
                experts_down = jnp.asarray(
                    np.transpose(_numpy_value(state[packed_down]), (0, 2, 1)), dtype
                )
            else:
                gate_up: list[jax.Array] = []
                down: list[jax.Array] = []
                for expert_index in range(config.n_routed_experts):
                    expert_prefix = mlp_prefix + f"experts.{expert_index}."
                    gate = get(expert_prefix + "gate_proj.weight")
                    up = get(expert_prefix + "up_proj.weight")
                    gate_up.append(jnp.concatenate((gate, up), axis=0).T)
                    down.append(get(expert_prefix + "down_proj.weight").T)
                experts_gate_up = jnp.stack(gate_up)
                experts_down = jnp.stack(down)
            layer["mlp"] = {
                "router": get(mlp_prefix + "gate.weight", transpose=True, target_dtype=jnp.float32),
                "router_correction_bias": get(
                    mlp_prefix + "gate.e_score_correction_bias", target_dtype=jnp.float32
                ),
                "experts_gate_up": experts_gate_up,
                "experts_down": experts_down,
                "shared_gate_proj": get(mlp_prefix + "shared_experts.gate_proj.weight", transpose=True),
                "shared_up_proj": get(mlp_prefix + "shared_experts.up_proj.weight", transpose=True),
                "shared_down_proj": get(mlp_prefix + "shared_experts.down_proj.weight", transpose=True),
            }
        layers.append(layer)

    params: ArrayTree = {
        "embed_tokens": get("embed_tokens.weight"),
        "layers": tuple(layers),
        "norm": get("norm.weight"),
        "lm_head": get_exact("lm_head.weight", transpose=True),
    }
    validate_params(params, config)
    if strict:
        ignored_prefixes = (
            "model.visual.",
            prefix + f"layers.{config.num_hidden_layers}.",
        )
        unexpected = [
            name
            for name in state
            if name not in consumed and not any(name.startswith(candidate) for candidate in ignored_prefixes)
        ]
        if unexpected:
            raise ValueError(f"unexpected GLM-5.3 checkpoint tensors: {unexpected[:20]}")
    return params


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
    "ArrayTree",
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
    "_PRECISION",
    "attention_lora_parameter_count",
    "attention_lora_target_paths",
    "convert_hf_state_dict",
    "forward",
    "init_params",
    "parameter_count",
    "recurrent_kimi_delta_attention",
    "tiny_config",
    "validate_params",
    "v4_32_lora_preflight",
]
