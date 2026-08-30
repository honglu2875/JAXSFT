import json

import pytest

from jaxsft.models.glm5_3_flash import (
    GIB,
    OFFICIAL_CHECKPOINT,
    CheckpointContract,
    Glm53TextConfig,
    SafetensorsIndex,
    attention_lora_parameter_count,
    attention_lora_target_paths,
    v4_32_lora_preflight,
)


def _config_dict(*, official_shapes: bool = False):
    if official_shapes:
        layer_types = tuple(
            "deepseek_sparse_attention" if index % 4 == 3 else "linear_attention"
            for index in range(45)
        )
        hidden_size = 4096
        num_heads = 64
        q_rank = 1536
        kv_rank = 512
        head_dim = 256
        linear_heads = 64
        linear_head_dim = 128
        experts = 288
        top_experts = 8
        intermediate = 12288
        moe_intermediate = 2048
        mlp_types = ("dense",) * 3 + ("sparse",) * 42
    else:
        layer_types = ("linear_attention", "deepseek_sparse_attention")
        hidden_size = 32
        num_heads = 4
        q_rank = 8
        kv_rank = 4
        head_dim = 8
        linear_heads = 4
        linear_head_dim = 8
        experts = 4
        top_experts = 2
        intermediate = 64
        moe_intermediate = 16
        mlp_types = ("dense", "sparse")
    kda = tuple(index for index, kind in enumerate(layer_types) if kind == "linear_attention")
    full = tuple(index for index, kind in enumerate(layer_types) if kind == "deepseek_sparse_attention")
    return {
        "model_type": "glm5_next",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": 154880 if official_shapes else 128,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate,
            "moe_intermediate_size": moe_intermediate,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads,
            "layer_types": layer_types,
            "mlp_layer_types": mlp_types,
            "linear_attn_config": {
                "num_heads": linear_heads,
                "head_dim": linear_head_dim,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5,
                "kda_layers": kda,
                "full_attn_layers": full,
            },
            "first_k_dense_replace": sum(kind == "dense" for kind in mlp_types),
            "q_lora_rank": q_rank,
            "kv_lora_rank": kv_rank,
            "qk_head_dim": head_dim,
            "qk_nope_head_dim": head_dim,
            "qk_rope_head_dim": 0,
            "v_head_dim": head_dim,
            "n_routed_experts": experts,
            "n_shared_experts": 1,
            "num_experts_per_tok": top_experts,
            "rms_norm_eps": 1e-5,
            "mhc": True,
            "hc_mult": 4,
            "index_n_heads": num_heads // 2,
            "index_head_dim": head_dim,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }


def _official_config() -> Glm53TextConfig:
    return Glm53TextConfig.from_dict(_config_dict(official_shapes=True))


def _official_sized_index() -> SafetensorsIndex:
    return SafetensorsIndex(
        total_size_bytes=OFFICIAL_CHECKPOINT.total_size_bytes,
        tensor_files=(("weight", "model-00001-of-00062.safetensors"),),
        sha256=OFFICIAL_CHECKPOINT.index_sha256,
    )


def test_config_parses_hybrid_layout_and_rejects_inconsistent_partition():
    config = Glm53TextConfig.from_dict(_config_dict())
    assert config.layer_types == ("linear_attention", "deepseek_sparse_attention")
    assert config.source_quant_block_shape == (128, 128)

    broken = _config_dict()
    broken["text_config"]["linear_attn_config"]["full_attn_layers"] = []
    with pytest.raises(ValueError, match="exact partition"):
        Glm53TextConfig.from_dict(broken)


def test_attention_lora_count_uses_architecture_specific_matrix_shapes():
    tiny = Glm53TextConfig.from_dict(_config_dict())
    # KDA: four 32x32 targets => 4 * rank * (32 + 32) = 1,024.
    # Sparse attention: q_a/q_b/kv_a/kv_b/o => rank * 248 = 992.
    assert attention_lora_parameter_count(tiny, rank=4) == 2_016
    official = _official_config()
    assert attention_lora_parameter_count(official, rank=8) == 20_578_304
    targets = attention_lora_target_paths(official)
    assert len(targets) == 34 * 4 + 11 * 5
    assert ("layers", 0, "self_attn", "q_proj") in targets
    assert ("layers", 3, "self_attn", "kv_b_proj") in targets


def test_safetensors_index_is_strict_and_verifies_contract(tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 17},
                "weight_map": {"b": "model-2.safetensors", "a": "model-1.safetensors"},
            }
        )
    )
    index = SafetensorsIndex.from_path(path)
    assert index.tensor_files == (
        ("a", "model-1.safetensors"),
        ("b", "model-2.safetensors"),
    )
    contract = CheckpointContract(
        repo_id="test/model",
        revision="0" * 40,
        config_sha256="1" * 64,
        index_sha256=index.sha256,
        total_size_bytes=17,
        tensor_count=2,
        shard_count=2,
        parameter_counts_by_dtype=(("BF16", 8),),
        maximum_source_shard_bytes=11,
    )
    index.verify(contract)

    path.write_text('{"metadata":{"total_size":1},"weight_map":{"x":"../escape.safetensors"}}')
    with pytest.raises(ValueError, match="basename"):
        SafetensorsIndex.from_path(path)


def test_bfloat16_expansion_fails_v4_32_before_activations():
    assert OFFICIAL_CHECKPOINT.logical_parameter_count == 321_323_031_390
    assert OFFICIAL_CHECKPOINT.expanded_parameter_bytes("bfloat16") == 642_646_653_816
    assert OFFICIAL_CHECKPOINT.expanded_parameter_bytes("bfloat16") > 16 * 32 * GIB

    plan = v4_32_lora_preflight(
        _official_config(),
        _official_sized_index(),
        execution_weight_format="bfloat16",
    )
    assert not plan.static_fit
    assert not plan.runnable
    assert any("persistent BF16" in blocker for blocker in plan.blockers)


def test_fp8_preflight_distinguishes_byte_fit_from_execution_evidence():
    config = _official_config()
    index = _official_sized_index()
    unproven = v4_32_lora_preflight(config, index, rank=8)
    assert unproven.static_fit
    assert not unproven.runnable
    assert unproven.adapter_parameter_count == 20_578_304
    assert unproven.free_per_device_bytes > 0
    assert len(unproven.blockers) == 2

    proven = v4_32_lora_preflight(
        config,
        index,
        rank=8,
        executable_kernel_proven=True,
        direct_loader_proven=True,
    )
    assert proven.static_fit
    assert proven.runnable
    assert not proven.blockers
