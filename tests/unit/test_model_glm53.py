import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.lora import LoRAConfig, format_parameter_path, init_lora_adapters, merge_lora_adapters
from jaxsft.models.glm5_3_flash import (
    BatchedBlockFP8LinearKernel,
    BlockFP8LinearKernel,
    GIB,
    OFFICIAL_CHECKPOINT,
    _PRECISION,
    CheckpointContract,
    Glm53TextConfig,
    SafetensorsIndex,
    SafetensorsShardHeader,
    attention_lora_parameter_count,
    attention_lora_target_paths,
    block_fp8_linear,
    checkpoint_text_tensor_specs,
    dequantize_block_fp8,
    forward,
    init_params,
    parameter_count,
    selected_block_fp8_linear,
    tiny_config,
    validate_params,
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
            "indexer_types": ("full",) * len(layer_types),
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
            "index_topk": 2048 if official_shapes else 4,
            "index_kpool": 4 if official_shapes else 2,
            "index_kpool_compress": True,
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


def _unit_scale_fp8_kernel(kernel, *, block_shape=(2, 2)):
    source = jnp.swapaxes(kernel, -1, -2).astype(jnp.float8_e4m3fn)
    bits = jax.lax.bitcast_convert_type(source, jnp.uint8)
    scales = jnp.ones(
        (source.shape[0] // block_shape[0], source.shape[1] // block_shape[1]),
        jnp.float32,
    )
    wrapper = BlockFP8LinearKernel(
        bits,
        scales,
        block_shape=block_shape,
        compute_dtype=jnp.float32,
    )
    reference = jnp.swapaxes(source.astype(jnp.float32), -1, -2)
    return wrapper, reference


def _unit_scale_batched_fp8_kernel(kernel, *, block_shape=(2, 2)):
    source = jnp.swapaxes(kernel, -1, -2).astype(jnp.float8_e4m3fn)
    bits = jax.lax.bitcast_convert_type(source, jnp.uint8)
    scales = jnp.ones(
        (
            source.shape[0],
            source.shape[1] // block_shape[0],
            source.shape[2] // block_shape[1],
        ),
        jnp.float32,
    )
    wrapper = BatchedBlockFP8LinearKernel(
        bits,
        scales,
        block_shape=block_shape,
        compute_dtype=jnp.float32,
    )
    reference = jnp.swapaxes(source.astype(jnp.float32), -1, -2)
    return wrapper, reference


def _quantize_tiny_linear_tree(value, path=()):
    if isinstance(value, tuple):
        pairs = [_quantize_tiny_linear_tree(item, path + (index,)) for index, item in enumerate(value)]
        return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    if isinstance(value, dict):
        quantized = {}
        reference = {}
        if "experts_gate_up" in value:
            gate, up = jnp.split(value["experts_gate_up"], 2, axis=-1)
            quantized["experts_gate"], gate_reference = _unit_scale_batched_fp8_kernel(gate)
            quantized["experts_up"], up_reference = _unit_scale_batched_fp8_kernel(up)
            quantized["experts_down"], down_reference = _unit_scale_batched_fp8_kernel(
                value["experts_down"]
            )
            reference["experts_gate_up"] = jnp.concatenate((gate_reference, up_reference), axis=-1)
            reference["experts_down"] = down_reference
        for key, item in value.items():
            if key in {"experts_gate_up", "experts_down"} and "experts_gate_up" in value:
                continue
            quantized[key], reference[key] = _quantize_tiny_linear_tree(item, path + (key,))
        return quantized, reference
    if (
        hasattr(value, "ndim")
        and value.ndim == 2
        and path[-1] not in {"embed_tokens", "conv1d", "index_kpool_compress_ape"}
    ):
        return _unit_scale_fp8_kernel(value)
    return value, value


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


def test_checkpoint_text_schema_names_every_logical_source_and_expert_pack_member():
    specs = checkpoint_text_tensor_specs(_official_config())
    assert len(specs) == 37_534
    assert len({spec.source_name for spec in specs}) == len(specs)
    by_name = {spec.source_name: spec for spec in specs}
    assert by_name["lm_head.weight"].source_shape == (154_880, 4_096)
    assert by_name["lm_head.weight"].transform == "transpose"
    conv = by_name["model.language_model.layers.0.self_attn.q_conv1d.weight"]
    assert conv.source_shape == (8_192, 1, 4)
    assert conv.transform == "squeeze_conv"
    expert = by_name["model.language_model.layers.3.mlp.experts.287.down_proj.weight"]
    assert expert.source_shape == (4_096, 2_048)
    assert expert.target_path[-1] == "experts_down"
    assert expert.pack_index == 287
    assert sum(spec.transform == "expert_transpose" for spec in specs) == 42 * 288 * 3


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


def test_safetensors_shard_header_plans_exact_http_ranges():
    header_json = json.dumps(
        {
            "weight": {"dtype": "F8_E4M3", "shape": [4, 6], "data_offsets": [0, 24]},
            "weight_scale_inv": {"dtype": "F32", "shape": [2, 2], "data_offsets": [24, 40]},
        },
        separators=(",", ":"),
    ).encode()
    prefix = len(header_json).to_bytes(8, "little") + header_json
    header = SafetensorsShardHeader.from_prefix(prefix)
    weight = header.tensor("weight")
    scale = header.tensor("weight_scale_inv")
    assert weight.shape == (4, 6)
    assert weight.nbytes == 24
    assert weight.absolute_start == 8 + len(header_json)
    assert weight.http_range == (8 + len(header_json), 8 + len(header_json) + 23)
    assert scale.http_range[0] == weight.http_range[1] + 1

    malformed = json.dumps(
        {"weight": {"dtype": "F8_E4M3", "shape": [4, 6], "data_offsets": [0, 23]}}
    ).encode()
    with pytest.raises(ValueError, match="byte count mismatch"):
        SafetensorsShardHeader.from_prefix(len(malformed).to_bytes(8, "little") + malformed)
    with pytest.raises(ValueError, match="incomplete"):
        SafetensorsShardHeader.from_prefix(prefix[:-1])

    gapped = json.dumps(
        {
            "weight": {"dtype": "F8_E4M3", "shape": [4, 6], "data_offsets": [1, 25]},
            "weight_scale_inv": {"dtype": "F32", "shape": [2, 2], "data_offsets": [25, 41]},
        }
    ).encode()
    with pytest.raises(ValueError, match="not contiguous"):
        SafetensorsShardHeader.from_prefix(len(gapped).to_bytes(8, "little") + gapped)


def test_block_fp8_dequant_and_tiled_linear_match_dense_reference():
    block_shape = (2, 3)
    quantized = jnp.asarray(
        [
            [1.0, -2.0, 0.5, 4.0, -1.0, 2.0],
            [-0.5, 1.5, 2.0, -3.0, 0.25, 1.0],
            [2.0, 0.0, -1.0, 0.5, 1.0, -2.0],
            [3.0, -0.25, 0.75, 2.0, -4.0, 0.5],
        ],
        dtype=jnp.float8_e4m3fn,
    )
    bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8)
    scales = jnp.asarray([[0.5, 2.0], [4.0, 0.25]], jnp.float32)
    dense = dequantize_block_fp8(bits, scales, block_shape=block_shape)
    expected = quantized.astype(jnp.float32).reshape(2, 2, 2, 3)
    expected = (expected * scales[:, None, :, None]).reshape(4, 6)
    assert np.array_equal(dense, expected)

    inputs = jnp.asarray(
        [[[1.0, 2.0, -1.0, 0.5, 3.0, -2.0]], [[-1.0, 0.25, 2.0, 4.0, -0.5, 1.0]]],
        jnp.float32,
    )
    expected_output = jnp.einsum("...k,nk->...n", inputs, dense, precision=_PRECISION)
    actual = jax.jit(
        lambda x, q, s: block_fp8_linear(
            x,
            q,
            s,
            block_shape=block_shape,
            compute_dtype=jnp.float32,
            output_dtype=jnp.float32,
        )
    )(inputs, bits, scales)
    assert np.allclose(actual, expected_output, atol=1e-6, rtol=1e-6)


def test_block_fp8_validation_fails_closed():
    bits = jnp.zeros((4, 6), jnp.uint8)
    scales = jnp.ones((2, 2), jnp.float32)
    with pytest.raises(ValueError, match="divisible"):
        block_fp8_linear(jnp.ones((1, 6)), bits, scales, block_shape=(3, 3))
    with pytest.raises(ValueError, match="scale grid"):
        block_fp8_linear(jnp.ones((1, 6)), bits, jnp.ones((1, 2)), block_shape=(2, 3))
    with pytest.raises(ValueError, match="end in dimension"):
        block_fp8_linear(jnp.ones((1, 5)), bits, scales, block_shape=(2, 3))
    with pytest.raises(TypeError, match="uint8 bits"):
        block_fp8_linear(jnp.ones((1, 6)), bits.astype(jnp.int8), scales, block_shape=(2, 3))
    with pytest.raises(TypeError, match="inputs must be floating"):
        block_fp8_linear(jnp.ones((1, 6), jnp.int32), bits, scales, block_shape=(2, 3))
    with pytest.raises(ValueError, match="reference dtype"):
        dequantize_block_fp8(bits, scales, block_shape=(2, 3), dtype=jnp.float16)


def test_block_fp8_kernel_pytrees_and_selected_experts_match_dense_reference():
    values = jnp.asarray(
        np.linspace(-3.0, 3.0, num=3 * 4 * 6).reshape(3, 4, 6),
        dtype=jnp.float8_e4m3fn,
    )
    bits = jax.lax.bitcast_convert_type(values, jnp.uint8)
    scales = jnp.asarray(
        [
            [[0.5, 1.0], [1.5, 2.0]],
            [[0.75, 1.25], [1.75, 2.25]],
            [[1.0, 1.5], [2.0, 2.5]],
        ],
        jnp.float32,
    )
    kernel = BatchedBlockFP8LinearKernel(
        bits,
        scales,
        block_shape=(2, 3),
        compute_dtype=jnp.float32,
    )
    assert kernel.shape == (3, 6, 4)
    mapped = jax.tree.map(lambda value: value, kernel)
    assert isinstance(mapped, BatchedBlockFP8LinearKernel)

    inputs = jnp.asarray([[1, 2, 3, 4, 5, 6], [-2, 1, 0.5, 3, -1, 2]], jnp.float32)
    indices = jnp.asarray([[0, 2], [1, 0]], jnp.int32)
    actual = jax.jit(selected_block_fp8_linear)(inputs, indices, kernel)
    dense = jax.vmap(
        lambda expert_bits, expert_scales: dequantize_block_fp8(
            expert_bits,
            expert_scales,
            block_shape=(2, 3),
            dtype=jnp.float32,
        )
    )(bits, scales)
    expected = jnp.einsum("ti,tkoi->tko", inputs, dense[indices], precision=_PRECISION)
    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-6)

    single = BlockFP8LinearKernel(
        bits[0], scales[0], block_shape=(2, 3), compute_dtype=jnp.float32
    )
    assert single.shape == (6, 4)
    assert single.ndim == 2
    assert single.size == 24


def test_bfloat16_expansion_fails_v4_32_before_activations():
    assert OFFICIAL_CHECKPOINT.logical_parameter_count == 321_323_031_390
    assert dict(OFFICIAL_CHECKPOINT.serialized_element_counts_by_dtype)["F32"] == 19_484_766
    assert OFFICIAL_CHECKPOINT.scale_metadata_elements == 19_189_248
    assert OFFICIAL_CHECKPOINT.expanded_parameter_bytes("bfloat16") == 642_723_410_808
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
    assert len(unproven.blockers) == 4

    schema_proven = v4_32_lora_preflight(
        config,
        index,
        rank=8,
        executable_kernel_proven=True,
        direct_loader_proven=True,
        execution_schema_proven=True,
        placed_base_per_device_bytes=20_234_287_352,
        staging_per_host_bytes=150_994_944,
    )
    assert schema_proven.static_fit
    assert not schema_proven.runnable
    assert schema_proven.memory[0].per_device_bytes == 20_234_287_352
    assert schema_proven.staging_per_host_bytes == 150_994_944
    assert schema_proven.blockers == (
        "the complete frozen text model has not passed a measured sharded TPU forward",
    )

    proven = v4_32_lora_preflight(
        config,
        index,
        rank=8,
        executable_kernel_proven=True,
        direct_loader_proven=True,
        execution_schema_proven=True,
        full_model_forward_proven=True,
        placed_base_per_device_bytes=20_234_287_352,
        staging_per_host_bytes=150_994_944,
    )
    assert proven.static_fit
    assert proven.runnable
    assert not proven.blockers


def test_reduced_hybrid_model_forward_backward_and_shapes():
    assert _PRECISION is jax.lax.Precision.HIGHEST
    config = tiny_config(vocab_size=64)
    params = init_params(jax.random.key(0), config, dtype=jnp.float32)
    validate_params(params, config)
    assert parameter_count(config) == sum(value.size for value in jax.tree.leaves(params))
    ids = jnp.array([[1, 2, 3, 4], [4, 3, 0, 0]], jnp.int32)
    mask = jnp.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool)
    logits = forward(params, config, ids, attention_mask=mask)
    assert logits.shape == (2, 4, 64)
    assert np.asarray(jnp.isfinite(logits).all())

    selected = {
        "embed_tokens": params["embed_tokens"],
        "lm_head": params["lm_head"],
    }

    def endpoint_loss(endpoints):
        replaced = dict(params)
        replaced.update(endpoints)
        return jnp.mean(forward(replaced, config, ids, attention_mask=mask) ** 2)

    gradients = jax.grad(endpoint_loss)(selected)
    assert all(np.asarray(jnp.isfinite(value).all()) for value in jax.tree.leaves(gradients))


def test_full_reduced_fp8_wrappers_and_expert_packs_match_dequantized_lora_reference():
    config = tiny_config(vocab_size=48)
    original = init_params(jax.random.key(21), config, dtype=jnp.float32)
    quantized, reference = _quantize_tiny_linear_tree(original)
    ids = jnp.asarray([[1, 2, 3], [4, 5, 0]], jnp.int32)
    mask = jnp.asarray([[1, 1, 1], [1, 1, 0]], bool)

    quantized_logits = forward(quantized, config, ids, attention_mask=mask)
    reference_logits = forward(reference, config, ids, attention_mask=mask)
    assert np.allclose(quantized_logits, reference_logits, atol=2e-5, rtol=2e-5)

    lora_config = LoRAConfig(rank=2, alpha=4.0)
    targets = attention_lora_target_paths(config)
    adapters = init_lora_adapters(
        jax.random.key(22),
        quantized,
        targets,
        eligible_paths=targets,
        config=lora_config,
        dtype=jnp.float32,
    )

    def adapter_loss(base, current_adapters):
        logits = forward(
            base,
            config,
            ids,
            attention_mask=mask,
            adapters=current_adapters,
            lora_config=lora_config,
        )
        return jnp.mean(jnp.square(logits.astype(jnp.float32)))

    quantized_loss, quantized_grad = jax.value_and_grad(adapter_loss, argnums=1)(
        quantized, adapters
    )
    reference_loss, reference_grad = jax.value_and_grad(adapter_loss, argnums=1)(
        reference, adapters
    )
    assert np.allclose(quantized_loss, reference_loss, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(
        jax.tree.leaves(quantized_grad), jax.tree.leaves(reference_grad), strict=True
    ):
        assert np.allclose(actual, expected, atol=3e-5, rtol=3e-5)


def test_glm_attention_lora_zero_identity_and_merged_equivalence():
    config = tiny_config(vocab_size=48)
    params = init_params(jax.random.key(5), config, dtype=jnp.float32)
    targets = attention_lora_target_paths(config)
    lora_config = LoRAConfig(rank=2, alpha=4)
    adapters = init_lora_adapters(
        jax.random.key(6),
        params,
        targets,
        eligible_paths=targets,
        config=lora_config,
        dtype=jnp.float32,
    )
    ids = jnp.array([[1, 2, 3, 4]], jnp.int32)
    base_logits = forward(params, config, ids)
    identity_logits = forward(params, config, ids, adapters=adapters, lora_config=lora_config)
    assert np.array_equal(base_logits, identity_logits)

    first_name = format_parameter_path(targets[0])
    adapters[first_name]["b"] = jnp.arange(adapters[first_name]["b"].size, dtype=jnp.float32).reshape(
        adapters[first_name]["b"].shape
    ) / 100
    unmerged = forward(params, config, ids, adapters=adapters, lora_config=lora_config)
    merged_params = merge_lora_adapters(params, adapters, targets, config=lora_config)
    merged = forward(merged_params, config, ids)
    assert np.allclose(unmerged, merged, atol=3e-6, rtol=3e-6)

    gradients = jax.grad(
        lambda trainable: jnp.mean(
            forward(params, config, ids, adapters=trainable, lora_config=lora_config) ** 2
        )
    )(adapters)
    assert all(np.asarray(jnp.isfinite(value).all()) for value in jax.tree.leaves(gradients))


def test_right_padding_does_not_change_valid_prefix_logits():
    config = tiny_config(vocab_size=32)
    params = init_params(jax.random.key(9), config)
    short = forward(params, config, jnp.array([[2, 3, 4]], jnp.int32))
    padded = forward(
        params,
        config,
        jnp.array([[2, 3, 4, 0, 0]], jnp.int32),
        attention_mask=jnp.array([[1, 1, 1, 0, 0]], bool),
    )
    assert np.allclose(short, padded[:, :3], atol=3e-6, rtol=3e-6)
