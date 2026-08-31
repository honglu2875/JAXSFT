"""Optional Transformers 5.16+ oracle for the reduced GLM-5.3 text path."""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.models.glm5_3_flash import convert_hf_state_dict, forward, tiny_config


pytestmark = pytest.mark.parity


def _error_metrics(actual, expected):
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    difference = actual - expected
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)),
    }


def _hf_config(transformers, config):
    hf_config = transformers.Glm5NextTextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        n_shared_experts=config.n_shared_experts,
        n_routed_experts=config.n_routed_experts,
        routed_scaling_factor=config.routed_scaling_factor,
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_nope_head_dim=config.qk_nope_head_dim,
        v_head_dim=config.v_head_dim,
        n_group=config.n_group,
        topk_group=config.topk_group,
        num_experts_per_tok=config.num_experts_per_tok,
        norm_topk_prob=config.norm_topk_prob,
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=config.initializer_range,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=False,
        mlp_layer_types=list(config.mlp_layer_types),
        attention_bias=False,
        attention_dropout=0.0,
        index_topk=config.index_topk,
        index_head_dim=config.index_head_dim,
        index_n_heads=config.index_n_heads,
        layer_types=list(config.layer_types),
        indexer_types=list(config.indexer_types),
        swiglu_limit=config.swiglu_limit,
        linear_attn_config={
            "num_heads": config.linear_attention.num_heads,
            "head_dim": config.linear_attention.head_dim,
            "short_conv_kernel_size": config.linear_attention.short_conv_kernel_size,
            "gate_lower_bound": config.linear_attention.gate_lower_bound,
        },
        hc_mult=config.hc_mult,
        hc_eps=config.hc_eps,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        index_kpool=config.index_kpool,
        index_kpool_always_select_tail=config.index_kpool_always_select_tail,
        pad_token_id=0,
        use_cache=False,
        output_router_logits=False,
    )
    hf_config._attn_implementation = "eager"
    return hf_config


def test_tiny_hybrid_forward_loss_and_selected_gradients_match_transformers():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Glm5NextTextModel"):
        pytest.skip("installed Transformers has no GLM-5.3 model")

    config = tiny_config(vocab_size=64)
    torch.manual_seed(37)
    reference = transformers.Glm5NextTextModel(_hf_config(transformers, config)).float().eval()
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).float().eval()
    state = {f"model.language_model.{name}": value for name, value in reference.state_dict().items()}
    state["lm_head.weight"] = lm_head.weight
    params = convert_hf_state_dict(state, config, dtype=jnp.float32)

    ids_np = np.array([[1, 5, 7, 9, 3], [4, 2, 8, 0, 0]], np.int64)
    mask_np = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], np.int64)
    weights_np = np.array([[0, 0, 1, 0.25, 1], [0, 1, 0.5, 0, 0]], np.float32)
    torch_ids, torch_mask = torch.tensor(ids_np), torch.tensor(mask_np)
    reference.zero_grad(set_to_none=True)
    lm_head.zero_grad(set_to_none=True)
    torch_hidden = reference(input_ids=torch_ids, attention_mask=torch_mask, use_cache=False).last_hidden_state
    torch_logits = lm_head(torch_hidden)
    targets = torch_ids[:, 1:]
    token_nll = torch.nn.functional.cross_entropy(
        torch_logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    torch_weights = torch.tensor(weights_np[:, 1:])
    torch_loss = (token_nll * torch_weights).sum() / torch_weights.sum()
    torch_loss.backward()

    def objective(tree):
        logits = forward(tree, config, jnp.asarray(ids_np), attention_mask=jnp.asarray(mask_np, bool))
        predictions = logits[:, :-1].astype(jnp.float32)
        targets = jnp.asarray(ids_np[:, 1:])
        selected = jnp.take_along_axis(predictions, targets[..., None], -1)[..., 0]
        nll = jax.nn.logsumexp(predictions, -1) - selected
        weights = jnp.asarray(weights_np[:, 1:])
        return jnp.sum(nll * weights) / jnp.sum(weights)

    jax_loss, gradients = jax.value_and_grad(objective)(params)
    jax_logits = np.asarray(forward(params, config, ids_np, attention_mask=mask_np))
    valid = mask_np.astype(bool)
    np.testing.assert_allclose(
        jax_logits[valid],
        torch_logits.detach().numpy()[valid],
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(float(jax_loss), float(torch_loss.detach()), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        np.asarray(gradients["layers"][0]["self_attn"]["q_proj"]),
        reference.layers[0].self_attn.q_proj.weight.grad.numpy().T,
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(gradients["layers"][1]["mlp"]["shared_up_proj"]),
        reference.layers[1].mlp.shared_experts.up_proj.weight.grad.numpy().T,
        rtol=2e-3,
        atol=2e-3,
    )

    if evidence_path := os.environ.get("JAXSFT_GLM53_PARITY_EVIDENCE"):
        evidence = {
            "schema_version": 1,
            "test": "glm53_reduced_hybrid_cpu_parity",
            "versions": {
                "jax": jax.__version__,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "backend": jax.default_backend(),
            "dtype": "float32",
            "config": {
                "layers": list(config.layer_types),
                "mlp_layers": list(config.mlp_layer_types),
                "hidden_size": config.hidden_size,
                "routed_experts": config.n_routed_experts,
                "experts_per_token": config.num_experts_per_tok,
                "hc_mult": config.hc_mult,
            },
            "metrics": {
                "valid_logits": _error_metrics(
                    jax_logits[valid], torch_logits.detach().numpy()[valid]
                ),
                "loss_abs": abs(float(jax_loss) - float(torch_loss.detach())),
                "kda_q_proj_gradient": _error_metrics(
                    gradients["layers"][0]["self_attn"]["q_proj"],
                    reference.layers[0].self_attn.q_proj.weight.grad.numpy().T,
                ),
                "shared_expert_up_gradient": _error_metrics(
                    gradients["layers"][1]["mlp"]["shared_up_proj"],
                    reference.layers[1].mlp.shared_experts.up_proj.weight.grad.numpy().T,
                ),
            },
        }
        target = Path(evidence_path)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
