"""Optional Hugging Face numerical and tokenizer oracles."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.data.adapters import AdapterContext, messages_adapter
from jaxsft.data.render import render_qwen3_5
from jaxsft.models.qwen3_5 import Qwen35Config, convert_hf_state_dict, forward, tiny_config
from jaxsft.optim import AdamWHyperparameters, adamw_init, adamw_update


pytestmark = pytest.mark.parity


def _snapshot():
    path = os.environ.get("JAXSFT_QWEN35_SNAPSHOT")
    if not path or not (Path(path) / "tokenizer.json").is_file():
        pytest.skip("set JAXSFT_QWEN35_SNAPSHOT to a pinned Qwen3.5 tokenizer snapshot")
    return path


def test_renderer_and_token_ids_match_transformers_for_tools_and_reasoning():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(_snapshot(), local_files_only=True)
    messages = [
        {"role": "system", "content": "Be useful."},
        {"role": "user", "content": "Weather in SF?"},
        {
            "role": "assistant",
            "reasoning_content": "Need weather.",
            "content": "I will check.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": {"city": "SF", "days": 2}}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
        {"role": "assistant", "reasoning_content": "Got it.", "content": "It is sunny."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                },
            },
        }
    ]
    context = AdapterContext("fixture", "revision", "default", "test", 0)
    sample = messages_adapter({"id": "fixture", "messages": messages, "tools": tools}, context)
    ours = render_qwen3_5(sample).text
    reference = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    assert ours == reference
    reference_tokens = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=False
    )
    if hasattr(reference_tokens, "input_ids"):
        reference_tokens = reference_tokens.input_ids
    elif isinstance(reference_tokens, dict):
        reference_tokens = reference_tokens["input_ids"]
    assert tokenizer(ours, add_special_tokens=False).input_ids == reference_tokens


def test_tiny_forward_loss_and_gradient_match_transformers():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):
        pytest.skip("installed Transformers has no dense Qwen3.5 model")
    config = tiny_config()
    hf_config = transformers.Qwen3_5TextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        layer_types=list(config.layer_types),
        linear_conv_kernel_dim=config.linear_conv_kernel_dim,
        linear_key_head_dim=config.linear_key_head_dim,
        linear_num_key_heads=config.linear_num_key_heads,
        linear_value_head_dim=config.linear_value_head_dim,
        linear_num_value_heads=config.linear_num_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        partial_rotary_factor=config.partial_rotary_factor,
        tie_word_embeddings=True,
        attn_output_gate=True,
        attention_dropout=0.0,
        attention_bias=False,
    )
    hf_config._attn_implementation = "eager"
    torch.manual_seed(7)
    reference = transformers.Qwen3_5ForCausalLM(hf_config).float().eval()
    jax_config = Qwen35Config.from_dict(hf_config.to_dict())
    params = convert_hf_state_dict(reference.state_dict(), jax_config, dtype=jnp.float32)
    ids_np = np.array([[1, 5, 7, 9, 3], [4, 2, 8, 0, 0]], np.int64)
    mask_np = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], np.int64)
    weights_np = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]], np.float32)
    torch_ids, torch_mask = torch.tensor(ids_np), torch.tensor(mask_np)
    reference.zero_grad(set_to_none=True)
    torch_logits = reference(input_ids=torch_ids, attention_mask=torch_mask, use_cache=False).logits
    target = torch_ids[:, 1:]
    token_nll = torch.nn.functional.cross_entropy(
        torch_logits[:, :-1].reshape(-1, config.vocab_size), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    torch_weights = torch.tensor(weights_np[:, 1:])
    torch_loss = (token_nll * torch_weights).sum() / torch_weights.sum()
    torch_loss.backward()

    def objective(tree):
        logits = forward(tree, jax_config, jnp.asarray(ids_np), attention_mask=jnp.asarray(mask_np, bool))
        prediction = logits[:, :-1].astype(jnp.float32)
        targets = jnp.asarray(ids_np[:, 1:])
        selected = jnp.take_along_axis(prediction, targets[..., None], -1)[..., 0]
        nll = jax.nn.logsumexp(prediction, -1) - selected
        weights = jnp.asarray(weights_np[:, 1:])
        return jnp.sum(nll * weights) / jnp.sum(weights)

    jax_loss, gradients = jax.value_and_grad(objective)(params)
    # Hugging Face continues to compute recurrent states at right-padding
    # positions; JAXSFT deliberately freezes them. Only valid-token logits are
    # semantically defined, and a unit test separately proves prefix invariance.
    jax_logits = np.asarray(forward(params, jax_config, ids_np, attention_mask=mask_np))
    np.testing.assert_allclose(
        jax_logits[mask_np.astype(bool)],
        torch_logits.detach().numpy()[mask_np.astype(bool)],
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(float(jax_loss), float(torch_loss.detach()), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        np.asarray(gradients["layers"][1]["mlp"]["up_proj"]),
        reference.model.layers[1].mlp.up_proj.weight.grad.numpy().T,
        rtol=5e-4,
        atol=5e-4,
    )

    learning_rate = 3e-4
    hyperparameters = AdamWHyperparameters(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.1,
        max_grad_norm=1.0,
    )
    torch_gradient_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), hyperparameters.max_grad_norm)
    torch_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=learning_rate,
        betas=(hyperparameters.beta1, hyperparameters.beta2),
        eps=hyperparameters.epsilon,
        weight_decay=hyperparameters.weight_decay,
    )
    torch_optimizer.step()
    updated, _, jax_gradient_norm = adamw_update(
        params,
        gradients,
        adamw_init(params),
        learning_rate=learning_rate,
        hyperparameters=hyperparameters,
    )
    np.testing.assert_allclose(
        float(jax_gradient_norm),
        float(torch_gradient_norm),
        rtol=5e-5,
        atol=5e-6,
    )
    # PyTorch decays every parameter by default; JAXSFT deliberately excludes
    # one-dimensional norm/recurrence state. A matrix kernel has identical
    # semantics and is the cross-framework optimizer-step oracle.
    np.testing.assert_allclose(
        np.asarray(updated["layers"][1]["mlp"]["up_proj"]),
        reference.model.layers[1].mlp.up_proj.weight.detach().numpy().T,
        rtol=5e-4,
        atol=5e-5,
    )
