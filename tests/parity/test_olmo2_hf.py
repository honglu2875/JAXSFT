"""Optional Hugging Face numerical and tokenizer oracles for OLMo 2."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.models.olmo2 import Olmo2Config, convert_hf_state_dict, forward, tiny_config
from jaxsft.data.adapters import AdapterContext, messages_adapter
from jaxsft.data.render import render_olmo2_instruct
from jaxsft.optim import AdamWHyperparameters, adamw_init, adamw_update


pytestmark = pytest.mark.parity


def _instruct_snapshot():
    path = os.environ.get("JAXSFT_OLMO2_INSTRUCT_SNAPSHOT")
    if not path or not (Path(path) / "tokenizer.json").is_file():
        pytest.skip("set JAXSFT_OLMO2_INSTRUCT_SNAPSHOT to the pinned OLMo 2 Instruct tokenizer")
    return path


def test_renderer_and_token_ids_match_transformers():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(_instruct_snapshot(), local_files_only=True)
    messages = [
        {"role": "system", "content": "  Keep whitespace.  "},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up"},
        {"role": "assistant", "content": "Final answer"},
    ]
    context = AdapterContext("fixture", "revision", "default", "test", 0)
    sample = messages_adapter({"id": "olmo-template", "messages": messages}, context)
    ours = render_olmo2_instruct(sample).text
    reference = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert ours == reference
    reference_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(reference_tokens, "input_ids"):
        reference_tokens = reference_tokens.input_ids
    elif isinstance(reference_tokens, dict):
        reference_tokens = reference_tokens["input_ids"]
    assert tokenizer(ours, add_special_tokens=False).input_ids == reference_tokens


def test_tiny_forward_loss_gradient_and_update_match_transformers():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Olmo2ForCausalLM"):
        pytest.skip("installed Transformers has no OLMo 2 model")

    config = tiny_config()
    hf_config = transformers.Olmo2Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        initializer_range=config.initializer_range,
        hidden_act="silu",
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=2,
    )
    hf_config._attn_implementation = "eager"
    torch.manual_seed(23)
    reference = transformers.Olmo2ForCausalLM(hf_config).float().eval()
    jax_config = Olmo2Config.from_dict(hf_config.to_dict())
    params = convert_hf_state_dict(reference.state_dict(), jax_config, dtype=jnp.float32)

    ids_np = np.array([[1, 5, 7, 9, 3], [4, 2, 8, 0, 0]], np.int64)
    mask_np = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], np.int64)
    weights_np = np.array([[0, 0, 1, 0.25, 1], [0, 1, 0.5, 0, 0]], np.float32)
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
    np.testing.assert_allclose(
        np.asarray(updated["layers"][1]["mlp"]["up_proj"]),
        reference.model.layers[1].mlp.up_proj.weight.detach().numpy().T,
        rtol=5e-4,
        atol=5e-5,
    )
