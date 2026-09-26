"""Optional Qwen2/Qwen2.5-Math numerical and pinned tokenizer oracles."""

from dataclasses import replace
import os
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsft.data.adapters import AdapterContext, messages_adapter
from jaxsft.data.render import render_qwen2_5_math
from jaxsft.models.qwen2 import Qwen2Config, convert_hf_state_dict, forward, load_hf_checkpoint, tiny_config
from jaxsft.optim import AdamWHyperparameters, adamw_init, adamw_update

pytestmark = pytest.mark.parity


def _snapshot():
    path = os.environ.get("JAXSFT_QWEN2_MATH_SNAPSHOT")
    if not path or not (Path(path) / "tokenizer.json").is_file():
        pytest.skip("set JAXSFT_QWEN2_MATH_SNAPSHOT to the pinned Qwen2.5-Math-1.5B snapshot")
    return path


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("generation_prompt", [False, True])
def test_math_renderer_matches_pinned_transformers_template(system, generation_prompt):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(_snapshot(), local_files_only=True)
    messages = [{"role": "user", "content": "  Question\n"},
                {"role": "assistant", "content": "\n  First answer  "},
                {"role": "user", "content": "Follow-up"},
                {"role": "assistant", "content": "Final answer"}]
    if system:
        messages.insert(0, {"role": "system", "content": "  Keep whitespace.\n"})
    sample = messages_adapter({"id": "math-template", "messages": messages},
                              AdapterContext("fixture", "revision", "default", "test", 0))
    ours = render_qwen2_5_math(sample, add_generation_prompt=generation_prompt)
    reference = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    assert ours.text == reference
    reference_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt)
    if hasattr(reference_ids, "input_ids"):
        reference_ids = reference_ids.input_ids
    elif isinstance(reference_ids, dict):
        reference_ids = reference_ids["input_ids"]
    assert tokenizer(ours.text, add_special_tokens=False).input_ids == reference_ids
    assert all(span.role == "assistant" for span in ours.spans if span.default_weight)


@pytest.mark.parametrize("tied", [False, True])
def test_tiny_forward_all_gradients_update_and_safetensors_match_transformers(tied, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    config = replace(tiny_config(), tie_word_embeddings=tied)
    hf_config = transformers.Qwen2Config(**config.__dict__, use_sliding_window=False)
    hf_config._attn_implementation = "eager"
    torch.manual_seed(23)
    reference = transformers.Qwen2ForCausalLM(hf_config).float().eval()
    config = Qwen2Config.from_dict(hf_config.to_dict())
    params = convert_hf_state_dict(reference.state_dict(), config, dtype=jnp.float32)
    assert sum(p.numel() for p in reference.parameters()) == sum(p.size for p in jax.tree.leaves(params))
    ids = np.array([[1, 5, 7, 9, 3], [4, 2, 8, 0, 0]], np.int64)
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], bool)
    positions = np.array([[3, 4, 5, 6, 7], [0, 1, 2, 3, 4]], np.int64)
    weights = np.array([[0, 0, 1, 0.25, 1], [0, 1, 0.5, 0, 0]], np.float32)
    torch_ids = torch.tensor(ids)
    logits = reference(input_ids=torch_ids, attention_mask=torch.tensor(mask),
                       position_ids=torch.tensor(positions), use_cache=False).logits
    nll = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size),
                                            torch_ids[:, 1:].reshape(-1), reduction="none").reshape(2, -1)
    torch_weights = torch.tensor(weights[:, 1:])
    loss = (nll * torch_weights).sum() / torch_weights.sum()
    loss.backward()

    def objective(tree):
        result = forward(tree, config, ids, attention_mask=mask, position_ids=positions, remat=True)
        predictions = result[:, :-1]
        selected = jnp.take_along_axis(predictions, jnp.asarray(ids[:, 1:, None]), -1)[..., 0]
        return jnp.sum((jax.nn.logsumexp(predictions, -1) - selected) * weights[:, 1:]) / weights[:, 1:].sum()

    jax_loss, gradients = jax.value_and_grad(objective)(params)
    actual_logits = forward(params, config, ids, attention_mask=mask, position_ids=positions)
    np.testing.assert_allclose(np.asarray(actual_logits)[mask], logits.detach().numpy()[mask], atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(jax_loss, loss.detach().numpy(), atol=2e-6, rtol=2e-5)
    reference_gradients = convert_hf_state_dict(
        {name: value.grad for name, value in reference.named_parameters()}, config, dtype=jnp.float32)
    for actual, expected in zip(jax.tree.leaves(gradients), jax.tree.leaves(reference_gradients), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=5e-4)

    hp = AdamWHyperparameters(beta1=0.9, beta2=0.95, epsilon=1e-8, weight_decay=0.1, max_grad_norm=1.0)
    torch_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), hp.max_grad_norm)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, betas=(hp.beta1, hp.beta2),
                                  eps=hp.epsilon, weight_decay=hp.weight_decay)
    optimizer.step()
    updated, _, grad_norm = adamw_update(params, gradients, adamw_init(params), learning_rate=3e-4, hyperparameters=hp)
    np.testing.assert_allclose(grad_norm, torch_norm, atol=5e-6, rtol=5e-5)
    converted_updated = convert_hf_state_dict(reference.state_dict(), config, dtype=jnp.float32)
    for actual, expected in zip(jax.tree.leaves(updated), jax.tree.leaves(converted_updated), strict=True):
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)

    reference.save_pretrained(tmp_path, max_shard_size="10KB" if tied else "1GB")
    loaded_config, loaded = load_hf_checkpoint(tmp_path, dtype=jnp.float32)
    assert loaded_config == config
    for actual, expected in zip(jax.tree.leaves(loaded), jax.tree.leaves(converted_updated), strict=True):
        np.testing.assert_array_equal(actual, expected)
    bad_state = dict(reference.state_dict())
    bad_state["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected Qwen2"):
        convert_hf_state_dict(bad_state, config)
    del bad_state["unexpected.weight"]
    del bad_state["model.layers.0.self_attn.q_proj.bias"]
    with pytest.raises(KeyError, match="missing required tensor"):
        convert_hf_state_dict(bad_state, config)


@pytest.mark.skipif(os.environ.get("JAXSFT_QWEN2_PUBLIC_PARITY") != "1", reason="opt-in 1.5B CPU comparison")
def test_public_math_checkpoint_all_logits_match_transformers():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(8)
    snapshot = _snapshot()
    reference = transformers.Qwen2ForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    config, params = load_hf_checkpoint(snapshot, dtype=jnp.float32)
    ids = np.array([[151644, 8948, 198, 9707, 151645]], np.int64)
    with torch.no_grad():
        expected = reference(input_ids=torch.tensor(ids), use_cache=False).logits.numpy()
    actual = np.asarray(forward(params, config, ids))
    assert sum(p.numel() for p in reference.parameters()) == sum(p.size for p in jax.tree.leaves(params))
    print(json.dumps({"parameters": sum(p.size for p in jax.tree.leaves(params)),
                      "logits_compared": actual.size,
                      "max_absolute_error": float(np.abs(actual - expected).max()),
                      "mean_absolute_error": float(np.abs(actual - expected).mean())}))
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
