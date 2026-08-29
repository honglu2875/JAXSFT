from dataclasses import dataclass

import pytest

from jaxsft.data.adapters import AdapterContext, AdapterError, messages_adapter, ultrachat_200k_adapter
from jaxsft.data.render import RenderedDocument, RenderedSpan, render_olmo2_instruct, render_qwen3_5
from jaxsft.data.tokenize import LossPolicy, TokenizerSnapshot, tokenize_document


CONTEXT = AdapterContext("HuggingFaceH4/ultrachat_200k", "a" * 40, "default", "train_sft", 0)


def test_ultrachat_adapter_and_qwen_render_are_semantic():
    row = {
        "prompt": "Hello",
        "prompt_id": "abc",
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}],
    }
    sample = ultrachat_200k_adapter(row, CONTEXT)
    rendered = render_qwen3_5(sample)
    assert rendered.text == (
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nHi!<|im_end|>\n"
    )
    assistant = [span for span in rendered.spans if span.role == "assistant" and span.default_weight]
    assert {span.span_class for span in assistant} == {"content", "assistant_end"}
    assert assistant[0].semantic_ref.message_index == 1


def test_olmo2_renderer_preserves_content_and_rejects_unsupported_semantics():
    row = {
        "id": "olmo-fixture",
        "messages": [
            {"role": "system", "content": "  Keep whitespace.  "},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }
    sample = messages_adapter(row, CONTEXT)
    rendered = render_olmo2_instruct(sample)
    assert rendered.text == (
        "<|endoftext|><|system|>\n  Keep whitespace.  \n"
        "<|user|>\nQuestion\n<|assistant|>\nAnswer<|endoftext|>"
    )
    selected = [span for span in rendered.spans if span.default_weight]
    assert [span.span_class for span in selected] == ["content", "assistant_end"]

    tool_row = {
        "id": "tool-row",
        "messages": [
            {"role": "user", "content": "call it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}}
                ],
            },
        ],
    }
    with pytest.raises(ValueError, match="cannot render part kinds"):
        render_olmo2_instruct(messages_adapter(tool_row, CONTEXT))


def test_tokenizer_snapshot_uses_pinned_pad_configuration_in_its_identity(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tokenizer = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<|endoftext|>": 1, "<|pad|>": 2, "hello": 3},
            unk_token="<unk>",
        )
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = tmp_path / "tokenizer_config.json"
    config.write_text('{"pad_token":"<|pad|>"}\n')
    olmo_snapshot, _ = TokenizerSnapshot.load(tmp_path)
    assert olmo_snapshot.pad_token_id == 2

    config.write_text('{"pad_token":{"content":"<|endoftext|>"}}\n')
    qwen_snapshot, _ = TokenizerSnapshot.load(tmp_path)
    assert qwen_snapshot.pad_token_id == 1
    assert qwen_snapshot.identity_hash != olmo_snapshot.identity_hash


def test_ultrachat_prompt_mismatch_is_rejected():
    row = {
        "prompt": "one",
        "prompt_id": "abc",
        "messages": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "answer"}],
    }
    with pytest.raises(AdapterError, match="prompt must equal"):
        ultrachat_200k_adapter(row, CONTEXT)


def test_openai_tool_call_remains_structured():
    row = {
        "id": "tool-row",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"SF"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
            {"role": "assistant", "content": "Sunny."},
        ],
    }
    sample = messages_adapter(row, CONTEXT)
    call = sample.messages[1].parts[-1]
    assert call.kind == "tool_call"
    assert call.tool_name == "weather"
    assert call.call_id == "call-1"
    rendered = render_qwen3_5(sample).text
    assert "<function=weather>" in rendered
    assert "<parameter=city>\nSF\n</parameter>" in rendered
    assert "<tool_response>\nsunny\n</tool_response>" in rendered


def test_tool_preamble_does_not_claim_first_user_message_metadata():
    row = {
        "id": "tool-row",
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "sunny"},
        ],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
    }
    rendered = render_qwen3_5(messages_adapter(row, CONTEXT))
    preamble_end = next(span for span in rendered.spans if span.span_class == "turn_end")
    assert preamble_end.role == "system"
    assert preamble_end.semantic_ref is None


@dataclass
class FakeEncoding:
    ids: list[int]
    offsets: list[tuple[int, int]]
    tokens: list[str]


class FakeEncoder:
    def encode(self, text, add_special_tokens=False):
        assert text == "ab"
        return FakeEncoding(ids=[7], offsets=[(0, 2)], tokens=["ab"])


def test_qwen_boundary_token_is_explicitly_owned_by_right_span():
    document = RenderedDocument(
        sample_id="s",
        spans=(
            RenderedSpan("a", None, "control", role="user", boundary_owner="right"),
            RenderedSpan("b", None, "content", role="assistant", default_weight=1.5, boundary_owner="right"),
        ),
        renderer="fixture",
        renderer_version=1,
    )
    tokenized = tokenize_document(document, FakeEncoder(), tokenizer_hash="hash")
    assert tokenized.metadata[0].span_class == "content"
    # Causal convention always clears the first token, even when its span is selected.
    assert tokenized.loss_weights == (0.0,)


def test_ordered_loss_policy_selects_only_assistant_content():
    metadata_policy = LossPolicy.from_config(
        {
            "rules": [
                {"select": {}, "weight": 0.0},
                {"select": {"role": "assistant", "span_class": "content"}, "weight": 2.0},
            ]
        }
    )
    row = {
        "prompt": "Q",
        "prompt_id": "abc",
        "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
    }
    rendered = render_qwen3_5(ultrachat_200k_adapter(row, CONTEXT))
    # A simple character encoder makes ownership and the causal shift inspectable.
    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    tokenized = tokenize_document(rendered, Characters(), tokenizer_hash="fixture", policy=metadata_policy)
    selected = [meta for weight, meta in zip(tokenized.loss_weights, tokenized.metadata) if weight]
    assert selected
    assert all(meta.role == "assistant" and meta.span_class == "content" for meta in selected)
    assert all(meta.loss_rule_indices == (0, 1) for meta in selected)
    assert set(weight for weight in tokenized.loss_weights if weight) == {2.0}


def test_loss_aware_truncation_retains_later_target_on_weight_tie_and_records_loss():
    document = RenderedDocument(
        sample_id="long",
        spans=(
            RenderedSpan("u" * 10, None, "content", role="user"),
            RenderedSpan("a" * 4, None, "content", role="assistant", default_weight=1.0),
            RenderedSpan("x" * 10, None, "content", role="user"),
            RenderedSpan("b" * 4, None, "content", role="assistant", default_weight=1.0),
        ),
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    tokenized = tokenize_document(
        document,
        Characters(),
        tokenizer_hash="fixture",
        max_length=10,
        truncation="loss_aware",
    )
    assert tokenized.input_ids == tuple(range(18, 28))
    assert tokenized.loss_weights == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    record = tokenized.truncation_record
    assert record is not None
    assert (record.start, record.end) == (18, 28)
    assert (record.original_selected_tokens, record.retained_selected_tokens) == (8, 4)
    assert (record.original_weight, record.retained_weight) == (8.0, 4.0)


def test_loss_aware_truncation_can_reserve_explicit_prefix_context():
    document = RenderedDocument(
        sample_id="context-budget",
        spans=(
            RenderedSpan("u" * 10, None, "content", role="user"),
            RenderedSpan("a" * 18, None, "content", role="assistant", default_weight=1.0),
        ),
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    tokenized = tokenize_document(
        document,
        Characters(),
        tokenizer_hash="fixture",
        max_length=10,
        truncation="loss_aware",
        truncation_min_context_tokens=4,
    )
    assert tokenized.input_ids == tuple(range(6, 16))
    assert tokenized.selected_tokens == 6
    record = tokenized.truncation_record
    assert record is not None
    assert record.retained_context_tokens == 4
    assert record.context_constraint_satisfied
