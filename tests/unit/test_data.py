from dataclasses import dataclass

import pytest

from jaxsft.data.adapters import AdapterContext, AdapterError, messages_adapter, ultrachat_200k_adapter
from jaxsft.data.render import RenderedDocument, RenderedSpan, render_olmo2_instruct, render_qwen3_5
from jaxsft.data.ir import Message, Part, SemanticRef
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
    assert sample.messages[2].parts[0].call_id == "call-1"
    rendered = render_qwen3_5(sample)
    assert "<function=weather>" in rendered.text
    assert "<parameter=city>\nSF\n</parameter>" in rendered.text
    assert "<tool_response>\nsunny\n</tool_response>" in rendered.text
    assert {
        span.call_id
        for span in rendered.spans
        if span.semantic_ref is not None and span.semantic_ref.message_index == 2
    } == {"call-1"}


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Part(kind="tool_call", value={}, call_id=""),
        lambda: Part(kind="tool_call", value={}, tool_name=7),
        lambda: Message(role="tool", parts=(), call_id=""),
    ],
)
def test_ir_rejects_malformed_tool_identifiers(constructor):
    with pytest.raises((TypeError, ValueError), match="non-empty string"):
        constructor()


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": None, "tool_calls": [{"id": 7, "function": {"name": "f"}}]},
        {"role": "tool", "tool_call_id": "", "content": "result"},
        {"role": "tool", "name": 3, "content": "result"},
    ],
)
def test_messages_adapter_reports_malformed_identifiers_as_adapter_errors(message):
    with pytest.raises(AdapterError, match="non-empty string"):
        messages_adapter(
            {"messages": [{"role": "user", "content": "question"}, message]},
            CONTEXT,
        )


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


def _message_span(
    text,
    message_index,
    role,
    *,
    part_index=0,
    weight=0.0,
    part_kind="text",
    call_id=None,
):
    return RenderedSpan(
        text,
        SemanticRef("semantic", message_index, part_index),
        "content",
        role=role,
        part_kind=part_kind,
        call_id=call_id,
        default_weight=weight,
    )


def test_semantic_loss_aware_truncation_chooses_complete_later_messages():
    document = RenderedDocument(
        sample_id="semantic",
        spans=(
            _message_span("uuuu", 0, "user"),
            _message_span("aaaa", 1, "assistant", weight=1.0),
            _message_span("xxxx", 2, "user"),
            _message_span("bbbb", 3, "assistant", weight=1.0),
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
        max_length=8,
        truncation="semantic_loss_aware",
        truncation_min_context_tokens=4,
    )
    assert tokenized.input_ids == tuple(range(8, 16))
    record = tokenized.truncation_record
    assert record is not None
    assert record.semantic_boundary_aligned
    assert record.original_message_indices == (0, 1, 2, 3)
    assert record.retained_message_indices == (2, 3)
    assert record.dropped_message_indices == (0, 1)
    assert record.retained_context_tokens == 4
    assert record.context_constraint_satisfied


def test_semantic_loss_aware_truncation_keeps_tool_exchange_atomic():
    document = RenderedDocument(
        sample_id="semantic",
        spans=(
            _message_span("uu", 0, "user"),
            _message_span("cc", 1, "assistant", weight=1.0, part_kind="tool_call", call_id="c1"),
            _message_span("tt", 2, "tool", part_kind="tool_result", call_id="c1"),
            _message_span("ff", 3, "assistant", weight=1.0),
            _message_span("qq", 4, "user"),
            _message_span("aa", 5, "assistant", weight=1.0),
        ),
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    tool_window = tokenize_document(
        document,
        Characters(),
        tokenizer_hash="fixture",
        max_length=6,
        truncation="semantic_loss_aware",
    )
    record = tool_window.truncation_record
    assert record is not None
    assert record.retained_message_indices == (1, 2, 3)
    assert record.original_tool_atomic_units == 1
    assert record.retained_tool_atomic_units == 1

    later_window = tokenize_document(
        document,
        Characters(),
        tokenizer_hash="fixture",
        max_length=4,
        truncation="semantic_loss_aware",
    )
    later_record = later_window.truncation_record
    assert later_record is not None
    assert later_record.retained_message_indices == (4, 5)
    assert later_record.retained_tool_atomic_units == 0
    assert set(later_record.retained_message_indices).isdisjoint({1, 2, 3})


def test_semantic_truncation_links_parallel_results_and_chained_call_by_id():
    document = RenderedDocument(
        sample_id="semantic",
        spans=(
            _message_span("u", 0, "user"),
            _message_span("a", 1, "assistant", part_index=0, weight=1.0, part_kind="tool_call", call_id="c1"),
            _message_span("b", 1, "assistant", part_index=1, weight=1.0, part_kind="tool_call", call_id="c2"),
            _message_span("x", 2, "tool", part_kind="tool_result", call_id="c2"),
            _message_span("y", 3, "tool", part_kind="tool_result", call_id="c1"),
            _message_span("c", 4, "assistant", weight=1.0, part_kind="tool_call", call_id="c3"),
            _message_span("z", 5, "tool", part_kind="tool_result", call_id="c3"),
            _message_span("f", 6, "assistant", weight=1.0),
            _message_span("q", 7, "user"),
            _message_span("r", 8, "assistant", weight=1.0),
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
        max_length=7,
        truncation="semantic_loss_aware",
    )
    record = tokenized.truncation_record
    assert record is not None
    assert record.retained_message_indices == (1, 2, 3, 4, 5, 6)
    assert record.original_tool_atomic_units == 1
    assert record.retained_tool_atomic_units == 1


def test_semantic_loss_aware_truncation_rejects_objective_message_that_cannot_fit():
    document = RenderedDocument(
        sample_id="semantic",
        spans=(_message_span("a" * 10, 0, "assistant", weight=1.0),),
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    with pytest.raises(ValueError, match="semantic truncation cannot retain"):
        tokenize_document(
            document,
            Characters(),
            tokenizer_hash="fixture",
            max_length=6,
            truncation="semantic_loss_aware",
        )


@pytest.mark.parametrize(
    ("tool_call_id", "tool_result_id", "include_final", "error"),
    [
        ("c1", "c2", True, "unknown call ID"),
        ("c1", "c1", False, "immediate final assistant"),
        (None, "c1", True, "every tool call"),
    ],
)
def test_semantic_truncation_rejects_unlinked_or_incomplete_tool_transactions(
    tool_call_id,
    tool_result_id,
    include_final,
    error,
):
    spans = [
        _message_span("uu", 0, "user"),
        _message_span("cc", 1, "assistant", weight=1.0, part_kind="tool_call", call_id=tool_call_id),
        _message_span("tt", 2, "tool", part_kind="tool_result", call_id=tool_result_id),
    ]
    if include_final:
        spans.append(_message_span("ff", 3, "assistant", weight=1.0))
    spans.extend((_message_span("qq", 4, "user"), _message_span("aa", 5, "assistant", weight=1.0)))
    document = RenderedDocument(
        sample_id="semantic",
        spans=tuple(spans),
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    with pytest.raises(ValueError, match=error):
        tokenize_document(
            document,
            Characters(),
            tokenizer_hash="fixture",
            max_length=64,
            truncation="semantic_loss_aware",
        )


@pytest.mark.parametrize(
    ("spans", "error"),
    [
        (
            (
                _message_span("cc", 0, "user", part_kind="tool_call", call_id="c1"),
                _message_span("tt", 1, "tool", part_kind="tool_result", call_id="c1"),
                _message_span("ff", 2, "assistant", weight=1.0),
            ),
            "tool calls to belong to an assistant message",
        ),
        (
            (
                _message_span("aa", 0, "assistant", part_kind="tool_call", call_id="c1"),
                _message_span("tt", 1, "tool", part_kind="tool_result", call_id="c1"),
                _message_span("bb", 2, "assistant", part_kind="tool_call", call_id="c1"),
                _message_span("uu", 3, "tool", part_kind="tool_result", call_id="c1"),
                _message_span("ff", 4, "assistant", weight=1.0),
            ),
            "reused tool-call ID",
        ),
    ],
)
def test_semantic_truncation_rejects_wrong_role_or_reused_tool_call_ids(spans, error):
    document = RenderedDocument(
        sample_id="semantic",
        spans=spans,
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    with pytest.raises(ValueError, match=error):
        tokenize_document(
            document,
            Characters(),
            tokenizer_hash="fixture",
            max_length=64,
            truncation="semantic_loss_aware",
        )


def test_semantic_truncation_never_retains_tool_exchange_without_its_preamble():
    document = RenderedDocument(
        sample_id="semantic",
        spans=(
            RenderedSpan("pp", None, "tool_preamble", role="system"),
            _message_span("uu", 0, "user"),
            _message_span("cc", 1, "assistant", weight=1.0, part_kind="tool_call", call_id="c1"),
            _message_span("tt", 2, "tool", part_kind="tool_result", call_id="c1"),
            _message_span("ff", 3, "assistant", weight=1.0),
        ),
        renderer="fixture",
        renderer_version=1,
    )

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return FakeEncoding(list(range(len(text))), [(i, i + 1) for i in range(len(text))], list(text))

    with pytest.raises(ValueError, match="semantic truncation cannot retain"):
        tokenize_document(
            document,
            Characters(),
            tokenizer_hash="fixture",
            max_length=8,
            truncation="semantic_loss_aware",
        )
