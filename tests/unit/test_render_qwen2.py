from dataclasses import replace

import pytest

from jaxsft.data.adapters import AdapterContext, messages_adapter
from jaxsft.data.ir import Message, Part, ToolDefinition
from jaxsft.data.render import get_renderer, render_qwen2_5_math, renderer_identity


def sample():
    return messages_adapter({"id": "math", "messages": [
        {"role": "system", "content": "  Keep whitespace.\n"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]}, AdapterContext("fixture", "revision", "default", "test", 0))


def test_math_renderer_keeps_parts_and_assistant_loss_ownership():
    original = sample()
    value = replace(original, messages=(*original.messages[:-1], Message("assistant", (
        Part("text", "  Reasoning\n", tags=frozenset({"explanation"})),
        Part("code", "by rfl\n"),
    ))))
    document = render_qwen2_5_math(value, add_generation_prompt=True)
    assert document.text == (
        "<|im_start|>system\n  Keep whitespace.\n<|im_end|>\n"
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\n  Reasoning\nby rfl\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    selected = [span for span in document.spans if span.default_weight]
    assert [span.span_class for span in selected] == ["content", "content", "assistant_end"]
    assert [span.semantic_ref.part_index for span in selected[:2]] == [0, 1]
    assert selected[0].tags == frozenset({"explanation"})
    assert get_renderer("qwen2") is render_qwen2_5_math
    assert renderer_identity("qwen2")["name"] == "qwen2_5_math"


@pytest.mark.parametrize("unsupported", [
    {"messages": (Message("developer", (Part("text", "instruction"),)),)},
    {"messages": (Message("assistant", (Part("reasoning", "separate reasoning"),)),)},
    {"tools": (ToolDefinition("calculator"),)},
])
def test_math_renderer_rejects_unsupported_semantics(unsupported):
    with pytest.raises(ValueError, match="Qwen2.5-Math text rendering"):
        render_qwen2_5_math(replace(sample(), **unsupported))
