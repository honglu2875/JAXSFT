"""Model-specific text chat rendering with semantic span ownership."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from .ir import FrozenMap, Part, Sample, SemanticRef, ToolDefinition, thaw_json

BoundaryOwner = Literal["left", "right", "reject"]

QWEN35_TEMPLATE_REPO = "Qwen/Qwen3.5-0.8B-Base"
QWEN35_TEMPLATE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


@dataclass(frozen=True)
class RenderedSpan:
    text: str
    semantic_ref: SemanticRef | None
    span_class: str
    role: str | None = None
    part_kind: str | None = None
    call_id: str | None = None
    tool_name: str | None = None
    tags: frozenset[str] = frozenset()
    default_weight: float = 0.0
    boundary_owner: BoundaryOwner = "right"
    attributes: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("RenderedSpan.text must be a string")
        if self.default_weight < 0:
            raise ValueError("RenderedSpan.default_weight must be non-negative")


@dataclass(frozen=True)
class RenderedDocument:
    sample_id: str
    spans: tuple[RenderedSpan, ...]
    renderer: str
    renderer_version: int
    options: FrozenMap = field(default_factory=FrozenMap)

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)

    @property
    def template_hash(self) -> str:
        identity = f"{self.renderer}\0{self.renderer_version}\0{self.text}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


_TOOL_PREAMBLE = (
    "# Tools\n\nYou have access to the following functions:\n\n<tools>"
)
_TOOL_INSTRUCTIONS = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1"
    "\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter"
    "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:"
    "\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags"
    "\n- Required parameters MUST be specified"
    "\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after"
    "\n- If there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n</IMPORTANT>"
)


def _json(value: object) -> str:
    # Transformers' sandboxed Jinja environment preserves input mapping order
    # and retains a space after separators.
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def _tool_json(tool: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": thaw_json(tool.parameters),
        },
    }


def _content_parts(parts: tuple[Part, ...], *, include: set[str]) -> list[tuple[int, Part, str]]:
    selected: list[tuple[int, Part, str]] = []
    for part_index, part in enumerate(parts):
        if part.kind in include:
            if not isinstance(part.value, str):
                raise TypeError(f"{part.kind} content must be a string")
            selected.append((part_index, part, part.value))
    if not selected:
        return []
    joined = "".join(text for _, _, text in selected)
    start = len(joined) - len(joined.lstrip())
    stop = len(joined.rstrip())
    result: list[tuple[int, Part, str]] = []
    cursor = 0
    for part_index, part, text in selected:
        left, right = cursor, cursor + len(text)
        clipped_left, clipped_right = max(left, start), min(right, stop)
        if clipped_left < clipped_right:
            result.append((part_index, part, text[clipped_left - left : clipped_right - left]))
        cursor = right
    return result


def _span(
    sample: Sample,
    text: str,
    span_class: str,
    *,
    message_index: int | None = None,
    part_index: int | None = None,
    part: Part | None = None,
    role: str | None = None,
    default_weight: float = 0.0,
) -> RenderedSpan:
    semantic_ref = None
    if message_index is not None:
        semantic_ref = SemanticRef(sample.id, message_index, part_index)
    return RenderedSpan(
        text=text,
        semantic_ref=semantic_ref,
        span_class=span_class,
        role=role,
        part_kind=None if part is None else part.kind,
        call_id=None if part is None else part.call_id,
        tool_name=None if part is None else part.tool_name,
        tags=frozenset() if part is None else part.tags,
        default_weight=default_weight,
        boundary_owner="right",
    )


def _append_content(
    spans: list[RenderedSpan],
    sample: Sample,
    message_index: int,
    *,
    include: set[str],
    span_class: str,
    default_weight: float,
) -> str:
    parts = _content_parts(sample.messages[message_index].parts, include=include)
    for part_index, part, text in parts:
        spans.append(
            _span(
                sample,
                text,
                span_class,
                message_index=message_index,
                part_index=part_index,
                part=part,
                role=sample.messages[message_index].role,
                default_weight=default_weight,
            )
        )
    return "".join(text for _, _, text in parts)


def render_qwen3_5(sample: Sample, *, add_generation_prompt: bool = False, enable_thinking: bool = False) -> RenderedDocument:
    """Render the text subset of Qwen3.5's pinned Hugging Face template.

    A token crossing a span boundary is explicitly owned by the span on its
    right. That Qwen-specific rule preserves assistant-content ownership for
    byte-level tokens that absorb leading whitespace.
    """

    if any(message.role == "developer" for message in sample.messages):
        raise ValueError("Qwen3.5's pinned template has no developer role")
    if any(part.kind == "media" for message in sample.messages for part in message.parts):
        raise ValueError("JAXSFT's Qwen3.5 renderer is text-only")

    spans: list[RenderedSpan] = []
    first = sample.messages[0]
    if sample.tools:
        spans.append(_span(sample, "<|im_start|>system\n" + _TOOL_PREAMBLE, "tool_preamble", role="system"))
        for tool in sample.tools:
            spans.append(_span(sample, "\n" + _json(_tool_json(tool)), "tool_definition", role="system"))
        spans.append(_span(sample, _TOOL_INSTRUCTIONS, "tool_preamble", role="system"))
        if first.role == "system":
            system_content = _content_parts(first.parts, include={"text", "code"})
            if system_content:
                spans.append(_span(sample, "\n\n", "template_control", message_index=0, role="system"))
                _append_content(
                    spans,
                    sample,
                    0,
                    include={"text", "code"},
                    span_class="content",
                    default_weight=0.0,
                )
        spans.append(
            _span(
                sample,
                "<|im_end|>\n",
                "turn_end",
                message_index=0 if first.role == "system" else None,
                role="system",
            )
        )
    elif first.role == "system":
        spans.append(_span(sample, "<|im_start|>system\n", "role_header", message_index=0, role="system"))
        _append_content(
            spans,
            sample,
            0,
            include={"text", "code"},
            span_class="content",
            default_weight=0.0,
        )
        spans.append(_span(sample, "<|im_end|>\n", "turn_end", message_index=0, role="system"))

    last_query_index = None
    for index in range(len(sample.messages) - 1, -1, -1):
        message = sample.messages[index]
        if message.role == "user":
            content = "".join(text for _, _, text in _content_parts(message.parts, include={"text", "code"}))
            if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
                last_query_index = index
                break
    if last_query_index is None:
        raise ValueError("Qwen3.5 rendering requires at least one user query")

    for message_index, message in enumerate(sample.messages):
        if message.role == "system":
            if message_index != 0:
                raise ValueError("system message must be first")
            continue
        if message.role == "user":
            spans.append(
                _span(sample, "<|im_start|>user\n", "role_header", message_index=message_index, role="user")
            )
            _append_content(
                spans,
                sample,
                message_index,
                include={"text", "code"},
                span_class="content",
                default_weight=0.0,
            )
            spans.append(_span(sample, "<|im_end|>\n", "turn_end", message_index=message_index, role="user"))
            continue
        if message.role == "assistant":
            spans.append(
                _span(sample, "<|im_start|>assistant\n", "role_header", message_index=message_index, role="assistant")
            )
            reasoning_parts = _content_parts(message.parts, include={"reasoning"})
            if message_index > last_query_index:
                spans.append(
                    _span(sample, "<think>\n", "reasoning_control", message_index=message_index, role="assistant")
                )
                for part_index, part, text in reasoning_parts:
                    spans.append(
                        _span(
                            sample,
                            text,
                            "reasoning",
                            message_index=message_index,
                            part_index=part_index,
                            part=part,
                            role="assistant",
                            default_weight=1.0,
                        )
                    )
                spans.append(
                    _span(sample, "\n</think>\n\n", "reasoning_control", message_index=message_index, role="assistant")
                )
            elif reasoning_parts:
                raise ValueError("the pinned Qwen3.5 template suppresses reasoning before the final user query")
            content = _append_content(
                spans,
                sample,
                message_index,
                include={"text", "code"},
                span_class="content",
                default_weight=1.0,
            )
            calls = [(part_index, part) for part_index, part in enumerate(message.parts) if part.kind == "tool_call"]
            for call_index, (part_index, part) in enumerate(calls):
                if not part.tool_name:
                    raise ValueError("tool_call part requires tool_name")
                prefix = ""
                if call_index == 0 and content.strip():
                    prefix = "\n\n"
                elif call_index > 0:
                    prefix = "\n"
                spans.append(
                    _span(
                        sample,
                        prefix + f"<tool_call>\n<function={part.tool_name}>\n",
                        "tool_call_control",
                        message_index=message_index,
                        part_index=part_index,
                        part=part,
                        role="assistant",
                        default_weight=1.0,
                    )
                )
                arguments = thaw_json(part.value)
                if not isinstance(arguments, dict):
                    raise ValueError("Qwen3.5 tool-call arguments must be a mapping")
                for name, value in arguments.items():
                    value_text = _json(value) if isinstance(value, (dict, list)) else str(value)
                    spans.append(
                        _span(
                            sample,
                            f"<parameter={name}>\n{value_text}\n</parameter>\n",
                            "tool_call",
                            message_index=message_index,
                            part_index=part_index,
                            part=part,
                            role="assistant",
                            default_weight=1.0,
                        )
                    )
                spans.append(
                    _span(
                        sample,
                        "</function>\n</tool_call>",
                        "tool_call_control",
                        message_index=message_index,
                        part_index=part_index,
                        part=part,
                        role="assistant",
                        default_weight=1.0,
                    )
                )
            spans.append(
                _span(
                    sample,
                    "<|im_end|>\n",
                    "assistant_end",
                    message_index=message_index,
                    role="assistant",
                    default_weight=1.0,
                )
            )
            continue
        if message.role == "tool":
            previous_role = sample.messages[message_index - 1].role if message_index else None
            if previous_role != "tool":
                spans.append(_span(sample, "<|im_start|>user", "role_header", message_index=message_index, role="tool"))
            spans.append(_span(sample, "\n<tool_response>\n", "tool_result_control", message_index=message_index, role="tool"))
            _append_content(
                spans,
                sample,
                message_index,
                include={"text", "tool_result"},
                span_class="tool_result",
                default_weight=0.0,
            )
            spans.append(_span(sample, "\n</tool_response>", "tool_result_control", message_index=message_index, role="tool"))
            next_role = sample.messages[message_index + 1].role if message_index + 1 < len(sample.messages) else None
            if next_role != "tool":
                spans.append(_span(sample, "<|im_end|>\n", "turn_end", message_index=message_index, role="tool"))
            continue
        raise ValueError(f"unsupported Qwen3.5 message role {message.role!r}")

    if add_generation_prompt:
        suffix = "<|im_start|>assistant\n<think>\n" if enable_thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        spans.append(_span(sample, suffix, "generation_prompt", role="assistant"))

    return RenderedDocument(
        sample_id=sample.id,
        spans=tuple(span for span in spans if span.text),
        renderer="qwen3_5_text",
        renderer_version=1,
        options=FrozenMap((("add_generation_prompt", add_generation_prompt), ("enable_thinking", enable_thinking))),
    )


OLMO2_INSTRUCT_TEMPLATE_REPO = "allenai/OLMo-2-0425-1B-Instruct"
OLMO2_INSTRUCT_TEMPLATE_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"


def render_olmo2_instruct(sample: Sample, *, add_generation_prompt: bool = False) -> RenderedDocument:
    """Render the pinned OLMo 2 Instruct template while retaining span metadata.

    The base and instruct repositories at the pinned revisions have identical
    ``tokenizer.json`` files. The instruct repository supplies the chat
    template used here. That template has no tool, reasoning, developer, or
    media syntax, so such semantics are rejected instead of silently dropped.
    """

    if sample.tools:
        raise ValueError("the pinned OLMo 2 Instruct template has no tool definitions")
    supported_parts = {"text", "code"}
    spans = [_span(sample, "<|endoftext|>", "bos")]
    for message_index, message in enumerate(sample.messages):
        if message.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported OLMo 2 Instruct message role {message.role!r}")
        unsupported = [part.kind for part in message.parts if part.kind not in supported_parts]
        if unsupported:
            raise ValueError(
                f"the pinned OLMo 2 Instruct template cannot render part kinds {sorted(set(unsupported))}"
            )
        spans.append(
            _span(
                sample,
                f"<|{message.role}|>\n",
                "role_header",
                message_index=message_index,
                role=message.role,
            )
        )
        content_weight = 1.0 if message.role == "assistant" else 0.0
        for part_index, part in enumerate(message.parts):
            if not isinstance(part.value, str):
                raise TypeError(f"{part.kind} content must be a string")
            spans.append(
                _span(
                    sample,
                    part.value,
                    "content",
                    message_index=message_index,
                    part_index=part_index,
                    part=part,
                    role=message.role,
                    default_weight=content_weight,
                )
            )
        if message.role == "assistant":
            suffix = "<|endoftext|>" + ("\n" if message_index + 1 < len(sample.messages) else "")
            spans.append(
                _span(
                    sample,
                    suffix,
                    "assistant_end",
                    message_index=message_index,
                    role="assistant",
                    default_weight=1.0,
                )
            )
        else:
            spans.append(
                _span(sample, "\n", "turn_end", message_index=message_index, role=message.role)
            )
    if add_generation_prompt:
        spans.append(_span(sample, "<|assistant|>\n", "generation_prompt", role="assistant"))
    return RenderedDocument(
        sample_id=sample.id,
        spans=tuple(span for span in spans if span.text),
        renderer="olmo2_instruct",
        renderer_version=1,
        options=FrozenMap(
            (
                ("add_generation_prompt", add_generation_prompt),
                ("template_repo_id", OLMO2_INSTRUCT_TEMPLATE_REPO),
                ("template_revision", OLMO2_INSTRUCT_TEMPLATE_REVISION),
            )
        ),
    )


def get_renderer(name: str):
    if name in {"qwen3_5", "qwen3_5_text"}:
        return render_qwen3_5
    if name in {"olmo2", "olmo2_instruct"}:
        return render_olmo2_instruct
    raise ValueError(f"unsupported renderer {name!r}")


def renderer_identity(name: str) -> dict[str, str | int]:
    if name in {"qwen3_5", "qwen3_5_text"}:
        return {
            "name": "qwen3_5_text",
            "version": 1,
            "template_repo_id": QWEN35_TEMPLATE_REPO,
            "template_revision": QWEN35_TEMPLATE_REVISION,
        }
    if name in {"olmo2", "olmo2_instruct"}:
        return {
            "name": "olmo2_instruct",
            "version": 1,
            "template_repo_id": OLMO2_INSTRUCT_TEMPLATE_REPO,
            "template_revision": OLMO2_INSTRUCT_TEMPLATE_REVISION,
        }
    raise ValueError(f"unsupported renderer {name!r}")
