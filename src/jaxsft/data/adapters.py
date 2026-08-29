"""Explicit adapters from common instruction-dataset rows into the IR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .ir import FrozenMap, Message, Part, Sample, SourceRef, ToolDefinition, freeze_json


class AdapterError(ValueError):
    """A row cannot be represented without guessing."""


@dataclass(frozen=True)
class AdapterContext:
    repo_id: str
    revision: str
    config: str
    split: str
    row_index: int


def _source(context: AdapterContext, row_id: str | None) -> SourceRef:
    return SourceRef(
        repo_id=context.repo_id,
        revision=context.revision,
        config=context.config,
        split=context.split,
        row_index=context.row_index,
        row_id=row_id,
    )


def _sample_id(context: AdapterContext, row: Mapping[str, Any]) -> str:
    for key in ("prompt_id", "id", "uuid"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value):
            return f"{context.repo_id}:{context.split}:{value}"
    identity = f"{context.repo_id}\0{context.revision}\0{context.config}\0{context.split}\0{context.row_index}"
    return f"{context.repo_id}:{context.split}:{hashlib.sha256(identity.encode()).hexdigest()}"


def _text_parts(content: Any, *, role: str, path: str) -> list[Part]:
    if isinstance(content, str):
        kind = "tool_result" if role == "tool" else "text"
        return [Part(kind=kind, value=content)]
    if content is None:
        return []
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray, str)):
        raise AdapterError(f"{path} must be a string, null, or typed content-block list")
    parts: list[Part] = []
    for block_index, block in enumerate(content):
        block_path = f"{path}[{block_index}]"
        if not isinstance(block, Mapping):
            raise AdapterError(f"{block_path} must be a mapping")
        block_type = block.get("type")
        if block_type in ("text", "output_text", "input_text"):
            value = block.get("text")
            if not isinstance(value, str):
                raise AdapterError(f"{block_path}.text must be a string")
            parts.append(Part(kind="text", value=value))
        elif block_type in ("thinking", "reasoning"):
            value = block.get("thinking", block.get("text"))
            if not isinstance(value, str):
                raise AdapterError(f"{block_path} reasoning value must be a string")
            parts.append(Part(kind="reasoning", value=value))
        elif block_type in ("tool_use", "tool_call"):
            name = block.get("name")
            arguments = block.get("input", block.get("arguments", {}))
            if not isinstance(name, str) or not name:
                raise AdapterError(f"{block_path}.name must be a non-empty string")
            parts.append(
                Part(
                    kind="tool_call",
                    value=freeze_json(arguments, path=f"{block_path}.arguments"),
                    call_id=block.get("id"),
                    tool_name=name,
                )
            )
        elif block_type == "tool_result":
            value = block.get("content")
            if not isinstance(value, str):
                raise AdapterError(f"{block_path}.content must be a string")
            parts.append(Part(kind="tool_result", value=value, call_id=block.get("tool_use_id")))
        elif block_type in ("image", "image_url", "video"):
            parts.append(Part(kind="media", value=freeze_json(block, path=block_path)))
        else:
            raise AdapterError(f"{block_path} has unsupported type {block_type!r}")
    return parts


def _tool_call_parts(raw_calls: Any, *, path: str) -> list[Part]:
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes, bytearray)):
        raise AdapterError(f"{path} must be a list")
    result: list[Part] = []
    for call_index, raw_call in enumerate(raw_calls):
        call_path = f"{path}[{call_index}]"
        if not isinstance(raw_call, Mapping):
            raise AdapterError(f"{call_path} must be a mapping")
        function = raw_call.get("function", raw_call)
        if not isinstance(function, Mapping):
            raise AdapterError(f"{call_path}.function must be a mapping")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise AdapterError(f"{call_path}.function.name must be a non-empty string")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise AdapterError(f"{call_path}.function.arguments is invalid JSON") from error
        result.append(
            Part(
                kind="tool_call",
                value=freeze_json(arguments, path=f"{call_path}.function.arguments"),
                call_id=raw_call.get("id"),
                tool_name=name,
            )
        )
    return result


def _adapt_messages(raw_messages: Any) -> tuple[Message, ...]:
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes, bytearray)):
        raise AdapterError("messages must be a list")
    if not raw_messages:
        raise AdapterError("messages must not be empty")
    messages: list[Message] = []
    role_aliases = {"human": "user", "gpt": "assistant", "function": "tool"}
    for message_index, raw_message in enumerate(raw_messages):
        path = f"messages[{message_index}]"
        if not isinstance(raw_message, Mapping):
            raise AdapterError(f"{path} must be a mapping")
        raw_role = raw_message.get("role", raw_message.get("from"))
        role = role_aliases.get(raw_role, raw_role)
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise AdapterError(f"{path}.role has unsupported value {raw_role!r}")
        content = raw_message.get("content", raw_message.get("value"))
        parts = _text_parts(content, role=role, path=f"{path}.content")
        reasoning = raw_message.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant" or not isinstance(reasoning, str):
                raise AdapterError(f"{path}.reasoning_content requires an assistant string")
            parts.insert(0, Part(kind="reasoning", value=reasoning))
        parts.extend(_tool_call_parts(raw_message.get("tool_calls"), path=f"{path}.tool_calls"))
        messages.append(
            Message(
                role=role,
                parts=tuple(parts),
                name=raw_message.get("name"),
                call_id=raw_message.get("tool_call_id", raw_message.get("call_id")),
            )
        )
    return tuple(messages)


def _adapt_tools(raw_tools: Any) -> tuple[ToolDefinition, ...]:
    if raw_tools is None:
        return ()
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes, bytearray)):
        raise AdapterError("tools must be a list")
    tools: list[ToolDefinition] = []
    for tool_index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, Mapping):
            raise AdapterError(f"tools[{tool_index}] must be a mapping")
        function = raw_tool.get("function", raw_tool)
        if not isinstance(function, Mapping):
            raise AdapterError(f"tools[{tool_index}].function must be a mapping")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise AdapterError(f"tools[{tool_index}] has no function name")
        params = function.get("parameters", {})
        frozen = freeze_json(params, path=f"tools[{tool_index}].parameters")
        if not isinstance(frozen, FrozenMap):
            raise AdapterError(f"tools[{tool_index}].parameters must be a mapping")
        tools.append(ToolDefinition(name=name, description=str(function.get("description", "")), parameters=frozen))
    return tuple(tools)


def messages_adapter(row: Mapping[str, Any], context: AdapterContext) -> Sample:
    if "messages" not in row:
        raise AdapterError("messages adapter requires a messages field")
    row_id = str(row.get("prompt_id", row.get("id"))) if row.get("prompt_id", row.get("id")) is not None else None
    return Sample(
        id=_sample_id(context, row),
        source=_source(context, row_id),
        messages=_adapt_messages(row["messages"]),
        tools=_adapt_tools(row.get("tools")),
    )


def ultrachat_200k_adapter(row: Mapping[str, Any], context: AdapterContext) -> Sample:
    """Strict adapter for HuggingFaceH4/ultrachat_200k."""

    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("prompt_id"), str):
        raise AdapterError("UltraChat rows require string prompt and prompt_id fields")
    sample = messages_adapter(row, context)
    if sample.messages[0].role != "user":
        raise AdapterError("UltraChat conversation must start with a user message")
    if sample.messages[0].parts and sample.messages[0].parts[0].value != row["prompt"]:
        raise AdapterError("UltraChat prompt must equal the first user message")
    if any(message.role not in {"user", "assistant"} for message in sample.messages):
        raise AdapterError("UltraChat supports only user/assistant messages")
    return sample


def prompt_completion_adapter(row: Mapping[str, Any], context: AdapterContext) -> Sample:
    prompt, completion = row.get("prompt"), row.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise AdapterError("prompt/completion adapter requires string prompt and completion fields")
    return Sample(
        id=_sample_id(context, row),
        source=_source(context, str(row["id"]) if "id" in row else None),
        messages=(
            Message(role="user", parts=(Part(kind="text", value=prompt),)),
            Message(role="assistant", parts=(Part(kind="text", value=completion),)),
        ),
    )


def sharegpt_adapter(row: Mapping[str, Any], context: AdapterContext) -> Sample:
    conversations = row.get("conversations")
    if conversations is None:
        raise AdapterError("ShareGPT adapter requires conversations")
    mapped = {"messages": conversations, "id": row.get("id")}
    return messages_adapter(mapped, context)


ADAPTERS: dict[str, Callable[[Mapping[str, Any], AdapterContext], Sample]] = {
    "messages": messages_adapter,
    "ultrachat_200k": ultrachat_200k_adapter,
    "prompt_completion": prompt_completion_adapter,
    "sharegpt": sharegpt_adapter,
}


def get_adapter(name: str) -> Callable[[Mapping[str, Any], AdapterContext], Sample]:
    try:
        return ADAPTERS[name]
    except KeyError as error:
        raise AdapterError(f"unknown adapter {name!r}; choose one of {sorted(ADAPTERS)}") from error
