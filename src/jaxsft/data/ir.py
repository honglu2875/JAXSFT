"""Model-independent instruction-data intermediate representation.

The types in this file intentionally retain semantic chunks until rendering.
Adapters must reject structures they cannot map instead of silently converting
them to strings.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "developer", "user", "assistant", "tool"]
PartKind = Literal["text", "reasoning", "code", "tool_call", "tool_result", "media", "control"]
JSONScalar = None | bool | int | float | str
# Recursive aliases are kept structural at runtime; validators below enforce
# the exact JSON domain and static type checkers still see each public field.
FrozenJSON = Any


@dataclass(frozen=True)
class FrozenMap(Mapping[str, FrozenJSON]):
    """Small immutable, deterministically ordered JSON mapping."""

    items_tuple: tuple[tuple[str, FrozenJSON], ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(key for key, _ in self.items_tuple)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("FrozenMap keys must be strings")
        if len(keys) != len(set(keys)):
            raise ValueError("FrozenMap keys must be unique")

    def __getitem__(self, key: str) -> FrozenJSON:
        for candidate, value in self.items_tuple:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.items_tuple)

    def __len__(self) -> int:
        return len(self.items_tuple)


def freeze_json(value: Any, *, path: str = "value") -> FrozenJSON:
    """Validate and recursively freeze a JSON-like value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(tuple((str(key), freeze_json(item, path=f"{path}.{key}")) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, path=f"{path}[]") for item in value)
    raise TypeError(f"{path} must contain only JSON values, got {type(value).__name__}")


def thaw_json(value: FrozenJSON) -> Any:
    if isinstance(value, FrozenMap):
        return {key: thaw_json(item) for key, item in value.items_tuple}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class SourceRef:
    repo_id: str
    revision: str
    config: str
    split: str
    row_index: int
    row_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("repo_id", "revision", "config", "split"):
            if not getattr(self, name):
                raise ValueError(f"SourceRef.{name} must be non-empty")
        if self.row_index < 0:
            raise ValueError("SourceRef.row_index must be non-negative")


@dataclass(frozen=True)
class Part:
    kind: PartKind
    value: str | FrozenJSON
    call_id: str | None = None
    tool_name: str | None = None
    tags: frozenset[str] = frozenset()
    attributes: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if self.kind not in {"text", "reasoning", "code", "tool_call", "tool_result", "media", "control"}:
            raise ValueError(f"unsupported part kind {self.kind!r}")
        if self.kind in {"text", "reasoning", "code", "tool_result", "control"} and not isinstance(
            self.value, str
        ):
            raise TypeError(f"{self.kind} part value must be a string")
        if not isinstance(self.tags, frozenset) or any(not isinstance(tag, str) for tag in self.tags):
            raise TypeError("Part.tags must be a frozenset of strings")
        for name in ("call_id", "tool_name"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"Part.{name} must be a non-empty string when present")


@dataclass(frozen=True)
class Message:
    role: Role
    parts: tuple[Part, ...]
    name: str | None = None
    call_id: str | None = None
    attributes: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if self.role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role {self.role!r}")
        if not isinstance(self.parts, tuple):
            raise TypeError("Message.parts must be a tuple")
        for field_name in ("name", "call_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"Message.{field_name} must be a non-empty string when present")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    parameters: FrozenMap = field(default_factory=FrozenMap)
    attributes: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDefinition.name must be non-empty")


@dataclass(frozen=True)
class Sample:
    id: str
    source: SourceRef
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    attributes: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Sample.id must be non-empty")
        if not self.messages:
            raise ValueError("Sample.messages must be non-empty")
        if not isinstance(self.messages, tuple) or not isinstance(self.tools, tuple):
            raise TypeError("Sample.messages and Sample.tools must be tuples")


@dataclass(frozen=True)
class SemanticRef:
    sample_id: str
    message_index: int | None
    part_index: int | None
