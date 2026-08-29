"""Small, explicit dispatch table for architecture-local model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelImplementation:
    architecture: str
    config_type: type
    forward: Callable[..., Any]
    init_params: Callable[..., Any]
    load_hf_checkpoint: Callable[..., Any]
    parameter_count: Callable[..., int]
    tiny_config: Callable[..., Any]
    validate_params: Callable[..., None]


def get_model_implementation(architecture: str) -> ModelImplementation:
    """Resolve one supported model without hiding its architecture-local code."""

    if architecture == "qwen3_5":
        from .qwen3_5 import (
            Qwen35Config,
            forward,
            init_params,
            load_hf_checkpoint,
            parameter_count,
            tiny_config,
            validate_params,
        )

        return ModelImplementation(
            architecture,
            Qwen35Config,
            forward,
            init_params,
            load_hf_checkpoint,
            parameter_count,
            tiny_config,
            validate_params,
        )
    if architecture == "olmo2":
        from .olmo2 import (
            Olmo2Config,
            forward,
            init_params,
            load_hf_checkpoint,
            parameter_count,
            tiny_config,
            validate_params,
        )

        return ModelImplementation(
            architecture,
            Olmo2Config,
            forward,
            init_params,
            load_hf_checkpoint,
            parameter_count,
            tiny_config,
            validate_params,
        )
    raise ValueError(f"unsupported model architecture {architecture!r}")


__all__ = ["ModelImplementation", "get_model_implementation"]
