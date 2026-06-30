"""Inference and deployment-facing interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from chess_llm.artifacts.schemas import ChatMessage


@dataclass(frozen=True)
class InferenceRequest:
    """A model generation request for chess move selection."""

    model_id: str
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResponse:
    """Raw text returned by an inference backend."""

    model_id: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceClient(Protocol):
    """Protocol implemented by concrete inference clients."""

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Return a raw model response for ``request``."""


__all__ = ["InferenceClient", "InferenceRequest", "InferenceResponse"]
