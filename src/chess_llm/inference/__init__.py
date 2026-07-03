"""Inference and deployment-facing interfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import chess

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


class DeterministicLegalMoveClient:
    """Local test client that always returns one deterministic legal move."""

    def __init__(self, *, model_id: str = "deterministic-legal-stub") -> None:
        self.model_id = model_id

    def complete(self, inference_request: InferenceRequest) -> InferenceResponse:
        fen = _fen_from_request(inference_request)
        move_uci: str | None = None
        if fen:
            try:
                board = chess.Board(fen)
            except ValueError:
                board = None
            if board is not None:
                legal_moves = sorted(move.uci() for move in board.legal_moves)
                move_uci = legal_moves[0] if legal_moves else None

        if move_uci is None:
            raw_text = "<think>Deterministic legal stub found no legal move.</think>"
        else:
            raw_text = (
                f"<think>Deterministic legal stub selected {move_uci} "
                "from the sorted legal move list.</think>\n"
                f"<move>{move_uci}</move>"
            )
        return InferenceResponse(
            model_id=inference_request.model_id or self.model_id,
            raw_text=raw_text,
            metadata={"client": "deterministic_legal_stub", "fen": fen},
        )


def _fen_from_request(inference_request: InferenceRequest) -> str | None:
    fen = inference_request.metadata.get("fen")
    if isinstance(fen, str) and fen.strip():
        return fen.strip()

    for message in reversed(inference_request.messages):
        match = re.search(r"FEN:\s*(.+)", message.content)
        if match:
            return match.group(1).splitlines()[0].strip()
    return None


__all__ = [
    "DeterministicLegalMoveClient",
    "InferenceClient",
    "InferenceRequest",
    "InferenceResponse",
]
