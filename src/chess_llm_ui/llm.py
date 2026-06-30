"""LLM clients for the UI backend."""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib import error, request

import chess

from chess_llm.inference import InferenceRequest, InferenceResponse


class LLMClientError(RuntimeError):
    """Raised when an inference backend fails before returning usable text."""


class StaticLLMClient:
    """Test and demo client that returns one static response."""

    def __init__(self, raw_text: str, *, model_id: str = "static-model") -> None:
        self.raw_text = raw_text
        self.model_id = model_id

    def complete(self, inference_request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            model_id=inference_request.model_id or self.model_id,
            raw_text=self.raw_text,
            metadata={"client": "static"},
        )


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


class OpenAICompatibleLLMClient:
    """Small chat-completions client for OpenAI-compatible endpoints."""

    def __init__(self, *, base_url: str, api_key: str | None = None, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(self, inference_request: InferenceRequest) -> InferenceResponse:
        payload = {
            "model": inference_request.model_id,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in inference_request.messages
            ],
            "temperature": inference_request.temperature,
            "max_tokens": inference_request.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"LLM endpoint request failed: {exc.reason}") from exc

        try:
            parsed: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM endpoint returned invalid JSON") from exc
        choices = parsed.get("choices") or []
        if not choices:
            raise LLMClientError("LLM endpoint returned no choices")
        message = choices[0].get("message") or {}
        raw_text = str(message.get("content") or "")
        return InferenceResponse(
            model_id=str(parsed.get("model") or inference_request.model_id),
            raw_text=raw_text,
            metadata={
                "client": "openai_compatible",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": parsed.get("usage"),
            },
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
