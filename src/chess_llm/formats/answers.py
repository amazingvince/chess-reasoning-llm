"""Flexible model answer parsing for chess move outputs."""

from __future__ import annotations

import json
import re

from chess_llm.artifacts.schemas import ParsedAnswer


_MOVE_TAG_RE = re.compile(
    r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>",
    re.IGNORECASE,
)
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


def parse_answer(text: str) -> ParsedAnswer:
    """Parse a model answer into a normalized chess move artifact.

    Precedence is:
    1. ``<move>...</move>`` tag
    2. JSON object containing ``move_uci``
    3. bare UCI
    4. unambiguous prose UCI
    """
    raw_text = text or ""

    tagged = _MOVE_TAG_RE.findall(raw_text)
    if tagged:
        unique = _unique_moves(tagged)
        if len(unique) == 1:
            return ParsedAnswer(
                raw_text=raw_text,
                move_uci=unique[0],
                format_type="move_tag",
            )
        return ParsedAnswer(
            raw_text=raw_text,
            format_type="none",
            parse_error="ambiguous move tags",
        )

    json_move = _parse_json_move(raw_text)
    if json_move is not None:
        return ParsedAnswer(
            raw_text=raw_text,
            move_uci=json_move,
            format_type="json",
        )

    stripped = raw_text.strip()
    if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", stripped, re.IGNORECASE):
        return ParsedAnswer(
            raw_text=raw_text,
            move_uci=stripped.lower(),
            format_type="bare_uci",
        )

    prose_moves = _unique_moves(_UCI_RE.findall(raw_text))
    if len(prose_moves) == 1:
        return ParsedAnswer(
            raw_text=raw_text,
            move_uci=prose_moves[0],
            format_type="prose_uci",
        )
    if len(prose_moves) > 1:
        return ParsedAnswer(
            raw_text=raw_text,
            format_type="none",
            parse_error="ambiguous UCI moves in prose",
        )
    return ParsedAnswer(
        raw_text=raw_text,
        format_type="none",
        parse_error="no UCI move found",
    )


def _parse_json_move(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    move = payload.get("move_uci")
    if not isinstance(move, str):
        return None
    move = move.strip().lower()
    if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move):
        return move
    return None


def _unique_moves(moves: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for move in moves:
        normalized = move.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
