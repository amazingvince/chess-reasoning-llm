"""Flexible model answer parsing for chess move outputs."""

from __future__ import annotations

import json
import re

from chess_llm.artifacts.schemas import ParsedAnswer


_MOVE_TAG_RE = re.compile(
    r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>",
    re.IGNORECASE,
)
_STRICT_MOVE_TAG_RE = re.compile(r"<move>([a-h][1-8][a-h][1-8][qrbn]?)</move>")
_THINK_MOVE_STRICT_RE = re.compile(
    r"^<think>.*?</think>\s*<move>[a-h][1-8][a-h][1-8][qrbn]?</move>\s*$",
    re.DOTALL,
)
_BARE_UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?", re.IGNORECASE)
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
# Complete FEN strings: piece placement, side, castling, en passant, and
# optional move counters.  FEN rank rows such as "b2b4" would otherwise parse
# as UCI moves during prose fallback extraction.
_FEN_STRING_RE = re.compile(
    r"(?:[prnbqk1-8]+/){7}[prnbqk1-8]+"
    r"\s+[wb]\s+(?:-|[a-hkq]+)\s+(?:-|[a-h][36])"
    r"(?:\s+\d+\s+\d+)?",
    re.IGNORECASE,
)


def strip_fen_strings(text: str) -> str:
    """Remove complete FEN strings so board rows cannot parse as UCI moves."""
    return _FEN_STRING_RE.sub(" ", text or "")


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

    prose_moves = _unique_moves(_UCI_RE.findall(strip_fen_strings(raw_text)))
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


def validate_think_move_format(text: str) -> bool:
    """Return True for strict ``<think>...</think><move>UCI</move>`` output."""
    return _THINK_MOVE_STRICT_RE.match(text or "") is not None


def extract_uci_from_move_tag(text: str) -> str | None:
    """Extract a strict lowercase UCI move from the first ``<move>`` tag."""
    match = _STRICT_MOVE_TAG_RE.search(text or "")
    return match.group(1) if match else None


def extract_move(text: str) -> str | None:
    """Extract the first benchmark-style UCI move from tag, bare text, or prose.

    This helper intentionally returns the first prose UCI mention.  Use
    ``parse_answer`` when ambiguous prose should be rejected instead.
    """
    raw_text = text or ""
    uci = extract_uci_from_move_tag(raw_text)
    if uci is not None:
        return uci

    stripped = raw_text.strip().lower()
    if _BARE_UCI_RE.fullmatch(stripped):
        return stripped

    match = _UCI_RE.search(strip_fen_strings(stripped))
    if match:
        return match.group(1).lower()
    return None


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
    if _BARE_UCI_RE.fullmatch(move):
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
