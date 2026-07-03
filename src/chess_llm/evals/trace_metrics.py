"""Reasoning-trace metrics for chess move protocols."""

from __future__ import annotations

from typing import Any
import re

import chess

from chess_llm.formats.answers import (
    extract_uci_from_move_tag,
    strip_fen_strings,
    validate_think_move_format,
)


_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_LINE_RE = re.compile(r"\b(?:Line|PV|Variation)\s*:\s*([^.\n<]+)", re.IGNORECASE)
_CANDIDATE_RE = re.compile(r"\bCandidates?\s*:\s*([^.\n<]+)", re.IGNORECASE)
_CONCLUSION_RE = re.compile(
    r"\b(?:Best|Conclusion|Final(?: move)?)\s*:\s*([a-h][1-8][a-h][1-8][qrbn]?)\b",
    re.IGNORECASE,
)
_BACKTRACK_RE = re.compile(r"\b(?:backtrack|back\s+to|return\s+to)\b", re.IGNORECASE)


def analyze_trace_metrics(
    fen: str,
    prediction: str,
    *,
    chess960: bool = False,
) -> dict[str, Any]:
    """Return lightweight faithfulness metrics for a `<think>/<move>` prediction."""

    metrics: dict[str, Any] = {
        "trace_referenced_move_accuracy": None,
        "trace_referenced_move_count": 0,
        "trace_illegal_referenced_move_count": 0,
        "trace_candidate_count": 0,
        "trace_line_depth": 0,
        "trace_backtrack_count": 0,
        "trace_step_accuracy": None,
        "trace_step_count": 0,
        "trace_conclusion_move_match": None,
        "trace_failure_buckets": [],
    }
    failure_buckets: list[str] = []
    text = str(prediction or "")
    think = _extract_think_text(text)
    if think is None:
        failure_buckets.append("missing_think")
        metrics["trace_failure_buckets"] = failure_buckets
        return metrics

    if not validate_think_move_format(text):
        failure_buckets.append("invalid_think_move_format")

    board = _board_from_fen(fen, chess960=chess960)
    if board is None:
        failure_buckets.append("invalid_fen")
        metrics["trace_failure_buckets"] = failure_buckets
        return metrics

    sanitized = strip_fen_strings(think)
    total_refs = 0
    valid_refs = 0
    line_spans: list[tuple[int, int]] = []
    step_total = 0
    step_valid = 0
    line_depth = 0

    for match in _LINE_RE.finditer(sanitized):
        line_spans.append(match.span())
        moves = _uci_tokens(match.group(1))
        line_depth = max(line_depth, len(moves))
        total_refs += len(moves)
        line_valid = _validate_line(board, moves)
        step_total += len(moves)
        step_valid += line_valid
        valid_refs += line_valid

    non_line_text = _remove_spans(sanitized, line_spans)
    non_line_moves = _uci_tokens(non_line_text)
    total_refs += len(non_line_moves)
    legal_root = {move.uci() for move in board.legal_moves}
    valid_refs += sum(1 for move in non_line_moves if move in legal_root)

    illegal_refs = total_refs - valid_refs
    if illegal_refs:
        failure_buckets.append("illegal_referenced_move")

    candidate_count = 0
    candidate_match = _CANDIDATE_RE.search(sanitized)
    if candidate_match is not None:
        candidate_count = len(_uci_tokens(candidate_match.group(1)))
    else:
        candidate_count = sum(
            1
            for line in sanitized.splitlines()
            if line.strip().lower().startswith("candidate ")
        )

    conclusion = _extract_conclusion_move(sanitized)
    move_tag = extract_uci_from_move_tag(text)
    conclusion_match = None
    if conclusion is not None and move_tag is not None:
        conclusion_match = 1.0 if conclusion == move_tag.lower() else 0.0
        if conclusion_match == 0.0:
            failure_buckets.append("conclusion_move_mismatch")
    elif conclusion is not None or move_tag is not None:
        failure_buckets.append("missing_conclusion_or_move_tag")

    metrics.update(
        {
            "trace_referenced_move_accuracy": (
                valid_refs / total_refs if total_refs else None
            ),
            "trace_referenced_move_count": total_refs,
            "trace_illegal_referenced_move_count": illegal_refs,
            "trace_candidate_count": candidate_count,
            "trace_line_depth": line_depth,
            "trace_backtrack_count": len(_BACKTRACK_RE.findall(sanitized)),
            "trace_step_accuracy": step_valid / step_total if step_total else None,
            "trace_step_count": step_total,
            "trace_conclusion_move_match": conclusion_match,
            "trace_failure_buckets": sorted(set(failure_buckets)),
        }
    )
    return metrics


def numeric_trace_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Return only numeric trace metrics suitable for benchmark score dicts."""

    numeric: dict[str, float | None] = {}
    for key, value in metrics.items():
        if key == "trace_failure_buckets":
            continue
        if value is None:
            numeric[key] = None
        elif isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


def _extract_think_text(text: str) -> str | None:
    match = _THINK_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def _uci_tokens(text: str) -> list[str]:
    return [match.lower() for match in _UCI_RE.findall(text or "")]


def _board_from_fen(fen: str, *, chess960: bool) -> chess.Board | None:
    try:
        board = chess.Board(fen, chess960=chess960)
    except (ValueError, TypeError):
        return None
    return board if board.is_valid() else None


def _validate_line(board: chess.Board, moves: list[str]) -> int:
    valid = 0
    replay = board.copy(stack=False)
    for move_uci in moves:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            continue
        if move not in replay.legal_moves:
            continue
        replay.push(move)
        valid += 1
    return valid


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    chunks: list[str] = []
    last = 0
    for start, end in sorted(spans):
        chunks.append(text[last:start])
        last = end
    chunks.append(text[last:])
    return " ".join(chunks)


def _extract_conclusion_move(text: str) -> str | None:
    matches = list(_CONCLUSION_RE.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).lower()


__all__ = ["analyze_trace_metrics", "numeric_trace_metrics"]
