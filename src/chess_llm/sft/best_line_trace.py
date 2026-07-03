"""Fixed-grammar best-line trace helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import chess

from chess_llm.sft.settings import DEFAULT_EVAL_BUCKETS


_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
_BEST_LINE_TRACE_RE = re.compile(
    r"^<think>\n"
    r"Root: (?P<root>[a-h][1-8][a-h][1-8][qrbn]?)\n"
    r"Eval: (?P<score>[+-]?\d+cp|M-?\d+); Bucket: (?P<bucket>[a-z_ ]+)\n"
    r"PV: (?P<pv>[a-h][1-8][a-h][1-8][qrbn]?"
    r"(?: [a-h][1-8][a-h][1-8][qrbn]?)*)\n"
    r"Best: (?P<best>[a-h][1-8][a-h][1-8][qrbn]?)\n"
    r"</think><move>(?P<move>[a-h][1-8][a-h][1-8][qrbn]?)</move>\s*$",
    re.IGNORECASE,
)


def best_line_bucket_label(cp: int | None = None, mate: int | None = None) -> str:
    """Return the side-to-move bucket label for a best-line eval."""
    if mate is not None:
        return "forced mate"
    if cp is None:
        return "equal"
    abs_cp = abs(int(cp))
    for low, high, label in DEFAULT_EVAL_BUCKETS:
        if low <= abs_cp < high:
            return label
    return DEFAULT_EVAL_BUCKETS[-1][2]


def format_best_line_trace_answer(
    root_uci: str,
    *,
    cp: int | None = None,
    mate: int | None = None,
    pv_moves: Sequence[str],
) -> str:
    """Return the exact fixed grammar for a seeded best-line trace."""
    root = root_uci.strip().lower()
    pv = [str(move).strip().lower() for move in pv_moves if str(move).strip()]
    score_text = f"M{int(mate)}" if mate is not None else f"{int(cp or 0):+d}cp"
    bucket = best_line_bucket_label(cp=cp, mate=mate)
    return "\n".join(
        [
            "<think>",
            f"Root: {root}",
            f"Eval: {score_text}; Bucket: {bucket}",
            f"PV: {' '.join(pv)}",
            f"Best: {root}",
            f"</think><move>{root}</move>",
        ]
    )


def parse_best_line_trace_answer(text: str) -> dict[str, object] | None:
    """Parse the fixed best-line trace grammar."""
    match = _BEST_LINE_TRACE_RE.match(text or "")
    if match is None:
        return None
    return {
        "root": match.group("root").lower(),
        "score": match.group("score"),
        "bucket": " ".join(match.group("bucket").lower().split()),
        "pv": [move.lower() for move in match.group("pv").split()],
        "best": match.group("best").lower(),
        "move": match.group("move").lower(),
    }


def normalize_pv_moves(value: object) -> list[str]:
    """Normalize a PV value from a list or space-delimited string."""
    if isinstance(value, str):
        candidates = value.split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = [str(item) for item in value]
    else:
        return []
    moves = [move.strip().lower() for move in candidates if move.strip()]
    if any(_UCI_RE.fullmatch(move) is None for move in moves):
        return []
    return moves


def validate_pv_moves(
    fen: str,
    pv_moves: Sequence[str],
    *,
    chess960: bool = False,
) -> bool:
    """Return True when every PV move is legal in sequence from the FEN."""
    if not pv_moves:
        return False
    try:
        board = chess.Board(fen, chess960=chess960)
    except (ValueError, TypeError):
        return False
    if not board.is_valid():
        return False
    for move_uci in pv_moves:
        try:
            move = chess.Move.from_uci(str(move_uci).lower())
        except ValueError:
            return False
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


def best_line_trace_payload(
    fen: str,
    rating: Mapping[str, object],
    *,
    chess960: bool = False,
) -> dict[str, object] | None:
    """Build a validated best-line trace payload from one MultiPV rating row."""
    uci = rating.get("uci") or rating.get("move") or rating.get("best_move")
    if not isinstance(uci, str) or not uci.strip():
        return None
    root = uci.strip().lower()
    cp = _optional_int(rating.get("cp", rating.get("centipawn")))
    mate = _optional_int(rating.get("mate", rating.get("mate_in")))
    if cp is None and mate is None:
        return None
    pv_moves = normalize_pv_moves(
        rating.get("pv") or rating.get("pv_line") or rating.get("principal_variation")
    )
    if not pv_moves or pv_moves[0] != root:
        return None
    if not validate_pv_moves(fen, pv_moves, chess960=chess960):
        return None
    answer = format_best_line_trace_answer(
        root,
        cp=cp,
        mate=mate,
        pv_moves=pv_moves,
    )
    payload: dict[str, object] = {
        "best_move": root,
        "pv": pv_moves,
        "pv_line": " ".join(pv_moves),
        "cp": cp,
        "mate": mate,
        "bucket": best_line_bucket_label(cp=cp, mate=mate),
        "expected_answer": answer,
    }
    return payload


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "best_line_bucket_label",
    "best_line_trace_payload",
    "format_best_line_trace_answer",
    "normalize_pv_moves",
    "parse_best_line_trace_answer",
    "validate_pv_moves",
]
