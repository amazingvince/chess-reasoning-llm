"""Small python-chess helpers shared by evaluation and Autodata code."""

from __future__ import annotations

import chess


def canonical_fen_key(fen: str, chess960: bool = False) -> str:
    """Return a FEN identity key that ignores halfmove/fullmove counters."""
    primary = _parse_fen(fen, chess960=chess960)
    normalized = primary[0] if primary and primary[1] else None
    if normalized is None and not chess960:
        fallback = _parse_fen(fen, chess960=True)
        if fallback and fallback[1]:
            normalized = fallback[0]
    if normalized is None and primary is not None:
        normalized = primary[0]
    if normalized is None:
        normalized = fen
    parts = normalized.strip().split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return normalized.strip()


def variant_fen_key(fen: str, chess960: bool = False) -> str:
    """Return a variant-aware FEN identity key for dedupe/blocklists.

    ``canonical_fen_key()`` deliberately returns only the normalized board
    state. That is useful for display and broad FEN grouping, but it can
    collide when the same four-field FEN is used under standard and Chess960
    rules. Data identity needs to keep those variants separate.
    """
    resolved_chess960 = chess960 or _fen_requires_chess960(fen)
    prefix = "960" if resolved_chess960 else "std"
    return f"{prefix}:{canonical_fen_key(fen, chess960=resolved_chess960)}"


def _fen_requires_chess960(fen: str) -> bool:
    """Return True when *fen* is valid as Chess960 but not standard chess."""
    standard = _parse_fen(fen, chess960=False)
    if standard and standard[1]:
        return False
    chess960 = _parse_fen(fen, chess960=True)
    return bool(chess960 and chess960[1])


def _parse_fen(fen: str, chess960: bool) -> tuple[str, bool] | None:
    try:
        board = chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return None
    return board.fen(en_passant="legal"), board.is_valid()


def validate_fen(fen: str, chess960: bool = False) -> bool:
    """Return True when *fen* can be parsed by python-chess."""
    try:
        board = chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return False
    return board.is_valid()


def legal_moves(fen: str, chess960: bool = False) -> list[str]:
    """Return sorted legal UCI moves, or an empty list for invalid FEN."""
    try:
        board = chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return []
    if not board.is_valid():
        return []
    return sorted(move.uci() for move in board.legal_moves)


def is_legal_move(fen: str, move_uci: str, chess960: bool = False) -> bool:
    """Return True when *move_uci* is legal in *fen*."""
    try:
        board = chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return False
    if not board.is_valid():
        return False
    return move_uci in {move.uci() for move in board.legal_moves}
