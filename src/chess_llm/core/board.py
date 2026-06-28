"""Small python-chess helpers shared by evaluation and Autodata code."""

from __future__ import annotations

import chess


def canonical_fen_key(fen: str) -> str:
    """Return a FEN identity key that ignores halfmove/fullmove counters."""
    parts = fen.strip().split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return fen.strip()


def validate_fen(fen: str, chess960: bool = False) -> bool:
    """Return True when *fen* can be parsed by python-chess."""
    try:
        chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return False
    return True


def legal_moves(fen: str, chess960: bool = False) -> list[str]:
    """Return sorted legal UCI moves, or an empty list for invalid FEN."""
    try:
        board = chess.Board(fen, chess960=chess960)
    except (TypeError, ValueError):
        return []
    return sorted(move.uci() for move in board.legal_moves)


def is_legal_move(fen: str, move_uci: str, chess960: bool = False) -> bool:
    """Return True when *move_uci* is legal in *fen*."""
    try:
        board = chess.Board(fen, chess960=chess960)
        move = chess.Move.from_uci(move_uci)
    except (TypeError, ValueError):
        return False
    return move in board.legal_moves
