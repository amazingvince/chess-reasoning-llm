"""Board rendering helpers shared by data generation and evaluation."""

from __future__ import annotations

import chess


def render_ascii_board(board: chess.Board) -> str:
    """Render a board as an ASCII diagram with rank and file labels."""
    lines: list[str] = []
    for rank in range(7, -1, -1):
        row: list[str] = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            row.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} {' '.join(row)}")
    lines.append("  a b c d e f g h")
    return "\n".join(lines)
