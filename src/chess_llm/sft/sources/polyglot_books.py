"""Source 5: Polyglot Opening Books.

Read .bin Polyglot book files and extract weighted moves for positions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import chess
import chess.polyglot


@contextmanager
def load_book(path: str):
    """Open a Polyglot book as a context manager."""
    reader = chess.polyglot.open_reader(path)
    try:
        yield reader
    finally:
        reader.close()


def get_weighted_moves(
    reader: chess.polyglot.MemoryMappedReader,
    board: chess.Board,
) -> list[tuple[str, int]]:
    """Return ``[(move_uci, weight), ...]`` sorted by weight descending.

    Returns an empty list if the position is not in the book.
    """
    try:
        entries = list(reader.find_all(board))
    except KeyError:
        return []

    result = []
    for entry in entries:
        move_uci = entry.move.uci()
        # Verify the move is legal in this position
        if entry.move in board.legal_moves:
            result.append((move_uci, entry.weight))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def scan_book_positions(
    reader: chess.polyglot.MemoryMappedReader,
    max_depth: int = 20,
) -> Iterator[dict]:
    """Walk the opening book by replaying top-weighted moves.

    Yields ``{fen, moves: [(uci, weight), ...], depth}`` for each
    position reachable by following the highest-weight move.
    """
    board = chess.Board()
    for depth in range(max_depth):
        moves = get_weighted_moves(reader, board)
        if not moves:
            break
        yield {
            "fen": board.fen(),
            "moves": moves,
            "depth": depth,
        }
        # Follow the top move
        top_move = chess.Move.from_uci(moves[0][0])
        board.push(top_move)


__all__ = ["get_weighted_moves", "load_book", "scan_book_positions"]
