"""Chess960 position generation helpers."""

from __future__ import annotations

from random import Random

import chess


def generate_random(rng: Random | None = None) -> tuple[chess.Board, int]:
    """Return a random Chess960 starting position and its Scharnagl ID."""
    rng = rng or Random()
    position_id = rng.randint(0, 959)
    board = chess.Board.from_chess960_pos(position_id)
    board.chess960 = True
    return board, position_id


def generate_all() -> list[tuple[chess.Board, int]]:
    """Return all 960 Chess960 starting positions."""
    positions: list[tuple[chess.Board, int]] = []
    for position_id in range(960):
        board = chess.Board.from_chess960_pos(position_id)
        board.chess960 = True
        positions.append((board, position_id))
    return positions


def apply_random_moves(
    board: chess.Board,
    rng: Random | None = None,
    n_moves: int = 10,
) -> tuple[chess.Board, int]:
    """Return a copy of *board* and the number of random legal moves pushed."""
    rng = rng or Random()
    current = board.copy()
    current.chess960 = bool(board.chess960)
    applied = 0
    for _ in range(n_moves):
        legal_moves = list(current.legal_moves)
        if not legal_moves:
            break
        current.push(rng.choice(legal_moves))
        applied += 1
    return current, applied


def sample_chess960_positions(
    n: int,
    n_random_moves: int = 0,
    rng: Random | None = None,
) -> list[dict]:
    """Sample Chess960 positions with optional random continuations."""
    rng = rng or Random()
    positions: list[dict] = []
    for _ in range(n):
        board, position_id = generate_random(rng)
        if n_random_moves > 0:
            requested_moves = rng.randint(0, n_random_moves)
            board, actual_moves = apply_random_moves(board, rng, requested_moves)
        else:
            actual_moves = 0
        positions.append(
            {
                "fen": board.fen(),
                "chess960_id": position_id,
                "is_chess960": True,
                "n_moves_applied": actual_moves,
            }
        )
    return positions


__all__ = [
    "apply_random_moves",
    "generate_all",
    "generate_random",
    "sample_chess960_positions",
]
