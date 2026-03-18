"""Source 8: Chess960 Position Generation.

Generate random or enumerated Chess960 starting positions and
derived mid-game positions.
"""

from __future__ import annotations

from random import Random

import chess


def generate_random(rng: Random | None = None) -> tuple[chess.Board, int]:
    """Return a random Chess960 starting position and its ID (0-959)."""
    rng = rng or Random()
    pos_id = rng.randint(0, 959)
    board = chess.Board.from_chess960_pos(pos_id)
    board.chess960 = True
    return board, pos_id


def generate_all() -> list[tuple[chess.Board, int]]:
    """Return all 960 starting positions."""
    result = []
    for pos_id in range(960):
        board = chess.Board.from_chess960_pos(pos_id)
        board.chess960 = True
        result.append((board, pos_id))
    return result


def apply_random_moves(
    board: chess.Board,
    rng: Random | None = None,
    n_moves: int = 10,
) -> chess.Board:
    """Play *n_moves* random legal moves to get a mid-game Chess960 position.

    Returns a copy of the board after the moves.  If the game ends before
    *n_moves*, returns the board at that point.
    """
    rng = rng or Random()
    b = board.copy()
    for _ in range(n_moves):
        legal = list(b.legal_moves)
        if not legal:
            break
        b.push(rng.choice(legal))
    return b


def sample_chess960_positions(
    n: int,
    n_random_moves: int = 0,
    rng: Random | None = None,
) -> list[dict]:
    """Sample *n* Chess960 positions with optional random moves applied.

    Returns list of ``{fen, chess960_id, is_chess960, n_moves_applied}``.
    """
    rng = rng or Random()
    results = []
    for _ in range(n):
        board, pos_id = generate_random(rng)
        if n_random_moves > 0:
            actual_moves = rng.randint(0, n_random_moves)
            board = apply_random_moves(board, rng, actual_moves)
        else:
            actual_moves = 0

        results.append({
            "fen": board.fen(),
            "chess960_id": pos_id,
            "is_chess960": True,
            "n_moves_applied": actual_moves,
        })
    return results
