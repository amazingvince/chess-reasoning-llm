"""Source 2: Lichess Chess Puzzles.

Load ~4M tactical puzzles. Critical: applies the setup-move preprocessing
so the FEN reflects the actual puzzle position (after opponent's move).
"""

from __future__ import annotations

import logging
from typing import Iterator

import chess
from datasets import load_dataset

from config.settings import HF_DATASETS

logger = logging.getLogger(__name__)


def load_puzzles(
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes: list[str] | None = None,
    max_puzzles: int | None = None,
) -> Iterator[dict]:
    """Yield preprocessed puzzle dicts.

    The raw Lichess puzzle FEN is BEFORE the opponent's setup move.
    This function pushes that move so the yielded FEN is the actual
    puzzle position the solver faces.

    Each dict::

        {fen, side, legal_moves, solution_first_move, solution_full,
         themes: list[str], rating, puzzle_id}
    """
    ds = load_dataset(
        HF_DATASETS["lichess_puzzles"], split="train", streaming=True
    )
    count = 0
    for row in ds:
        if max_puzzles is not None and count >= max_puzzles:
            return

        rating = row.get("Rating", row.get("rating", 0))
        if isinstance(rating, str):
            try:
                rating = int(rating)
            except ValueError:
                continue

        if min_rating is not None and rating < min_rating:
            continue
        if max_rating is not None and rating > max_rating:
            continue

        raw_themes_str = row.get("Themes", row.get("themes", ""))
        if isinstance(raw_themes_str, list):
            puzzle_themes = raw_themes_str
        else:
            puzzle_themes = raw_themes_str.split() if raw_themes_str else []

        # Theme filter
        if themes is not None:
            if not any(t in puzzle_themes for t in themes):
                continue

        fen = row.get("FEN", row.get("fen", ""))
        moves_str = row.get("Moves", row.get("moves", ""))
        if not fen or not moves_str:
            continue

        result = _preprocess_puzzle(
            fen, moves_str, puzzle_themes, rating,
            row.get("PuzzleId", row.get("puzzle_id", "")),
        )
        if result is not None:
            yield result
            count += 1


def _preprocess_puzzle(
    fen: str,
    moves_str: str,
    themes: list[str],
    rating: int,
    puzzle_id: str,
) -> dict | None:
    """Apply setup-move and return preprocessed puzzle dict."""
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return None

    moves = moves_str.split()
    if len(moves) < 2:
        return None

    # Step 1: push the opponent's setup move
    try:
        setup_move = chess.Move.from_uci(moves[0])
        if setup_move not in board.legal_moves:
            return None
        board.push(setup_move)
    except (ValueError, chess.InvalidMoveError):
        return None

    # Step 2: puzzle position
    puzzle_fen = board.fen()
    side = "white" if board.turn == chess.WHITE else "black"

    # Step 3: solution moves
    solution_moves = moves[1:]
    first_solution = solution_moves[0]

    # Step 4: validate first solution move
    try:
        sol_move = chess.Move.from_uci(first_solution)
        if sol_move not in board.legal_moves:
            return None
    except (ValueError, chess.InvalidMoveError):
        return None

    legal_moves = [m.uci() for m in board.legal_moves]

    return {
        "fen": puzzle_fen,
        "side": side,
        "legal_moves": legal_moves,
        "solution_first_move": first_solution,
        "solution_full": solution_moves,
        "themes": themes,
        "rating": rating,
        "puzzle_id": puzzle_id,
    }
