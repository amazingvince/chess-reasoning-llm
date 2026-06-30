"""Pure helpers for preprocessing Lichess puzzle rows."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Mapping

import chess

from chess_llm.sft.settings import DEFAULT_HF_CACHE_DIR, DEFAULT_HF_DATASETS
from chess_llm.sft.sources._streaming import (
    close_streaming_dataset,
    streaming_shutdown_wait_seconds,
)

DatasetLoader = Callable[..., Iterable[Mapping]]


def preprocess_puzzle(
    fen: str,
    moves_str: str,
    themes: list[str],
    rating: int,
    puzzle_id: str,
) -> dict | None:
    """Apply the setup move and validate the full solution continuation."""
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return None

    moves = moves_str.split()
    if len(moves) < 2:
        return None

    try:
        setup_move = chess.Move.from_uci(moves[0])
    except (ValueError, chess.InvalidMoveError):
        return None
    if setup_move not in board.legal_moves:
        return None
    board.push(setup_move)

    puzzle_fen = board.fen()
    side = "white" if board.turn == chess.WHITE else "black"
    legal_moves = [move.uci() for move in board.legal_moves]

    solution_moves: list[str] = []
    for raw_move in moves[1:]:
        try:
            move = chess.Move.from_uci(raw_move)
        except (ValueError, chess.InvalidMoveError):
            return None
        if move not in board.legal_moves:
            return None
        solution_moves.append(move.uci())
        board.push(move)

    return {
        "fen": puzzle_fen,
        "side": side,
        "legal_moves": legal_moves,
        "solution_first_move": solution_moves[0],
        "solution_full": solution_moves,
        "themes": themes,
        "rating": rating,
        "puzzle_id": puzzle_id,
    }


def load_puzzles(
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes: list[str] | None = None,
    max_puzzles: int | None = None,
    *,
    dataset_loader: DatasetLoader | None = None,
    dataset_name: str = DEFAULT_HF_DATASETS["lichess_puzzles"],
) -> Iterator[dict]:
    """Yield preprocessed Lichess puzzle rows from the HF dataset."""
    if max_puzzles is not None and max_puzzles <= 0:
        return

    uses_default_loader = dataset_loader is None
    loader = dataset_loader or _default_dataset_loader
    count = 0
    rows = loader(dataset_name, split="train", streaming=True)
    row_iter = iter(rows)
    try:
        for row in row_iter:
            if max_puzzles is not None and count >= max_puzzles:
                return

            rating = _coerce_int(row.get("Rating", row.get("rating")))
            if rating is None:
                continue
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

            puzzle_themes = _normalize_themes(row.get("Themes", row.get("themes", "")))
            if themes is not None and not any(theme in puzzle_themes for theme in themes):
                continue

            fen = str(row.get("FEN", row.get("fen", "")) or "")
            moves_str = str(row.get("Moves", row.get("moves", "")) or "")
            if not fen or not moves_str:
                continue

            parsed = preprocess_puzzle(
                fen,
                moves_str,
                puzzle_themes,
                rating,
                str(row.get("PuzzleId", row.get("puzzle_id", "")) or ""),
            )
            if parsed is None:
                continue
            yield parsed
            count += 1
    finally:
        close_streaming_dataset(
            row_iter,
            rows,
            shutdown_wait_seconds=(
                streaming_shutdown_wait_seconds() if uses_default_loader else 0.0
            ),
        )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_themes(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(theme) for theme in value if str(theme)]
    text = str(value or "")
    return text.split() if text else []


def _default_dataset_loader(*args, **kwargs):
    os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE_DIR)
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


__all__ = [
    "load_puzzles",
    "preprocess_puzzle",
]
