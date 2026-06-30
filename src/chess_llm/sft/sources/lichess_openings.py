"""Lichess opening dataset loader and row parser."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator, Mapping

import chess

from chess_llm.sft.settings import DEFAULT_HF_CACHE_DIR, DEFAULT_HF_DATASETS

logger = logging.getLogger(__name__)

DatasetLoader = Callable[..., Iterable[Mapping]]


def parse_opening_row(row: Mapping) -> dict | None:
    """Parse one Lichess opening row and replay UCI moves to final FEN."""
    uci_text = str(row.get("uci", "") or "")
    if not uci_text.strip():
        return None

    board = chess.Board()
    uci_moves: list[str] = []
    for uci in uci_text.split():
        try:
            move = chess.Move.from_uci(uci)
        except (ValueError, chess.InvalidMoveError):
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)
        uci_moves.append(move.uci())

    parsed = {
        "eco_volume": row.get("eco-volume", row.get("eco_volume", "")),
        "eco": row.get("eco", ""),
        "name": row.get("name", ""),
        "pgn": row.get("pgn", ""),
        "uci_moves": uci_moves,
        "epd": row.get("epd", ""),
        "fen": board.fen(),
    }
    source_epd = str(row.get("epd", "") or "").strip()
    if source_epd:
        if source_epd != " ".join(board.fen().split()[:4]):
            parsed["epd_mismatch"] = True
    return parsed


def load_openings(
    max_openings: int | None = None,
    *,
    dataset_loader: DatasetLoader | None = None,
    dataset_name: str = DEFAULT_HF_DATASETS["lichess_openings"],
) -> Iterator[dict]:
    """Yield parsed opening rows from the Lichess opening dataset."""
    loader = dataset_loader or _default_dataset_loader
    count = 0
    for row in loader(dataset_name, split="train"):
        if max_openings is not None and count >= max_openings:
            return
        parsed = parse_opening_row(row)
        if parsed is None:
            logger.warning(
                "Invalid move sequence for opening %s: %s",
                row.get("name", ""),
                row.get("uci", ""),
            )
            continue
        yield parsed
        count += 1


def _default_dataset_loader(*args, **kwargs):
    os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE_DIR)
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


__all__ = [
    "load_openings",
    "parse_opening_row",
]
