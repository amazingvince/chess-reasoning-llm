"""Source 3: Lichess Chess Openings.

3,630 named openings with ECO codes, PGN, UCI move sequences, and EPD.
"""

from __future__ import annotations

import logging
from typing import Iterator

import chess
from datasets import load_dataset

from config.settings import HF_DATASETS

logger = logging.getLogger(__name__)


def load_openings(max_openings: int | None = None) -> Iterator[dict]:
    """Yield opening dicts with replayed final FEN.

    Each dict::

        {eco_volume, eco, name, pgn, uci_moves, epd, fen}

    The ``fen`` field is computed by replaying the UCI moves with
    python-chess, ensuring it's a valid position.
    """
    ds = load_dataset(HF_DATASETS["lichess_openings"], split="train")
    count = 0
    for row in ds:
        if max_openings is not None and count >= max_openings:
            return

        eco_volume = row.get("eco-volume", row.get("eco_volume", ""))
        eco = row.get("eco", "")
        name = row.get("name", "")
        pgn = row.get("pgn", "")
        uci_str = row.get("uci", "")
        epd = row.get("epd", "")

        if not uci_str:
            continue

        # Replay UCI moves to get final FEN
        board = chess.Board()
        uci_moves = uci_str.split()
        valid = True
        for uci in uci_moves:
            try:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    valid = False
                    break
                board.push(move)
            except (ValueError, chess.InvalidMoveError):
                valid = False
                break

        if not valid:
            logger.warning("Invalid move sequence for opening %s: %s", name, uci_str)
            continue

        yield {
            "eco_volume": eco_volume,
            "eco": eco,
            "name": name,
            "pgn": pgn,
            "uci_moves": uci_moves,
            "epd": epd,
            "fen": board.fen(),
        }
        count += 1
