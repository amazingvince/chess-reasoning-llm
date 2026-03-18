"""Source 9: MATE Dataset.

Load the OutFlankShu MATE dataset with SAN -> UCI conversion.
Subsets: MATE-N (no explanation), MATE-S (strategy), MATE-T (tactic),
MATE-ST (strategy + tactic).
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import chess
from datasets import load_dataset

from config.settings import HF_DATASETS

logger = logging.getLogger(__name__)


def load_mate(
    subset: str = "all",
    max_rows: int | None = None,
) -> Iterator[dict]:
    """Yield MATE dataset rows with moves converted to UCI.

    Parameters
    ----------
    subset : str
        One of "all", "N", "S", "T", "ST".
    max_rows : int, optional
        Limit number of rows.

    Yields
    ------
    dict
        ``{fen, move_a, move_b, better_move, strategy, tactic}``
    """
    count = 0

    # Try HuggingFace datasets first, fall back to raw download
    try:
        yield from _load_via_hf(subset, max_rows)
        return
    except Exception as exc:
        logger.warning("HF datasets loader failed (%s), trying raw JSON", exc)

    yield from _load_via_raw_json(subset, max_rows)


def _load_via_hf(
    subset: str, max_rows: int | None
) -> Iterator[dict]:
    """Load via HuggingFace datasets library."""
    ds = load_dataset(HF_DATASETS["mate"], split="train", streaming=True)
    count = 0
    for row in ds:
        if max_rows is not None and count >= max_rows:
            return

        result = _process_row(row, subset)
        if result is not None:
            yield result
            count += 1


def _load_via_raw_json(
    subset: str, max_rows: int | None
) -> Iterator[dict]:
    """Fallback: load from cached JSON files if HF parser fails."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=HF_DATASETS["mate"],
            filename="data/train.jsonl",
            repo_type="dataset",
        )
    except Exception:
        # Try alternative file patterns
        try:
            path = hf_hub_download(
                repo_id=HF_DATASETS["mate"],
                filename="train.jsonl",
                repo_type="dataset",
            )
        except Exception:
            logger.error("Could not download MATE dataset")
            return

    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if max_rows is not None and count >= max_rows:
                return
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            result = _process_row(row, subset)
            if result is not None:
                yield result
                count += 1


def _process_row(row: dict, subset: str) -> dict | None:
    """Process a single MATE row, converting SAN to UCI."""
    fen = row.get("fen", row.get("FEN", ""))
    if not fen:
        return None

    # Subset filter
    strategy = row.get("strategy", row.get("Strategy", ""))
    tactic = row.get("tactic", row.get("Tactic", ""))
    has_strategy = bool(strategy)
    has_tactic = bool(tactic)

    if subset == "N" and (has_strategy or has_tactic):
        return None
    elif subset == "S" and not has_strategy:
        return None
    elif subset == "T" and not has_tactic:
        return None
    elif subset == "ST" and not (has_strategy and has_tactic):
        return None

    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return None

    # Convert moves: try UCI first, fall back to SAN -> UCI
    move_a = _to_uci(board, row.get("move_a", row.get("Move_A", "")))
    move_b = _to_uci(board, row.get("move_b", row.get("Move_B", "")))
    better = row.get("better_move", row.get("Better_Move", row.get("label", "")))

    if not move_a or not move_b:
        return None

    # Normalize better_move to UCI
    if better in ("A", "a", "move_a", "Move_A"):
        better_move = move_a
    elif better in ("B", "b", "move_b", "Move_B"):
        better_move = move_b
    else:
        better_move = _to_uci(board, better) if better else ""

    return {
        "fen": fen,
        "move_a": move_a,
        "move_b": move_b,
        "better_move": better_move,
        "strategy": strategy or "",
        "tactic": tactic or "",
    }


def _to_uci(board: chess.Board, move_str: str) -> str:
    """Convert a move string (SAN or UCI) to UCI. Returns '' on failure."""
    if not move_str:
        return ""
    move_str = move_str.strip()

    # Try UCI first
    try:
        m = chess.Move.from_uci(move_str)
        if m in board.legal_moves:
            return move_str
    except (ValueError, chess.InvalidMoveError):
        pass

    # Try SAN
    try:
        m = board.parse_san(move_str)
        return m.uci()
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        pass

    return ""
