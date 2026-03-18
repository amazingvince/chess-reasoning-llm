"""Source 9: MATE Dataset.

Load the OutFlankShu MATE dataset from zip archives on HuggingFace.
The dataset is stored as instruction/input/output JSONL inside zip files,
not as a standard HF datasets table.

Subsets available: no_explain, strategy, tactic, both.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from typing import Iterator

import chess
from huggingface_hub import hf_hub_download

from config.settings import HF_DATASETS

logger = logging.getLogger(__name__)

# Zip files available in the MATE repo
_ZIP_FILES = ["no_explain.zip", "both.zip"]


def load_mate(
    max_rows: int | None = None,
) -> Iterator[dict]:
    """Yield MATE dataset rows with UCI moves.

    Downloads zip archives from HuggingFace, extracts JSONL files,
    and parses the instruction/input/output format.

    Yields
    ------
    dict
        ``{fen, move_a, move_b, better_move}``
    """
    count = 0
    for zip_name in _ZIP_FILES:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASETS["mate"],
                filename=zip_name,
                repo_type="dataset",
            )
        except Exception as exc:
            logger.warning("Could not download %s: %s", zip_name, exc)
            continue

        try:
            with zipfile.ZipFile(path) as zf:
                jsonl_files = [
                    n for n in zf.namelist()
                    if n.endswith(".jsonl") and not n.startswith("__MACOSX")
                ]
                for jf in sorted(jsonl_files):
                    with zf.open(jf) as fh:
                        for raw_line in fh:
                            if max_rows is not None and count >= max_rows:
                                return
                            try:
                                row = json.loads(raw_line)
                            except json.JSONDecodeError:
                                continue
                            result = _process_row(row)
                            if result is not None:
                                yield result
                                count += 1
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning("Failed to read %s: %s", zip_name, exc)
            continue

    logger.info("MATE dataset: yielded %d rows from %d zip(s)", count, len(_ZIP_FILES))


# Regex to extract FEN, MoveA, MoveB from the input field
_INPUT_RE = re.compile(
    r'FEN[^"]*"([^"]+)".*?'
    r'MoveA[:\s]*([a-h][1-8][a-h][1-8][qrbn]?).*?'
    r'MoveB[:\s]*([a-h][1-8][a-h][1-8][qrbn]?)',
    re.IGNORECASE,
)

# Regex to extract the chosen move from output field
_OUTPUT_RE = re.compile(
    r'Move([AB])[:\s]*([a-h][1-8][a-h][1-8][qrbn]?)',
    re.IGNORECASE,
)


def _process_row(row: dict) -> dict | None:
    """Process a single MATE instruction/input/output row."""
    input_text = row.get("input", "")
    output_text = row.get("output", "")

    m = _INPUT_RE.search(input_text)
    if not m:
        return None

    fen, move_a, move_b = m.group(1), m.group(2), m.group(3)

    # Validate FEN and moves
    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return None

    move_a_obj = _validate_uci(board, move_a)
    move_b_obj = _validate_uci(board, move_b)
    if move_a_obj is None or move_b_obj is None:
        return None

    # Determine better move from output
    om = _OUTPUT_RE.search(output_text)
    if not om:
        return None

    choice_label = om.group(1).upper()
    better_move = move_a if choice_label == "A" else move_b

    return {
        "fen": fen,
        "move_a": move_a,
        "move_b": move_b,
        "better_move": better_move,
    }


def _validate_uci(board: chess.Board, uci: str) -> chess.Move | None:
    """Validate a UCI move is legal. Returns Move or None."""
    try:
        m = chess.Move.from_uci(uci)
        return m if m in board.legal_moves else None
    except (ValueError, chess.InvalidMoveError):
        return None
