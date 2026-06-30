"""Helpers for parsing MATE dataset rows."""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from collections.abc import Callable, Iterator

import chess

from chess_llm.sft.settings import DEFAULT_HF_CACHE_DIR, DEFAULT_HF_DATASETS

logger = logging.getLogger(__name__)

DEFAULT_MATE_ZIP_FILES = ("no_explain.zip", "both.zip")
HubDownloader = Callable[..., str]


_INPUT_RE = re.compile(
    r'FEN[^"]*"([^"]+)".*?'
    r"MoveA[:\s]*([a-h][1-8][a-h][1-8][qrbn]?).*?"
    r"MoveB[:\s]*([a-h][1-8][a-h][1-8][qrbn]?)",
    re.IGNORECASE,
)

_OUTPUT_RE = re.compile(
    r"Move([AB])[:\s]*([a-h][1-8][a-h][1-8][qrbn]?)",
    re.IGNORECASE,
)


def process_mate_row(row: dict) -> dict | None:
    """Parse one MATE instruction row into a validated move-choice dict."""
    input_text = row.get("input", "")
    output_text = row.get("output", "")

    match = _INPUT_RE.search(input_text)
    if not match:
        return None

    fen, move_a, move_b = match.group(1), match.group(2), match.group(3)

    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return None
    if not board.is_valid():
        return None

    move_a_obj = validate_uci(board, move_a)
    move_b_obj = validate_uci(board, move_b)
    if move_a_obj is None or move_b_obj is None:
        return None

    output_match = _OUTPUT_RE.search(output_text)
    if not output_match:
        return None

    choice_label = output_match.group(1).upper()
    output_move = output_match.group(2)
    selected_move = move_a if choice_label == "A" else move_b
    if output_move != selected_move:
        return None

    parsed = {
        "fen": fen,
        "move_a": move_a,
        "move_b": move_b,
        "better_move": selected_move,
    }
    for key in (
        "strategy",
        "tactic",
        "strategy_a",
        "strategy_b",
        "tactic_a",
        "tactic_b",
        "source_subset",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parsed[key] = value.strip()
    return parsed


def load_mate(
    max_rows: int | None = None,
    *,
    downloader: HubDownloader | None = None,
    dataset_name: str = DEFAULT_HF_DATASETS["mate"],
    zip_files: list[str] | tuple[str, ...] = DEFAULT_MATE_ZIP_FILES,
) -> Iterator[dict]:
    """Yield parsed rows from MATE ZIP archives on Hugging Face."""
    if max_rows is not None and max_rows <= 0:
        return

    download = downloader or _default_hub_download
    count = 0
    for zip_name in zip_files:
        try:
            path = download(dataset_name, zip_name, repo_type="dataset")
        except Exception as exc:
            logger.warning("Could not download %s: %s", zip_name, exc)
            continue

        try:
            with zipfile.ZipFile(path) as archive:
                for jsonl_name in _jsonl_names(archive):
                    with archive.open(jsonl_name) as fh:
                        for raw_line in fh:
                            if max_rows is not None and count >= max_rows:
                                return
                            try:
                                row = json.loads(raw_line)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                            if not isinstance(row, dict):
                                continue
                            parsed = process_mate_row(row)
                            if parsed is None:
                                continue
                            yield parsed
                            count += 1
        except (zipfile.BadZipFile, OSError) as exc:
            logger.warning("Failed to read %s: %s", zip_name, exc)
            continue

    logger.info("MATE dataset: yielded %d rows from %d zip(s)", count, len(zip_files))


def validate_uci(board: chess.Board, uci: str) -> chess.Move | None:
    """Return a move when ``uci`` is legal on ``board``; otherwise ``None``."""
    try:
        move = chess.Move.from_uci(uci)
    except (ValueError, chess.InvalidMoveError):
        return None
    return move if move in board.legal_moves else None


def _jsonl_names(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if name.endswith(".jsonl") and "__MACOSX" not in name.split("/")
    )


def _default_hub_download(repo_id: str, filename: str, *, repo_type: str) -> str:
    os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE_DIR)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


__all__ = [
    "DEFAULT_MATE_ZIP_FILES",
    "load_mate",
    "process_mate_row",
    "validate_uci",
]
