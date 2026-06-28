"""Shared fixtures for training tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the training package root is on sys.path so that absolute imports
# like ``from config.phases import ...`` work regardless of pytest CWD.
_TRAINING_ROOT = str(Path(__file__).resolve().parent.parent)
if _TRAINING_ROOT not in sys.path:
    sys.path.insert(0, _TRAINING_ROOT)

import json
from pathlib import Path

import pytest


SYSTEM_PROMPT = (
    "You are a chess reasoning engine. You understand chess positions "
    "in FEN notation and express all moves in UCI notation (e.g., e2e4, "
    "g1f3, a7a8q for promotion). When analyzing positions, think step by step."
)

SAMPLE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def make_example(task: str = "1.1_fen_to_board", tier: int = 1, fen: str = SAMPLE_FEN) -> dict:
    """Create a single training example in pipeline output format."""
    return {
        "task": task,
        "tier": tier,
        "fen": fen,
        "is_chess960": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Position (FEN): {fen}\nShow me the board."},
            {"role": "assistant", "content": "Here is the board."},
        ],
        "metadata": {},
    }


def _synthetic_fen(tier: int, idx: int) -> str:
    """Generate a unique but plausible-looking FEN string for testing.

    The mixer canonicalizes FENs by ignoring move counters, so uniqueness
    must live in the first four FEN fields.
    """
    serial = (tier - 1) * 1000 + idx
    grid = [["" for _ in range(8)] for _ in range(8)]
    grid[0][7] = "k"
    grid[7][0] = "K"

    available = [
        (rank, file)
        for rank in range(8)
        for file in range(8)
        if (rank, file) not in {(0, 7), (7, 0)}
    ]
    knight_square = available[serial % len(available)]
    bishop_choices = [square for square in available if square != knight_square]
    bishop_square = bishop_choices[(serial // len(available)) % len(bishop_choices)]
    grid[knight_square[0]][knight_square[1]] = "N"
    grid[bishop_square[0]][bishop_square[1]] = "b"

    rows: list[str] = []
    for rank in grid:
        row = ""
        empties = 0
        for piece in rank:
            if piece:
                if empties:
                    row += str(empties)
                    empties = 0
                row += piece
            else:
                empties += 1
        if empties:
            row += str(empties)
        rows.append(row)

    side = "w" if (serial // (len(available) * len(bishop_choices))) % 2 == 0 else "b"
    return f"{'/'.join(rows)} {side} - - 0 1"


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """Create a temporary data root with synthetic tier data.

    Each example gets a unique FEN so that FEN-based eval splitting
    produces a clean partition.
    """
    for tier in range(1, 8):
        tier_dir = tmp_path / f"tier{tier}"
        tier_dir.mkdir()

        counts = {1: 100, 2: 80, 3: 60, 4: 40, 5: 10, 6: 30, 7: 50}
        count = counts.get(tier, 20)

        task_map = {
            1: "1.1_fen_to_board",
            2: "2.1_legal_move_gen",
            3: "3.1_available_captures",
            4: "4.1_material_balance",
            5: "5.1_opening_identification",
            6: "6.1_endgame_classification",
            7: "7.1_best_move_selection",
        }

        jsonl_path = tier_dir / f"{task_map[tier]}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for i in range(count):
                fen = _synthetic_fen(tier, i)
                example = make_example(task=task_map[tier], tier=tier, fen=fen)
                fh.write(json.dumps(example) + "\n")

    return tmp_path
