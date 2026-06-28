"""Tests for data/loader.py — tier JSONL loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.loader import load_tier_data


def test_load_tier_returns_correct_columns(tmp_data_root: Path):
    """Dataset should have messages, task, tier, and fen columns."""
    ds = load_tier_data(1, tmp_data_root)
    assert "messages" in ds.column_names
    assert "task" in ds.column_names
    assert "tier" in ds.column_names
    # fen is retained for the mixer's FEN-based eval split
    assert "fen" in ds.column_names


def test_load_tier_correct_count(tmp_data_root: Path):
    """Should load exactly the number of examples in the JSONL."""
    ds = load_tier_data(1, tmp_data_root)
    assert len(ds) == 100

    ds2 = load_tier_data(5, tmp_data_root)
    assert len(ds2) == 10


def test_load_tier_messages_format(tmp_data_root: Path):
    """Each messages entry should be a list of 3 role dicts."""
    ds = load_tier_data(1, tmp_data_root)
    messages = ds[0]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_load_tier_drops_pipeline_only_columns(tmp_data_root: Path):
    """Columns like is_chess960 and metadata should be dropped."""
    ds = load_tier_data(1, tmp_data_root)
    assert "is_chess960" not in ds.column_names
    assert "metadata" not in ds.column_names


def test_load_tier_missing_raises(tmp_data_root: Path):
    """Loading a tier with no data should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No JSONL files found"):
        load_tier_data(8, tmp_data_root)


def _make_example(task: str = "1.1_fen_to_board", tier: int = 1, idx: int = 0) -> dict:
    """Inline helper (avoids conftest import issues)."""
    fen = f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {idx}"
    return {
        "task": task, "tier": tier, "fen": fen, "is_chess960": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {fen}"},
            {"role": "assistant", "content": "answer"},
        ],
        "metadata": {},
    }


def test_load_tier_multiple_jsonl_files(tmp_data_root: Path):
    """Should load from multiple JSONL files in one tier directory."""
    tier_dir = tmp_data_root / "tier1"

    # Add a second JSONL file
    second_file = tier_dir / "1.2_board_to_fen.jsonl"
    with open(second_file, "w", encoding="utf-8") as fh:
        for i in range(50):
            example = _make_example(task="1.2_board_to_fen", tier=1, idx=10000 + i)
            fh.write(json.dumps(example) + "\n")

    ds = load_tier_data(1, tmp_data_root)
    assert len(ds) == 150  # 100 + 50


def test_load_tier_multiple_jsonl_files_with_incompatible_metadata(tmp_path: Path):
    """Should tolerate per-file metadata schema differences once dropped."""
    tier_dir = tmp_path / "tier6"
    tier_dir.mkdir(parents=True)

    file_a = tier_dir / "6.1_endgame_classification.jsonl"
    with open(file_a, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            **_make_example(task="6.1_endgame_classification", tier=6, idx=1),
            "metadata": {"material": "KQ vs K", "source": "generator_a"},
        }) + "\n")

    file_b = tier_dir / "6.2_endgame_wdl.jsonl"
    with open(file_b, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            **_make_example(task="6.2_endgame_wdl", tier=6, idx=2),
            "metadata": {"wdl": 1, "dtz": 7, "source": "generator_b"},
        }) + "\n")

    ds = load_tier_data(6, tmp_path)
    assert len(ds) == 2
    assert set(ds.column_names) == {"messages", "task", "tier", "fen"}


def test_load_tier_single_jsonl_with_incompatible_metadata_rows(tmp_path: Path):
    """Should tolerate metadata schema differences within one JSONL file."""
    tier_dir = tmp_path / "tier7"
    tier_dir.mkdir(parents=True)

    path = tier_dir / "7.1_best_move_selection.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            **_make_example(task="7.1_best_move_selection", tier=7, idx=1),
            "metadata": {"source": "sf", "depth": 18, "stockfish_eval_cp": 42, "stockfish_eval_mate": 0},
        }) + "\n")
        fh.write(json.dumps({
            **_make_example(task="7.1_best_move_selection", tier=7, idx=2),
            "metadata": {"source": "strategy", "strategy": "fork", "tactic": "pin"},
        }) + "\n")

    ds = load_tier_data(7, tmp_path)
    assert len(ds) == 2
    assert set(ds.column_names) == {"messages", "task", "tier", "fen"}


def test_load_all_tiers(tmp_data_root: Path):
    """Smoke test: load every tier."""
    for tier in range(1, 8):
        ds = load_tier_data(tier, tmp_data_root)
        assert len(ds) > 0
        assert "messages" in ds.column_names
