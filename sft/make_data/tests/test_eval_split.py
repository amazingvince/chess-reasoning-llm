"""Tests for pool/eval_split.py — split exclusivity, sizing, blocklist (~7 cases)."""

import pytest

from pool import eval_split


def test_mutual_exclusivity(monkeypatch):
    """No FEN appears in two splits — critical fix."""
    monkeypatch.setattr(eval_split, "EVAL_SPLIT_SIZES", {
        "perception": 5,
        "rules": 5,
    })
    # Same FEN pool offered to both splits
    sources = {
        "perception": [{"fen": f"pos_{i}"} for i in range(20)],
        "rules": [{"fen": f"pos_{i}"} for i in range(20)],
    }
    splits = eval_split.generate_all_eval_splits(sources, seed=42)

    perc_fens = {ex["fen"] for ex in splits["perception"]}
    rules_fens = {ex["fen"] for ex in splits["rules"]}
    overlap = perc_fens & rules_fens
    assert overlap == set(), f"Cross-split contamination: {overlap}"


def test_respects_target_size(monkeypatch):
    monkeypatch.setattr(eval_split, "EVAL_SPLIT_SIZES", {"perception": 3})
    sources = {"perception": [{"fen": f"pos_{i}"} for i in range(100)]}
    splits = eval_split.generate_all_eval_splits(sources, seed=42)
    assert len(splits["perception"]) == 3


def test_target_capped_by_available(monkeypatch):
    monkeypatch.setattr(eval_split, "EVAL_SPLIT_SIZES", {"perception": 100})
    sources = {"perception": [{"fen": f"pos_{i}"} for i in range(5)]}
    splits = eval_split.generate_all_eval_splits(sources, seed=42)
    assert len(splits["perception"]) == 5


def test_reproducible_with_same_seed(monkeypatch):
    monkeypatch.setattr(eval_split, "EVAL_SPLIT_SIZES", {"perception": 5})
    sources = {"perception": [{"fen": f"pos_{i}"} for i in range(50)]}
    s1 = eval_split.generate_all_eval_splits(sources, seed=123)
    s2 = eval_split.generate_all_eval_splits(sources, seed=123)
    assert [e["fen"] for e in s1["perception"]] == [e["fen"] for e in s2["perception"]]


def test_build_blocklist_excludes_empty():
    splits = {
        "a": [{"fen": "f1"}, {"fen": "f2"}, {"fen": ""}],
        "b": [{"fen": "f3"}],
    }
    blocklist = eval_split.build_blocklist(splits)
    assert "f1" in blocklist
    assert "f2" in blocklist
    assert "f3" in blocklist
    assert "" not in blocklist


def test_save_load_blocklist(tmp_path):
    splits = {"test_split": [{"fen": "fen_a"}, {"fen": "fen_b"}]}
    eval_split.save_eval_splits(splits, str(tmp_path / "splits"))
    blocklist = eval_split.load_blocklist(str(tmp_path / "splits" / "blocklist.txt"))
    assert "fen_a" in blocklist
    assert "fen_b" in blocklist


def test_empty_source(monkeypatch):
    monkeypatch.setattr(eval_split, "EVAL_SPLIT_SIZES", {"perception": 5})
    sources = {"perception": []}
    splits = eval_split.generate_all_eval_splits(sources, seed=42)
    assert splits["perception"] == []


def test_intra_split_dedup(monkeypatch):
    """Duplicate FENs within one split's candidate list are collapsed.

    The planning split concatenates puzzles + evals which may share
    FENs.  Without intra-split dedup, the same FEN can appear twice
    in the benchmark file.
    """
    monkeypatch.setattr(eval_split, "EVAL_SPLIT_SIZES", {"planning": 10})
    # 5 unique FENs, each appearing twice (simulates puzzles + evals overlap)
    candidates = [{"fen": f"pos_{i}"} for i in range(5)] * 2
    sources = {"planning": candidates}
    splits = eval_split.generate_all_eval_splits(sources, seed=42)

    fens = [ex["fen"] for ex in splits["planning"]]
    assert len(fens) == len(set(fens)), f"Intra-split duplicates: {fens}"
    assert len(fens) == 5  # only 5 unique FENs available
