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


def test_blocklist_uses_position_key_not_move_counters():
    """Same board state with different clocks must still be blocked."""
    fen_eval = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    fen_train_same_position = "8/8/8/8/8/8/4K3/4k3 w - - 17 42"
    splits = {"perception": [{"fen": fen_eval}]}

    blocklist = eval_split.build_blocklist(splits)

    assert eval_split.canonical_fen_key(fen_train_same_position) in blocklist


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


# ── partition_eco_codes ──────────────────────────────────────────────


def test_partition_eco_no_overlap():
    """Eval and train openings share no ECO codes."""
    openings = [
        {"fen": f"pos_{i}", "eco": f"B{i:02d}"} for i in range(20)
    ]
    from pool.eval_split import partition_eco_codes

    eval_o, train_o = partition_eco_codes(openings, holdout_fraction=0.25, seed=42)
    eval_ecos = {r["eco"] for r in eval_o}
    train_ecos = {r["eco"] for r in train_o}
    assert eval_ecos & train_ecos == set()


def test_partition_eco_deterministic():
    """Same seed produces same partition."""
    openings = [
        {"fen": f"pos_{i}", "eco": f"C{i:02d}"} for i in range(30)
    ]
    from pool.eval_split import partition_eco_codes

    e1, t1 = partition_eco_codes(openings, seed=99)
    e2, t2 = partition_eco_codes(openings, seed=99)
    assert [r["eco"] for r in e1] == [r["eco"] for r in e2]


def test_partition_eco_all_assigned():
    """Every opening goes to either eval or train."""
    openings = [
        {"fen": f"pos_{i}", "eco": f"A{i:02d}"} for i in range(15)
    ]
    from pool.eval_split import partition_eco_codes

    eval_o, train_o = partition_eco_codes(openings, seed=42)
    assert len(eval_o) + len(train_o) == len(openings)


def test_partition_eco_fraction_respected():
    """Holdout fraction roughly respected (within 1 ECO of target)."""
    openings = [
        {"fen": f"pos_{i}", "eco": f"D{i:02d}"} for i in range(100)
    ]
    from pool.eval_split import partition_eco_codes

    eval_o, train_o = partition_eco_codes(
        openings, holdout_fraction=0.15, seed=42,
    )
    # 100 unique ECOs * 0.15 = 15 eval ECOs
    eval_ecos = {r["eco"] for r in eval_o}
    assert 14 <= len(eval_ecos) <= 16
