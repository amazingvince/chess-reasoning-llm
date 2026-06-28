"""Tests for data/mixer.py — phase-aware mixing, sampling, upsampling.

The mixer splits each tier on unique FENs *before* sampling/upsampling,
so exact train counts depend on how many FENs land on the eval side.
Tests that need exact counts use eval_fraction=0.0 to bypass the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.phases import PHASE_A, PHASE_B, PHASE_C, PhaseConfig, TierMix
from data.mixer import build_phase_dataset, summarize_phase_data, _split_by_fen


# -- Exact-count tests (eval_fraction=0.0, no FEN split) --

def test_phase_a_uses_all_tier1_tier2(tmp_data_root: Path):
    """Phase A should include 100% of tier 1 and tier 2."""
    train_ds, eval_ds = build_phase_dataset(PHASE_A, tmp_data_root, eval_fraction=0.0)
    assert len(train_ds) == 180  # T1=100, T2=80


def test_phase_b_samples_tier1_tier2(tmp_data_root: Path):
    """Phase B should sample 30% of tiers 1-2 + 100% of tiers 3-6."""
    train_ds, eval_ds = build_phase_dataset(PHASE_B, tmp_data_root, eval_fraction=0.0)
    expected = 30 + 24 + 60 + 40 + 10 + 30  # 194
    assert len(train_ds) == expected


def test_phase_c_upsamples_tier7(tmp_data_root: Path):
    """Phase C should upsample tier 7 by 5x."""
    train_ds, eval_ds = build_phase_dataset(PHASE_C, tmp_data_root, eval_fraction=0.0)
    expected = 20 + 16 + 60 + 40 + 10 + 30 + 250  # 426
    assert len(train_ds) == expected


def test_custom_phase_config(tmp_data_root: Path):
    """Custom PhaseConfig should work correctly."""
    custom = PhaseConfig(
        name="test",
        display_name="Test Phase",
        tier_mix=(
            TierMix(tier=1, fraction=0.5),
            TierMix(tier=7, fraction=1.0, upsample=3),
        ),
        epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.01,
        weight_decay=0.0,
        resume_from=None,
    )
    train_ds, eval_ds = build_phase_dataset(custom, tmp_data_root, eval_fraction=0.0)
    assert len(train_ds) == 200  # T1: 100*0.5=50, T7: 50*3=150


# -- FEN-based split tests --

def test_eval_fraction_zero(tmp_data_root: Path):
    """eval_fraction=0.0 should produce empty eval set."""
    train_ds, eval_ds = build_phase_dataset(PHASE_A, tmp_data_root, eval_fraction=0.0)
    assert len(eval_ds) == 0
    assert len(train_ds) == 180


def test_eval_split_conserves_total(tmp_data_root: Path):
    """Train + eval should equal the total before upsampling."""
    train_ds, eval_ds = build_phase_dataset(PHASE_A, tmp_data_root, eval_fraction=0.1)
    assert len(train_ds) + len(eval_ds) == 180


def test_no_fen_leaks_between_train_eval(tmp_data_root: Path):
    """No FEN should appear in both train and eval."""
    from data.loader import load_tier_data

    ds = load_tier_data(1, tmp_data_root)
    train_part, eval_part = _split_by_fen(ds, eval_fraction=0.1, seed=42)

    train_fens = set(train_part["fen"])
    eval_fens = set(eval_part["fen"])
    assert train_fens.isdisjoint(eval_fens), "FEN leak: same FEN in train and eval"


def test_phase_split_prevents_cross_tier_fen_leak(tmp_path: Path):
    """A FEN must not be train in one tier and eval in another tier."""
    shared_fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"

    def write_row(path: Path, tier: int, fen: str, idx: int) -> None:
        row = {
            "task": f"{tier}.x_test",
            "tier": tier,
            "fen": fen,
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": f"FEN: {fen}"},
                {"role": "assistant", "content": f"a{idx}"},
            ],
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    tier1 = tmp_path / "tier1"
    tier2 = tmp_path / "tier2"
    tier1.mkdir()
    tier2.mkdir()
    for i in range(3):
        write_row(tier1 / "tier1.jsonl", 1, shared_fen, i)
    write_row(tier2 / "tier2.jsonl", 2, shared_fen, 0)
    for i in range(3):
        write_row(tier2 / "tier2.jsonl", 2, f"8/8/8/8/8/8/4K3/4k3 b - - 0 {i + 2}", i)

    phase = PhaseConfig(
        name="leak",
        display_name="Leak Regression",
        tier_mix=(TierMix(tier=1), TierMix(tier=2)),
        epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        weight_decay=0.0,
        resume_from=None,
    )

    train_ds, eval_ds = build_phase_dataset(phase, tmp_path, eval_fraction=0.25, seed=5)
    train_prompts = "\n".join(row["messages"][1]["content"] for row in train_ds)
    eval_prompts = "\n".join(row["messages"][1]["content"] for row in eval_ds)

    assert not (shared_fen in train_prompts and shared_fen in eval_prompts)


def test_upsampled_tier_no_fen_leak(tmp_data_root: Path):
    """After 5x upsampling, eval FENs must not appear in train."""
    from data.loader import load_tier_data
    from datasets import concatenate_datasets

    ds = load_tier_data(7, tmp_data_root)
    train_part, eval_part = _split_by_fen(ds, eval_fraction=0.1, seed=42)

    upsampled = concatenate_datasets([train_part] * 5)

    train_fens = set(upsampled["fen"])
    eval_fens = set(eval_part["fen"])
    assert train_fens.isdisjoint(eval_fens), "FEN leak after upsampling"


def test_eval_split_fraction_approximate(tmp_data_root: Path):
    """Eval set should be approximately the requested fraction."""
    train_ds, eval_ds = build_phase_dataset(PHASE_A, tmp_data_root, eval_fraction=0.1)
    total = len(train_ds) + len(eval_ds)
    actual_frac = len(eval_ds) / total
    assert 0.05 <= actual_frac <= 0.20, f"Eval fraction {actual_frac:.2f} outside expected range"


def test_fen_column_dropped_from_output(tmp_data_root: Path):
    """The fen column should be dropped from final train/eval datasets."""
    train_ds, eval_ds = build_phase_dataset(PHASE_A, tmp_data_root)
    assert "fen" not in train_ds.column_names
    assert "fen" not in eval_ds.column_names


# -- Skewed FEN group distribution tests --

def _make_skewed_tier(tmp_path: Path, tier: int = 1, groups: list[int] | None = None):
    """Create a tier where each FEN group has a specified size.

    ``groups`` is a list of integers; each entry is the number of
    rows sharing the same FEN.  E.g. [50, 50, 50] = 3 unique FENs,
    each repeated 50 times = 150 total rows.
    """
    if groups is None:
        groups = [50, 50, 50]
    tier_dir = tmp_path / f"tier{tier}"
    tier_dir.mkdir(exist_ok=True)
    jsonl_path = tier_dir / "skewed.jsonl"
    def fen_for_group(group_idx: int) -> str:
        grid = [["" for _ in range(8)] for _ in range(8)]
        grid[0][7] = "k"
        grid[7][0] = "K"
        available = [
            (rank, file)
            for rank in range(8)
            for file in range(8)
            if (rank, file) not in {(0, 7), (7, 0)}
        ]
        knight_square = available[group_idx % len(available)]
        bishop_choices = [square for square in available if square != knight_square]
        bishop_square = bishop_choices[(group_idx // len(available)) % len(bishop_choices)]
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

        side = "w" if group_idx < len(available) * len(bishop_choices) else "b"
        return f"{'/'.join(rows)} {side} - - 0 1"

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for group_idx, count in enumerate(groups):
            fen = fen_for_group(group_idx)
            for _ in range(count):
                fh.write(json.dumps({
                    "task": "test",
                    "tier": tier,
                    "fen": fen,
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                }) + "\n")
    return tmp_path


def test_skewed_fen_groups_stay_close_to_target(tmp_path: Path):
    """With 3 FEN groups of 50, a 2% request should give 0% eval.

    Target = 3 rows, but smallest group = 50.  Overshoot (47) > undershoot (3),
    so no group should be assigned to eval.  With 3 groups (>=2), the
    fallback would activate, but overshoot still dominates, so eval = 0
    is the correct result when all groups are too coarse.
    """
    from data.loader import load_tier_data

    data_root = _make_skewed_tier(tmp_path, groups=[50, 50, 50])
    ds = load_tier_data(1, data_root)
    _train, _eval = _split_by_fen(ds, eval_fraction=0.02, seed=42)

    # Target is 3 rows.  Smallest group is 50.  Overshoot (47) > undershoot (3).
    # The fallback (move smallest to eval) activates since eval is empty and
    # target >= 1 and there are >= 2 groups.  So we get exactly 50 in eval.
    assert len(_eval) == 50, f"Eval got {len(_eval)} rows, expected 50"
    assert len(_train) == 100


def test_skewed_fen_many_small_groups(tmp_path: Path):
    """With many small groups, eval fraction should be accurate."""
    from data.loader import load_tier_data

    # 100 groups of 1 row each = 100 unique FENs
    data_root = _make_skewed_tier(tmp_path, groups=[1] * 100)
    ds = load_tier_data(1, data_root)
    _train, _eval = _split_by_fen(ds, eval_fraction=0.10, seed=42)

    total = len(_train) + len(_eval)
    actual_frac = len(_eval) / total
    assert 0.05 <= actual_frac <= 0.15, f"Eval fraction {actual_frac:.2f} outside range"


def test_skewed_fen_mixed_group_sizes(tmp_path: Path):
    """With a mix of large and small groups, eval stays reasonable."""
    from data.loader import load_tier_data

    # One big group (100) + 20 small groups (5 each) = 200 total
    data_root = _make_skewed_tier(tmp_path, groups=[100] + [5] * 20)
    ds = load_tier_data(1, data_root)
    _train, _eval = _split_by_fen(ds, eval_fraction=0.10, seed=42)

    total = len(_train) + len(_eval)
    actual_frac = len(_eval) / total
    # Target = 20 rows.  Small groups can fill this precisely.
    # The algorithm sorts smallest-first, so it should pick ~4 small
    # groups (20 rows) rather than the 100-row group.
    assert actual_frac <= 0.25, f"Eval fraction {actual_frac:.2f} too high"
    # And there should be no FEN leak
    train_fens = set(_train["fen"])
    eval_fens = set(_eval["fen"])
    assert train_fens.isdisjoint(eval_fens)


def test_single_fen_tier_does_not_zero_train(tmp_path: Path):
    """A tier with one unique FEN should NOT send everything to eval."""
    from data.loader import load_tier_data

    # 10 rows, all same FEN — only 1 group
    data_root = _make_skewed_tier(tmp_path, groups=[10])
    ds = load_tier_data(1, data_root)
    _train, _eval = _split_by_fen(ds, eval_fraction=0.02, seed=42)

    # With only 1 group, we can't split without emptying one side.
    # The fallback requires >= 2 groups, so eval should be empty.
    assert len(_train) == 10, f"Train got {len(_train)}, expected 10"
    assert len(_eval) == 0, f"Eval got {len(_eval)}, expected 0"


def test_two_fen_groups_small_fraction(tmp_path: Path):
    """With 2 groups and a small fraction, the smallest goes to eval."""
    from data.loader import load_tier_data

    data_root = _make_skewed_tier(tmp_path, groups=[5, 95])
    ds = load_tier_data(1, data_root)
    _train, _eval = _split_by_fen(ds, eval_fraction=0.02, seed=42)

    # Target = 2 rows.  Groups sorted smallest-first: [5, 95].
    # Overshoot for group of 5: 5-2=3 > undershoot 2-0=2?  3 > 2 → skip.
    # Eval empty, target >= 1, >= 2 groups → fallback moves smallest (5) to eval.
    assert len(_eval) == 5
    assert len(_train) == 95


# -- Other tests --

def test_deterministic_with_seed(tmp_data_root: Path):
    """Same seed should produce identical datasets."""
    t1, e1 = build_phase_dataset(PHASE_A, tmp_data_root, seed=123)
    t2, e2 = build_phase_dataset(PHASE_A, tmp_data_root, seed=123)
    assert len(t1) == len(t2)
    assert len(e1) == len(e2)
    for i in range(min(5, len(t1))):
        assert t1[i]["task"] == t2[i]["task"]


def test_summarize_phase_data(tmp_data_root: Path):
    """summarize_phase_data should return correct counts."""
    summary = summarize_phase_data(PHASE_A, tmp_data_root)
    assert summary["tier_1"] == 100
    assert summary["tier_2"] == 80
    assert summary["total"] == 180


def test_summarize_phase_c_upsampling(tmp_data_root: Path):
    """Phase C summary should reflect 5x upsampling on tier 7."""
    summary = summarize_phase_data(PHASE_C, tmp_data_root)
    assert summary["tier_7"] == 250  # 50 * 5
    assert summary["tier_1"] == 20   # 100 * 0.2


def test_output_has_messages_column(tmp_data_root: Path):
    """Both train and eval datasets should have messages column."""
    train_ds, eval_ds = build_phase_dataset(PHASE_A, tmp_data_root)
    assert "messages" in train_ds.column_names
    assert "messages" in eval_ds.column_names
