"""Tests for sources/lichess_evals.py — streaming dedup and partitioning."""

import pytest
import chess

from conftest import KRK_FEN, STARTING_FEN
import sources.lichess_evals as lichess_evals
from sources.lichess_evals import (
    _flush_batch,
    _init_dedup_db,
    _normalize_pv_line,
    partition_evals,
)

# ── partition_evals ──────────────────────────────────────────────────


def test_normalize_pv_line_converts_castling_king_to_rook_notation():
    fen = "r3k2r/8/8/8/8/8/5P2/R3K2R w KQkq - 0 1"

    normalized = _normalize_pv_line(fen, "e1h1 e8h8")

    assert normalized == "e1g1 e8g8"


def test_partition_mate():
    rows = [{"fen": STARTING_FEN, "cp": None, "mate": 5, "depth": 35}]
    parts = partition_evals(iter(rows))
    assert len(parts["mate"]) == 1
    assert len(parts["balanced"]) == 0


def test_partition_high_eval():
    rows = [{"fen": STARTING_FEN, "cp": 300, "mate": None, "depth": 20}]
    parts = partition_evals(iter(rows))
    assert len(parts["high_eval"]) == 1
    assert len(parts["balanced"]) == 0


def test_partition_balanced_and_best_move():
    rows = [{"fen": STARTING_FEN, "cp": 30, "mate": None, "depth": 35}]
    parts = partition_evals(iter(rows))
    assert len(parts["balanced"]) == 1
    assert len(parts["best_move"]) == 1   # depth >= 30


def test_partition_endgame():
    # KRK has 3 pieces (<=10)
    rows = [{"fen": KRK_FEN, "cp": 500, "mate": None, "depth": 20}]
    parts = partition_evals(iter(rows))
    assert len(parts["endgame"]) == 1
    assert len(parts["high_eval"]) == 1   # also high eval


def test_partition_evals_uses_row_board_helper_for_chess960(monkeypatch):
    fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    calls = []

    def fake_board_from_raw(row):
        calls.append(row)
        return chess.Board(row["fen"], chess960=True)

    monkeypatch.setattr(lichess_evals, "board_from_raw", fake_board_from_raw)

    rows = [{
        "fen": fen,
        "cp": 30,
        "mate": None,
        "depth": 35,
        "metadata": {"chess960_id": 321},
    }]

    parts = partition_evals(iter(rows))

    assert calls == rows
    assert len(parts["best_move"]) == 1


# ── _flush_batch + DB dedup ──────────────────────────────────────────


def test_flush_batch_keeps_highest_depth(tmp_path):
    db_path = tmp_path / "dedup.db"
    conn = _init_dedup_db(db_path)

    batch1 = [(STARTING_FEN, "e2e4", "e2e4 e7e5", 20, 1000, 30, None)]
    _flush_batch(conn, batch1)

    batch2 = [(STARTING_FEN, "d2d4", "d2d4 d7d5", 25, 2000, 35, None)]
    _flush_batch(conn, batch2)

    row = conn.execute(
        "SELECT depth, best_move FROM evals WHERE fen = ?", (STARTING_FEN,)
    ).fetchone()
    assert row[0] == 25
    assert row[1] == "d2d4"
    conn.close()


def test_flush_batch_same_depth_higher_knodes_replaces(tmp_path):
    """Same depth, higher knodes — _flush_batch replaces the stored row."""
    db_path = tmp_path / "dedup.db"
    conn = _init_dedup_db(db_path)

    batch1 = [(STARTING_FEN, "e2e4", "e2e4 e7e5", 20, 100, 30, None)]
    _flush_batch(conn, batch1)

    batch2 = [(STARTING_FEN, "d2d4", "d2d4 d7d5", 20, 200, 35, None)]
    _flush_batch(conn, batch2)

    row = conn.execute(
        "SELECT depth, knodes, best_move FROM evals WHERE fen = ?", (STARTING_FEN,)
    ).fetchone()
    assert row[0] == 20
    assert row[1] == 200
    assert row[2] == "d2d4"
    conn.close()


# ── stream_evals dedup ───────────────────────────────────────────────


def test_stream_dedup_cross_run_same_depth_prefers_higher_knodes(monkeypatch, tmp_path):
    """Cross-run: DB has depth 20 / knodes 100, stream has depth 20 / knodes 200.

    The streamed row must replace the DB row — not be silently skipped.
    """
    from sources import lichess_evals

    # Pre-populate DB with the lower-knodes row
    db_path = tmp_path / "dedup.db"
    conn = _init_dedup_db(db_path)
    _flush_batch(conn, [(STARTING_FEN, "e2e4", "e2e4 e7e5", 20, 100, 30, None)])
    conn.close()

    # Stream a higher-knodes row at the same depth
    rows = [
        {
            "fen": STARTING_FEN, "depth": 20,
            "line": "d2d4 d7d5", "cp": 35, "mate": None, "knodes": 200,
        },
    ]
    results = list(
        lichess_evals.stream_evals(
            min_depth=20,
            dedup_db_path=db_path,
            dataset_loader=lambda *a, **kw: iter(rows),
        )
    )
    assert len(results) == 1
    assert results[0]["knodes"] == 200
    assert results[0]["best_move"] == "d2d4"

    # DB should also be updated
    conn2 = _init_dedup_db(db_path)
    row = conn2.execute(
        "SELECT knodes, best_move FROM evals WHERE fen = ?", (STARTING_FEN,)
    ).fetchone()
    assert row[0] == 200
    assert row[1] == "d2d4"
    conn2.close()


def test_stream_dedup_skips_lower_depth(monkeypatch, tmp_path):
    """Higher depth first, then lower depth for same FEN — only 1 yielded."""
    from sources import lichess_evals

    rows = [
        {
            "fen": STARTING_FEN, "depth": 25,
            "line": "d2d4 d7d5", "cp": 35, "mate": None, "knodes": 2000,
        },
        {
            "fen": STARTING_FEN, "depth": 20,
            "line": "e2e4 e7e5", "cp": 30, "mate": None, "knodes": 1000,
        },
    ]

    db_path = tmp_path / "dedup.db"
    results = list(
        lichess_evals.stream_evals(
            min_depth=20,
            dedup_db_path=db_path,
            dataset_loader=lambda *a, **kw: iter(rows),
        )
    )
    assert len(results) == 1
    assert results[0]["depth"] == 25


def test_stream_dedup_lower_then_higher_yields_only_higher(monkeypatch, tmp_path):
    """Low depth first, then higher depth for same FEN — only higher yielded.

    This was the original bug: the old code yielded both rows because it
    emitted each row as soon as it passed the in-memory dedup check.
    """
    from sources import lichess_evals

    rows = [
        {
            "fen": STARTING_FEN, "depth": 20,
            "line": "e2e4 e7e5", "cp": 30, "mate": None, "knodes": 1000,
        },
        {
            "fen": STARTING_FEN, "depth": 25,
            "line": "d2d4 d7d5", "cp": 35, "mate": None, "knodes": 2000,
        },
    ]

    db_path = tmp_path / "dedup.db"
    results = list(
        lichess_evals.stream_evals(
            min_depth=20,
            dedup_db_path=db_path,
            dataset_loader=lambda *a, **kw: iter(rows),
        )
    )
    assert len(results) == 1
    assert results[0]["depth"] == 25
    assert results[0]["best_move"] == "d2d4"


def test_stream_dedup_same_depth_prefers_higher_knodes(monkeypatch, tmp_path):
    """Same FEN at same depth — knodes tie-breaks, higher knodes wins."""
    from sources import lichess_evals

    rows = [
        {
            "fen": STARTING_FEN, "depth": 20,
            "line": "e2e4 e7e5", "cp": 30, "mate": None, "knodes": 100,
        },
        {
            "fen": STARTING_FEN, "depth": 20,
            "line": "d2d4 d7d5", "cp": 35, "mate": None, "knodes": 200,
        },
    ]

    db_path = tmp_path / "dedup.db"
    results = list(
        lichess_evals.stream_evals(
            min_depth=20,
            dedup_db_path=db_path,
            dataset_loader=lambda *a, **kw: iter(rows),
        )
    )
    assert len(results) == 1
    assert results[0]["knodes"] == 200
    assert results[0]["best_move"] == "d2d4"


def test_stream_dedup_distinct_fens_both_yielded(monkeypatch, tmp_path):
    """Two different FENs — both yielded."""
    from sources import lichess_evals

    rows = [
        {
            "fen": STARTING_FEN, "depth": 20,
            "line": "e2e4 e7e5", "cp": 30, "mate": None, "knodes": 1000,
        },
        {
            "fen": KRK_FEN, "depth": 20,
            "line": "e1f2", "cp": 500, "mate": None, "knodes": 500,
        },
    ]

    db_path = tmp_path / "dedup.db"
    results = list(
        lichess_evals.stream_evals(
            min_depth=20,
            dedup_db_path=db_path,
            dataset_loader=lambda *a, **kw: iter(rows),
        )
    )
    assert len(results) == 2
    fens = {r["fen"] for r in results}
    assert fens == {STARTING_FEN, KRK_FEN}
