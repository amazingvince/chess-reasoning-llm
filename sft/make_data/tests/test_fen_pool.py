"""Tests for pool/fen_pool.py — dedup, ordering, sampling, persistence (~10 cases)."""

from random import Random

import pytest

from conftest import KRK_FEN, PROMOTION_FEN, STARTING_FEN
from pool.fen_pool import FENPool


def test_dedup_same_fen_twice():
    """Adding the same FEN twice keeps only the first entry."""
    pool = FENPool()
    pool.add(STARTING_FEN, source="lichess")
    pool.add(STARTING_FEN, source="stockfish")
    assert len(pool) == 1
    assert pool.get_tags(STARTING_FEN)["source"] == "lichess"


def test_all_fens_preserves_order():
    pool = FENPool()
    pool.add(STARTING_FEN)
    pool.add(KRK_FEN)
    pool.add(PROMOTION_FEN)
    assert pool.all_fens() == [STARTING_FEN, KRK_FEN, PROMOTION_FEN]


def test_contains():
    pool = FENPool()
    pool.add(STARTING_FEN)
    assert STARTING_FEN in pool
    assert KRK_FEN not in pool


def test_get_tags_missing():
    pool = FENPool()
    assert pool.get_tags("nonexistent_fen") == {}


def test_sample_with_filter():
    pool = FENPool()
    pool.add(STARTING_FEN, source="lichess")
    pool.add(KRK_FEN, source="syzygy")
    pool.add(PROMOTION_FEN, source="lichess")
    results = pool.sample(10, filters={"source": "lichess"})
    assert len(results) == 2
    fens = {r["fen"] for r in results}
    assert fens == {STARTING_FEN, PROMOTION_FEN}


def test_sample_count():
    pool = FENPool()
    pool.add(STARTING_FEN)
    pool.add(KRK_FEN)
    pool.add(PROMOTION_FEN)
    results = pool.sample(2, rng=Random(42))
    assert len(results) == 2


def test_sample_rng_reproducibility():
    pool = FENPool()
    pool.add(STARTING_FEN)
    pool.add(KRK_FEN)
    pool.add(PROMOTION_FEN)
    r1 = pool.sample(2, rng=Random(42))
    r2 = pool.sample(2, rng=Random(42))
    assert [r["fen"] for r in r1] == [r["fen"] for r in r2]


def test_sample_returns_all_when_n_exceeds_pool():
    pool = FENPool()
    pool.add(STARTING_FEN)
    pool.add(KRK_FEN)
    results = pool.sample(100)
    assert len(results) == 2


def test_remove():
    pool = FENPool()
    pool.add(STARTING_FEN)
    pool.add(KRK_FEN)
    pool.remove(STARTING_FEN)
    assert len(pool) == 1
    assert STARTING_FEN not in pool


def test_save_load_roundtrip(tmp_path):
    pool = FENPool()
    pool.add(STARTING_FEN, source="lichess")
    pool.add(KRK_FEN, source="syzygy")

    path = str(tmp_path / "pool.jsonl")
    pool.save(path)

    pool2 = FENPool()
    pool2.load(path)
    assert len(pool2) == 2
    assert pool2.all_fens() == [STARTING_FEN, KRK_FEN]
    assert pool2.get_tags(STARTING_FEN)["source"] == "lichess"
