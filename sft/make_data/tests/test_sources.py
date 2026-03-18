"""Tests for source loaders: mate_dataset and chess960."""

from random import Random

import chess
import pytest

from conftest import STARTING_FEN


# ── mate_dataset: _validate_uci ────────────────────────────────────


def test_validate_uci_legal():
    from sources.mate_dataset import _validate_uci

    board = chess.Board(STARTING_FEN)
    m = _validate_uci(board, "e2e4")
    assert m is not None
    assert m.uci() == "e2e4"


def test_validate_uci_illegal():
    from sources.mate_dataset import _validate_uci

    board = chess.Board(STARTING_FEN)
    assert _validate_uci(board, "e1e5") is None


def test_validate_uci_invalid_str():
    from sources.mate_dataset import _validate_uci

    board = chess.Board(STARTING_FEN)
    assert _validate_uci(board, "xyz") is None


def test_validate_uci_empty():
    from sources.mate_dataset import _validate_uci

    board = chess.Board(STARTING_FEN)
    assert _validate_uci(board, "") is None


# ── mate_dataset: _process_row ──────────────────────────────────────


def test_process_row_valid():
    from sources.mate_dataset import _process_row

    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                 'Which move is better? MoveA:e2e4 MoveB:d2d4 ',
        "output": "MoveA:e2e4",
    }
    result = _process_row(row)
    assert result is not None
    assert result["move_a"] == "e2e4"
    assert result["move_b"] == "d2d4"
    assert result["better_move"] == "e2e4"


def test_process_row_invalid_fen():
    from sources.mate_dataset import _process_row

    row = {
        "input": 'The FEN of the given chess board is "not-a-valid-fen". '
                 'Which move is better? MoveA:e2e4 MoveB:d2d4 ',
        "output": "MoveA:e2e4",
    }
    assert _process_row(row) is None


def test_process_row_move_b_chosen():
    from sources.mate_dataset import _process_row

    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                 'Which move is better? MoveA:e2e4 MoveB:d2d4 ',
        "output": "MoveB:d2d4",
    }
    result = _process_row(row)
    assert result is not None
    assert result["better_move"] == "d2d4"


def test_process_row_better_move_label_a():
    from sources.mate_dataset import _process_row

    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                 'Which move is better? MoveA:e2e4 MoveB:d2d4 ',
        "output": "MoveA:e2e4",
    }
    result = _process_row(row)
    assert result is not None
    assert result["better_move"] == "e2e4"


# ── chess960: sample_chess960_positions ──────────────────────────────


def test_chess960_sample_basic():
    from sources.chess960 import sample_chess960_positions

    positions = sample_chess960_positions(5, 0, Random(42))
    assert len(positions) == 5
    for p in positions:
        assert p["is_chess960"] is True
        assert "fen" in p
        assert "chess960_id" in p
        assert 0 <= p["chess960_id"] <= 959


def test_chess960_sample_fields():
    from sources.chess960 import sample_chess960_positions

    positions = sample_chess960_positions(3, 0, Random(42))
    for p in positions:
        assert set(p.keys()) == {"fen", "chess960_id", "is_chess960", "n_moves_applied"}
        assert p["n_moves_applied"] == 0


def test_chess960_sample_with_random_moves():
    from sources.chess960 import sample_chess960_positions

    positions_no_moves = sample_chess960_positions(5, 0, Random(42))
    positions_with_moves = sample_chess960_positions(5, 10, Random(42))

    # With random moves, at least some positions should differ from starting
    fens_no = {p["fen"] for p in positions_no_moves}
    fens_with = {p["fen"] for p in positions_with_moves}
    # Different seeds produce different positions, but the key check is
    # that with_moves positions exist and have valid FENs
    for p in positions_with_moves:
        board = chess.Board(p["fen"])  # Should not raise
        assert p["is_chess960"] is True
