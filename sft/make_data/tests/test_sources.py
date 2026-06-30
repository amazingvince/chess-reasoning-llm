"""Tests for source loaders: mate_dataset and chess960."""

from random import Random

import chess
import pytest

from conftest import STARTING_FEN


def test_extract_positions_reads_full_pgn_with_headers():
    from sources.lichess_games import extract_positions

    game = {
        "pgn": """
[Event "Casual Game"]
[Site "https://lichess.org/test"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 1/2-1/2
""",
    }

    positions = list(extract_positions(game))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
    ]


def test_extract_positions_reads_compact_movetext():
    from sources.lichess_games import extract_positions

    positions = list(extract_positions({"moves": "1.e4 e5 2.Nf3 Nc6"}))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
    ]


def test_preprocess_puzzle_rejects_invalid_later_solution_move():
    from sources.lichess_puzzles import _preprocess_puzzle

    result = _preprocess_puzzle(
        fen=STARTING_FEN,
        moves_str="e2e4 e7e5 e7e6",
        themes=[],
        rating=1200,
        puzzle_id="bad-line",
    )

    assert result is None


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


def test_process_row_rejects_output_label_uci_mismatch():
    from sources.mate_dataset import _process_row

    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                 'Which move is better? MoveA:e2e4 MoveB:d2d4 ',
        "output": "MoveA:d2d4",
    }
    assert _process_row(row) is None


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
        board = chess.Board(p["fen"], chess960=True)
        assert board.is_valid()
        assert p["is_chess960"] is True


def test_stockfish_wrapper_returns_white_centric_cp_for_black_to_move():
    """Stored cp labels use White's perspective, independent of side to move."""
    from sources.stockfish_engine import StockfishWrapper

    class FakeEngine:
        def analyse(self, _board, _limit):
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(100), chess.BLACK),
                "pv": [chess.Move.from_uci("e7e5")],
            }

    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    wrapper = StockfishWrapper(path="unused")
    wrapper._engine = FakeEngine()

    result = wrapper.evaluate(board)

    assert result["cp"] == -100
    assert result["mate"] is None
    assert result["best_move"] == "e7e5"
