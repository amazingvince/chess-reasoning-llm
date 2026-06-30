from random import Random
import sys
from pathlib import Path

import chess


def test_package_chess960_sample_positions_are_chess960_parseable():
    from chess_llm.sft.sources.chess960 import sample_chess960_positions

    rows = sample_chess960_positions(5, n_random_moves=4, rng=Random(42))

    assert len(rows) == 5
    for row in rows:
        board = chess.Board(row["fen"], chess960=True)
        assert board.is_valid()
        assert row["is_chess960"] is True
        assert 0 <= row["chess960_id"] <= 959
        assert 0 <= row["n_moves_applied"] <= 4


def test_known_chess960_castling_position_requires_chess960_parsing():
    fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"

    assert chess.Board(fen).is_valid() is False
    board = chess.Board(fen, chess960=True)

    assert board.is_valid()
    assert "d1c1" in {move.uci() for move in board.legal_moves}


def test_chess960_sample_reports_actual_moves_when_rollout_stops_early(monkeypatch):
    from chess_llm.sft.sources import chess960

    terminal_board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1", chess960=True)

    def fake_generate_random(_rng):
        return terminal_board, 0

    monkeypatch.setattr(chess960, "generate_random", fake_generate_random)

    rows = chess960.sample_chess960_positions(1, n_random_moves=10, rng=Random(3))

    assert rows[0]["n_moves_applied"] == 0
    assert rows[0]["fen"] == terminal_board.fen()


def test_package_chess960_generate_all_covers_960_ids():
    from chess_llm.sft.sources.chess960 import generate_all

    generated = generate_all()

    assert len(generated) == 960
    assert [pos_id for _, pos_id in generated] == list(range(960))
    assert all(board.chess960 for board, _ in generated)


def test_legacy_chess960_wrapper_delegates_to_package():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.sft.sources import chess960 as package_chess960
    from sources import chess960 as legacy_chess960

    assert legacy_chess960.sample_chess960_positions is package_chess960.sample_chess960_positions
    assert legacy_chess960.generate_random is package_chess960.generate_random
