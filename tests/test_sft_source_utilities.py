from __future__ import annotations

import importlib
import sys
from pathlib import Path

import chess
import pytest


KRK_FEN = "8/8/8/4k3/8/8/8/4K2R w - - 0 1"


def _clear_legacy_utility_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "pool.annotator"
            or module_name == "sources.polyglot_books"
            or module_name == "sources.syzygy_probing"
            or module_name.startswith("sft.make_data.pool.annotator")
            or module_name.startswith("sft.make_data.sources.polyglot_books")
            or module_name.startswith("sft.make_data.sources.syzygy_probing")
        ):
            sys.modules.pop(module_name, None)


def test_package_annotation_imports_without_legacy_modules():
    _clear_legacy_utility_modules()

    module = importlib.import_module("chess_llm.sft.annotation")

    assert module.BatchAnnotator
    assert "pool.annotator" not in sys.modules


def test_close_streaming_dataset_closes_iterator_source_and_collects_garbage():
    from chess_llm.sft.sources._streaming import close_streaming_dataset

    calls = []

    class Closeable:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(self.name)

    close_streaming_dataset(
        Closeable("iterator"),
        Closeable("source"),
        collect=lambda: calls.append("gc"),
        shutdown_wait_seconds=0.0,
    )

    assert calls == ["iterator", "source", "gc"]


def test_close_streaming_dataset_waits_when_configured():
    from chess_llm.sft.sources._streaming import close_streaming_dataset

    calls = []

    close_streaming_dataset(
        object(),
        object(),
        collect=lambda: calls.append("gc"),
        sleep=lambda seconds: calls.append(f"sleep:{seconds}"),
        shutdown_wait_seconds=3.0,
    )

    assert calls == ["gc", "sleep:3.0"]


def test_batch_annotator_parses_chess960_fen_before_stockfish():
    from chess_llm.sft.annotation import BatchAnnotator

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"

    class CapturingStockfish:
        def __init__(self):
            self.board = None

        def evaluate(self, board, _depth=None):
            self.board = board
            return {"cp": 0, "mate": None, "best_move": "d1d3", "pv_line": "d1d3"}

    stockfish = CapturingStockfish()
    annotator = BatchAnnotator(stockfish=stockfish)

    assert annotator.annotate(chess960_fen)["best_move"] == "d1d3"
    assert stockfish.board is not None
    assert stockfish.board.chess960 is True
    assert stockfish.board.is_valid()


def test_batch_annotator_cache_preload_and_lookup(tmp_path: Path):
    from chess_llm.sft.annotation import BatchAnnotator

    annotator = BatchAnnotator(cache_path=tmp_path / "annotations.db")
    count = annotator.preload_from_evals(
        [
            {
                "fen": KRK_FEN,
                "cp": 120,
                "mate": None,
                "best_move": "g1g7",
                "pv_line": "g1g7 h2h1",
                "depth": 30,
            }
        ]
    )

    assert count == 1
    assert annotator.annotate(KRK_FEN) == {
        "cp": 120,
        "mate": None,
        "best_move": "g1g7",
        "pv_line": "g1g7 h2h1",
    }


def test_package_polyglot_filters_illegal_moves_and_sorts_by_weight():
    from chess_llm.sft.sources.polyglot_books import get_weighted_moves

    board = chess.Board()

    class Entry:
        def __init__(self, move: str, weight: int):
            self.move = chess.Move.from_uci(move)
            self.weight = weight

    class Reader:
        def find_all(self, _board):
            return [
                Entry("g1f3", 5),
                Entry("e2e4", 20),
                Entry("a1a8", 100),
            ]

    assert get_weighted_moves(Reader(), board) == [("e2e4", 20), ("g1f3", 5)]


def test_package_syzygy_best_dtz_move_uses_package_probe(monkeypatch):
    from chess_llm.sft.sources import syzygy_probing

    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)
    probe_map = {}
    for move, result in [
        (moves[0], {"wdl": -2, "dtz": -7}),
        (moves[1], {"wdl": -2, "dtz": -3}),
    ]:
        board.push(move)
        probe_map[board.fen()] = result
        board.pop()

    monkeypatch.setattr(
        syzygy_probing,
        "probe",
        lambda _tb, probed_board: probe_map.get(probed_board.fen()),
    )

    assert syzygy_probing.best_dtz_move(None, board) == moves[1].uci()


def test_package_syzygy_prefers_unconditional_win_over_cursed_win(monkeypatch):
    from chess_llm.sft.sources import syzygy_probing

    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)
    probe_map = {}
    for move, result in [
        (moves[0], {"wdl": -1, "dtz": -1}),    # our WDL = cursed win
        (moves[1], {"wdl": -2, "dtz": -100}),  # our WDL = unconditional win
    ]:
        board.push(move)
        probe_map[board.fen()] = result
        board.pop()

    monkeypatch.setattr(
        syzygy_probing,
        "probe",
        lambda _tb, probed_board: probe_map.get(probed_board.fen()),
    )

    assert syzygy_probing.best_dtz_move(None, board) == moves[1].uci()


def test_package_syzygy_prefers_blessed_loss_over_unconditional_loss(monkeypatch):
    from chess_llm.sft.sources import syzygy_probing

    board = chess.Board(KRK_FEN)
    moves = list(board.legal_moves)
    probe_map = {}
    for move, result in [
        (moves[0], {"wdl": 2, "dtz": 100}),  # our WDL = unconditional loss
        (moves[1], {"wdl": 1, "dtz": 1}),    # our WDL = blessed loss
    ]:
        board.push(move)
        probe_map[board.fen()] = result
        board.pop()

    monkeypatch.setattr(
        syzygy_probing,
        "probe",
        lambda _tb, probed_board: probe_map.get(probed_board.fen()),
    )

    assert syzygy_probing.best_dtz_move(None, board) == moves[1].uci()


def test_legacy_utility_imports_alias_package_modules(monkeypatch):
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    _clear_legacy_utility_modules()

    package_annotator = importlib.import_module("chess_llm.sft.annotation")
    package_polyglot = importlib.import_module("chess_llm.sft.sources.polyglot_books")
    package_syzygy = importlib.import_module("chess_llm.sft.sources.syzygy_probing")

    legacy_annotator = importlib.import_module("pool.annotator")
    legacy_polyglot = importlib.import_module("sources.polyglot_books")
    legacy_syzygy = importlib.import_module("sources.syzygy_probing")

    assert legacy_annotator is package_annotator
    assert legacy_polyglot is package_polyglot
    assert legacy_syzygy is package_syzygy
