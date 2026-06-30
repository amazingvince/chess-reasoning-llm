import importlib
import sys
from pathlib import Path

import chess
import chess.engine
import pytest

from chess_llm.external.stockfish import (
    StockfishEngineConfig,
    StockfishWrapper,
    configure_stockfish_engine,
    open_stockfish,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FakeEngine:
    def __init__(self, info: dict | None = None) -> None:
        self.info = info or {
            "score": chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE),
            "pv": [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
        }
        self.configured: list[dict] = []
        self.analysed: list[tuple[str, int | None]] = []
        self.boards: list[chess.Board] = []
        self.quit_called = False

    def configure(self, options: dict) -> None:
        self.configured.append(options)

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        self.boards.append(board)
        self.analysed.append((board.fen(), limit.depth))
        return self.info

    def quit(self) -> None:
        self.quit_called = True


def test_stockfish_wrapper_evaluate_returns_white_centric_cp_for_black_pov_score():
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    engine = FakeEngine(
        {
            "score": chess.engine.PovScore(chess.engine.Cp(100), chess.BLACK),
            "pv": [chess.Move.from_uci("e7e5")],
        }
    )
    wrapper = StockfishWrapper(path="unused")
    wrapper._engine = engine

    result = wrapper.evaluate(board, depth=18)

    assert result == {
        "cp": -100,
        "mate": None,
        "best_move": "e7e5",
        "pv_line": "e7e5",
    }
    assert engine.analysed == [(board.fen(), 18)]


def test_stockfish_wrapper_evaluate_returns_white_centric_mate_score():
    board = chess.Board(STARTING_FEN)
    wrapper = StockfishWrapper(path="unused")
    wrapper._engine = FakeEngine(
        {
            "score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE),
            "pv": [chess.Move.from_uci("e2e4")],
        }
    )

    result = wrapper.evaluate(board)

    assert result["cp"] is None
    assert result["mate"] == 3
    assert result["best_move"] == "e2e4"


def test_stockfish_wrapper_batch_evaluate_skips_invalid_and_parseable_invalid_fens():
    valid_engine = FakeEngine()
    wrapper = StockfishWrapper(path="unused")
    wrapper._engine = valid_engine

    rows = list(
        wrapper.batch_evaluate(
            [
                "not a fen",
                "8/8/8/8/8/8/8/8 w - - 0 1",
                STARTING_FEN,
            ],
            depth=12,
        )
    )

    assert rows[0] == {"cp": None, "mate": None, "best_move": None, "pv_line": ""}
    assert rows[1] == {"cp": None, "mate": None, "best_move": None, "pv_line": ""}
    assert rows[2]["best_move"] == "e2e4"
    assert len(valid_engine.analysed) == 1
    assert valid_engine.analysed[0][1] == 12


def test_stockfish_wrapper_batch_evaluate_accepts_chess960_fens():
    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    engine = FakeEngine()
    wrapper = StockfishWrapper(path="unused")
    wrapper._engine = engine

    rows = list(wrapper.batch_evaluate([chess960_fen], depth=10))

    assert rows[0]["best_move"] == "e2e4"
    assert len(engine.boards) == 1
    assert engine.boards[0].chess960 is True
    assert engine.boards[0].is_valid()


def test_stockfish_wrapper_open_configures_engine_and_context_manager_closes():
    engine = FakeEngine()

    with StockfishWrapper(
        path="stockfish",
        depth=10,
        threads=2,
        hash_mb=128,
        syzygy_path="tb",
        engine_factory=lambda _path: engine,
    ) as wrapper:
        assert wrapper._engine is engine
        assert engine.configured == [{"Threads": 2, "Hash": 128, "SyzygyPath": "tb"}]

    assert engine.quit_called is True
    assert wrapper._engine is None


def test_open_stockfish_closes_engine_when_configuration_fails(tmp_path):
    stockfish_path = tmp_path / "stockfish"
    stockfish_path.write_text("", encoding="utf-8")
    engine = FakeEngine()

    def fail_configure(_engine, **_kwargs):
        raise RuntimeError("bad option")

    with pytest.raises(RuntimeError):
        open_stockfish(
            StockfishEngineConfig(stockfish_path),
            engine_factory=lambda _path: engine,
            configure=fail_configure,
        )

    assert engine.quit_called is True


def test_configure_stockfish_engine_skips_syzygy_when_not_configured():
    engine = FakeEngine()

    configure_stockfish_engine(engine, threads=3, hash_mb=64)

    assert engine.configured == [{"Threads": 3, "Hash": 64}]


def test_legacy_stockfish_wrapper_delegates_to_package(monkeypatch):
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    sys.modules.pop("sources.stockfish_engine", None)

    legacy = importlib.import_module("sources.stockfish_engine")
    package = importlib.import_module("chess_llm.external.stockfish")

    assert issubclass(legacy.StockfishWrapper, package.StockfishWrapper)
    wrapper = legacy.StockfishWrapper()
    assert wrapper.path
