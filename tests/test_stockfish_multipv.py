from __future__ import annotations

import chess
import chess.engine

from chess_llm.external.multipv import (
    SqliteMultipvCache,
    analyze_multipv,
    score_move_wpd,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_CASTLE_FEN = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"


class FakeMultipvEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        *,
        multipv: int | None = None,
    ):
        self.calls.append(
            {
                "fen": board.fen(),
                "depth": limit.depth,
                "multipv": multipv,
                "chess960": board.chess960,
            }
        )
        if board.fen() == STARTING_FEN:
            return [
                {
                    "score": chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE),
                    "pv": [
                        chess.Move.from_uci("e2e4"),
                        chess.Move.from_uci("e7e5"),
                        chess.Move.from_uci("g1f3"),
                    ],
                },
                {
                    "score": chess.engine.PovScore(chess.engine.Cp(12), chess.WHITE),
                    "pv": [chess.Move.from_uci("d2d4")],
                },
            ]
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(-30), board.turn),
            "pv": [next(iter(board.legal_moves), None)] if any(board.legal_moves) else [],
        }


def test_analyze_multipv_parses_and_reuses_sqlite_cache(tmp_path):
    engine = FakeMultipvEngine()
    cache = SqliteMultipvCache(tmp_path / "multipv.sqlite")

    first = analyze_multipv(
        engine,
        STARTING_FEN,
        depth=12,
        k=2,
        pv_len=2,
        cache=cache,
        engine_config={"Threads": 1, "Hash": 64},
    )
    second = analyze_multipv(
        engine,
        STARTING_FEN,
        depth=12,
        k=2,
        pv_len=2,
        cache=cache,
        engine_config={"Threads": 1, "Hash": 64},
    )

    assert len(engine.calls) == 1
    assert first == second
    assert first.depth == 12
    assert first.k == 2
    assert [move.uci for move in first.moves] == ["e2e4", "d2d4"]
    assert first.moves[0].cp == 42
    assert first.moves[0].mate is None
    assert first.moves[0].pv == ["e2e4", "e7e5"]
    assert first.moves[0].expectation > first.moves[1].expectation


def test_analyze_multipv_handles_mate_scores_chess960_and_missing_pvs(tmp_path):
    class MateEngine:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def analyse(self, board, limit, *, multipv=None):
            self.calls.append({"chess960": board.chess960, "depth": limit.depth})
            return [
                {
                    "score": chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE),
                    "pv": [chess.Move.from_uci("d1c1")],
                },
                {
                    "score": chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE),
                    "pv": [],
                },
            ]

    engine = MateEngine()
    analysis = analyze_multipv(
        engine,
        CHESS960_CASTLE_FEN,
        depth=8,
        k=2,
        pv_len=4,
        cache=SqliteMultipvCache(tmp_path / "multipv.sqlite"),
        chess960=True,
    )

    assert engine.calls == [{"chess960": True, "depth": 8}]
    assert len(analysis.moves) == 1
    assert analysis.moves[0].uci == "d1c1"
    assert analysis.moves[0].cp is None
    assert analysis.moves[0].mate == 3
    assert analysis.moves[0].expectation == 1.0


def test_score_move_wpd_reuses_cached_multipv_before_postmove_fallback(tmp_path):
    engine = FakeMultipvEngine()
    cache = SqliteMultipvCache(tmp_path / "multipv.sqlite")

    cached = score_move_wpd(
        engine,
        STARTING_FEN,
        "d2d4",
        depth=12,
        k=2,
        pv_len=2,
        cache=cache,
    )
    fallback = score_move_wpd(
        engine,
        STARTING_FEN,
        "g1f3",
        depth=12,
        k=2,
        pv_len=2,
        cache=cache,
    )

    assert cached.multipv_hit is True
    assert cached.postmove_hit is False
    assert cached.wpd > 0
    assert fallback.multipv_hit is False
    assert fallback.postmove_hit is True
    assert fallback.predicted_move == "g1f3"
    assert len(engine.calls) == 2
