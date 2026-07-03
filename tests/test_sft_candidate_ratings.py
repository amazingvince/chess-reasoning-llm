from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.engine


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FakeMultipvEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, int | None]] = []

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, *, multipv=None):
        self.calls.append((board.fen(), limit.depth, multipv))
        moves = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3"]
        cps = [42, 15, 5, -20, -80]
        return [
            {
                "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
                "pv": [chess.Move.from_uci(move)],
            }
            for move, cp in zip(moves, cps, strict=True)
        ]


def test_build_candidate_rating_rows_uses_cached_true_multipv(tmp_path: Path):
    from chess_llm.sft.candidate_ratings import build_candidate_rating_rows

    engine = FakeMultipvEngine()
    cache_path = tmp_path / "multipv.sqlite"
    rows = [{"fen": STARTING_FEN, "source": "unit"}]

    first = build_candidate_rating_rows(
        rows,
        engine=engine,
        cache_path=cache_path,
        depth=8,
        multipv=5,
        pv_len=1,
    )
    second = build_candidate_rating_rows(
        rows,
        engine=engine,
        cache_path=cache_path,
        depth=8,
        multipv=5,
        pv_len=1,
    )

    assert len(engine.calls) == 1
    assert len(first.rows) == 1
    assert len(second.rows) == 1
    row = first.rows[0]
    assert row["fen"] == STARTING_FEN
    assert row["source"] == "stockfish_multipv"
    assert row["best_move"] == "e2e4"
    assert row["multipv_depth"] == 8
    assert row["multipv_k"] == 5
    assert row["pv_len"] == 1
    assert [item["uci"] for item in row["candidate_ratings"]] == [
        "e2e4",
        "d2d4",
        "g1f3",
        "c2c4",
        "b1c3",
    ]
    assert row["candidate_ratings"][0]["cp"] == 42


def test_write_candidate_rating_jsonl_skips_invalid_and_short_multipv(tmp_path: Path):
    from chess_llm.sft.candidate_ratings import write_candidate_rating_jsonl

    class ShortEngine(FakeMultipvEngine):
        def analyse(self, board, limit, *, multipv=None):
            return super().analyse(board, limit, multipv=multipv)[:4]

    output_path = tmp_path / "candidate_ratings.jsonl"
    result = write_candidate_rating_jsonl(
        [
            {"fen": STARTING_FEN},
            {"fen": "not a fen"},
        ],
        output_path=output_path,
        engine=ShortEngine(),
        cache_path=tmp_path / "multipv.sqlite",
        depth=8,
        multipv=5,
        pv_len=1,
    )

    assert result.row_count == 0
    assert result.skipped_count == 2
    assert output_path.read_text(encoding="utf-8") == ""


def test_load_candidate_rating_evals_reads_jsonl_and_ignores_bad_rows(tmp_path: Path):
    from chess_llm.sft.candidate_ratings import load_candidate_rating_evals

    path = tmp_path / "candidate_ratings.jsonl"
    path.write_text(
        json.dumps({"fen": STARTING_FEN, "candidate_ratings": [{"uci": "e2e4", "cp": 1}]})
        + "\n"
        + "{not-json}\n"
        + json.dumps({"fen": STARTING_FEN})
        + "\n",
        encoding="utf-8",
    )

    assert load_candidate_rating_evals(path) == [
        {"fen": STARTING_FEN, "candidate_ratings": [{"uci": "e2e4", "cp": 1}]}
    ]


def test_jsonl_loaders_keep_bom_prefixed_first_row(tmp_path: Path):
    from chess_llm.sft.candidate_ratings import (
        load_candidate_rating_evals,
        load_input_rows,
    )

    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"fen": STARTING_FEN}) + "\n",
        encoding="utf-8-sig",
    )

    ratings_path = tmp_path / "candidate_ratings.jsonl"
    ratings_path.write_text(
        json.dumps(
            {
                "fen": STARTING_FEN,
                "candidate_ratings": [{"uci": "e2e4", "cp": 1}],
            }
        )
        + "\n",
        encoding="utf-8-sig",
    )

    assert load_input_rows(input_path) == [{"fen": STARTING_FEN}]
    assert load_candidate_rating_evals(ratings_path) == [
        {"fen": STARTING_FEN, "candidate_ratings": [{"uci": "e2e4", "cp": 1}]}
    ]


def test_candidate_ratings_cli_parser_exposes_stockfish_efficient_defaults():
    from chess_llm.sft.candidate_ratings import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--input-jsonl",
            "positions.jsonl",
            "--output-jsonl",
            "candidate_ratings.jsonl",
            "--stockfish-path",
            "stockfish",
        ]
    )

    assert args.multipv == 5
    assert args.threads == 1
    assert args.pv_len == 8
