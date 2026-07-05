import logging
from pathlib import Path

import chess

from chess_llm.sft.settings import SftDataSettings
from chess_llm.sft.sources import lichess_evals
from chess_llm.sft.sources.lichess_evals import (
    EVAL_PERSPECTIVE,
    _default_dedup_db_path,
    _flush_batch,
    _init_dedup_db,
    partition_evals,
    parse_lichess_eval_row,
    stream_evals,
)

CASTLING_FEN = "r3k2r/8/8/8/8/8/5P2/R3K2R w KQkq - 0 1"


BLACK_TO_MOVE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_parse_lichess_eval_row_marks_cp_as_white_perspective():
    result = parse_lichess_eval_row(
        {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "e7e5 g1f3",
            "depth": 30,
            "knodes": 123,
            "cp": 100,
            "mate": None,
        }
    )

    assert result is not None
    assert result["fen"] == BLACK_TO_MOVE_FEN
    assert result["best_move"] == "e7e5"
    assert result["pv_line"] == "e7e5 g1f3"
    assert result["cp"] == 100
    assert result["mate"] is None
    assert result["eval_perspective"] == EVAL_PERSPECTIVE == "white"


def test_parse_lichess_eval_row_marks_mate_as_white_perspective():
    result = parse_lichess_eval_row(
        {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "e7e5",
            "depth": 30,
            "knodes": 123,
            "cp": None,
            "mate": 3,
        }
    )

    assert result is not None
    assert result["mate"] == 3
    assert result["eval_perspective"] == "white"


def test_parse_lichess_eval_row_rejects_parseable_but_invalid_fen():
    result = parse_lichess_eval_row(
        {
            "fen": "7K/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/q6k b - -",
            "line": "a1a2",
            "depth": 30,
            "knodes": 123,
            "cp": 100,
            "mate": None,
        }
    )

    assert result is None


def test_stream_evals_uses_injected_loader_and_actual_hf_schema(tmp_path):
    rows = [
        {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "e7e5 g1f3",
            "depth": 19,
            "knodes": 999,
            "cp": 5,
            "mate": None,
        },
        {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "e7e5 g1f3",
            "depth": 30,
            "knodes": 100,
            "cp": 100,
            "mate": None,
        },
        {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "c7c5 g1f3",
            "depth": 30,
            "knodes": 200,
            "cp": 80,
            "mate": None,
        },
    ]
    calls = []

    def fake_loader(name: str, *, split: str, streaming: bool):
        calls.append((name, split, streaming))
        return rows

    result = list(
        stream_evals(
            min_depth=20,
            dedup_db_path=tmp_path / "evals.db",
            dataset_loader=fake_loader,
        )
    )

    assert calls == [("Lichess/chess-position-evaluations", "train", True)]
    assert len(result) == 1
    assert result[0]["best_move"] == "c7c5"
    assert result[0]["depth"] == 30
    assert result[0]["knodes"] == 200
    assert result[0]["eval_perspective"] == "white"


def test_stream_evals_zero_limit_does_not_open_dataset(tmp_path):
    def fail_loader(*_args, **_kwargs):
        raise AssertionError("dataset loader should not be called")

    assert list(
        stream_evals(
            max_rows=0,
            dedup_db_path=tmp_path / "evals.db",
            dataset_loader=fail_loader,
        )
    ) == []


def test_default_dedup_db_path_uses_sft_settings_default_root(tmp_path, monkeypatch):
    monkeypatch.delenv("CHESS_SFT_OUTPUT", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = (
        SftDataSettings.from_env(Path.cwd(), env={}).annotations_dir / "evals_dedup.db"
    )

    assert _default_dedup_db_path() == expected
    assert not _default_dedup_db_path().is_relative_to(tmp_path)


def test_stream_evals_closes_streaming_dataset_after_early_stop(tmp_path):
    class CloseableRows:
        def __init__(self, rows):
            self.rows = rows
            self.closed = False

        def __iter__(self):
            return iter(self.rows)

        def close(self):
            self.closed = True

    rows = CloseableRows(
        [
            {
                "fen": BLACK_TO_MOVE_FEN,
                "line": "e7e5 g1f3",
                "depth": 30,
                "knodes": 100,
                "cp": 100,
                "mate": None,
            },
            {
                "fen": STARTING_FEN,
                "line": "e2e4 e7e5",
                "depth": 30,
                "knodes": 100,
                "cp": 20,
                "mate": None,
            },
        ]
    )

    result = list(
        stream_evals(
            min_depth=20,
            max_rows=1,
            dedup_db_path=tmp_path / "evals.db",
            dataset_loader=lambda *_args, **_kwargs: rows,
        )
    )

    assert len(result) == 1
    assert rows.closed is True


def test_stream_evals_rerun_yields_existing_db_rows(tmp_path):
    db_path = tmp_path / "evals.db"
    conn = _init_dedup_db(db_path)
    _flush_batch(conn, [(BLACK_TO_MOVE_FEN, "e7e5", "e7e5 g1f3", 30, 100, 100, None)])
    conn.close()

    result = list(
        stream_evals(
            min_depth=20,
            max_rows=1,
            dedup_db_path=db_path,
            dataset_loader=lambda *_args, **_kwargs: (),
        )
    )

    assert result == [
        {
            "fen": BLACK_TO_MOVE_FEN,
            "best_move": "e7e5",
            "pv_line": "e7e5 g1f3",
            "depth": 30,
            "knodes": 100,
            "cp": 100,
            "mate": None,
            "eval_perspective": "white",
        }
    ]


def test_stream_evals_with_new_better_row_yields_merged_cache_subset(tmp_path):
    db_path = tmp_path / "evals.db"
    conn = _init_dedup_db(db_path)
    _flush_batch(
        conn,
        [
            (BLACK_TO_MOVE_FEN, "e7e5", "e7e5 g1f3", 30, 100, 100, None),
            (STARTING_FEN, "e2e4", "e2e4 e7e5", 29, 90, 20, None),
        ],
    )
    conn.close()

    rows = [
        {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "c7c5 g1f3",
            "depth": 31,
            "knodes": 200,
            "cp": 80,
            "mate": None,
        },
    ]

    result = list(
        stream_evals(
            min_depth=20,
            max_rows=2,
            dedup_db_path=db_path,
            dataset_loader=lambda *_args, **_kwargs: rows,
        )
    )

    assert sorted(row["fen"] for row in result) == sorted([BLACK_TO_MOVE_FEN, STARTING_FEN])
    by_fen = {row["fen"]: row for row in result}
    assert by_fen[BLACK_TO_MOVE_FEN]["best_move"] == "c7c5"
    assert by_fen[BLACK_TO_MOVE_FEN]["depth"] == 31
    assert by_fen[STARTING_FEN]["best_move"] == "e2e4"


def test_stream_evals_rerun_normalizes_old_castling_pv_from_db_rows(tmp_path):
    db_path = tmp_path / "evals.db"
    conn = _init_dedup_db(db_path)
    _flush_batch(
        conn,
        [
            (BLACK_TO_MOVE_FEN, "e7e5", "e7e5 g1f3", 31, 100, 100, None),
            (CASTLING_FEN, "e1h1", "e1h1 e8h8", 40, 500, 300, None),
        ],
    )
    conn.close()

    result = list(
        stream_evals(
            min_depth=20,
            max_rows=2,
            dedup_db_path=db_path,
            dataset_loader=lambda *_args, **_kwargs: (),
        )
    )

    assert sorted(row["depth"] for row in result) == [31, 40]
    castling = next(row for row in result if row["fen"] == CASTLING_FEN)
    assert castling["depth"] == 40
    assert castling["best_move"] == "e1g1"
    assert castling["pv_line"] == "e1g1 e8g8"


def test_stream_evals_cached_order_is_seed_stable_shuffle_not_depth_ranked(tmp_path):
    db_path = tmp_path / "evals.db"
    conn = _init_dedup_db(db_path)
    board = chess.Board()
    batch = []
    for depth in range(21, 31):
        move = next(iter(board.legal_moves))
        batch.append((board.fen(), move.uci(), move.uci(), depth, 100, 10, None))
        board.push(move)
    _flush_batch(conn, batch)
    conn.close()

    def run() -> list[dict]:
        return list(
            stream_evals(
                min_depth=20,
                dedup_db_path=db_path,
                dataset_loader=lambda *_args, **_kwargs: (),
            )
        )

    first = run()
    second = run()

    depths = [row["depth"] for row in first]
    assert sorted(depths) == list(range(21, 31))
    assert depths != sorted(depths, reverse=True)
    assert [row["fen"] for row in first] == [row["fen"] for row in second]


def test_stream_evals_skips_invalid_cached_fens(tmp_path):
    invalid_fen = "7K/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/q6k b - -"
    db_path = tmp_path / "evals.db"
    conn = _init_dedup_db(db_path)
    _flush_batch(
        conn,
        [
            (invalid_fen, "a1a2", "a1a2", 30, 100, 100, None),
            (BLACK_TO_MOVE_FEN, "e7e5", "e7e5 g1f3", 30, 100, 20, None),
        ],
    )
    conn.close()

    result = list(
        stream_evals(
            min_depth=20,
            dedup_db_path=db_path,
            dataset_loader=lambda *_args, **_kwargs: (),
        )
    )

    assert [row["fen"] for row in result] == [BLACK_TO_MOVE_FEN]


def test_stream_evals_flushes_batches_and_survives_mid_stream_failure(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(lichess_evals, "_FLUSH_INTERVAL", 1)

    def exploding_rows(*_args, **_kwargs):
        yield {
            "fen": BLACK_TO_MOVE_FEN,
            "line": "e7e5 g1f3",
            "depth": 30,
            "knodes": 100,
            "cp": 100,
            "mate": None,
        }
        yield {
            "fen": STARTING_FEN,
            "line": "e2e4 e7e5",
            "depth": 30,
            "knodes": 100,
            "cp": 20,
            "mate": None,
        }
        raise RuntimeError("stream died mid-scan")

    with caplog.at_level(logging.ERROR):
        result = list(
            stream_evals(
                min_depth=20,
                dedup_db_path=tmp_path / "evals.db",
                dataset_loader=exploding_rows,
            )
        )

    assert sorted(row["fen"] for row in result) == sorted([BLACK_TO_MOVE_FEN, STARTING_FEN])
    assert any("mid-stream" in record.getMessage() for record in caplog.records)


def test_partition_evals_uses_valid_board_rows_only():
    parts = partition_evals(
        iter(
            [
                {"fen": BLACK_TO_MOVE_FEN, "cp": 30, "mate": None, "depth": 35},
                {"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "cp": 300, "mate": None, "depth": 35},
            ]
        )
    )

    assert len(parts["balanced"]) == 1
    assert len(parts["best_move"]) == 1
    assert parts["high_eval"] == []
