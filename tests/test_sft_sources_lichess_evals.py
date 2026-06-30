from chess_llm.sft.sources.lichess_evals import (
    EVAL_PERSPECTIVE,
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

    assert [row["fen"] for row in result] == [BLACK_TO_MOVE_FEN, STARTING_FEN]
    assert result[0]["best_move"] == "c7c5"
    assert result[0]["depth"] == 31
    assert result[1]["best_move"] == "e2e4"


def test_stream_evals_rerun_orders_db_rows_and_normalizes_old_castling_pv(tmp_path):
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

    assert [row["depth"] for row in result] == [40, 31]
    assert result[0]["fen"] == CASTLING_FEN
    assert result[0]["best_move"] == "e1g1"
    assert result[0]["pv_line"] == "e1g1 e8g8"


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
