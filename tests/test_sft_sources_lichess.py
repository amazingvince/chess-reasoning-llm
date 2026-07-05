import hashlib
import importlib
import sys
from pathlib import Path

from chess_llm.sft.sources.lichess_games import extract_game_positions
from chess_llm.sft.sources.lichess_puzzles import preprocess_puzzle


def test_extract_game_positions_reads_full_pgn_with_headers():
    game = {
        "pgn": """
[Event "Casual Game"]
[Site "https://lichess.org/test"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 1/2-1/2
""",
    }

    positions = list(extract_game_positions(game))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
    ]
    assert positions[0]["fen"].startswith("rnbqkbnr/pppppppp/")


def test_extract_game_positions_reads_compact_movetext():
    positions = list(extract_game_positions({"moves": "1.e4 e5 2.Nf3 Nc6"}))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
    ]


def test_extract_game_positions_preserves_prefix_history_before_each_move():
    positions = list(extract_game_positions({"moves": "1.e4 e5 2.Nf3 Nc6"}))

    assert [pos["move_history"] for pos in positions] == [
        "",
        "e2e4",
        "e2e4 e7e5",
        "e2e4 e7e5 g1f3",
    ]
    assert positions[2]["ply"] == 2
    assert positions[2]["move_played_uci"] == "g1f3"


def test_extract_game_positions_accepts_pgn_comments_and_variations():
    game = {
        "movetext": """
1. e4 {[%clk 0:10:00]} (1. d4 d5) e5 $1
2. Nf3 Nc6 *
""",
    }

    positions = list(extract_game_positions(game))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
    ]


def test_extract_game_positions_emits_sha256_game_id_on_every_position():
    moves = "1. e4 e5 2. Nf3 Nc6 1/2-1/2"
    expected_game_id = hashlib.sha256(moves.encode("utf-8")).hexdigest()[:16]

    positions = list(extract_game_positions({"moves": moves}))

    assert len(positions) == 4
    assert all(pos["game_id"] == expected_game_id for pos in positions)


def test_extract_game_positions_keeps_valid_prefix_of_partially_corrupt_pgn():
    moves = "1. e4 e5 2. Nf3 Nc6 3. Bb5 NotAMove 4. O-O"

    positions = list(extract_game_positions({"movetext": moves}))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
    ]
    expected_game_id = hashlib.sha256(moves.encode("utf-8")).hexdigest()[:16]
    assert all(pos["game_id"] == expected_game_id for pos in positions)


def test_extract_game_positions_keeps_exactly_four_ply_recovered_prefix():
    positions = list(extract_game_positions({"movetext": "1. e4 e5 2. Nf3 Nc6 3. Ke3"}))

    assert [pos["move_played_uci"] for pos in positions] == [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
    ]


def test_extract_game_positions_rejects_malformed_partial_pgn():
    # Recovered prefix is only 3 plies, below MIN_RECOVERED_PLIES.
    positions = list(extract_game_positions({"movetext": "1. e4 e5 2. Nf3 NotAMove 3. Bb5"}))

    assert positions == []


def test_extract_game_positions_rejects_malformed_compact_movetext():
    positions = list(extract_game_positions({"moves": "e2e4 e7e5 g1f3 badmove f1b5"}))

    assert positions == []


def test_stream_games_uses_injected_loader_and_actual_hf_schema():
    from chess_llm.sft.sources.lichess_games import stream_games

    rows = [
        {"WhiteElo": "1199", "BlackElo": "1600", "Result": "1-0", "movetext": "1. e4"},
        {"WhiteElo": "1600", "BlackElo": "bad", "Result": "1-0", "movetext": "1. e4"},
        {
            "WhiteElo": "1600",
            "BlackElo": "1700",
            "Result": "1/2-1/2",
            "movetext": "1. e4 e5 2. Nf3 Nc6 1/2-1/2",
        },
        {
            "WhiteElo": 1800,
            "BlackElo": 1900,
            "Result": "0-1",
            "movetext": "1. d4 d5",
        },
    ]
    calls = []

    def fake_loader(name: str, *, split: str, streaming: bool):
        calls.append((name, split, streaming))
        return rows

    games = list(stream_games(min_elo=1500, max_games=1, dataset_loader=fake_loader))

    assert calls == [("Lichess/standard-chess-games", "train", True)]
    assert games == [
        {
            "pgn": "1. e4 e5 2. Nf3 Nc6 1/2-1/2",
            "white_elo": 1600,
            "black_elo": 1700,
            "result": "1/2-1/2",
            "moves": "1. e4 e5 2. Nf3 Nc6 1/2-1/2",
        },
    ]


def test_stream_games_can_restrict_hf_data_files_for_bounded_runs():
    from chess_llm.sft.sources.lichess_games import stream_games

    rows = [
        {
            "WhiteElo": "1600",
            "BlackElo": "1700",
            "Result": "1-0",
            "movetext": "1. e4 e5",
        },
    ]
    calls = []

    def fake_loader(name: str, *, split: str, streaming: bool, data_files):
        calls.append(
            {
                "name": name,
                "split": split,
                "streaming": streaming,
                "data_files": data_files,
            }
        )
        return rows

    games = list(
        stream_games(
            min_elo=1500,
            max_games=1,
            dataset_loader=fake_loader,
            data_files=["data/year=2013/month=01/train-00000-of-00001.parquet"],
        )
    )

    assert len(games) == 1
    assert calls == [
        {
            "name": "Lichess/standard-chess-games",
            "split": "train",
            "streaming": True,
            "data_files": [
                "data/year=2013/month=01/train-00000-of-00001.parquet"
            ],
        }
    ]


def test_stream_games_zero_limit_does_not_open_dataset():
    from chess_llm.sft.sources.lichess_games import stream_games

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("dataset loader should not be called")

    assert list(stream_games(max_games=0, dataset_loader=fail_loader)) == []


def test_stream_games_closes_streaming_dataset_after_early_stop():
    from chess_llm.sft.sources.lichess_games import stream_games

    class CloseableRows:
        def __init__(self, rows):
            self.rows = rows
            self.closed = False
            self.iterator_closed = False

        def __iter__(self):
            return CloseableIterator(self.rows, self)

        def close(self):
            self.closed = True

    class CloseableIterator:
        def __init__(self, rows, owner):
            self.iterator = iter(rows)
            self.owner = owner

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterator)

        def close(self):
            self.owner.iterator_closed = True

    rows = CloseableRows(
        [
            {
                "WhiteElo": "1600",
                "BlackElo": "1700",
                "Result": "1-0",
                "movetext": "1. e4 e5",
            },
            {
                "WhiteElo": "1800",
                "BlackElo": "1900",
                "Result": "0-1",
                "movetext": "1. d4 d5",
            },
        ]
    )

    games = list(
        stream_games(
            min_elo=1500,
            max_games=1,
            dataset_loader=lambda *_args, **_kwargs: rows,
        )
    )

    assert len(games) == 1
    assert rows.iterator_closed is True
    assert rows.closed is True


def test_stream_games_accepts_lowercase_schema_and_min_elo_boundary():
    from chess_llm.sft.sources.lichess_games import stream_games

    rows = [
        {"white_elo": 1500, "black_elo": 1500, "result": "1-0", "moves": "1. e4 e5"},
    ]

    games = list(
        stream_games(
            min_elo=1500,
            dataset_loader=lambda *_args, **_kwargs: rows,
        )
    )

    assert len(games) == 1
    assert games[0]["white_elo"] == 1500
    assert games[0]["black_elo"] == 1500
    assert games[0]["result"] == "1-0"


def test_load_puzzles_uses_injected_loader_filters_and_counts_valid_rows():
    from chess_llm.sft.sources.lichess_puzzles import load_puzzles

    rows = [
        {
            "PuzzleId": "too-low",
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "e2e4 e7e5",
            "Rating": "900",
            "Themes": "opening",
        },
        {
            "PuzzleId": "not-theme",
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "e2e4 e7e5",
            "Rating": "1200",
            "Themes": ["opening"],
        },
        {
            "PuzzleId": "valid",
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "e2e4 e7e5 g1f3",
            "Rating": "1300",
            "Themes": "middlegame fork",
        },
        {
            "PuzzleId": "valid-2",
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "d2d4 d7d5",
            "Rating": 1400,
            "Themes": ["fork"],
        },
    ]
    calls = []

    def fake_loader(name: str, *, split: str, streaming: bool):
        calls.append((name, split, streaming))
        return rows

    puzzles = list(
        load_puzzles(
            min_rating=1000,
            max_rating=1500,
            themes=["fork"],
            max_puzzles=1,
            dataset_loader=fake_loader,
        )
    )

    assert calls == [("Lichess/chess-puzzles", "train", True)]
    assert len(puzzles) == 1
    assert puzzles[0]["puzzle_id"] == "valid"
    assert puzzles[0]["rating"] == 1300
    assert puzzles[0]["themes"] == ["middlegame", "fork"]
    assert puzzles[0]["solution_first_move"] == "e7e5"


def test_load_puzzles_accepts_lowercase_schema_and_inclusive_rating_bounds():
    from chess_llm.sft.sources.lichess_puzzles import load_puzzles

    rows = [
        {
            "puzzle_id": "valid",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": "e2e4 e7e5",
            "rating": "1500",
            "themes": ["fork", "mate"],
        },
        {
            "puzzle_id": "bad-rating",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves": "e2e4 e7e5",
            "rating": "?",
            "themes": ["fork"],
        },
    ]

    puzzles = list(
        load_puzzles(
            min_rating=1500,
            max_rating=1500,
            themes=["mate"],
            dataset_loader=lambda *_args, **_kwargs: rows,
        )
    )

    assert len(puzzles) == 1
    assert puzzles[0]["puzzle_id"] == "valid"
    assert puzzles[0]["rating"] == 1500
    assert puzzles[0]["themes"] == ["fork", "mate"]


def test_load_puzzles_zero_limit_does_not_open_dataset():
    from chess_llm.sft.sources.lichess_puzzles import load_puzzles

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("dataset loader should not be called")

    assert list(load_puzzles(max_puzzles=0, dataset_loader=fail_loader)) == []


def test_load_puzzles_closes_streaming_dataset_after_early_stop():
    from chess_llm.sft.sources.lichess_puzzles import load_puzzles

    class CloseableRows:
        def __init__(self, rows):
            self.rows = rows
            self.closed = False
            self.iterator_closed = False

        def __iter__(self):
            return CloseableIterator(self.rows, self)

        def close(self):
            self.closed = True

    class CloseableIterator:
        def __init__(self, rows, owner):
            self.iterator = iter(rows)
            self.owner = owner

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterator)

        def close(self):
            self.owner.iterator_closed = True

    rows = CloseableRows(
        [
            {
                "PuzzleId": "valid",
                "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "Moves": "e2e4 e7e5 g1f3",
                "Rating": "1300",
                "Themes": "middlegame fork",
            },
            {
                "PuzzleId": "valid-2",
                "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "Moves": "d2d4 d7d5 g1f3",
                "Rating": "1400",
                "Themes": "fork",
            },
        ]
    )

    puzzles = list(
        load_puzzles(
            min_rating=1000,
            max_puzzles=1,
            dataset_loader=lambda *_args, **_kwargs: rows,
        )
    )

    assert len(puzzles) == 1
    assert rows.iterator_closed is True
    assert rows.closed is True


def test_preprocess_puzzle_rejects_invalid_later_solution_move():
    result = preprocess_puzzle(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        moves_str="e2e4 e7e5 e7e6",
        themes=[],
        rating=1200,
        puzzle_id="bad-line",
    )

    assert result is None
def test_package_lichess_sources_do_not_eagerly_import_datasets(monkeypatch):
    for module_name in [
        "chess_llm.sft.sources",
        "chess_llm.sft.sources.lichess_games",
        "chess_llm.sft.sources.lichess_puzzles",
        "datasets",
    ]:
        sys.modules.pop(module_name, None)

    importlib.import_module("chess_llm.sft.sources")
    importlib.import_module("chess_llm.sft.sources.lichess_games")
    importlib.import_module("chess_llm.sft.sources.lichess_puzzles")

    assert "datasets" not in sys.modules
