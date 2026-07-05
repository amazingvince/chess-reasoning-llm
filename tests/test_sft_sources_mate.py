import importlib
import json
import sys
import zipfile
from pathlib import Path

import chess

from chess_llm.sft.sources.mate import load_mate, process_mate_row, validate_uci


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_process_mate_row_rejects_contradictory_label_and_uci():
    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
        "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
        "output": "MoveA:d2d4",
    }

    assert process_mate_row(row) is None


def test_process_mate_row_rejects_parseable_but_invalid_fen():
    row = {
        "input": 'The FEN of the given chess board is "8/8/8/8/8/8/8/8 w - - 0 1". '
        "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
        "output": "MoveA:e2e4",
    }

    assert process_mate_row(row) is None


def test_process_mate_row_accepts_matching_output_move():
    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
        "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
        "output": "MoveB:d2d4",
    }

    result = process_mate_row(row)

    assert result is not None
    assert result["move_a"] == "e2e4"
    assert result["move_b"] == "d2d4"
    assert result["better_move"] == "d2d4"


def test_process_mate_row_preserves_strategy_and_tactic_annotations():
    row = {
        "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
        "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
        "output": "MoveA:e2e4",
        "strategy": "control the center",
        "tactic": "central pawn push",
        "strategy_a": "claim central space",
        "strategy_b": "queen pawn alternative",
        "source_subset": "strategy",
    }

    result = process_mate_row(row)

    assert result is not None
    assert result["strategy"] == "control the center"
    assert result["tactic"] == "central pawn push"
    assert result["strategy_a"] == "claim central space"
    assert result["strategy_b"] == "queen pawn alternative"
    assert result["source_subset"] == "strategy"


def test_validate_uci_rejects_illegal_moves():
    board = chess.Board(STARTING_FEN)

    assert validate_uci(board, "e1e5") is None


def test_load_mate_uses_injected_downloader_and_counts_valid_rows(tmp_path):
    archive = _mate_archive(
        tmp_path / "mate.zip",
        {
            "data/rows.jsonl": [
                {"input": "bad", "output": "bad"},
                {
                    "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                    "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
                    "output": "MoveA:e2e4",
                    "strategy": "take the center",
                },
                {
                    "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                    "Which move is better? MoveA:g1f3 MoveB:c2c4 ",
                    "output": "MoveB:c2c4",
                    "tactic_b": "space gain",
                },
            ],
        },
    )
    calls = []

    def fake_downloader(repo_id: str, filename: str, *, repo_type: str):
        calls.append((repo_id, filename, repo_type))
        return str(archive)

    rows = list(load_mate(max_rows=1, downloader=fake_downloader, zip_files=["both.zip"]))

    assert calls == [("OutFlankShu/MATE_DATASET", "both.zip", "dataset")]
    assert rows == [
        {
            "fen": STARTING_FEN,
            "move_a": "e2e4",
            "move_b": "d2d4",
            "better_move": "e2e4",
            "strategy": "take the center",
        }
    ]


def test_load_mate_skips_bad_archives_and_malformed_jsonl(tmp_path):
    good_archive = _mate_archive(
        tmp_path / "good.zip",
        {
            "__MACOSX/ignored.jsonl": [
                {
                    "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                    "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
                    "output": "MoveA:e2e4",
                }
            ],
            "nested/__MACOSX/ignored.jsonl": [
                {
                    "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                    "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
                    "output": "MoveA:e2e4",
                }
            ],
            "rows.jsonl": [
                "{bad json",
                ["not", "a", "dict"],
                {
                    "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
                    "Which move is better? MoveA:g1f3 MoveB:c2c4 ",
                    "output": "MoveA:g1f3",
                    "source_subset": "no_explain",
                },
            ],
        },
    )
    bad_archive = tmp_path / "bad.zip"
    bad_archive.write_text("not a zip", encoding="utf-8")

    def fake_downloader(_repo_id: str, filename: str, *, repo_type: str):
        return str(bad_archive if filename == "bad.zip" else good_archive)

    rows = list(load_mate(downloader=fake_downloader, zip_files=["bad.zip", "good.zip"]))

    assert len(rows) == 1
    assert rows[0]["better_move"] == "g1f3"
    assert rows[0]["source_subset"] == "no_explain"


def test_load_mate_zero_limit_does_not_download():
    def fail_downloader(*_args, **_kwargs):
        raise AssertionError("downloader should not be called")

    assert list(load_mate(max_rows=0, downloader=fail_downloader)) == []
def test_package_mate_loader_does_not_eagerly_import_huggingface_hub(monkeypatch):
    for module_name in [
        "chess_llm.sft.sources.mate",
        "chess_llm.sft.sources",
        "huggingface_hub",
    ]:
        sys.modules.pop(module_name, None)

    importlib.import_module("chess_llm.sft.sources.mate")
    importlib.import_module("chess_llm.sft.sources")

    assert "huggingface_hub" not in sys.modules


def _mate_archive(path: Path, files: dict[str, list[dict | str]]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, rows in files.items():
            content = "\n".join(
                row if isinstance(row, str) else json.dumps(row)
                for row in rows
            )
            archive.writestr(name, content)
    return path
