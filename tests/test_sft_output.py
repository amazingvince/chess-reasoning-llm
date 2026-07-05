import importlib
import json
import sys

import chess

from chess_llm.formats import render_ascii_board
from chess_llm.sft import build_sft_row


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _clear_legacy_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "config"
            or name.startswith("config.")
            or name == "validation"
            or name.startswith("validation.")
            or name == "output"
            or name.startswith("output.")
        ):
            sys.modules.pop(name, None)


def _row(task: str = "1.1_fen_to_board") -> dict:
    assistant_content = (
        render_ascii_board(chess.Board(STARTING_FEN))
        if task == "1.1_fen_to_board"
        else "board"
    )
    return build_sft_row(
        task=task,
        tier=int(task.split(".", 1)[0]),
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=assistant_content,
    )


def test_package_output_imports_without_legacy_modules():
    _clear_legacy_modules()

    module = importlib.import_module("chess_llm.sft.output")

    assert module.JSONLWriter
    assert module.PipelineStats
    assert "validation.validator" not in sys.modules
    assert "config.settings" not in sys.modules


def test_jsonl_writer_validates_writes_and_discards_tmp_on_close(tmp_path):
    from chess_llm.sft.output import JSONLWriter

    output_path = tmp_path / "tier1" / "1.1_fen_to_board.jsonl"
    with JSONLWriter(
        output_path,
        commit_on_close=False,
        expected_task_id="1.1_fen_to_board",
    ) as writer:
        assert writer.write(_row("1.1_fen_to_board")) is True
        assert writer.write(_row("1.2_board_to_fen")) is False
        assert writer.count == 1
        assert writer.error_count == 1
        assert not output_path.exists()
        assert output_path.with_name(output_path.name + ".tmp").exists()

    assert not output_path.exists()
    assert not output_path.with_name(output_path.name + ".tmp").exists()


def test_jsonl_writer_commit_publishes_only_valid_rows(tmp_path):
    from chess_llm.sft.output import JSONLWriter

    output_path = tmp_path / "rows.jsonl"
    with JSONLWriter(output_path, expected_task_id="1.1_fen_to_board") as writer:
        assert writer.write(_row("1.1_fen_to_board"))
        assert not writer.write({"task": "bad", "fen": "not a fen", "messages": []})

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [_row("1.1_fen_to_board")]


def test_jsonl_writer_counts_malformed_rows_as_validation_errors(tmp_path):
    from chess_llm.sft.output import JSONLWriter

    output_path = tmp_path / "rows.jsonl"
    malformed = {
        "task": "1.1_fen_to_board",
        "fen": STARTING_FEN,
        "messages": [{"content": "missing role"}],
        "metadata": [],
    }

    with JSONLWriter(output_path, expected_task_id="1.1_fen_to_board") as writer:
        assert writer.write(malformed) is False
        assert writer.error_count == 1

    assert output_path.read_text(encoding="utf-8") == ""


def test_pipeline_stats_report_and_chess960_mix_use_injected_settings():
    from chess_llm.sft.output import PipelineStats

    stats = PipelineStats(
        volumes={"2.1_legal_move_gen": 10},
        chess960_ratios={2: 0.25},
    )
    stats.record("2.1_legal_move_gen", is_chess960=True)
    stats.record("2.1_legal_move_gen", passed_validation=False)
    stats.record("2.1_legal_move_gen")

    report = stats.report()

    assert "2.1_legal_move_gen" in report
    assert "       10 " in report
    assert "       3 " in report
    assert "       2 " in report
    assert "       1 " in report
    assert "33.3%" in report
    assert stats.verify_chess960_mix() == {
        "tier_2": {"target": 0.25, "actual": 0.3333, "delta": 0.0833}
    }
