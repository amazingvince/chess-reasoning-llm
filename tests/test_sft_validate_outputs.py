from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import chess

from chess_llm.formats import render_ascii_board
from chess_llm.sft.examples import build_sft_row


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _starting_board_answer() -> str:
    return render_ascii_board(chess.Board(STARTING_FEN))


def _clear_legacy_validate_outputs_modules() -> None:
    for module_name in (
        "config.settings",
        "pool.eval_split",
        "validation.validator",
        "validation.decontamination",
        "validation.completeness",
        "scripts.validate_outputs",
        "sft.make_data.scripts.validate_outputs",
    ):
        sys.modules.pop(module_name, None)


def test_package_validate_outputs_imports_without_legacy_script_module():
    _clear_legacy_validate_outputs_modules()

    module = importlib.import_module("chess_llm.sft.validate_outputs")

    assert module.main
    assert module.validate_outputs
    for legacy_name in (
        "config.settings",
        "pool.eval_split",
        "validation.validator",
        "validation.decontamination",
        "validation.completeness",
        "scripts.validate_outputs",
        "sft.make_data.scripts.validate_outputs",
    ):
        assert legacy_name not in sys.modules
    assert "scripts.validate_outputs" not in sys.modules
    assert "sft.make_data.scripts.validate_outputs" not in sys.modules


def test_package_validate_outputs_import_does_not_set_hf_home(monkeypatch):
    sys.modules.pop("chess_llm.sft.validate_outputs", None)
    monkeypatch.delenv("HF_HOME", raising=False)

    importlib.import_module("chess_llm.sft.validate_outputs")

    assert "HF_HOME" not in sys.modules
    assert "HF_HOME" not in __import__("os").environ


def test_package_validate_outputs_missing_output_dir_returns_failure(tmp_path: Path):
    from chess_llm.sft import validate_outputs

    result = validate_outputs.validate_outputs(
        tmp_path / "missing",
        skip_completeness=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 1
    assert result.total_files == 0
    assert result.total_examples == 0
    assert result.total_errors == 1


def test_package_validate_outputs_accepts_valid_jsonl_when_optional_audits_skipped(
    tmp_path: Path,
):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        skip_completeness=True,
        skip_decontamination=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 0
    assert result.total_examples == 1
    assert result.total_errors == 0
    stats_key = str(Path("tier1") / "1.1_fen_to_board.jsonl").replace("\\", "/")
    assert result.file_stats[stats_key]["count"] == 1


def test_package_validate_outputs_missing_blocklist_fails_by_default(
    tmp_path: Path,
):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        skip_completeness=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 1
    assert result.total_examples == 1
    assert result.total_errors == 1


def test_package_validate_outputs_reports_json_parse_errors(tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    task_path = output_dir / "tier1" / "bad.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text("{not-json}\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        skip_completeness=True,
        skip_decontamination=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 1
    assert result.total_errors == 1
    stats_key = str(Path("tier1") / "bad.jsonl").replace("\\", "/")
    assert result.file_stats[stats_key]["errors"] == 1


def test_package_validate_outputs_reports_malformed_rows_without_crashing(tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    task_path = output_dir / "tier1" / "malformed.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        json.dumps(
            {
                "task": "1.1_fen_to_board",
                "fen": STARTING_FEN,
                "messages": [{"content": "missing role"}],
                "metadata": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_outputs.validate_outputs(
        output_dir,
        skip_completeness=True,
        skip_decontamination=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 1
    assert result.total_examples == 1
    assert result.total_errors == 1
    stats_key = str(Path("tier1") / "malformed.jsonl").replace("\\", "/")
    assert result.file_stats[stats_key]["errors"] == 1


def test_package_validate_outputs_completeness_honors_expected_volume(tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        expected_volume=2,
        skip_decontamination=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 1
    assert result.completeness_issues["tier1/1.1_fen_to_board.jsonl"] == (
        "underfilled: 1 / 2"
    )


def test_package_validate_outputs_completeness_can_scope_to_tiers(tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        expected_volume=1,
        tiers=[1],
        skip_decontamination=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 1
    assert "tier1/1.2_board_to_fen.jsonl" in result.completeness_issues
    assert not any(path.startswith("tier2/") for path in result.completeness_issues)


def test_package_validate_outputs_ignores_hidden_cache_jsonl_files(tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache_path = output_dir / ".training_cache" / "tier1" / "1.1_fen_to_board.stripped.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{not-json}\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        skip_completeness=True,
        skip_decontamination=True,
        blocklist_path=tmp_path / "missing_blocklist.txt",
    )

    assert result.exit_code == 0
    assert result.total_files == 1
    assert result.total_examples == 1
    assert "tier1/1.1_fen_to_board.jsonl" in result.file_stats
    assert not any(".training_cache" in path for path in result.file_stats)


def test_package_validate_outputs_counts_each_contaminated_fen(tmp_path: Path):
    from chess_llm.core.board import variant_fen_key
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    rows = [
        build_sft_row(
            task="1.1_fen_to_board",
            tier=1,
            fen=STARTING_FEN,
            user_prompt=f"Render board {idx}.",
            assistant_content=_starting_board_answer(),
        )
        for idx in range(2)
    ]
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text(variant_fen_key(STARTING_FEN) + "\n", encoding="utf-8")

    result = validate_outputs.validate_outputs(
        output_dir,
        skip_completeness=True,
        blocklist_path=blocklist_path,
    )

    assert result.exit_code == 1
    assert result.total_errors == 2
    assert len(next(iter(result.contamination.values()))) == 2


def test_package_validate_outputs_main_accepts_argv(monkeypatch, tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(validate_outputs, "EVAL_SPLITS_DIR", tmp_path / "splits")

    assert (
        validate_outputs.main(
            [
                "--output-dir",
                str(output_dir),
                "--skip-completeness",
                "--skip-decontamination",
            ]
        )
        == 0
    )


def test_package_validate_outputs_main_accepts_blocklist_path(tmp_path: Path):
    from chess_llm.core.board import variant_fen_key
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    task_path = output_dir / "tier1" / "1.1_fen_to_board.jsonl"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text(variant_fen_key(STARTING_FEN) + "\n", encoding="utf-8")

    assert (
        validate_outputs.main(
            [
                "--output-dir",
                str(output_dir),
                "--skip-completeness",
                "--blocklist",
                str(blocklist_path),
            ]
        )
        == 1
    )


def test_package_validate_outputs_main_accepts_tier_scope(monkeypatch, tmp_path: Path):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(validate_outputs, "EVAL_SPLITS_DIR", tmp_path / "splits")

    assert (
        validate_outputs.main(
            [
                "--output-dir",
                str(output_dir),
                "--expected-volume",
                "1",
                "--tier",
                "1",
                "--skip-decontamination",
            ]
        )
        == 1
    )


def test_package_validate_outputs_main_prints_legacy_summary(
    monkeypatch,
    capsys,
    tmp_path: Path,
):
    from chess_llm.sft import validate_outputs

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    monkeypatch.setattr(validate_outputs, "EVAL_SPLITS_DIR", tmp_path / "splits")

    assert (
        validate_outputs.main(
            [
                "--output-dir",
                str(output_dir),
                "--skip-completeness",
                "--skip-decontamination",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "Validation Summary" in out
    assert "Total files:" in out
    assert "Total examples:" in out


def test_legacy_validate_outputs_import_aliases_package_module(monkeypatch):
    package_module = importlib.import_module("chess_llm.sft.validate_outputs")
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    _clear_legacy_validate_outputs_modules()

    legacy_module = importlib.import_module("scripts.validate_outputs")

    assert legacy_module is package_module
