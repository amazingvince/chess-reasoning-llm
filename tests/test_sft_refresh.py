import json
import logging
from pathlib import Path

import pytest

from chess_llm.artifacts.jsonl import write_jsonl
from chess_llm.artifacts.schemas import (
    ChatMessage,
    JudgmentArtifact,
    ParsedAnswer,
    PromptArtifact,
    RolloutArtifact,
)
from chess_llm.autodata.failure_buckets import (
    ILLEGAL_MOVE,
    INVALID_FEN,
    LEGAL_UNSCORED,
    MISSING_FEN,
    PARSE_FAILURE,
)
from chess_llm.autodata.sft_refresh import build_sft_refresh, main


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_FEN = "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1"


def _blocklist_file(tmp_path: Path, fens: tuple[str, ...] = ()) -> Path:
    path = tmp_path / "blocklist.txt"
    path.write_text("".join(f"{fen}\n" for fen in fens), encoding="utf-8")
    return path


def _prompt(
    prompt_id: str,
    *,
    fen: str | None = STARTING_FEN,
    task_type: str = "best_move",
    gold_answer: str = "e2e4",
) -> PromptArtifact:
    return PromptArtifact(
        prompt_id=prompt_id,
        messages=[ChatMessage(role="user", content=f"FEN: {fen}\nChoose the best move.")],
        fen=fen,
        task_type=task_type,
        metadata={
            "split": "planning",
            "task_type": task_type,
            "gold_answer": gold_answer,
            "metric_type": "move_extraction",
        },
    )


def _rollout(
    rollout_id: str,
    prompt_id: str,
    *,
    raw_output: str = "I cannot decide.",
    move_uci: str | None = None,
    model_id: str = "model-a",
) -> RolloutArtifact:
    return RolloutArtifact(
        rollout_id=rollout_id,
        prompt_id=prompt_id,
        model_id=model_id,
        raw_output=raw_output,
        parsed_answer=ParsedAnswer(
            raw_text=raw_output,
            move_uci=move_uci,
            format_type="move_tag" if move_uci else "none",
            parse_error=None if move_uci else "no UCI move found",
        ),
    )


def _judgment(
    judgment_id: str,
    rollout_id: str,
    *,
    legal: bool | None = False,
    failure_bucket: str | None = PARSE_FAILURE,
    teacher_move_uci: str | None = None,
    regret_cp: float | None = None,
) -> JudgmentArtifact:
    return JudgmentArtifact(
        judgment_id=judgment_id,
        rollout_id=rollout_id,
        legal=legal,
        failure_bucket=failure_bucket,
        teacher_move_uci=teacher_move_uci,
        regret_cp=regret_cp,
        feedback="unit feedback",
    )


def _write_artifacts(
    tmp_path: Path,
    prompts: list[PromptArtifact],
    rollouts: list[RolloutArtifact],
    judgments: list[JudgmentArtifact],
) -> tuple[Path, Path, Path]:
    prompts_path = tmp_path / "prompts.jsonl"
    rollouts_path = tmp_path / "rollouts.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    write_jsonl(prompts_path, prompts)
    write_jsonl(rollouts_path, rollouts)
    write_jsonl(judgments_path, judgments)
    return prompts_path, rollouts_path, judgments_path


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parse_failure_creates_format_repair_row_that_passes_tier7_validation(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000")],
        [_judgment("judgment-1", "rollout-1", failure_bucket=PARSE_FAILURE)],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    rows = _read_rows(result.format_repair_path)
    assert result.format_repair_count == 1
    assert result.move_correction_count == 0
    assert rows[0]["task"] == "7.4_autodata_format_repair"
    assert rows[0]["tier"] == 7
    assert rows[0]["fen"] == STARTING_FEN
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["content"].startswith("FEN:")
    assert rows[0]["messages"][2]["content"] == (
        "<think>The previous answer did not provide a single parseable UCI move. "
        "Use the verified target move.</think>\n<move>e2e4</move>"
    )
    assert rows[0]["metadata"]["source_prompt_id"] == "planning_00000"
    assert rows[0]["metadata"]["target_source"] == "gold_answer"

    from chess_llm.sft.validation import validate_example

    passed, errors = validate_example(rows[0])
    assert passed, errors


def test_illegal_move_creates_move_correction_row_from_gold_answer(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000", raw_output="<move>e2e5</move>", move_uci="e2e5")],
        [_judgment("judgment-1", "rollout-1", legal=False, failure_bucket=ILLEGAL_MOVE)],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    rows = _read_rows(result.move_correction_path)
    assert result.format_repair_count == 0
    assert result.move_correction_count == 1
    assert rows[0]["task"] == "7.5_autodata_move_correction"
    assert rows[0]["messages"][2]["content"].endswith("<move>e2e4</move>")
    assert rows[0]["metadata"]["parsed_model_move_uci"] == "e2e5"
    assert rows[0]["metadata"]["failure_bucket"] == ILLEGAL_MOVE


def test_high_regret_legal_move_creates_move_correction_from_teacher_move(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000", gold_answer="d2d4")],
        [_rollout("rollout-1", "planning_00000", raw_output="<move>d2d4</move>", move_uci="d2d4")],
        [
            _judgment(
                "judgment-1",
                "rollout-1",
                legal=True,
                failure_bucket=None,
                teacher_move_uci="e2e4",
                regret_cp=125.0,
            )
        ],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        min_regret_cp=100.0,
        blocklist_path=_blocklist_file(tmp_path),
    )

    rows = _read_rows(result.move_correction_path)
    assert result.move_correction_count == 1
    assert rows[0]["messages"][2]["content"].endswith("<move>e2e4</move>")
    assert rows[0]["metadata"]["target_source"] == "teacher_move_uci"
    assert rows[0]["metadata"]["regret_cp"] == 125.0


def test_high_regret_legal_move_without_teacher_is_skipped(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000", gold_answer="e2e4")],
        [_rollout("rollout-1", "planning_00000", raw_output="<move>d2d4</move>", move_uci="d2d4")],
        [
            _judgment(
                "judgment-1",
                "rollout-1",
                legal=True,
                failure_bucket=None,
                teacher_move_uci=None,
                regret_cp=125.0,
            )
        ],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    assert result.move_correction_count == 0
    assert result.skipped_count == 1
    assert result.skip_reasons["missing_target_move"] == 1


def test_skips_low_signal_rows_and_records_reasons(tmp_path):
    prompts = [
        _prompt("low_regret"),
        _prompt("legal_unscored"),
        _prompt("missing_fen", fen=None),
        _prompt("non_move", task_type="legal_moves"),
        _prompt("invalid_target", gold_answer="e2e4 d2d4"),
        _prompt("duplicate_a"),
        _prompt("duplicate_b"),
    ]
    rollouts = [
        _rollout("r-low", "low_regret", move_uci="d2d4"),
        _rollout("r-legal-unscored", "legal_unscored", move_uci="e2e4"),
        _rollout("r-missing-fen", "missing_fen"),
        _rollout("r-non-move", "non_move"),
        _rollout("r-invalid-target", "invalid_target"),
        _rollout("r-duplicate-a", "duplicate_a"),
        _rollout("r-duplicate-b", "duplicate_b"),
    ]
    judgments = [
        _judgment("j-low", "r-low", legal=True, failure_bucket=None, teacher_move_uci="e2e4", regret_cp=25.0),
        _judgment("j-legal-unscored", "r-legal-unscored", legal=True, failure_bucket=LEGAL_UNSCORED),
        _judgment("j-missing-fen", "r-missing-fen", legal=None, failure_bucket=MISSING_FEN),
        _judgment("j-non-move", "r-non-move", failure_bucket=PARSE_FAILURE),
        _judgment("j-invalid-target", "r-invalid-target", failure_bucket=PARSE_FAILURE),
        _judgment("j-duplicate-a", "r-duplicate-a", failure_bucket=PARSE_FAILURE),
        _judgment("j-duplicate-b", "r-duplicate-b", failure_bucket=PARSE_FAILURE),
    ]
    prompts_path, rollouts_path, judgments_path = _write_artifacts(tmp_path, prompts, rollouts, judgments)

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    assert result.format_repair_count == 1
    assert result.move_correction_count == 0
    assert result.skipped_count == 6
    assert result.skip_reasons["low_regret"] == 1
    assert result.skip_reasons["legal_unscored"] == 1
    assert result.skip_reasons["missing_fen"] == 1
    assert result.skip_reasons["non_move_task"] == 1
    assert result.skip_reasons["missing_target_move"] == 1
    assert result.skip_reasons["duplicate"] == 1


def test_cli_writes_outputs_and_manifest(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000")],
        [_judgment("judgment-1", "rollout-1")],
    )
    output_dir = tmp_path / "refresh"

    blocklist_path = _blocklist_file(tmp_path)

    exit_code = main(
        [
            "--prompts",
            str(prompts_path),
            "--rollouts",
            str(rollouts_path),
            "--judgments",
            str(judgments_path),
            "--output-dir",
            str(output_dir),
            "--blocklist-path",
            str(blocklist_path),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "tier7" / "7.4_autodata_format_repair.jsonl").exists()
    assert (output_dir / "tier7" / "7.5_autodata_move_correction.jsonl").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_repair_count"] == 1
    assert manifest["move_correction_count"] == 0
    assert manifest["chess960"] is False
    assert manifest["blocklist_path"] == str(blocklist_path)
    assert manifest["blocklisted_count"] == 0


def test_blocklisted_fen_rows_are_dropped_and_logged(tmp_path, caplog):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000")],
        [_judgment("judgment-1", "rollout-1", failure_bucket=PARSE_FAILURE)],
    )

    with caplog.at_level(logging.WARNING, logger="chess_llm.autodata.sft_refresh"):
        result = build_sft_refresh(
            prompts_path,
            rollouts_path,
            judgments_path,
            tmp_path / "refresh",
            blocklist_path=_blocklist_file(tmp_path, (STARTING_FEN,)),
        )

    assert result.format_repair_count == 0
    assert result.move_correction_count == 0
    assert result.skip_reasons["blocklisted_fen"] == 1
    assert "Dropped 1 refresh row(s)" in caplog.text
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["blocklisted_count"] == 1
    assert manifest["skip_reasons"]["blocklisted_fen"] == 1


def test_missing_explicit_blocklist_raises(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000")],
        [_judgment("judgment-1", "rollout-1", failure_bucket=PARSE_FAILURE)],
    )

    with pytest.raises(FileNotFoundError):
        build_sft_refresh(
            prompts_path,
            rollouts_path,
            judgments_path,
            tmp_path / "refresh",
            blocklist_path=tmp_path / "does_not_exist.txt",
        )


def test_refresh_rows_include_previous_answer_in_user_prompt(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000"), _prompt("planning_00001", gold_answer="d2d4")],
        [
            _rollout("rollout-1", "planning_00000", raw_output="I cannot decide."),
            _rollout(
                "rollout-2",
                "planning_00001",
                raw_output="<move>d2d4</move>",
                move_uci="d2d4",
            ),
        ],
        [
            _judgment("judgment-1", "rollout-1", failure_bucket=PARSE_FAILURE),
            _judgment(
                "judgment-2",
                "rollout-2",
                legal=True,
                failure_bucket=None,
                teacher_move_uci="e2e4",
                regret_cp=125.0,
            ),
        ],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    repair_rows = _read_rows(result.format_repair_path)
    correction_rows = _read_rows(result.move_correction_path)
    assert repair_rows[0]["messages"][1]["content"] == (
        f"FEN: {STARTING_FEN}\nChoose the best move.\n\n"
        "Previous answer:\nI cannot decide.\n\n"
        "The previous answer was rejected. Reply with the corrected move."
    )
    assert correction_rows[0]["messages"][1]["content"] == (
        f"FEN: {STARTING_FEN}\nChoose the best move.\n\n"
        "Previous answer:\n<move>d2d4</move>\n\n"
        "The previous answer was rejected. Reply with the corrected move."
    )


def test_chess960_flag_validates_and_labels_refresh_rows(tmp_path):
    explicit_standard = _prompt("planning_00001")
    explicit_standard.metadata["is_chess960"] = False
    prompts = [
        _prompt("chess960_00000", fen=CHESS960_FEN),
        _prompt("planning_00000"),
        explicit_standard,
    ]
    rollouts = [
        _rollout("rollout-960", "chess960_00000"),
        _rollout("rollout-std", "planning_00000"),
        _rollout("rollout-explicit-std", "planning_00001"),
    ]
    judgments = [
        _judgment("judgment-960", "rollout-960", failure_bucket=PARSE_FAILURE),
        _judgment("judgment-std", "rollout-std", failure_bucket=PARSE_FAILURE),
        _judgment("judgment-explicit-std", "rollout-explicit-std", failure_bucket=PARSE_FAILURE),
    ]
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path, prompts, rollouts, judgments
    )
    blocklist_path = _blocklist_file(tmp_path)

    without_flag = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh_std",
        blocklist_path=blocklist_path,
    )
    with_flag = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh_960",
        chess960=True,
        blocklist_path=blocklist_path,
    )

    assert without_flag.skip_reasons["invalid_fen"] == 1

    assert with_flag.format_repair_count == 3
    rows = _read_rows(with_flag.format_repair_path)
    assert [row["fen"] for row in rows] == [CHESS960_FEN, STARTING_FEN, STARTING_FEN]
    # Run flag is the default; explicit prompt metadata still wins, matching
    # the judging-time convention in chess_llm.evals.batch_judge.
    assert [row["is_chess960"] for row in rows] == [True, True, False]
    manifest = json.loads(with_flag.manifest_path.read_text(encoding="utf-8"))
    assert manifest["chess960"] is True


def test_move_correction_skipped_when_target_equals_model_move(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000", raw_output="<move>d2d4</move>", move_uci="d2d4")],
        [
            _judgment(
                "judgment-1",
                "rollout-1",
                legal=True,
                failure_bucket=None,
                teacher_move_uci="d2d4",
                regret_cp=150.0,
            )
        ],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    assert result.move_correction_count == 0
    assert result.skip_reasons["target_equals_model_move"] == 1


def test_invalid_fen_judgment_bucket_is_skipped_as_data_error(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000", fen="8/8/8/8/8/8/8/8 w - - 0 1")],
        [_rollout("rollout-1", "planning_00000")],
        [_judgment("judgment-1", "rollout-1", legal=None, failure_bucket=INVALID_FEN)],
    )

    result = build_sft_refresh(
        prompts_path,
        rollouts_path,
        judgments_path,
        tmp_path / "refresh",
        blocklist_path=_blocklist_file(tmp_path),
    )

    assert result.format_repair_count == 0
    assert result.move_correction_count == 0
    assert result.skip_reasons["invalid_fen"] == 1
