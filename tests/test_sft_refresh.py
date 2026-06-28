import json
import sys
from pathlib import Path

from chess_llm.artifacts.jsonl import write_jsonl
from chess_llm.artifacts.schemas import (
    ChatMessage,
    JudgmentArtifact,
    ParsedAnswer,
    PromptArtifact,
    RolloutArtifact,
)
from chess_llm.autodata.failure_buckets import ILLEGAL_MOVE, LEGAL_UNSCORED, MISSING_FEN, PARSE_FAILURE
from chess_llm.autodata.sft_refresh import build_sft_refresh, main


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


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

    result = build_sft_refresh(prompts_path, rollouts_path, judgments_path, tmp_path / "refresh")

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

    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))
    from validation.validator import validate_example

    passed, errors = validate_example(rows[0])
    assert passed, errors


def test_illegal_move_creates_move_correction_row_from_gold_answer(tmp_path):
    prompts_path, rollouts_path, judgments_path = _write_artifacts(
        tmp_path,
        [_prompt("planning_00000")],
        [_rollout("rollout-1", "planning_00000", raw_output="<move>e2e5</move>", move_uci="e2e5")],
        [_judgment("judgment-1", "rollout-1", legal=False, failure_bucket=ILLEGAL_MOVE)],
    )

    result = build_sft_refresh(prompts_path, rollouts_path, judgments_path, tmp_path / "refresh")

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

    result = build_sft_refresh(prompts_path, rollouts_path, judgments_path, tmp_path / "refresh")

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

    result = build_sft_refresh(prompts_path, rollouts_path, judgments_path, tmp_path / "refresh")

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
        ]
    )

    assert exit_code == 0
    assert (output_dir / "tier7" / "7.4_autodata_format_repair.jsonl").exists()
    assert (output_dir / "tier7" / "7.5_autodata_move_correction.jsonl").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_repair_count"] == 1
    assert manifest["move_correction_count"] == 0
