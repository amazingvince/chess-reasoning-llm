from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _passing_results() -> dict:
    return {
        "planning": {
            "candidate_ratings": 0.80,
            "candidate_ratings_candidate_set_jaccard": 0.90,
            "candidate_ratings_best_move_match": 0.75,
            "candidate_ratings_cp_bucket_accuracy": 0.70,
            "candidate_ratings_cp_bucket_mae": 1.0,
            "step_verification": 0.75,
            "step_verification_verdict_accuracy": 0.85,
            "step_verification_faulty_line_accuracy": 0.70,
            "step_verification_error_type_accuracy": 0.75,
            "step_verification_correction_match": 0.70,
            "best_move_trace_referenced_move_accuracy": 0.92,
            "best_move_trace_step_accuracy": 0.82,
            "best_move_trace_conclusion_move_match": 0.97,
            "best_move_wpd": 0.08,
            "best_move_multipv_hit_rate": 0.80,
            "best_move_postmove_hit_rate": 0.15,
            "best_move_cache_hit_rate": 0.60,
            "best_move_reward_std_per_prompt": 0.12,
            "wpd": 0.08,
            "reward_std_per_prompt": 0.12,
        }
    }


def _passing_analysis() -> dict:
    return {
        "schema_version": "prediction_analysis.v1",
        "tasks": {
            "step_verification": {
                "metadata_breakdowns": {
                    "source_task": {"7.8_candidate_ratings": 5},
                    "error_type": {"wrong_bucket": 2, "none": 3},
                    "corruption_kind": {"candidate_rating_wrong_bucket": 2},
                }
            }
        },
    }


def test_r0_gate_builds_passing_report_from_eval_artifacts(tmp_path: Path):
    from chess_llm.training.r0_gate import build_r0_gate_report

    results_path = tmp_path / "predictions.results.json"
    analysis_path = tmp_path / "predictions.analysis.json"
    _write_json(results_path, _passing_results())
    _write_json(analysis_path, _passing_analysis())

    report = build_r0_gate_report(results_path, analysis_path)

    assert report["schema_version"] == "r0_gate_report.v1"
    assert report["artifact_type"] == "r0_gate_report"
    assert report["passed"] is True
    assert report["failure_count"] == 0
    assert report["results_path"] == str(results_path)
    assert report["analysis_path"] == str(analysis_path)
    assert report["metrics"]["candidate_ratings_best_move_match"] == 0.75
    assert report["metrics"]["reward_std_per_prompt"] == 0.12
    assert all(check["passed"] for check in report["checks"])


def test_r0_gate_reports_missing_required_measurement_sections(tmp_path: Path):
    from chess_llm.training.r0_gate import build_r0_gate_report

    results_path = tmp_path / "predictions.results.json"
    analysis_path = tmp_path / "predictions.analysis.json"
    _write_json(results_path, {"planning": {"candidate_ratings_best_move_match": 0.90}})
    _write_json(analysis_path, {"tasks": {"step_verification": {}}})

    report = build_r0_gate_report(results_path, analysis_path)
    failed_check_names = {
        check["name"] for check in report["checks"] if not check["passed"]
    }

    assert report["passed"] is False
    assert "candidate_ratings_cp_bucket_accuracy" in failed_check_names
    assert "step_verification_verdict_accuracy" in failed_check_names
    assert "step_verification_metadata_breakdowns" in failed_check_names
    assert "trace_referenced_move_accuracy" in failed_check_names
    assert "wpd_present" in failed_check_names


def test_r0_gate_cli_writes_default_report_and_respects_soft_gate(tmp_path: Path):
    from chess_llm.training import r0_gate

    results_path = tmp_path / "predictions.results.json"
    analysis_path = tmp_path / "predictions.analysis.json"
    _write_json(results_path, {"planning": {}})
    _write_json(analysis_path, {"tasks": {}})

    hard_rc = r0_gate.main(
        [
            "--results",
            str(results_path),
            "--analysis",
            str(analysis_path),
        ]
    )
    report_path = results_path.with_suffix(".r0_report.json")
    hard_report = json.loads(report_path.read_text(encoding="utf-8"))

    soft_rc = r0_gate.main(
        [
            "--results",
            str(results_path),
            "--analysis",
            str(analysis_path),
            "--soft-gate",
        ]
    )

    assert hard_rc == 1
    assert soft_rc == 0
    assert hard_report["passed"] is False
    assert hard_report["failure_count"] > 0
