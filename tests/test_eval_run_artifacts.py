import json
from pathlib import Path


def test_finalize_evaluation_artifacts_appends_ledger_and_mirrors_sidecars(
    tmp_path: Path,
):
    from chess_llm.artifacts.eval_runs import finalize_evaluation_artifacts

    output = tmp_path / "eval_predictions.jsonl"
    results = output.with_suffix(".results.json")
    analysis = output.with_suffix(".analysis.json")
    eval_run = output.with_suffix(".eval_run.json")
    multipv = output.with_suffix(".multipv.sqlite")
    for path, text in (
        (output, '{"example_id":"planning_00000"}\n'),
        (results, '{"planning":{"best_move":1.0}}\n'),
        (analysis, '{"tasks":{}}\n'),
        (multipv, "sqlite bytes"),
    ):
        path.write_text(text, encoding="utf-8")
    eval_run.write_text(
        json.dumps(
            {
                "schema_version": "artifact.v1",
                "artifact_type": "evaluation_run",
                "run_id": "eval-test",
                "created_at_utc": "2026-07-05T00:00:00+00:00",
                "model_id": "model-id",
                "phase": "c",
                "benchmark_dir": "benchmark",
                "benchmark_manifest_path": "benchmark/manifest.json",
                "benchmark_version": "unit-v1",
                "predictions_path": str(output),
                "results_path": str(results),
                "return_code": 0,
                "split_counts": {"planning": 1},
                "has_acpl": False,
                "n_failures": 0,
                "inference": {},
                "scoring": {},
                "gate": {},
                "metadata": {"prediction_analysis_path": str(analysis)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = tmp_path / "runs" / "runs.jsonl"
    mirror_root = tmp_path / "mirror"

    result = finalize_evaluation_artifacts(
        eval_run,
        ledger_path=ledger,
        mirror_dir=mirror_root,
    )

    ledger_rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mirror_dir = mirror_root / "eval-test"

    assert result["run_id"] == "eval-test"
    assert len(ledger_rows) == 1
    assert ledger_rows[0]["run_id"] == "eval-test"
    assert ledger_rows[0]["metadata"]["artifact_mirror_dir"] == str(mirror_dir)
    assert (mirror_dir / output.name).read_text(encoding="utf-8") == output.read_text(
        encoding="utf-8"
    )
    assert (mirror_dir / results.name).exists()
    assert (mirror_dir / analysis.name).exists()
    assert (mirror_dir / eval_run.name).exists()
    assert (mirror_dir / multipv.name).exists()
    mirrored_eval_run = json.loads((mirror_dir / eval_run.name).read_text(encoding="utf-8"))
    assert mirrored_eval_run["metadata"]["artifact_mirror_dir"] == str(mirror_dir)
    assert "eval_run" in mirrored_eval_run["metadata"]["mirrored_artifacts"]


def test_finalize_evaluation_artifacts_appends_without_mirror(tmp_path: Path):
    import chess_llm.artifacts as artifacts
    from chess_llm.artifacts.eval_runs import finalize_evaluation_artifacts

    assert artifacts.finalize_evaluation_artifacts is finalize_evaluation_artifacts

    output = tmp_path / "predictions.jsonl"
    eval_run = output.with_suffix(".eval_run.json")
    eval_run.write_text(
        json.dumps(
            {
                "schema_version": "artifact.v1",
                "artifact_type": "evaluation_run",
                "run_id": "eval-no-mirror",
                "created_at_utc": "2026-07-05T00:00:00+00:00",
                "model_id": "model-id",
                "phase": None,
                "benchmark_dir": "benchmark",
                "benchmark_manifest_path": None,
                "benchmark_version": "unknown",
                "predictions_path": str(output),
                "results_path": str(output.with_suffix(".results.json")),
                "return_code": 1,
                "split_counts": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    ledger = tmp_path / "runs.jsonl"
    finalize_evaluation_artifacts(eval_run, ledger_path=ledger)
    finalize_evaluation_artifacts(eval_run, ledger_path=ledger)

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]

    assert [row["run_id"] for row in rows] == ["eval-no-mirror", "eval-no-mirror"]
