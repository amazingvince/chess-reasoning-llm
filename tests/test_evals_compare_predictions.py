import json
from pathlib import Path

import pytest

from chess_llm.evals.benchmark import BenchmarkExample, save_benchmark


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _planning_example(example_id: str, gold: str) -> BenchmarkExample:
    return BenchmarkExample(
        example_id=example_id,
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...\nWhat is the best move?",
        gold_answer=gold,
        metric_type="move_extraction",
        metadata={},
    )


def test_mcnemar_exact_p_value_uses_two_sided_binomial_tail():
    from chess_llm.evals.compare_predictions import mcnemar_exact_p_value

    assert mcnemar_exact_p_value(0, 0) == 1.0
    assert mcnemar_exact_p_value(0, 5) == 0.0625
    assert mcnemar_exact_p_value(1, 4) == pytest.approx(0.375)


def test_compare_predictions_scores_same_examples_with_paired_counts(tmp_path):
    from chess_llm.evals.compare_predictions import compare_prediction_files

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    examples = [
        _planning_example("planning_00000", "e2e4"),
        _planning_example("planning_00001", "d2d4"),
        _planning_example("planning_00002", "g1f3"),
        _planning_example("planning_00003", "c2c4"),
    ]
    save_benchmark(examples, benchmark_dir / "planning.jsonl")

    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline_path,
        [
            {"example_id": "planning_00000", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00001", "prediction": "<move>d2d4</move>"},
            {"example_id": "planning_00002", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00003", "prediction": "<move>e2e4</move>"},
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            {"example_id": "planning_00000", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00001", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00002", "prediction": "<move>g1f3</move>"},
            {"example_id": "planning_00003", "prediction": "<move>c2c4</move>"},
        ],
    )

    report = compare_prediction_files(
        benchmark_dir=benchmark_dir,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )

    overall = report["overall"]
    assert overall["n"] == 4
    assert overall["baseline_accuracy"] == 0.5
    assert overall["candidate_accuracy"] == 0.75
    assert overall["delta"] == 0.25
    assert overall["baseline_only"] == 1
    assert overall["candidate_only"] == 2
    assert overall["both_correct"] == 1
    assert overall["both_wrong"] == 0
    assert overall["mcnemar_p"] == 1.0
    assert report["tasks"]["planning/best_move"]["candidate_only"] == 2


def test_compare_predictions_cli_prints_and_writes_json_report(tmp_path, capsys):
    from chess_llm.evals import compare_predictions

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    save_benchmark(
        [
            _planning_example("planning_00000", "e2e4"),
            _planning_example("planning_00001", "d2d4"),
        ],
        benchmark_dir / "planning.jsonl",
    )
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output_path = tmp_path / "comparison.json"
    _write_jsonl(
        baseline_path,
        [
            {"example_id": "planning_00000", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00001", "prediction": "<move>e2e4</move>"},
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            {"example_id": "planning_00000", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00001", "prediction": "<move>d2d4</move>"},
        ],
    )

    exit_code = compare_predictions.main(
        [
            "--benchmark-dir",
            str(benchmark_dir),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output-json",
            str(output_path),
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Paired Prediction Comparison" in out
    assert "Overall" in out
    assert "candidate_only" in out
    assert payload["overall"]["candidate_accuracy"] == 1.0
    assert payload["overall"]["baseline_accuracy"] == 0.5
