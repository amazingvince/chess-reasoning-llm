"""Paired comparison for two benchmark prediction files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from chess_llm.evals.benchmark import (
    BenchmarkExample,
    load_benchmark,
    score_prediction,
)
from chess_llm.evals.run_benchmark import load_predictions

_RAW_PROTOCOL_TASK_TYPES = frozenset({
    "best_move",
    "puzzle_solve",
    "best_line_trace",
})


def mcnemar_exact_p_value(baseline_only: int, candidate_only: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant paired wins."""
    discordant = int(baseline_only) + int(candidate_only)
    if discordant <= 0:
        return 1.0

    tail_end = min(int(baseline_only), int(candidate_only))
    log_half = math.log(0.5)
    log_terms = [
        (
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            + discordant * log_half
        )
        for k in range(tail_end + 1)
    ]
    max_log = max(log_terms)
    tail_probability = math.exp(max_log) * sum(
        math.exp(term - max_log) for term in log_terms
    )
    return min(1.0, 2.0 * tail_probability)


def compare_prediction_files(
    *,
    benchmark_dir: Path,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, object]:
    """Score two prediction JSONLs on the same benchmark examples."""
    examples = load_benchmark_dir(benchmark_dir)
    baseline_scores = score_prediction_file(baseline_path, examples)
    candidate_scores = score_prediction_file(candidate_path, examples)

    splits: dict[str, dict[str, object]] = {}
    for split_name, split_examples in _group_by(examples, lambda example: example.split).items():
        splits[split_name] = paired_score_summary(
            split_examples,
            baseline_scores,
            candidate_scores,
        )

    tasks: dict[str, dict[str, object]] = {}
    for key, task_examples in _group_by(
        examples,
        lambda example: f"{example.split}/{example.task_type}",
    ).items():
        tasks[key] = paired_score_summary(
            task_examples,
            baseline_scores,
            candidate_scores,
        )

    return {
        "benchmark_dir": str(benchmark_dir),
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "overall": paired_score_summary(examples, baseline_scores, candidate_scores),
        "splits": splits,
        "tasks": tasks,
    }


def load_benchmark_dir(benchmark_dir: Path) -> list[BenchmarkExample]:
    """Load all benchmark examples from a frozen benchmark directory."""
    manifest_path = benchmark_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        split_names = sorted((manifest.get("splits") or {}).keys())
    else:
        split_names = [path.stem for path in sorted(benchmark_dir.glob("*.jsonl"))]

    examples: list[BenchmarkExample] = []
    for split_name in split_names:
        path = benchmark_dir / f"{split_name}.jsonl"
        if path.exists():
            examples.extend(load_benchmark(path))
    return examples


def score_prediction_file(
    predictions_path: Path,
    examples: Sequence[BenchmarkExample],
) -> dict[str, float]:
    """Return per-example primary scores for a prediction JSONL."""
    predictions_raw, raw_predictions_raw = load_predictions(predictions_path)
    predictions = _first_prediction_map(predictions_raw)
    raw_predictions = _first_prediction_map(raw_predictions_raw)

    scores: dict[str, float] = {}
    for example in examples:
        prediction = predictions.get(example.example_id, "")
        if example.task_type in _RAW_PROTOCOL_TASK_TYPES:
            prediction = raw_predictions.get(example.example_id, prediction)
        scores[example.example_id] = float(
            score_prediction(example, prediction).get("primary", 0.0) or 0.0
        )
    return scores


def paired_score_summary(
    examples: Sequence[BenchmarkExample],
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
) -> dict[str, object]:
    """Summarize paired per-example scores and exact McNemar counts."""
    n = len(examples)
    baseline_sum = 0.0
    candidate_sum = 0.0
    baseline_correct = 0
    candidate_correct = 0
    both_correct = 0
    both_wrong = 0
    baseline_only = 0
    candidate_only = 0

    for example in examples:
        baseline_score = float(baseline_scores.get(example.example_id, 0.0))
        candidate_score = float(candidate_scores.get(example.example_id, 0.0))
        baseline_sum += baseline_score
        candidate_sum += candidate_score

        baseline_hit = baseline_score == 1.0
        candidate_hit = candidate_score == 1.0
        baseline_correct += int(baseline_hit)
        candidate_correct += int(candidate_hit)
        if baseline_hit and candidate_hit:
            both_correct += 1
        elif baseline_hit and not candidate_hit:
            baseline_only += 1
        elif candidate_hit and not baseline_hit:
            candidate_only += 1
        else:
            both_wrong += 1

    baseline_accuracy = baseline_correct / n if n else 0.0
    candidate_accuracy = candidate_correct / n if n else 0.0
    baseline_mean = baseline_sum / n if n else 0.0
    candidate_mean = candidate_sum / n if n else 0.0
    return {
        "n": n,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta": candidate_accuracy - baseline_accuracy,
        "baseline_mean_score": baseline_mean,
        "candidate_mean_score": candidate_mean,
        "mean_score_delta": candidate_mean - baseline_mean,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "mcnemar_p": mcnemar_exact_p_value(baseline_only, candidate_only),
    }


def print_report(report: dict[str, object]) -> None:
    """Print a compact paired comparison report."""
    print("\n=== Paired Prediction Comparison ===\n")
    _print_summary("Overall", report["overall"])

    splits = report.get("splits", {})
    if isinstance(splits, dict) and splits:
        print("Splits:")
        for split_name in sorted(splits):
            _print_summary(f"  {split_name}", splits[split_name])

    tasks = report.get("tasks", {})
    if isinstance(tasks, dict) and tasks:
        print("Tasks:")
        for task_name in sorted(tasks):
            _print_summary(f"  {task_name}", tasks[task_name])


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``chess-llm-compare-predictions``."""
    parser = argparse.ArgumentParser(
        description="Compare two prediction JSONLs with paired benchmark tests"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    benchmark_dir = Path(args.benchmark_dir)
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    for label, path in (
        ("Benchmark directory", benchmark_dir),
        ("Baseline predictions", baseline_path),
        ("Candidate predictions", candidate_path),
    ):
        if not path.exists():
            print(f"[FAIL] {label} does not exist: {path}")
            return 1

    report = compare_prediction_files(
        benchmark_dir=benchmark_dir,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    print_report(report)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0


def _first_prediction_map(
    predictions: dict[str, str | list[str]],
) -> dict[str, str]:
    return {
        example_id: value[0] if isinstance(value, list) else value
        for example_id, value in predictions.items()
    }


def _group_by(
    examples: Sequence[BenchmarkExample],
    key_fn,
) -> dict[str, list[BenchmarkExample]]:
    grouped: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        grouped[key_fn(example)].append(example)
    return dict(grouped)


def _print_summary(label: str, summary: object) -> None:
    if not isinstance(summary, dict):
        return
    print(f"{label} (n={summary['n']}):")
    print(f"  {'baseline_accuracy':<24} {summary['baseline_accuracy']:>7.1%}")
    print(f"  {'candidate_accuracy':<24} {summary['candidate_accuracy']:>7.1%}")
    print(f"  {'delta':<24} {summary['delta']:>+7.1%}")
    print(f"  {'baseline_mean_score':<24} {summary['baseline_mean_score']:>7.3f}")
    print(f"  {'candidate_mean_score':<24} {summary['candidate_mean_score']:>7.3f}")
    print(f"  {'mean_score_delta':<24} {summary['mean_score_delta']:>+7.3f}")
    print(f"  {'baseline_only':<24} {summary['baseline_only']:>7}")
    print(f"  {'candidate_only':<24} {summary['candidate_only']:>7}")
    print(f"  {'both_correct':<24} {summary['both_correct']:>7}")
    print(f"  {'both_wrong':<24} {summary['both_wrong']:>7}")
    print(f"  {'mcnemar_p':<24} {summary['mcnemar_p']:>7.4f}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
