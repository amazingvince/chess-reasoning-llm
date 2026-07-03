"""R-0 measurement-readiness gate for chess reasoning traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


R0_GATE_SCHEMA_VERSION = "r0_gate_report.v1"
R0_GATE_ARTIFACT_TYPE = "r0_gate_report"


@dataclass(frozen=True)
class GateCheckSpec:
    """One numeric R-0 gate check."""

    name: str
    metric_key: str
    threshold: float | None = None
    max_value: float | None = None


NUMERIC_CHECKS: tuple[GateCheckSpec, ...] = (
    GateCheckSpec(
        "candidate_ratings_best_move_match",
        "candidate_ratings_best_move_match",
        threshold=0.70,
    ),
    GateCheckSpec(
        "candidate_ratings_candidate_set_jaccard",
        "candidate_ratings_candidate_set_jaccard",
        threshold=0.70,
    ),
    GateCheckSpec(
        "candidate_ratings_cp_bucket_accuracy",
        "candidate_ratings_cp_bucket_accuracy",
        threshold=0.65,
    ),
    GateCheckSpec(
        "candidate_ratings_cp_bucket_mae",
        "candidate_ratings_cp_bucket_mae",
        max_value=1.25,
    ),
    GateCheckSpec(
        "step_verification_verdict_accuracy",
        "step_verification_verdict_accuracy",
        threshold=0.80,
    ),
    GateCheckSpec(
        "step_verification_faulty_line_accuracy",
        "step_verification_faulty_line_accuracy",
        threshold=0.65,
    ),
    GateCheckSpec(
        "step_verification_error_type_accuracy",
        "step_verification_error_type_accuracy",
        threshold=0.70,
    ),
    GateCheckSpec(
        "trace_referenced_move_accuracy",
        "trace_referenced_move_accuracy",
        threshold=0.90,
    ),
    GateCheckSpec(
        "trace_step_accuracy",
        "trace_step_accuracy",
        threshold=0.80,
    ),
    GateCheckSpec(
        "trace_conclusion_move_match",
        "trace_conclusion_move_match",
        threshold=0.95,
    ),
)

WPD_REQUIRED_KEYS = (
    "wpd",
    "multipv_hit_rate",
    "postmove_hit_rate",
    "cache_hit_rate",
    "reward_std_per_prompt",
)


def build_r0_gate_report(
    results_path: str | Path,
    analysis_path: str | Path,
) -> dict[str, Any]:
    """Build an R-0 gate report from eval results and prediction analysis artifacts."""
    results_file = Path(results_path)
    analysis_file = Path(analysis_path)
    results = _read_json_object(results_file)
    analysis = _read_json_object(analysis_file)
    metrics = _flatten_results_metrics(results)

    checks: list[dict[str, Any]] = []
    for spec in NUMERIC_CHECKS:
        checks.append(_numeric_check(spec, metrics))
    checks.append(_step_verification_metadata_check(analysis))
    checks.extend(_wpd_checks(metrics))

    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": R0_GATE_SCHEMA_VERSION,
        "artifact_type": R0_GATE_ARTIFACT_TYPE,
        "passed": not failed,
        "failure_count": len(failed),
        "checks": checks,
        "metrics": metrics,
        "results_path": str(results_file),
        "analysis_path": str(analysis_file),
    }


def write_r0_gate_report(
    results_path: str | Path,
    analysis_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Write the R-0 gate report and return its path."""
    results_file = Path(results_path)
    output_file = (
        Path(output_path)
        if output_path is not None
        else results_file.with_suffix(".r0_report.json")
    )
    report = build_r0_gate_report(results_file, analysis_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_file


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for ``chess-llm-r0-report``."""
    parser = argparse.ArgumentParser(
        description="Build an R-0 measurement-readiness gate report."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--soft-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the R-0 gate report."""
    args = build_arg_parser().parse_args(argv)
    output_path = write_r0_gate_report(
        args.results,
        args.analysis,
        args.output,
    )
    report = _read_json_object(output_path)
    _print_report(report, output_path)
    if report.get("passed") or args.soft_gate:
        return 0
    return 1


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _flatten_results_metrics(results: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name, split_metrics in results.items():
        if not isinstance(split_metrics, Mapping):
            continue
        for key, value in split_metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            metric_key = str(key)
            metrics.setdefault(metric_key, float(value))
            metrics[f"{split_name}.{metric_key}"] = float(value)
    return metrics


def _numeric_check(
    spec: GateCheckSpec,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    value = _find_metric(metrics, spec.metric_key)
    if value is None:
        return {
            "name": spec.name,
            "passed": False,
            "metric": spec.metric_key,
            "value": None,
            "reason": "missing_metric",
        }
    if spec.threshold is not None:
        passed = value >= spec.threshold
        return {
            "name": spec.name,
            "passed": passed,
            "metric": spec.metric_key,
            "value": value,
            "threshold": spec.threshold,
            "comparison": ">=",
        }
    if spec.max_value is not None:
        passed = value <= spec.max_value
        return {
            "name": spec.name,
            "passed": passed,
            "metric": spec.metric_key,
            "value": value,
            "threshold": spec.max_value,
            "comparison": "<=",
        }
    raise ValueError(f"Gate check {spec.name!r} has no threshold")


def _step_verification_metadata_check(analysis: Mapping[str, Any]) -> dict[str, Any]:
    breakdowns = (
        analysis.get("tasks", {})
        if isinstance(analysis.get("tasks"), Mapping)
        else {}
    )
    step_task = (
        breakdowns.get("step_verification")
        if isinstance(breakdowns, Mapping)
        else None
    )
    metadata_breakdowns = (
        step_task.get("metadata_breakdowns", {})
        if isinstance(step_task, Mapping)
        else {}
    )
    required = ("source_task", "error_type", "corruption_kind")
    missing = [
        key
        for key in required
        if not isinstance(metadata_breakdowns, Mapping)
        or not isinstance(metadata_breakdowns.get(key), Mapping)
        or not metadata_breakdowns.get(key)
    ]
    return {
        "name": "step_verification_metadata_breakdowns",
        "passed": not missing,
        "required": list(required),
        "missing": missing,
    }


def _wpd_checks(metrics: Mapping[str, float]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for key in WPD_REQUIRED_KEYS:
        value = _find_metric(metrics, key)
        checks.append(
            {
                "name": "wpd_present" if key == "wpd" else key,
                "passed": value is not None,
                "metric": key,
                "value": value,
                "reason": "missing_metric" if value is None else "present",
            }
        )
    return checks


def _find_metric(metrics: Mapping[str, float], suffix: str) -> float | None:
    if suffix in metrics:
        return metrics[suffix]
    matches = [
        value
        for key, value in sorted(metrics.items())
        if key.endswith(f"_{suffix}") or key.endswith(f".{suffix}")
    ]
    if matches:
        return matches[0]
    return None


def _print_report(report: Mapping[str, Any], output_path: Path) -> None:
    status = "PASS" if report.get("passed") else "FAIL"
    print(f"R-0 gate: {status} ({report.get('failure_count', 0)} failure(s))")
    print(f"Report: {output_path}")
    for check in report.get("checks", []):
        if not isinstance(check, Mapping):
            continue
        if check.get("passed"):
            continue
        print(f"  [FAIL] {check.get('name')}: {check.get('reason', 'threshold')}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "R0_GATE_ARTIFACT_TYPE",
    "R0_GATE_SCHEMA_VERSION",
    "build_arg_parser",
    "build_r0_gate_report",
    "main",
    "write_r0_gate_report",
]
