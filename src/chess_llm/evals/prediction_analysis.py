"""Lightweight diagnostics for benchmark prediction JSONL files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from chess_llm.core.legality import parse_legality_yes_no_answer
from chess_llm.evals.benchmark import (
    BenchmarkExample,
    _UCI_RE,
    _extract_canonical_fen,
    normalize_prediction,
    score_prediction,
)

PREDICTION_ANALYSIS_SCHEMA_VERSION = "prediction_analysis.v1"
_PREDICTION_FAILURE_EXAMPLE_LIMIT = 5
_PREDICTION_FORMAT_BLEED_LIMIT = 20
_PREDICTION_PREFIX_LIMIT = 10
_PREDICTION_EXCERPT_CHARS = 240
_MOVE_EDIT_TRACE_TASK_TYPES = frozenset({
    "move_square_edits",
    "fen_assembly",
    "state_tracking",
})
_THINK_MOVE_PROTOCOL_TASK_TYPES = frozenset({"best_move", "puzzle_solve"})


def write_prediction_analysis_report(
    predictions_path: Path,
    output_path: Path | None = None,
    *,
    examples_by_id: Mapping[str, BenchmarkExample] | None = None,
) -> Path:
    """Write a diagnostic summary for an eval predictions JSONL file."""
    predictions_path = Path(predictions_path)
    output_path = output_path or predictions_path.with_suffix(".analysis.json")
    report = analyze_prediction_file(predictions_path, examples_by_id=examples_by_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def analyze_prediction_file(
    predictions_path: Path,
    *,
    examples_by_id: Mapping[str, BenchmarkExample] | None = None,
) -> dict[str, Any]:
    """Summarize prediction failures and output-format families by task."""
    predictions_path = Path(predictions_path)
    task_buckets: dict[str, dict[str, Any]] = defaultdict(_new_prediction_analysis_bucket)
    split_buckets: dict[str, dict[str, Any]] = defaultdict(_new_prediction_analysis_bucket)
    format_bleed: list[dict[str, Any]] = []
    total_rows = 0
    example_ids: set[str] = set()

    with predictions_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = _enrich_prediction_row(
                json.loads(line),
                examples_by_id=examples_by_id,
            )
            total_rows += 1
            example_id = str(row.get("example_id", ""))
            if example_id:
                example_ids.add(example_id)

            task_type = str(row.get("task_type") or "unknown")
            split = str(row.get("split") or "unknown")
            prediction = str(row.get("raw_prediction", row.get("prediction", "")))
            family = prediction_format_family(prediction)
            prefix = _prediction_prefix(prediction)
            primary_score = _prediction_primary_score(row)
            is_failure = primary_score is not None and primary_score < 1.0
            is_bleed = is_prediction_format_bleed(task_type, family)

            for bucket in (task_buckets[task_type], split_buckets[split]):
                _update_prediction_analysis_bucket(
                    bucket,
                    row,
                    family=family,
                    prefix=prefix,
                    primary_score=primary_score,
                    is_failure=is_failure,
                    is_format_bleed=is_bleed,
                )

            if is_bleed and len(format_bleed) < _PREDICTION_FORMAT_BLEED_LIMIT:
                format_bleed.append(_prediction_report_example(row, family=family))

    return {
        "schema_version": PREDICTION_ANALYSIS_SCHEMA_VERSION,
        "predictions_path": str(predictions_path),
        "row_count": total_rows,
        "example_count": len(example_ids),
        "tasks": {
            task_type: _finalize_prediction_analysis_bucket(bucket)
            for task_type, bucket in sorted(task_buckets.items())
        },
        "splits": {
            split: _finalize_prediction_analysis_bucket(bucket)
            for split, bucket in sorted(split_buckets.items())
        },
        "format_bleed": format_bleed,
    }


def prediction_format_family(prediction: str) -> str:
    """Classify a prediction into the broad output family it resembles."""
    text = prediction.strip()
    if not text:
        return "empty"

    lower_text = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if "<think" in lower_text or "<move" in lower_text:
        return "think_move_protocol"
    if _looks_like_fen_row_rewrite_trace(lines):
        return "fen_row_rewrite_trace"
    if _looks_like_square_edit_trace(lines):
        return "square_edit_trace"
    if _UCI_RE.search(normalize_prediction(text).lower()):
        return "uci_moves"
    if parse_legality_yes_no_answer(normalize_prediction(text)) is not None:
        return "legality_answer"
    if _extract_canonical_fen(text, chess960=False) is not None:
        if lower_text.startswith("result fen:"):
            return "result_fen"
        return "fen"
    if _looks_like_board_render(lines):
        return "board_render"
    return "text"


def is_prediction_format_bleed(task_type: str, family: str) -> bool:
    """Return True when a task receives another task family's answer shape."""
    if family == "fen_row_rewrite_trace" and task_type != "fen_row_application":
        return True
    if family == "square_edit_trace" and task_type not in _MOVE_EDIT_TRACE_TASK_TYPES:
        return True
    if family == "think_move_protocol" and task_type not in _THINK_MOVE_PROTOCOL_TASK_TYPES:
        return True
    return False


def _enrich_prediction_row(
    row: dict[str, Any],
    *,
    examples_by_id: Mapping[str, BenchmarkExample] | None,
) -> dict[str, Any]:
    if not examples_by_id:
        return row

    example_id = row.get("example_id")
    example = examples_by_id.get(str(example_id)) if example_id is not None else None
    if example is None:
        return row

    enriched = dict(row)
    enriched.setdefault("split", example.split)
    enriched.setdefault("task_type", example.task_type)
    enriched.setdefault("metric_type", example.metric_type)
    enriched.setdefault("fen", example.fen)
    enriched.setdefault("prompt", example.prompt)
    enriched.setdefault("gold_answer", example.gold_answer)
    if "score" not in enriched:
        enriched["score"] = score_prediction(
            example,
            str(enriched.get("raw_prediction", enriched.get("prediction", ""))),
        )
    return enriched


def _new_prediction_analysis_bucket() -> dict[str, Any]:
    return {
        "row_count": 0,
        "example_ids": set(),
        "score_sum": 0.0,
        "scored_count": 0,
        "failure_count": 0,
        "format_bleed_count": 0,
        "format_families": Counter(),
        "prefixes": Counter(),
        "failure_examples": [],
    }


def _update_prediction_analysis_bucket(
    bucket: dict[str, Any],
    row: Mapping[str, Any],
    *,
    family: str,
    prefix: str,
    primary_score: float | None,
    is_failure: bool,
    is_format_bleed: bool,
) -> None:
    bucket["row_count"] += 1
    example_id = row.get("example_id")
    if example_id:
        bucket["example_ids"].add(str(example_id))
    if primary_score is not None:
        bucket["score_sum"] += primary_score
        bucket["scored_count"] += 1
    if is_failure:
        bucket["failure_count"] += 1
        if len(bucket["failure_examples"]) < _PREDICTION_FAILURE_EXAMPLE_LIMIT:
            bucket["failure_examples"].append(_prediction_report_example(row, family=family))
    if is_format_bleed:
        bucket["format_bleed_count"] += 1
    bucket["format_families"][family] += 1
    if prefix:
        bucket["prefixes"][prefix] += 1


def _finalize_prediction_analysis_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    scored_count = int(bucket["scored_count"])
    return {
        "row_count": int(bucket["row_count"]),
        "example_count": len(bucket["example_ids"]),
        "scored_count": scored_count,
        "primary_accuracy": (
            bucket["score_sum"] / scored_count if scored_count else None
        ),
        "failure_count": int(bucket["failure_count"]),
        "format_bleed_count": int(bucket["format_bleed_count"]),
        "format_families": dict(sorted(bucket["format_families"].items())),
        "prefixes": dict(bucket["prefixes"].most_common(_PREDICTION_PREFIX_LIMIT)),
        "failure_examples": list(bucket["failure_examples"]),
    }


def _prediction_report_example(
    row: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "example_id": row.get("example_id"),
        "sample_index": row.get("sample_index"),
        "split": row.get("split"),
        "task_type": row.get("task_type"),
        "metric_type": row.get("metric_type"),
        "format_family": family,
        "primary_score": _prediction_primary_score(row),
        "prediction_excerpt": _prediction_excerpt(
            str(row.get("raw_prediction", row.get("prediction", "")))
        ),
        "gold_excerpt": _prediction_excerpt(str(row.get("gold_answer", ""))),
    }
    if row.get("diagnostics"):
        result["diagnostics"] = row["diagnostics"]
    return result


def _prediction_primary_score(row: Mapping[str, Any]) -> float | None:
    score = row.get("score")
    if not isinstance(score, Mapping):
        return None
    value = score.get("primary")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _looks_like_fen_row_rewrite_trace(lines: list[str]) -> bool:
    return (
        bool(lines)
        and lines[0].lower().startswith("rows:")
        and "->" in lines[0]
        and any(line.lower().startswith("result fen:") for line in lines)
    )


def _looks_like_square_edit_trace(lines: list[str]) -> bool:
    if not lines:
        return False
    source_or_destination = sum(
        1
        for line in lines
        if line.lower().startswith(("source ", "destination "))
    )
    rank_edits = sum(1 for line in lines if line.lower().startswith("rank "))
    result_fen = any(line.lower().startswith("result fen:") for line in lines)
    legacy_trace = any(
        line.lower().startswith(("lookup:", "squares:", "ranks:", "move "))
        for line in lines
    )
    inline_trace = any(
        marker in " ".join(lines).lower()
        for marker in ("lookup:", "squares:", "ranks:", "result fen:")
    )
    return (
        source_or_destination + rank_edits >= 2
        or result_fen and source_or_destination + rank_edits >= 1
        or legacy_trace
        or inline_trace and result_fen
    )


def _looks_like_board_render(lines: list[str]) -> bool:
    if len(lines) < 4:
        return False
    board_like_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("+", "|")) or any(piece in stripped for piece in "KQRBNPkqrbnp"):
            board_like_lines += 1
    return board_like_lines >= 4


def _prediction_prefix(prediction: str) -> str:
    for line in prediction.splitlines():
        prefix = line.strip()
        if prefix:
            return _prediction_excerpt(prefix, max_chars=80)
    return ""


def _prediction_excerpt(
    value: str,
    *,
    max_chars: int = _PREDICTION_EXCERPT_CHARS,
) -> str:
    compact = " ".join(value.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
