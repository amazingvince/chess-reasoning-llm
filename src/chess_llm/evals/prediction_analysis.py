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
from chess_llm.formats.answers import strip_fen_strings

PREDICTION_ANALYSIS_SCHEMA_VERSION = "prediction_analysis.v1"
_PREDICTION_FAILURE_EXAMPLE_LIMIT = 5
_PREDICTION_FORMAT_BLEED_LIMIT = 20
_PREDICTION_PREFIX_LIMIT = 10
_PREDICTION_EXCERPT_CHARS = 240
_LEGAL_MOVES_BY_PIECE_DIAGNOSTIC_KEYS = frozenset({
    "section_completeness",
    "all_legal_line_present",
    "piece_inventory_accuracy",
    "all_moves_precision",
    "all_moves_recall",
    "all_moves_jaccard",
    "per_piece_group_jaccard",
    "illegal_extra_count",
    "missing_move_count",
})
_MOVE_EDIT_TRACE_TASK_TYPES = frozenset({
    "move_square_edits",
    "fen_assembly",
    "state_tracking",
})
_THINK_MOVE_PROTOCOL_TASK_TYPES = frozenset({
    "best_move",
    "puzzle_solve",
    "best_line_trace",
})
_TRACE_METRIC_KEYS = frozenset({
    "trace_referenced_move_accuracy",
    "trace_referenced_move_count",
    "trace_illegal_referenced_move_count",
    "trace_candidate_count",
    "trace_line_depth",
    "trace_backtrack_count",
    "trace_step_accuracy",
    "trace_step_count",
    "trace_conclusion_move_match",
})
_STEP_VERIFICATION_METADATA_KEYS = (
    "source_task",
    "error_type",
    "corruption_kind",
    "difficulty",
)
_PUZZLE_RATING_BUCKETS = (
    (None, 800, "<800"),
    (800, 1200, "800-1199"),
    (1200, 1600, "1200-1599"),
    (1600, 2000, "1600-1999"),
    (2000, 2400, "2000-2399"),
    (2400, 2800, "2400-2799"),
    (2800, None, "2800+"),
)
_PUZZLE_RATING_BUCKET_ORDER = tuple(label for _low, _high, label in _PUZZLE_RATING_BUCKETS)


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
    puzzle_strata = _new_puzzle_strata()
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
            _update_puzzle_strata(puzzle_strata, row, primary_score)

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
        "puzzle_strata": _finalize_puzzle_strata(puzzle_strata),
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
    if _UCI_RE.search(strip_fen_strings(normalize_prediction(text).lower())):
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
    if example.metadata:
        enriched.setdefault("metadata", dict(example.metadata))
    if "score" not in enriched or _prediction_score_needs_refresh(enriched, example):
        enriched["score"] = score_prediction(
            example,
            str(enriched.get("raw_prediction", enriched.get("prediction", ""))),
        )
    return enriched


def _prediction_score_needs_refresh(
    row: Mapping[str, Any],
    example: BenchmarkExample,
) -> bool:
    score = row.get("score")
    if not isinstance(score, Mapping):
        return True
    if example.task_type in _THINK_MOVE_PROTOCOL_TASK_TYPES:
        return not _TRACE_METRIC_KEYS.issubset(score.keys())
    if example.task_type != "legal_moves_by_piece":
        return False
    return not _LEGAL_MOVES_BY_PIECE_DIAGNOSTIC_KEYS.issubset(score.keys())


def _new_prediction_analysis_bucket() -> dict[str, Any]:
    return {
        "row_count": 0,
        "example_ids": set(),
        "score_sum": 0.0,
        "scored_count": 0,
        "score_metric_sums": defaultdict(float),
        "score_metric_counts": defaultdict(int),
        "failure_count": 0,
        "format_bleed_count": 0,
        "format_families": Counter(),
        "prefixes": Counter(),
        "metadata_breakdowns": defaultdict(Counter),
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
    score = row.get("score")
    if isinstance(score, Mapping):
        for key, value in score.items():
            if key == "primary" or not isinstance(value, (int, float)):
                continue
            bucket["score_metric_sums"][str(key)] += float(value)
            bucket["score_metric_counts"][str(key)] += 1
    if is_failure:
        bucket["failure_count"] += 1
        if len(bucket["failure_examples"]) < _PREDICTION_FAILURE_EXAMPLE_LIMIT:
            bucket["failure_examples"].append(_prediction_report_example(row, family=family))
    if is_format_bleed:
        bucket["format_bleed_count"] += 1
    bucket["format_families"][family] += 1
    if prefix:
        bucket["prefixes"][prefix] += 1
    if row.get("task_type") == "step_verification":
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        for key in _STEP_VERIFICATION_METADATA_KEYS:
            value = metadata.get(key, row.get(key))
            if isinstance(value, (str, int)) and str(value):
                bucket["metadata_breakdowns"][key][str(value)] += 1


def _finalize_prediction_analysis_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    scored_count = int(bucket["scored_count"])
    metric_sums: Mapping[str, float] = bucket["score_metric_sums"]
    metric_counts: Mapping[str, int] = bucket["score_metric_counts"]
    metadata_breakdowns: Mapping[str, Counter] = bucket["metadata_breakdowns"]
    return {
        "row_count": int(bucket["row_count"]),
        "example_count": len(bucket["example_ids"]),
        "scored_count": scored_count,
        "primary_accuracy": (
            bucket["score_sum"] / scored_count if scored_count else None
        ),
        "score_metrics": {
            key: metric_sums[key] / metric_counts[key]
            for key in sorted(metric_sums)
            if metric_counts[key]
        },
        "failure_count": int(bucket["failure_count"]),
        "format_bleed_count": int(bucket["format_bleed_count"]),
        "format_families": dict(sorted(bucket["format_families"].items())),
        "prefixes": dict(bucket["prefixes"].most_common(_PREDICTION_PREFIX_LIMIT)),
        "metadata_breakdowns": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(metadata_breakdowns.items())
        },
        "failure_examples": list(bucket["failure_examples"]),
    }


def _new_puzzle_strata() -> dict[str, Any]:
    return {
        "rating_buckets": defaultdict(_new_stratum_bucket),
        "themes": defaultdict(_new_stratum_bucket),
        "missing_rating_count": 0,
        "missing_themes_count": 0,
    }


def _new_stratum_bucket() -> dict[str, Any]:
    return {
        "row_count": 0,
        "example_ids": set(),
        "score_sum": 0.0,
        "scored_count": 0,
    }


def _update_puzzle_strata(
    strata: dict[str, Any],
    row: Mapping[str, Any],
    primary_score: float | None,
) -> None:
    if row.get("task_type") != "puzzle_solve":
        return

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}

    rating = _coerce_puzzle_rating(metadata.get("rating", row.get("rating")))
    rating_bucket = _puzzle_rating_bucket(rating)
    if rating_bucket is None:
        strata["missing_rating_count"] += 1
    else:
        _update_stratum_bucket(
            strata["rating_buckets"][rating_bucket],
            row,
            primary_score,
        )

    themes = _normalize_puzzle_themes(metadata.get("themes", row.get("themes")))
    if not themes:
        strata["missing_themes_count"] += 1
    for theme in themes:
        _update_stratum_bucket(strata["themes"][theme], row, primary_score)


def _update_stratum_bucket(
    bucket: dict[str, Any],
    row: Mapping[str, Any],
    primary_score: float | None,
) -> None:
    bucket["row_count"] += 1
    example_id = row.get("example_id")
    if example_id:
        bucket["example_ids"].add(str(example_id))
    if primary_score is None:
        return
    bucket["score_sum"] += primary_score
    bucket["scored_count"] += 1


def _finalize_puzzle_strata(strata: Mapping[str, Any]) -> dict[str, Any]:
    rating_buckets: Mapping[str, dict[str, Any]] = strata["rating_buckets"]
    themes: Mapping[str, dict[str, Any]] = strata["themes"]
    return {
        "rating_buckets": {
            label: _finalize_stratum_bucket(rating_buckets[label])
            for label in _PUZZLE_RATING_BUCKET_ORDER
            if label in rating_buckets
        },
        "themes": {
            theme: _finalize_stratum_bucket(bucket)
            for theme, bucket in sorted(themes.items(), key=lambda item: item[0].lower())
        },
        "missing_rating_count": int(strata["missing_rating_count"]),
        "missing_themes_count": int(strata["missing_themes_count"]),
    }


def _finalize_stratum_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    scored_count = int(bucket["scored_count"])
    return {
        "row_count": int(bucket["row_count"]),
        "example_count": len(bucket["example_ids"]),
        "scored_count": scored_count,
        "primary_accuracy": (
            float(bucket["score_sum"]) / scored_count if scored_count else None
        ),
    }


def _coerce_puzzle_rating(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _puzzle_rating_bucket(rating: int | None) -> str | None:
    if rating is None:
        return None
    for low, high, label in _PUZZLE_RATING_BUCKETS:
        if (low is None or rating >= low) and (high is None or rating < high):
            return label
    return None


def _normalize_puzzle_themes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        return []
    return sorted({item.strip() for item in items if item.strip()})


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
    if row.get("score"):
        result["score"] = row["score"]
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
        for marker in ("lookup:", "squares:", "ranks:")
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
