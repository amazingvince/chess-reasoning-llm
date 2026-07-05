"""Helpers for preparing already-loaded sources for held-out eval splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from chess_llm.sft.step_verification import (
    StepVerificationLabel,
    corrupt_candidate_ratings_best_line,
    format_step_verification_answer,
    number_trace_lines,
    sound_step_verification_label,
)
from chess_llm.sft.context import raw_is_chess960
from chess_llm.sft.eval_split import partition_eco_codes
from chess_llm.sft.settings import DEFAULT_MIN_DEPTH_EVAL_BENCHMARK

# FEN-pool sources that must never seed eval splits.  Self-play positions
# depend on the current checkpoint, so letting them into eval splits would
# churn the frozen benchmark (and its manifest) between harvest runs.
EVAL_SPLIT_EXCLUDED_SOURCES: frozenset[str] = frozenset({"self_play"})


@dataclass(frozen=True)
class EvalSplitSourcePreparation:
    """Prepared source map plus partition details used by eval split generation."""

    sources: dict[str, list[dict]]
    eval_openings: list[dict]
    train_openings: list[dict]
    eval_benchmark_evals: list[dict]
    standard_fen_pool: list[dict]
    chess960_fen_pool: list[dict]


def build_eval_split_sources(
    config: dict,
    *,
    min_depth_eval_benchmark: int = DEFAULT_MIN_DEPTH_EVAL_BENCHMARK,
    opening_holdout_fraction: float = 0.15,
    seed: int = 42,
    mutate_config: bool = False,
    include_chess960_in_standard_splits: bool = False,
) -> EvalSplitSourcePreparation:
    """Build the source map consumed by ``generate_all_eval_splits``."""
    all_openings = config.get("openings", [])
    if all_openings:
        eval_openings_raw, train_openings = partition_eco_codes(
            all_openings,
            holdout_fraction=opening_holdout_fraction,
            seed=seed,
        )
    else:
        eval_openings_raw, train_openings = [], []

    if mutate_config:
        config["openings"] = train_openings

    book_moves = config.get("book_moves", {})
    eval_openings: list[dict] = []
    for opening in eval_openings_raw:
        row = dict(opening)
        fen = row.get("fen", "")
        if fen in book_moves:
            row["book_moves"] = book_moves[fen]
        eval_openings.append(row)

    all_evals = config.get("position_evals", [])
    eval_benchmark_evals = [
        row
        for row in all_evals
        if row.get("depth", 0) >= min_depth_eval_benchmark
    ]

    fen_pool = [
        row
        for row in config.get("fen_pool", [])
        if not _is_eval_split_excluded(row)
    ]
    chess960_fen_pool = [row for row in fen_pool if raw_is_chess960(row)]
    if include_chess960_in_standard_splits:
        standard_fen_pool = list(fen_pool)
    else:
        standard_fen_pool = [row for row in fen_pool if not raw_is_chess960(row)]

    sources = {
        "perception": standard_fen_pool,
        "rules": standard_fen_pool,
        "tactics": config.get("puzzles", []),
        "evaluation": eval_benchmark_evals,
        "openings": eval_openings,
        "endgames": config.get("endgame_positions", []),
        "planning": _planning_eval_sources(config),
        "chess960": chess960_fen_pool,
        "mate": config.get("mate_rows", []),
    }

    return EvalSplitSourcePreparation(
        sources=sources,
        eval_openings=eval_openings,
        train_openings=train_openings,
        eval_benchmark_evals=eval_benchmark_evals,
        standard_fen_pool=standard_fen_pool,
        chess960_fen_pool=chess960_fen_pool,
    )


def reserve_training_rows_for_volume(
    split_sizes: Mapping[str, int],
    prepared_sources: EvalSplitSourcePreparation,
    *,
    volume_override: int | None,
) -> dict[str, int]:
    """Cap smoke eval split sizes while leaving FEN candidates for training."""
    adjusted = dict(split_sizes)
    if volume_override is None:
        return adjusted

    _reserve_shared_source_budget(
        adjusted,
        split_names=("perception", "rules"),
        source_count=len(prepared_sources.standard_fen_pool),
        reserve_count=volume_override,
    )
    _reserve_shared_source_budget(
        adjusted,
        split_names=("chess960",),
        source_count=len(prepared_sources.chess960_fen_pool),
        reserve_count=volume_override,
    )
    return adjusted


def _is_eval_split_excluded(row: object) -> bool:
    if not isinstance(row, Mapping):
        return False
    return row.get("source") in EVAL_SPLIT_EXCLUDED_SOURCES


def _planning_eval_sources(config: Mapping[str, object]) -> list[dict]:
    """Return planning eval candidates with MultiPV task coverage."""
    sources: list[dict] = []
    candidate_rows = config.get("candidate_rating_evals", [])
    if isinstance(candidate_rows, Sequence) and not isinstance(candidate_rows, (str, bytes)):
        sources.extend(_multipv_planning_eval_sources(candidate_rows))
    for key in ("puzzles", "best_move_evals"):
        rows = config.get(key, [])
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            sources.extend(row for row in rows if isinstance(row, dict))
    return sources


def _multipv_planning_eval_sources(rows: Sequence[object]) -> list[dict]:
    """Map cached MultiPV rows onto distinct planning benchmark task types."""
    sources: list[dict] = []
    task_index = 0
    for row in rows:
        if not isinstance(row, Mapping) or not _has_candidate_ratings(row):
            continue
        task_slot = task_index % 3
        task_index += 1
        if task_slot == 0:
            candidate_row = dict(row)
            candidate_row["task_type"] = "candidate_ratings"
            sources.append(candidate_row)
        elif task_slot == 1:
            best_line_row = dict(row)
            best_line_row["task_type"] = "best_line_trace"
            best_line_row["best_line_trace"] = True
            sources.append(best_line_row)
        else:
            verifier_row = _candidate_rating_verifier_source(row)
            if verifier_row is not None:
                sources.append(verifier_row)
    return sources


def _has_candidate_ratings(row: Mapping[str, object]) -> bool:
    ratings = row.get("candidate_ratings")
    if not isinstance(ratings, list):
        ratings = row.get("move_evaluations")
    return isinstance(ratings, list) and len(ratings) >= 5


def _candidate_rating_verifier_source(row: Mapping[str, object]) -> dict | None:
    from chess_llm.evals.benchmark import format_candidate_ratings_answer

    try:
        clean_trace = format_candidate_ratings_answer(
            list(row.get("candidate_ratings") or row.get("move_evaluations") or [])[:5]
        )
    except (TypeError, ValueError):
        return None
    if not clean_trace:
        return None
    corrupted = corrupt_candidate_ratings_best_line(clean_trace)
    if corrupted is None:
        trace = clean_trace
        label = sound_step_verification_label()
        corruption_kind = "candidate_rating_sound"
        difficulty = "sound"
    else:
        trace, label = corrupted
        corruption_kind = "candidate_rating_wrong_best"
        difficulty = "hard"
    verifier_row = dict(row)
    verifier_row["task_type"] = "step_verification"
    verifier_row["verification_trace"] = number_trace_lines(trace)
    verifier_row["expected_answer"] = format_step_verification_answer(label)
    verifier_row["metadata"] = _step_verification_metadata(
        verifier_row.get("metadata"),
        clean_trace=clean_trace,
        trace=trace,
        label=label,
        corruption_kind=corruption_kind,
        difficulty=difficulty,
    )
    return verifier_row


def _step_verification_metadata(
    existing: object,
    *,
    clean_trace: str,
    trace: str,
    label: StepVerificationLabel,
    corruption_kind: str,
    difficulty: str,
) -> dict:
    metadata = dict(existing) if isinstance(existing, Mapping) else {}
    metadata.update(
        {
            "source": "step_verification",
            "source_task": "7.8_candidate_ratings",
            "source_trace": clean_trace,
            "displayed_trace": trace,
            "corruption_kind": corruption_kind,
            "difficulty": difficulty,
            "verification_verdict": label.verdict,
            "faulty_line": "none" if label.faulty_line is None else label.faulty_line,
            "error_type": label.error_type,
            "correction": label.correction,
            "expected_answer": format_step_verification_answer(label),
        }
    )
    return metadata


def _reserve_shared_source_budget(
    split_sizes: dict[str, int],
    *,
    split_names: Sequence[str],
    source_count: int,
    reserve_count: int,
) -> None:
    eval_budget = max(0, source_count - reserve_count)
    for split_name in split_names:
        if split_name not in split_sizes:
            continue
        target = split_sizes[split_name]
        split_sizes[split_name] = min(target, eval_budget)
        eval_budget -= split_sizes[split_name]


__all__ = [
    "EVAL_SPLIT_EXCLUDED_SOURCES",
    "EvalSplitSourcePreparation",
    "build_eval_split_sources",
    "reserve_training_rows_for_volume",
]
