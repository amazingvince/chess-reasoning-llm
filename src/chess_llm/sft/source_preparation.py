"""Helpers for preparing already-loaded sources for held-out eval splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from chess_llm.sft.context import raw_is_chess960
from chess_llm.sft.eval_split import partition_eco_codes
from chess_llm.sft.settings import DEFAULT_MIN_DEPTH_EVAL_BENCHMARK


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

    fen_pool = config.get("fen_pool", [])
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
        "planning": config.get("puzzles", []) + config.get("best_move_evals", []),
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
    "EvalSplitSourcePreparation",
    "build_eval_split_sources",
    "reserve_training_rows_for_volume",
]
