"""Split-level benchmark scoring orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chess_llm.evals.benchmark import (
    BenchmarkExample,
    format_compliance,
    legal_move_rate,
    pass_at_k,
)


def score_evaluation_splits(
    *,
    split_examples: Mapping[str, list[BenchmarkExample]],
    predictions: Mapping[str, Sequence[str]],
    sampled_predictions: Mapping[str, Sequence[str]],
    flat_preds: Mapping[str, str],
    flat_raw_preds: Mapping[str, str],
    acpl_splits: set[str],
    output_path: Path,
    stockfish_path: str,
    acpl_depth: int,
    acpl_workers: int,
    pass_k: int,
    open_stockfish: Callable[[str], chess.engine.SimpleEngine | None],
    resolve_stockfish_path: Callable[[str], str | None],
    compute_acpl: Callable[..., dict[str, float]],
    compute_wpd: Callable[..., dict[str, dict[str, object]]],
    score_split_with_protocol_predictions: Callable[
        [
            list[BenchmarkExample],
            Mapping[str, str],
            Mapping[str, str],
            dict[str, float] | None,
            dict[str, dict[str, object]] | None,
        ],
        dict[str, float],
    ],
    example_is_chess960: Callable[[BenchmarkExample], bool],
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, float]], dict[str, int], bool]:
    """Score each benchmark split and close Stockfish even when scoring fails."""
    engine: chess.engine.SimpleEngine | None = None
    has_acpl = False
    if acpl_splits:
        engine = open_stockfish(stockfish_path)
        has_acpl = engine is not None
        if engine is not None:
            logger.info("ACPL enabled for splits: %s", ", ".join(sorted(acpl_splits)))
    else:
        logger.info(
            "ACPL disabled for this run. It is computed by default only for "
            "Phase C planning eval; use --full-acpl-report for all splits."
        )

    acpl_engine_factory: Callable[[], chess.engine.SimpleEngine] | None = None
    if engine is not None and acpl_workers > 1:
        resolved_stockfish_path = resolve_stockfish_path(stockfish_path)
        if resolved_stockfish_path is not None:
            acpl_engine_factory = (
                lambda path=resolved_stockfish_path: chess.engine.SimpleEngine.popen_uci(
                    path
                )
            )

    split_results: dict[str, dict[str, float]] = {}
    split_counts: dict[str, int] = {}

    try:
        for split_name, examples in split_examples.items():
            acpl_scores: dict[str, float] | None = None
            wpd_scores: dict[str, dict[str, object]] | None = None
            if engine and split_name in acpl_splits:
                eval_cache_path = output_path.with_suffix(".multipv.sqlite")
                acpl_scores = compute_acpl(
                    engine,
                    examples,
                    flat_preds,
                    acpl_depth,
                    cache_path=eval_cache_path,
                    workers=acpl_workers,
                    engine_factory=acpl_engine_factory,
                )
                wpd_predictions: dict[str, Any] = dict(flat_preds)
                if sampled_predictions:
                    wpd_predictions = {}
                    for ex in examples:
                        per_prompt_preds: list[str] = []
                        if ex.example_id in flat_preds:
                            per_prompt_preds.append(flat_preds[ex.example_id])
                        per_prompt_preds.extend(sampled_predictions.get(ex.example_id, []))
                        if len(per_prompt_preds) == 1:
                            wpd_predictions[ex.example_id] = per_prompt_preds[0]
                        elif per_prompt_preds:
                            wpd_predictions[ex.example_id] = per_prompt_preds
                wpd_scores = compute_wpd(
                    engine,
                    examples,
                    wpd_predictions,
                    depth=acpl_depth,
                    cache_path=eval_cache_path,
                )

            metrics = score_split_with_protocol_predictions(
                examples,
                flat_preds,
                flat_raw_preds,
                acpl_scores,
                wpd_scores,
            )

            puzzle_examples = [e for e in examples if e.task_type == "puzzle_solve"]
            if puzzle_examples and pass_k > 1:
                # Preserve the historical pass@k behavior from evaluate.py:
                # greedy first, followed by sampled candidates.
                for k in (1, pass_k):
                    hits = 0
                    for ex in puzzle_examples:
                        preds = list(predictions.get(ex.example_id, [""]))
                        preds.extend(sampled_predictions.get(ex.example_id, []))
                        hits += pass_at_k(preds[:k], ex.gold_answer)
                    metrics[f"puzzle_pass_at_{k}"] = hits / len(puzzle_examples)

            planning_preds = [
                (
                    ex,
                    flat_raw_preds.get(ex.example_id, flat_preds.get(ex.example_id, "")),
                )
                for ex in examples
                if ex.task_type in ("best_move", "puzzle_solve", "best_line_trace")
            ]
            if planning_preds:
                fc_scores = [format_compliance(p) for _, p in planning_preds]
                lm_raw_scores = [
                    legal_move_rate(
                        p,
                        ex.fen,
                        chess960=example_is_chess960(ex),
                    )
                    for ex, p in planning_preds
                ]
                lm_scores = [0.0 if v is None else v for v in lm_raw_scores]
                metrics["format_compliance"] = sum(fc_scores) / len(fc_scores)
                metrics["legal_move_rate"] = sum(lm_scores) / len(lm_scores)
                metrics["missing_move_tag_count"] = float(
                    sum(1 for v in lm_raw_scores if v is None)
                )

            split_results[split_name] = metrics
            split_counts[split_name] = len(examples)
    finally:
        if engine:
            engine.quit()

    return split_results, split_counts, has_acpl
