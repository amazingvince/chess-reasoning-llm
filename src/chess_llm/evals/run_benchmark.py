"""CLI and helpers for scoring predictions against frozen benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import chess
import chess.engine

from chess_llm.evals.benchmark import (
    ACPL_INVALID_MOVE_PENALTY,
    BenchmarkExample,
    centipawn_loss,
    example_is_chess960,
    extract_move,
    format_compliance,
    legal_move_rate,
    load_benchmark,
    pass_at_k,
    score_split,
)
from chess_llm.evals.prediction_analysis import write_prediction_analysis_report
from chess_llm.external.multipv import SqliteMultipvCache, score_move_wpd

logger = logging.getLogger(__name__)

_MOVE_TASK_TYPES = frozenset({
    "best_move",
    "puzzle_solve",
    "best_line_trace",
    "endgame_best_move",
})


def load_predictions(
    path: Path,
) -> tuple[dict[str, str | list[str]], dict[str, str | list[str]]]:
    """Load model predictions from JSONL.

    Returns ``(predictions, raw_predictions)``: the normalized ``prediction``
    field used for answer accuracy, and the ``raw_prediction`` field (falling
    back to ``prediction``) used for protocol metrics such as format
    compliance.
    """
    predictions: dict[str, list[str]] = defaultdict(list)
    raw_predictions: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            predictions[row["example_id"]].append(row["prediction"])
            raw_predictions[row["example_id"]].append(
                row.get("raw_prediction", row["prediction"])
            )

    def _flatten(values: dict[str, list[str]]) -> dict[str, str | list[str]]:
        return {
            example_id: entries[0] if len(entries) == 1 else entries
            for example_id, entries in values.items()
        }

    return _flatten(predictions), _flatten(raw_predictions)


def example_is_chess960(example: BenchmarkExample) -> bool:
    """Return True when benchmark metadata marks an example as Chess960."""
    metadata = example.metadata or {}
    return bool(metadata.get("is_chess960") or metadata.get("chess960_id") is not None)


_example_is_chess960 = example_is_chess960


def evaluate_predicted_move(
    engine,
    fen: str,
    uci_move: str,
    depth: int = 20,
    chess960: bool = False,
) -> int | None:
    """Evaluate the position after a predicted move with Stockfish."""
    try:
        board = chess.Board(fen, chess960=chess960)
        original_turn = board.turn
        move = chess.Move.from_uci(uci_move)
        if uci_move not in {legal_move.uci() for legal_move in board.legal_moves}:
            return None
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(original_turn)
        return score.score(mate_score=10000)
    except Exception:
        return None


_evaluate_predicted_move = evaluate_predicted_move


def _cached_evaluate_predicted_move(
    engine,
    cache: SqliteMultipvCache | None,
    fen: str,
    uci_move: str,
    depth: int,
    chess960: bool = False,
) -> int | None:
    if cache is not None:
        hit, cp = cache.get_scalar_evaluation(
            kind="post_move",
            fen=fen,
            chess960=chess960,
            depth=depth,
            move_uci=uci_move,
            pov="original_side_to_move",
        )
        if hit:
            return cp
    cp = evaluate_predicted_move(
        engine,
        fen,
        uci_move,
        depth,
        chess960=chess960,
    )
    if cache is not None:
        cache.put_scalar_evaluation(
            kind="post_move",
            fen=fen,
            chess960=chess960,
            depth=depth,
            move_uci=uci_move,
            pov="original_side_to_move",
            cp=cp,
        )
    return cp


def white_cp_to_side_to_move_cp(
    fen: str,
    cp: float | int,
    chess960: bool = False,
) -> float:
    """Convert a White-perspective cp label to side-to-move perspective."""
    board = chess.Board(fen, chess960=chess960)
    value = float(cp)
    return value if board.turn == chess.WHITE else -value


_white_cp_to_side_to_move_cp = white_cp_to_side_to_move_cp


def compute_acpl(
    engine,
    examples: list[BenchmarkExample],
    flat_preds: dict[str, str],
    depth: int = 20,
    cache_path: Path | None = None,
) -> dict[str, float]:
    """Compute per-example centipawn loss for move-prediction tasks."""
    acpl_scores: dict[str, float] = {}
    cache = SqliteMultipvCache(cache_path) if cache_path is not None else None
    evaluated = 0

    for example in examples:
        if example.task_type not in _MOVE_TASK_TYPES:
            continue
        gold_cp = example.metadata.get("cp")
        if gold_cp is None:
            continue
        chess960 = example_is_chess960(example)

        prediction = flat_preds.get(example.example_id, "")
        uci = extract_move(prediction)
        if uci is None:
            acpl_scores[example.example_id] = ACPL_INVALID_MOVE_PENALTY
            continue

        predicted_cp = _cached_evaluate_predicted_move(
            engine,
            cache,
            example.fen,
            uci,
            depth,
            chess960=chess960,
        )
        if predicted_cp is None:
            acpl_scores[example.example_id] = ACPL_INVALID_MOVE_PENALTY
            continue

        gold_side_cp = white_cp_to_side_to_move_cp(
            example.fen,
            gold_cp,
            chess960=chess960,
        )
        acpl_scores[example.example_id] = centipawn_loss(gold_side_cp, float(predicted_cp))
        evaluated += 1

    logger.info("ACPL: evaluated %d positions with Stockfish", evaluated)
    return acpl_scores


def compute_wpd(
    engine,
    examples: list[BenchmarkExample],
    flat_preds: dict[str, str],
    *,
    depth: int = 20,
    cache_path: Path | None = None,
    multipv: int = 5,
    pv_len: int = 8,
) -> dict[str, dict[str, object]]:
    """Compute WPD diagnostics using cached MultiPV top-N first."""
    cache = SqliteMultipvCache(cache_path) if cache_path is not None else None
    wpd_scores: dict[str, dict[str, object]] = {}
    for example in examples:
        if example.task_type not in _MOVE_TASK_TYPES:
            continue
        prediction = flat_preds.get(example.example_id, "")
        uci = extract_move(prediction)
        if uci is None:
            wpd_scores[example.example_id] = {
                "wpd": None,
                "best_expectation": None,
                "predicted_expectation": 0.0,
                "reward": 0.0,
                "reward_bucket": "missing_move",
                "multipv_hit": False,
                "postmove_hit": False,
                "cache_hit": False,
            }
            continue
        diagnostics = score_move_wpd(
            engine,
            example.fen,
            uci,
            depth=depth,
            k=multipv,
            pv_len=pv_len,
            cache=cache,
            chess960=example_is_chess960(example),
        )
        wpd_scores[example.example_id] = diagnostics.to_metric_dict()
    return wpd_scores


def print_report(
    version: str,
    split_results: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    has_acpl: bool = False,
    split_coverage: dict[str, tuple[int, int]] | None = None,
    uncovered_splits: list[str] | None = None,
) -> None:
    """Print benchmark evaluation report."""
    print(f"\n=== Benchmark Evaluation ({version}) ===\n")
    split_order = [
        "perception", "rules", "tactics", "evaluation",
        "openings", "endgames", "planning", "chess960", "mate",
    ]
    uncovered = set(uncovered_splits or [])

    for split_name in split_order:
        if split_name in uncovered:
            count = split_counts.get(split_name, 0)
            print(
                f"{split_name.capitalize()} ({count} examples): "
                "UNCOVERED (0 predictions) -- excluded from scoring and ACPL\n"
            )
            continue
        if split_name not in split_results:
            continue
        metrics = split_results[split_name]
        count = split_counts.get(split_name, 0)
        print(f"{split_name.capitalize()} ({count} examples):")
        if split_coverage and split_name in split_coverage:
            covered, total = split_coverage[split_name]
            print(f"  {'coverage':<35} {covered:>4}/{total}")
        for key, value in sorted(metrics.items()):
            if key in ("overall", "acpl"):
                continue
            if isinstance(value, float):
                if "acpl" in key:
                    print(f"  {key:<35} {value:>7.1f} cp")
                else:
                    print(f"  {key:<35} {value:>7.1%}")
        overall = metrics.get("overall")
        if overall is not None:
            print(f"  {'overall':<35} {overall:>7.1%}")
        acpl = metrics.get("acpl")
        if acpl is not None:
            print(f"  {'ACPL':<35} {acpl:>7.1f} cp")
        print()

    if not has_acpl:
        print("Note: ACPL not computed (pass --stockfish-path to enable)\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``chess-llm-run-benchmark``."""
    parser = argparse.ArgumentParser(
        description="Score model predictions against frozen benchmark"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--acpl-depth", type=int, default=20)
    args = parser.parse_args(argv)

    benchmark_dir = Path(args.benchmark_dir)
    predictions_path = Path(args.predictions)
    if not benchmark_dir.exists():
        print(f"[FAIL] Benchmark directory does not exist: {benchmark_dir}")
        return 1
    if not predictions_path.exists():
        print(f"[FAIL] Predictions file does not exist: {predictions_path}")
        return 1

    manifest_path = benchmark_dir / "manifest.json"
    manifest: dict = {}
    version = "unknown"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        version = manifest.get("version", "unknown")

    predictions_raw, raw_predictions_raw = load_predictions(predictions_path)
    flat_predictions = {
        example_id: value if isinstance(value, str) else value[0]
        for example_id, value in predictions_raw.items()
    }
    flat_raw_predictions = {
        example_id: value if isinstance(value, str) else value[0]
        for example_id, value in raw_predictions_raw.items()
    }

    engine = None
    has_acpl = False
    if args.stockfish_path:
        stockfish_path = Path(args.stockfish_path)
        if stockfish_path.exists():
            try:
                engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
                has_acpl = True
                logger.info("Stockfish loaded for ACPL from %s", stockfish_path)
            except Exception as exc:
                print(f"[WARN] Could not load Stockfish: {exc}")

    split_results: dict[str, dict[str, float]] = {}
    split_counts: dict[str, int] = {}
    split_coverage: dict[str, tuple[int, int]] = {}
    uncovered_splits: list[str] = []
    examples_by_id: dict[str, BenchmarkExample] = {}
    manifest_splits = manifest.get("splits", {})
    score_splits = (
        sorted(manifest_splits.keys())
        if manifest_splits
        else [path.stem for path in sorted(benchmark_dir.glob("*.jsonl"))]
    )

    try:
        for split_name in score_splits:
            jsonl_path = benchmark_dir / f"{split_name}.jsonl"
            if not jsonl_path.exists():
                continue
            examples = load_benchmark(jsonl_path)
            if not examples:
                continue
            examples_by_id.update({example.example_id: example for example in examples})

            covered = sum(
                1 for example in examples if example.example_id in flat_predictions
            )
            missing = len(examples) - covered
            split_coverage[split_name] = (covered, len(examples))
            if covered == 0:
                uncovered_splits.append(split_name)
                split_counts[split_name] = len(examples)
                print(
                    f"[WARN] split '{split_name}': 0/{len(examples)} examples have "
                    "predictions; split is uncovered and excluded from scoring and ACPL"
                )
                continue
            if missing:
                print(
                    f"[WARN] split '{split_name}': MISSING predictions for "
                    f"{missing}/{len(examples)} examples; they score 0"
                )

            acpl_scores = None
            wpd_scores = None
            if engine is not None:
                cache_path = predictions_path.with_suffix(".multipv.sqlite")
                acpl_scores = compute_acpl(
                    engine,
                    examples,
                    flat_predictions,
                    args.acpl_depth,
                    cache_path=cache_path,
                )
                wpd_scores = compute_wpd(
                    engine,
                    examples,
                    flat_predictions,
                    depth=args.acpl_depth,
                    cache_path=cache_path,
                )

            scoring_predictions = dict(flat_predictions)
            for example in examples:
                if example.task_type in ("best_move", "puzzle_solve", "best_line_trace"):
                    raw = flat_raw_predictions.get(example.example_id)
                    if raw is not None:
                        scoring_predictions[example.example_id] = raw
            metrics = score_split(examples, scoring_predictions, acpl_scores, wpd_scores)

            puzzle_examples = [example for example in examples if example.task_type == "puzzle_solve"]
            if puzzle_examples:
                for k in (1, 8):
                    hits = 0
                    for example in puzzle_examples:
                        value = predictions_raw.get(example.example_id, "")
                        candidates = value if isinstance(value, list) else [value]
                        hits += pass_at_k(candidates[:k], example.gold_answer)
                    metrics[f"puzzle_pass_at_{k}"] = hits / len(puzzle_examples)

            planning_predictions = [
                (
                    example,
                    flat_raw_predictions.get(
                        example.example_id,
                        flat_predictions.get(example.example_id, ""),
                    ),
                )
                for example in examples
                if example.task_type in ("best_move", "puzzle_solve", "best_line_trace")
            ]
            if planning_predictions:
                format_scores = [format_compliance(pred) for _, pred in planning_predictions]
                raw_legal_scores = [
                    legal_move_rate(
                        pred,
                        example.fen,
                        chess960=example_is_chess960(example),
                    )
                    for example, pred in planning_predictions
                ]
                # A prediction with no move tag counts as 0.0: a missing move
                # is not a legal move.
                legal_scores = [
                    0.0 if score is None else score for score in raw_legal_scores
                ]
                metrics["format_compliance"] = sum(format_scores) / len(format_scores)
                metrics["legal_move_rate"] = sum(legal_scores) / len(legal_scores)
                missing_tags = sum(1 for score in raw_legal_scores if score is None)
                if missing_tags:
                    metrics["missing_move_tag_count"] = float(missing_tags)

            split_results[split_name] = metrics
            split_counts[split_name] = len(examples)
    finally:
        if engine is not None:
            engine.quit()

    print_report(
        version,
        split_results,
        split_counts,
        has_acpl,
        split_coverage=split_coverage,
        uncovered_splits=uncovered_splits,
    )
    analysis_path = write_prediction_analysis_report(
        predictions_path,
        examples_by_id=examples_by_id,
    )
    logger.info("Saved prediction analysis report to %s", analysis_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
