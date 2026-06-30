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
    BenchmarkExample,
    centipawn_loss,
    extract_move,
    format_compliance,
    legal_move_rate,
    load_benchmark,
    pass_at_k,
    score_split,
)
from chess_llm.evals.prediction_analysis import write_prediction_analysis_report

logger = logging.getLogger(__name__)

_MOVE_TASK_TYPES = frozenset({"best_move", "puzzle_solve", "endgame_best_move"})
_ACPL_INVALID_MOVE_PENALTY = 150.0


def load_predictions(path: Path) -> dict[str, str | list[str]]:
    """Load model predictions from JSONL."""
    predictions: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            predictions[row["example_id"]].append(row["prediction"])

    result: dict[str, str | list[str]] = {}
    for example_id, values in predictions.items():
        result[example_id] = values[0] if len(values) == 1 else values
    return result


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
) -> dict[str, float]:
    """Compute per-example centipawn loss for move-prediction tasks."""
    acpl_scores: dict[str, float] = {}
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
            acpl_scores[example.example_id] = max(
                _ACPL_INVALID_MOVE_PENALTY,
                min(abs(gold_cp), 500.0),
            )
            continue

        predicted_cp = evaluate_predicted_move(
            engine,
            example.fen,
            uci,
            depth,
            chess960=chess960,
        )
        if predicted_cp is None:
            acpl_scores[example.example_id] = max(
                _ACPL_INVALID_MOVE_PENALTY,
                min(abs(gold_cp), 500.0),
            )
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


def print_report(
    version: str,
    split_results: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    has_acpl: bool = False,
) -> None:
    """Print benchmark evaluation report."""
    print(f"\n=== Benchmark Evaluation ({version}) ===\n")
    split_order = [
        "perception", "rules", "tactics", "evaluation",
        "openings", "endgames", "planning", "chess960", "mate",
    ]

    for split_name in split_order:
        if split_name not in split_results:
            continue
        metrics = split_results[split_name]
        count = split_counts.get(split_name, 0)
        print(f"{split_name.capitalize()} ({count} examples):")
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

    predictions_raw = load_predictions(predictions_path)
    flat_predictions = {
        example_id: value if isinstance(value, str) else value[0]
        for example_id, value in predictions_raw.items()
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

            acpl_scores = None
            if engine is not None:
                acpl_scores = compute_acpl(engine, examples, flat_predictions, args.acpl_depth)

            metrics = score_split(examples, flat_predictions, acpl_scores)

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
                (example, flat_predictions.get(example.example_id, ""))
                for example in examples
                if example.task_type in ("best_move", "puzzle_solve")
            ]
            if planning_predictions:
                format_scores = [format_compliance(pred) for _, pred in planning_predictions]
                legal_scores = [
                    score
                    for score in (
                        legal_move_rate(
                            pred,
                            example.fen,
                            chess960=example_is_chess960(example),
                        )
                        for example, pred in planning_predictions
                    )
                    if score is not None
                ]
                metrics["format_compliance"] = sum(format_scores) / len(format_scores)
                if legal_scores:
                    metrics["legal_move_rate"] = sum(legal_scores) / len(legal_scores)

            split_results[split_name] = metrics
            split_counts[split_name] = len(examples)
    finally:
        if engine is not None:
            engine.quit()

    print_report(version, split_results, split_counts, has_acpl)
    analysis_path = write_prediction_analysis_report(
        predictions_path,
        examples_by_id=examples_by_id,
    )
    logger.info("Saved prediction analysis report to %s", analysis_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
