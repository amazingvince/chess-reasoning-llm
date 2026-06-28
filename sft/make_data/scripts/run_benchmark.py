#!/usr/bin/env python3
"""Score model predictions against frozen benchmark.

Usage:
    python run_benchmark.py --benchmark-dir /path --predictions /path/output.jsonl
    python run_benchmark.py --benchmark-dir /path --predictions /path --stockfish-path /path/to/sf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import chess

from validation.benchmark import (
    BenchmarkExample,
    load_benchmark,
    score_split,
    centipawn_loss,
    format_compliance,
    legal_move_rate,
    pass_at_k,
    _extract_move,
)
from validation.validator import validate_move_legal

logger = logging.getLogger(__name__)

# Task types that predict moves (candidates for ACPL)
_MOVE_TASK_TYPES = frozenset({
    "best_move", "puzzle_solve", "endgame_best_move",
})


def load_predictions(path: Path) -> dict[str, str | list[str]]:
    """Load model predictions from JSONL.

    Format: {"example_id": "perception_00042", "prediction": "..."}

    When multiple predictions exist for the same example_id (pass@k),
    returns a list of strings.
    """
    preds: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                preds[obj["example_id"]].append(obj["prediction"])

    result: dict[str, str | list[str]] = {}
    for eid, vals in preds.items():
        result[eid] = vals[0] if len(vals) == 1 else vals
    return result


def _evaluate_predicted_move(engine, fen: str, uci_move: str, depth: int = 20) -> int | None:
    """Evaluate position after a predicted move using Stockfish.

    Returns centipawn score from the original side-to-move's perspective,
    or None if the move is invalid.
    """
    try:
        board = chess.Board(fen)
        original_turn = board.turn
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
            return None
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(original_turn)
        return score.score(mate_score=10000)
    except Exception:
        return None


# Floor penalty (centipawns) for missing/illegal move predictions.
# Ensures positions with gold_cp near 0 still get a meaningful penalty
# when the model fails to produce a legal move.
_ACPL_INVALID_MOVE_PENALTY = 150.0


def compute_acpl(
    engine,
    examples: list[BenchmarkExample],
    flat_preds: dict[str, str],
    depth: int = 20,
) -> dict[str, float]:
    """Compute per-example centipawn loss using Stockfish.

    Returns mapping of example_id -> centipawn loss for move-prediction tasks
    that have gold_cp in metadata.
    """
    acpl_scores: dict[str, float] = {}
    evaluated = 0

    for ex in examples:
        if ex.task_type not in _MOVE_TASK_TYPES:
            continue
        gold_cp = ex.metadata.get("cp")
        if gold_cp is None:
            continue

        pred = flat_preds.get(ex.example_id, "")
        uci = _extract_move(pred)
        if uci is None:
            # No valid move tag — penalty is at least the floor
            acpl_scores[ex.example_id] = max(
                _ACPL_INVALID_MOVE_PENALTY, min(abs(gold_cp), 500.0),
            )
            continue

        predicted_cp = _evaluate_predicted_move(engine, ex.fen, uci, depth)
        if predicted_cp is None:
            # Illegal move — same floor-guaranteed penalty
            acpl_scores[ex.example_id] = max(
                _ACPL_INVALID_MOVE_PENALTY, min(abs(gold_cp), 500.0),
            )
            continue

        acpl_scores[ex.example_id] = centipawn_loss(float(gold_cp), float(predicted_cp))
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score model predictions against frozen benchmark"
    )
    parser.add_argument(
        "--benchmark-dir", type=str, required=True,
        help="Directory containing frozen benchmark JSONL files",
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--stockfish-path", type=str, default=None,
        help="Path to Stockfish binary for ACPL computation",
    )
    parser.add_argument(
        "--acpl-depth", type=int, default=20,
        help="Stockfish search depth for ACPL (default: 20)",
    )
    args = parser.parse_args()

    bench_dir = Path(args.benchmark_dir)
    pred_path = Path(args.predictions)

    if not bench_dir.exists():
        print(f"[FAIL] Benchmark directory does not exist: {bench_dir}")
        return 1
    if not pred_path.exists():
        print(f"[FAIL] Predictions file does not exist: {pred_path}")
        return 1

    # Load manifest
    manifest_path = bench_dir / "manifest.json"
    manifest: dict = {}
    version = "unknown"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        version = manifest.get("version", "unknown")

    # Load predictions
    predictions_raw = load_predictions(pred_path)

    # Flatten for score_split (use first prediction if multiple)
    flat_preds: dict[str, str] = {}
    for eid, val in predictions_raw.items():
        flat_preds[eid] = val if isinstance(val, str) else val[0]

    # Optional Stockfish engine for ACPL
    engine = None
    has_acpl = False
    if args.stockfish_path:
        sf_path = Path(args.stockfish_path)
        if sf_path.exists():
            try:
                import chess.engine
                engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
                has_acpl = True
                logger.info("Stockfish loaded for ACPL from %s", sf_path)
            except Exception as exc:
                print(f"[WARN] Could not load Stockfish: {exc}")

    split_results: dict[str, dict[str, float]] = {}
    split_counts: dict[str, int] = {}

    # Score only splits listed in the manifest (not stale leftovers)
    manifest_splits = manifest.get("splits", {})
    score_splits = sorted(manifest_splits.keys()) if manifest_splits else [
        p.stem for p in sorted(bench_dir.glob("*.jsonl"))
    ]
    for split_name in score_splits:
        jsonl_path = bench_dir / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            continue
        examples = load_benchmark(str(jsonl_path))
        if not examples:
            continue

        # Compute ACPL if engine available
        acpl_scores: dict[str, float] | None = None
        if engine:
            acpl_scores = compute_acpl(engine, examples, flat_preds, args.acpl_depth)

        # Score the split
        metrics = score_split(examples, flat_preds, acpl_scores)

        # Always compute pass@1 and pass@8 for puzzle solving per spec
        puzzle_examples = [
            e for e in examples if e.task_type == "puzzle_solve"
        ]
        if puzzle_examples:
            for k in (1, 8):
                hits = 0
                for ex in puzzle_examples:
                    pval = predictions_raw.get(ex.example_id, "")
                    plist = pval if isinstance(pval, list) else [pval]
                    # pass@k: use up to k predictions
                    hits += pass_at_k(plist[:k], ex.gold_answer)
                metrics[f"puzzle_pass_at_{k}"] = hits / len(puzzle_examples)

        # Compute aggregate secondary metrics
        planning_preds = [
            (ex, flat_preds.get(ex.example_id, ""))
            for ex in examples
            if ex.task_type in ("best_move", "puzzle_solve")
        ]
        if planning_preds:
            fc_scores = [format_compliance(p) for _, p in planning_preds]
            lm_scores = [
                v
                for v in (
                    legal_move_rate(
                        p,
                        ex.fen,
                        chess960=bool(ex.metadata.get("is_chess960")),
                    )
                    for ex, p in planning_preds
                )
                if v is not None
            ]
            metrics["format_compliance"] = sum(fc_scores) / len(fc_scores)
            if lm_scores:
                metrics["legal_move_rate"] = sum(lm_scores) / len(lm_scores)

        split_results[split_name] = metrics
        split_counts[split_name] = len(examples)

    if engine:
        engine.quit()

    print_report(version, split_results, split_counts, has_acpl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
