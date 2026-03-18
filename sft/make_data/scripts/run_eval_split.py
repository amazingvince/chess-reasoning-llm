#!/usr/bin/env python3
"""Generate eval splits, freeze benchmark, and save blocklist.

Applies the same filtering as the main pipeline:
  - ECO-based holdout for openings (same ECO family stays together)
  - depth >= 40 filter for evaluation split
  - Book-move enrichment for opening continuations
  - Frozen benchmark JSONL + manifest

Usage:
    python run_eval_split.py
    python run_eval_split.py --volume 50  # small test splits
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    EVAL_SPLITS_DIR,
    BENCHMARK_DIR,
    MASTER_SEED,
    MIN_DEPTH_EVAL_BENCHMARK,
)
from pool.eval_split import (
    generate_all_eval_splits,
    save_eval_splits,
    build_blocklist,
    partition_eco_codes,
)
from validation.benchmark import freeze_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _freeze_from_disk() -> None:
    """Re-freeze benchmark from existing eval split JSONL on disk."""
    from config.settings import EVAL_SPLIT_SIZES

    splits: dict[str, list[dict]] = {}
    for split_name in EVAL_SPLIT_SIZES:
        path = EVAL_SPLITS_DIR / f"{split_name}.jsonl"
        if not path.exists():
            continue
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        splits[split_name] = rows

    if not splits:
        logger.warning("No eval split files found in %s", EVAL_SPLITS_DIR)
        return

    freeze_and_save(
        splits, str(BENCHMARK_DIR), seed=MASTER_SEED,
        strict_coverage=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval splits")
    parser.add_argument("--volume", type=int, help="Override max items per source (for testing)")
    args = parser.parse_args()

    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
    if blocklist_path.exists():
        logger.info("Eval splits already exist at %s", EVAL_SPLITS_DIR)
        # Backfill frozen benchmark if missing (matches run_pipeline rerun path)
        if not (BENCHMARK_DIR / "manifest.json").exists():
            logger.info("Benchmark missing — freezing from existing eval splits...")
            _freeze_from_disk()
        else:
            logger.info("Benchmark already exists. Delete the directory to regenerate.")
        return 0

    # Import sources dynamically to load data
    from scripts.run_pipeline import load_sources

    config = load_sources(volume_override=args.volume)

    # ECO-based holdout for openings
    all_openings = config.get("openings", [])
    if all_openings:
        eval_openings, train_openings = partition_eco_codes(all_openings)
    else:
        eval_openings, train_openings = [], []

    # Enrich eval openings with book moves
    book_moves = config.get("book_moves", {})
    for opening in eval_openings:
        fen = opening.get("fen", "")
        if fen in book_moves:
            opening["book_moves"] = book_moves[fen]

    # Evaluation split: depth >= 40
    eval_benchmark_evals = [
        e for e in config.get("position_evals", [])
        if e.get("depth", 0) >= MIN_DEPTH_EVAL_BENCHMARK
    ]

    sources = {
        "perception": config.get("fen_pool", []),
        "rules": config.get("fen_pool", []),
        "tactics": config.get("puzzles", []),
        "evaluation": eval_benchmark_evals,
        "openings": eval_openings,
        "endgames": config.get("endgame_positions", []),
        "planning": config.get("puzzles", []) + config.get("best_move_evals", []),
        "chess960": [e for e in config.get("fen_pool", []) if e.get("is_chess960")],
        "mate": config.get("mate_rows", []),
    }

    splits = generate_all_eval_splits(sources, seed=MASTER_SEED)
    save_eval_splits(splits, str(EVAL_SPLITS_DIR))

    # Freeze benchmark
    freeze_and_save(splits, str(BENCHMARK_DIR), seed=MASTER_SEED)

    blocklist = build_blocklist(splits)
    total = sum(len(v) for v in splits.values())
    logger.info(
        "Generated %d eval examples, froze %d benchmark, blocklist has %d FENs",
        sum(len(v) for v in splits.values()), total, len(blocklist),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
