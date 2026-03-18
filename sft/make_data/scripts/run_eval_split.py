#!/usr/bin/env python3
"""Generate and freeze eval splits independently.

Usage:
    python run_eval_split.py
    python run_eval_split.py --volume 50  # small test splits
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import EVAL_SPLITS_DIR, MASTER_SEED
from pool.eval_split import generate_all_eval_splits, save_eval_splits, build_blocklist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval splits")
    parser.add_argument("--volume", type=int, help="Override max items per source (for testing)")
    args = parser.parse_args()

    blocklist_path = EVAL_SPLITS_DIR / "blocklist.txt"
    if blocklist_path.exists():
        logger.info("Eval splits already exist at %s", EVAL_SPLITS_DIR)
        logger.info("Delete the directory to regenerate.")
        return 0

    # Import sources dynamically to load data
    from scripts.run_pipeline import load_sources

    config = load_sources(volume_override=args.volume)

    sources = {
        "perception": config.get("fen_pool", []),
        "rules": config.get("fen_pool", []),
        "tactics": config.get("puzzles", []),
        "evaluation": config.get("position_evals", []),
        "openings": config.get("openings", []),
        "endgames": config.get("endgame_positions", []),
        "planning": config.get("puzzles", []) + config.get("best_move_evals", []),
        "chess960": [e for e in config.get("fen_pool", []) if e.get("is_chess960")],
        "mate": config.get("mate_rows", []),
    }

    splits = generate_all_eval_splits(sources, seed=MASTER_SEED)
    save_eval_splits(splits, str(EVAL_SPLITS_DIR))

    blocklist = build_blocklist(splits)
    total = sum(len(v) for v in splits.values())
    logger.info("Generated %d eval examples, blocklist has %d FENs", total, len(blocklist))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
