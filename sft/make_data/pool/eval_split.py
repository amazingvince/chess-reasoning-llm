"""Eval split generation and FEN blocklist management.

Eval splits are created FIRST, before any training data, so that
no training example can share a FEN with any eval example.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from random import Random

from config.settings import EVAL_SPLIT_SIZES, MASTER_SEED

logger = logging.getLogger(__name__)


def generate_all_eval_splits(
    sources: dict[str, list[dict]],
    seed: int = MASTER_SEED,
) -> dict[str, list[dict]]:
    """Create all 9 benchmark splits from source data.

    Parameters
    ----------
    sources : dict
        Keys match EVAL_SPLIT_SIZES keys. Values are lists of candidate
        dicts (each must have a ``"fen"`` key).
    seed : int
        Master seed for reproducibility.

    Returns
    -------
    dict[str, list[dict]]
        Split name -> list of held-out examples.

    The total is ~13K examples across 9 splits:
        perception: 2K, rules: 2K, tactics: 2K, evaluation: 1.5K,
        openings: 500, endgames: 1.5K, planning: 2K, chess960: 500,
        mate: 1K.
    """
    rng = Random(seed)
    splits: dict[str, list[dict]] = {}
    used_fens: set[str] = set()

    for split_name, target_size in EVAL_SPLIT_SIZES.items():
        candidates = sources.get(split_name, [])
        if not candidates:
            logger.warning(
                "No source data for eval split %r — skipping", split_name
            )
            splits[split_name] = []
            continue

        # Filter out FENs already used by earlier splits
        available = [c for c in candidates if c.get("fen", "") not in used_fens]
        n = min(target_size, len(available))
        sampled = rng.sample(available, n)
        splits[split_name] = sampled

        # Track used FENs to prevent cross-split contamination
        for ex in sampled:
            fen = ex.get("fen", "")
            if fen:
                used_fens.add(fen)

        logger.info(
            "Eval split %r: %d / %d target (%d candidates after dedup)",
            split_name, n, target_size, len(available),
        )

    return splits


def build_blocklist(eval_splits: dict[str, list[dict]]) -> frozenset[str]:
    """Extract all FENs from eval splits into a frozen blocklist."""
    fens: set[str] = set()
    for split_examples in eval_splits.values():
        for ex in split_examples:
            fen = ex.get("fen", "")
            if fen:
                fens.add(fen)
    logger.info("Built eval blocklist with %d FENs", len(fens))
    return frozenset(fens)


def save_eval_splits(splits: dict[str, list[dict]], path: str) -> None:
    """Save eval splits to disk (one JSONL per split)."""
    base = Path(path)
    base.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        out = base / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved eval split %r (%d examples) to %s", name, len(examples), out)

    # Also save flat blocklist
    blocklist_path = base / "blocklist.txt"
    all_fens = build_blocklist(splits)
    with open(blocklist_path, "w", encoding="utf-8") as fh:
        for fen in sorted(all_fens):
            fh.write(fen + "\n")
    logger.info("Saved blocklist (%d FENs) to %s", len(all_fens), blocklist_path)


def load_blocklist(path: str) -> frozenset[str]:
    """Load the FEN blocklist from a text file (one FEN per line)."""
    fens: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                fens.add(line)
    return frozenset(fens)
