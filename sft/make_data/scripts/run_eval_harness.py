#!/usr/bin/env python3
"""CLI for running the eval harness against held-out eval splits.

Usage:
    python run_eval_harness.py                     # all splits
    python run_eval_harness.py --split evaluation  # one split
    python run_eval_harness.py --split-dir /path   # custom directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import EVAL_SPLITS_DIR, EVAL_SPLIT_SIZES
from validation.eval_harness import SPLIT_CHECKS, evaluate_split


def load_split(path: Path) -> list[dict]:
    """Load a JSONL eval split file."""
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval harness on held-out splits")
    parser.add_argument(
        "--split", type=str, default=None,
        help="Run only this split (e.g. 'evaluation')",
    )
    parser.add_argument(
        "--split-dir", type=str, default=str(EVAL_SPLITS_DIR),
        help="Directory containing eval split JSONL files",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        print(f"[FAIL] Split directory does not exist: {split_dir}")
        return 1

    # Default to all 9 splits (not just those with checks defined), so
    # missing coverage is surfaced as a skip rather than silently omitted.
    splits_to_check = [args.split] if args.split else list(EVAL_SPLIT_SIZES.keys())
    any_fail = False

    print(f"{'Split':<15} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Status'}")
    print("-" * 62)

    for split_name in splits_to_check:
        path = split_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"{split_name:<15} {'—':>6} {'—':>6} {'—':>6} {'—':>6} [SKIP] file not found")
            continue

        examples = load_split(path)
        result = evaluate_split(split_name, examples)

        status = "[PASS]" if result.failed == 0 else "[FAIL]"
        if result.failed > 0:
            any_fail = True

        print(
            f"{result.split:<15} {result.total:>6} {result.passed:>6} "
            f"{result.failed:>6} {result.skipped:>6} {status}"
        )

        if result.errors:
            for err in result.errors[:5]:
                print(f"  ! {err}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more errors")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
