#!/usr/bin/env python3
"""Freeze eval splits into canonical benchmark JSONL files.

Usage:
    python freeze_benchmark.py                          # all splits
    python freeze_benchmark.py --split evaluation       # one split
    python freeze_benchmark.py --version chess-sft-eval-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import EVAL_SPLITS_DIR, EVAL_SPLIT_SIZES, MASTER_SEED
from validation.benchmark import freeze_and_save


def load_raw_split(path: Path) -> list[dict]:
    """Load a raw eval split JSONL file."""
    examples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze eval splits into canonical benchmark JSONL"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Freeze only this split (e.g. 'evaluation')",
    )
    parser.add_argument(
        "--split-dir", type=str, default=str(EVAL_SPLITS_DIR),
        help="Directory containing raw eval split JSONL files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for frozen benchmark (default: split-dir/../benchmark)",
    )
    parser.add_argument(
        "--version", type=str, default="chess-sft-eval-v1",
        help="Benchmark version string",
    )
    parser.add_argument(
        "--seed", type=int, default=MASTER_SEED,
        help="Seed for deterministic gold derivation",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        print(f"[FAIL] Split directory does not exist: {split_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else split_dir.parent / "benchmark"

    splits_to_freeze = [args.split] if args.split else list(EVAL_SPLIT_SIZES.keys())

    splits: dict[str, list[dict]] = {}
    for split_name in splits_to_freeze:
        path = split_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  {split_name}: [SKIP] file not found")
            continue
        splits[split_name] = load_raw_split(path)

    if not splits:
        print("[FAIL] No split files found")
        return 1

    is_partial = args.split is not None
    manifest = freeze_and_save(
        splits, str(output_dir), seed=args.seed, version=args.version,
        clean=not is_partial,
    )

    total = sum(manifest["splits"].values())
    print(f"\nFrozen {total} total examples across {len(manifest['splits'])} splits")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
