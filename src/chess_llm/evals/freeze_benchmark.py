"""CLI for freezing raw eval splits into benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chess_llm.evals.benchmark import freeze_and_save
from chess_llm.sft.settings import (
    DEFAULT_EVAL_SPLIT_SIZES,
    DEFAULT_MASTER_SEED,
    SftDataSettings,
)


def load_raw_split(path: Path) -> list[dict]:
    """Load a raw eval split JSONL file."""
    examples: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``chess-llm-freeze-benchmark``."""
    settings = SftDataSettings.from_env(Path.cwd())
    parser = argparse.ArgumentParser(
        description="Freeze eval splits into canonical benchmark JSONL"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Freeze only this split, for example 'evaluation'",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(settings.eval_splits_dir),
        help="Directory containing raw eval split JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for frozen benchmark (default: split-dir/../benchmark)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="chess-sft-eval-v1",
        help="Benchmark version string",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help="Seed for deterministic gold derivation",
    )
    coverage_group = parser.add_mutually_exclusive_group()
    coverage_group.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail if any planned task type has zero frozen examples.",
    )
    coverage_group.add_argument(
        "--allow-coverage-gaps",
        action="store_true",
        help=(
            "Allow planned task types with zero frozen examples. "
            "Single-split freezes allow this by default."
        ),
    )
    args = parser.parse_args(argv)

    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        print(f"[FAIL] Split directory does not exist: {split_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else split_dir.parent / "benchmark"
    splits_to_freeze = [args.split] if args.split else list(DEFAULT_EVAL_SPLIT_SIZES)

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
    strict_coverage = not is_partial
    if args.strict_coverage:
        strict_coverage = True
    if args.allow_coverage_gaps:
        strict_coverage = False

    try:
        manifest = freeze_and_save(
            splits,
            output_dir,
            seed=args.seed,
            version=args.version,
            clean=not is_partial,
            strict_coverage=strict_coverage,
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    total = sum(manifest["splits"].values())
    print(f"\nFrozen {total} total examples across {len(manifest['splits'])} splits")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
