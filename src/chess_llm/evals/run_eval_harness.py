"""CLI for running eval-split oracle checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chess_llm.evals.eval_harness import evaluate_split
from chess_llm.sft.settings import DEFAULT_EVAL_SPLIT_SIZES, SftDataSettings


def load_split(path: Path) -> list[dict]:
    """Load a JSONL eval split file."""
    examples: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``chess-llm-run-eval-harness``."""
    settings = SftDataSettings.from_env(Path.cwd())
    parser = argparse.ArgumentParser(description="Run eval harness on held-out splits")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Run only this split, for example 'evaluation'",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(settings.eval_splits_dir),
        help="Directory containing eval split JSONL files",
    )
    args = parser.parse_args(argv)

    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        print(f"[FAIL] Split directory does not exist: {split_dir}")
        return 1

    splits_to_check = [args.split] if args.split else list(DEFAULT_EVAL_SPLIT_SIZES)
    any_fail = False

    print(f"{'Split':<15} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Status'}")
    print("-" * 62)

    for split_name in splits_to_check:
        path = split_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"{split_name:<15} {'-':>6} {'-':>6} {'-':>6} {'-':>6} [SKIP] file not found")
            continue

        result = evaluate_split(split_name, load_split(path))
        status = "[PASS]" if result.failed == 0 else "[FAIL]"
        if result.failed > 0:
            any_fail = True

        print(
            f"{result.split:<15} {result.total:>6} {result.passed:>6} "
            f"{result.failed:>6} {result.skipped:>6} {status}"
        )
        if result.errors:
            for error in result.errors[:5]:
                print(f"  ! {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more errors")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
