"""Standalone CLI for generating held-out eval splits and frozen benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chess_llm.evals.benchmark import freeze_and_save
from chess_llm.sft.eval_split import (
    build_blocklist,
    effective_eval_split_sizes,
    generate_all_eval_splits,
    save_eval_splits,
)
from chess_llm.sft.settings import SftDataSettings
from chess_llm.sft.source_preparation import (
    build_eval_split_sources,
    reserve_training_rows_for_volume,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MAKE_DATA_ROOT = _REPO_ROOT / "sft" / "make_data"
_SETTINGS_ROOT = _LEGACY_MAKE_DATA_ROOT if _LEGACY_MAKE_DATA_ROOT.exists() else Path.cwd()
SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)

EVAL_SPLITS_DIR = SETTINGS.eval_splits_dir
BENCHMARK_DIR = SETTINGS.benchmark_dir
MASTER_SEED = SETTINGS.master_seed
MIN_DEPTH_EVAL_BENCHMARK = SETTINGS.min_depth_eval_benchmark
EVAL_SPLIT_SIZES = dict(SETTINGS.eval_split_sizes)


@dataclass(frozen=True)
class EvalSplitRunResult:
    """Summary of one eval-split generation or backfill run."""

    eval_splits_dir: Path
    benchmark_dir: Path
    total_eval_examples: int
    blocklist_size: int
    benchmark_backfilled: bool = False


def load_sources(volume_override: int | None = None) -> dict:
    """Load source config through the package pipeline seam."""
    from chess_llm.sft import pipeline

    return pipeline.load_sources(volume_override=volume_override)


def freeze_existing_eval_splits(
    eval_splits_dir: str | Path = EVAL_SPLITS_DIR,
    benchmark_dir: str | Path = BENCHMARK_DIR,
    *,
    split_names: Sequence[str] | None = None,
    seed: int | None = None,
) -> int:
    """Re-freeze a benchmark from existing eval split JSONL files."""
    split_root = Path(eval_splits_dir)
    effective_seed = MASTER_SEED if seed is None else seed
    names = tuple(split_names or EVAL_SPLIT_SIZES)
    splits: dict[str, list[dict]] = {}
    for split_name in names:
        path = split_root / f"{split_name}.jsonl"
        if not path.exists():
            continue
        rows: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        splits[split_name] = rows

    if not splits:
        logger.warning("No eval split files found in %s", split_root)
        return 0

    freeze_and_save(
        splits,
        Path(benchmark_dir),
        seed=effective_seed,
        strict_coverage=False,
    )
    return sum(len(rows) for rows in splits.values())


def generate_eval_splits(
    *,
    volume_override: int | None = None,
    eval_splits_dir: str | Path = EVAL_SPLITS_DIR,
    benchmark_dir: str | Path = BENCHMARK_DIR,
    seed: int | None = None,
    min_depth_eval_benchmark: int | None = None,
) -> EvalSplitRunResult:
    """Load sources, generate eval splits, freeze benchmark, and save blocklist."""
    effective_seed = MASTER_SEED if seed is None else seed
    effective_min_depth = (
        MIN_DEPTH_EVAL_BENCHMARK
        if min_depth_eval_benchmark is None
        else min_depth_eval_benchmark
    )
    config = load_sources(volume_override=volume_override)
    prepared_sources = build_eval_split_sources(
        config,
        min_depth_eval_benchmark=effective_min_depth,
        seed=effective_seed,
    )
    config["openings"] = prepared_sources.train_openings

    split_sizes = effective_eval_split_sizes(
        EVAL_SPLIT_SIZES,
        volume_override=volume_override,
    )
    split_sizes = reserve_training_rows_for_volume(
        split_sizes,
        prepared_sources,
        volume_override=volume_override,
    )
    splits = generate_all_eval_splits(
        prepared_sources.sources,
        seed=effective_seed,
        split_sizes=split_sizes,
    )
    save_eval_splits(splits, Path(eval_splits_dir))
    freeze_and_save(
        splits,
        Path(benchmark_dir),
        seed=effective_seed,
        strict_coverage=(volume_override is None),
    )
    blocklist = build_blocklist(splits)
    total = sum(len(rows) for rows in splits.values())
    logger.info(
        "Generated %d eval examples, froze %d benchmark, blocklist has %d FENs",
        total,
        total,
        len(blocklist),
    )
    return EvalSplitRunResult(
        eval_splits_dir=Path(eval_splits_dir),
        benchmark_dir=Path(benchmark_dir),
        total_eval_examples=total,
        blocklist_size=len(blocklist),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate eval splits")
    parser.add_argument(
        "--volume",
        type=int,
        help="Override max items per source and relax benchmark coverage checks",
    )
    parser.add_argument(
        "--eval-splits-dir",
        type=Path,
        default=EVAL_SPLITS_DIR,
        help="Directory for raw eval split JSONL files",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Directory for frozen benchmark output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    blocklist_path = Path(args.eval_splits_dir) / "blocklist.txt"
    manifest_path = Path(args.benchmark_dir) / "manifest.json"
    if blocklist_path.exists():
        logger.info("Eval splits already exist at %s", args.eval_splits_dir)
        if not manifest_path.exists():
            logger.info("Benchmark missing; freezing from existing eval splits...")
            freeze_existing_eval_splits(
                args.eval_splits_dir,
                args.benchmark_dir,
                seed=MASTER_SEED,
            )
        else:
            logger.info("Benchmark already exists. Delete the directory to regenerate.")
        return 0

    generate_eval_splits(
        volume_override=args.volume,
        eval_splits_dir=args.eval_splits_dir,
        benchmark_dir=args.benchmark_dir,
        seed=MASTER_SEED,
        min_depth_eval_benchmark=MIN_DEPTH_EVAL_BENCHMARK,
    )
    return 0


__all__ = [
    "BENCHMARK_DIR",
    "EVAL_SPLITS_DIR",
    "EVAL_SPLIT_SIZES",
    "EvalSplitRunResult",
    "MASTER_SEED",
    "MIN_DEPTH_EVAL_BENCHMARK",
    "SETTINGS",
    "build_arg_parser",
    "build_blocklist",
    "effective_eval_split_sizes",
    "freeze_and_save",
    "freeze_existing_eval_splits",
    "generate_all_eval_splits",
    "generate_eval_splits",
    "load_sources",
    "main",
    "save_eval_splits",
]


if __name__ == "__main__":
    raise SystemExit(main())
