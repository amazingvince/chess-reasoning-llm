"""Standalone CLI for generating held-out eval splits and frozen benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from chess_llm.evals.benchmark import freeze_and_save
from chess_llm.sft.context import raw_fen_identity_key
from chess_llm.sft.eval_split import (
    build_blocklist,
    effective_eval_split_sizes,
    generate_all_eval_splits,
    load_blocklist,
    save_eval_splits,
)
from chess_llm.sft.settings import SftDataSettings
from chess_llm.sft.source_preparation import (
    build_eval_split_sources,
    reserve_training_rows_for_volume,
)

logger = logging.getLogger(__name__)

EVAL_SPLIT_MANIFEST_NAME = "manifest.json"

_SETTINGS_ROOT = Path.cwd()
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


def _manifest_split_names(split_root: Path) -> set[str] | None:
    """Return split names in the eval-splits manifest, or None when unusable."""
    manifest_path = split_root / EVAL_SPLIT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    split_sizes = manifest.get("split_sizes") if isinstance(manifest, dict) else None
    if not isinstance(split_sizes, dict):
        return None
    return set(split_sizes)


def _rows_covered_by_blocklist(rows: list[dict], blocklist: frozenset[str]) -> bool:
    for row in rows:
        if not row.get("fen"):
            continue
        if raw_fen_identity_key(row) not in blocklist:
            return False
    return True


def freeze_existing_eval_splits(
    eval_splits_dir: str | Path = EVAL_SPLITS_DIR,
    benchmark_dir: str | Path = BENCHMARK_DIR,
    *,
    split_names: Sequence[str] | None = None,
    seed: int | None = None,
) -> int:
    """Re-freeze a benchmark from existing eval split JSONL files.

    Only splits present in the current eval-splits manifest and fully covered
    by ``blocklist.txt`` are frozen; stale or uncovered split files are
    skipped with a warning so training-visible positions never enter the
    benchmark.
    """
    split_root = Path(eval_splits_dir)
    effective_seed = MASTER_SEED if seed is None else seed
    names = tuple(split_names or EVAL_SPLIT_SIZES)
    manifest_names = _manifest_split_names(split_root)
    if manifest_names is None:
        logger.warning(
            "No usable eval-splits manifest in %s; freezing based on "
            "blocklist coverage only",
            split_root,
        )
    blocklist_path = split_root / "blocklist.txt"
    blocklist = load_blocklist(blocklist_path) if blocklist_path.exists() else None

    splits: dict[str, list[dict]] = {}
    for split_name in names:
        path = split_root / f"{split_name}.jsonl"
        if not path.exists():
            continue
        if manifest_names is not None and split_name not in manifest_names:
            logger.warning(
                "Skipping stale eval split %r: not in the current manifest",
                split_name,
            )
            continue
        rows: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if blocklist is None or not _rows_covered_by_blocklist(rows, blocklist):
            logger.warning(
                "Skipping eval split %r: positions not covered by blocklist.txt",
                split_name,
            )
            continue
        splits[split_name] = rows

    if not splits:
        logger.warning("No freezable eval split files found in %s", split_root)
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
    eval_splits_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
    seed: int | None = None,
    min_depth_eval_benchmark: int | None = None,
) -> EvalSplitRunResult:
    """Load sources, generate eval splits, freeze benchmark, and save blocklist."""
    eval_splits_dir = EVAL_SPLITS_DIR if eval_splits_dir is None else eval_splits_dir
    benchmark_dir = BENCHMARK_DIR if benchmark_dir is None else benchmark_dir
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
    fen_pool = config.get("fen_pool", [])
    splits = generate_all_eval_splits(
        prepared_sources.sources,
        seed=effective_seed,
        split_sizes=split_sizes,
    )
    save_eval_splits(splits, Path(eval_splits_dir), fen_pool=fen_pool)
    _write_pipeline_manifest(
        prepared_sources,
        split_sizes=split_sizes,
        volume_override=volume_override,
        master_seed=effective_seed,
        min_depth_eval_benchmark=effective_min_depth,
        eval_splits_dir=Path(eval_splits_dir),
    )
    freeze_and_save(
        splits,
        Path(benchmark_dir),
        seed=effective_seed,
        strict_coverage=(volume_override is None),
    )
    blocklist = build_blocklist(splits, fen_pool)
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


def _write_pipeline_manifest(
    prepared_sources,
    *,
    split_sizes: dict[str, int],
    volume_override: int | None,
    master_seed: int,
    min_depth_eval_benchmark: int,
    eval_splits_dir: Path,
) -> None:
    """Write the eval-splits manifest the pipeline uses for split reuse."""
    from chess_llm.sft import pipeline

    manifest = pipeline.build_eval_split_manifest(
        prepared_sources,
        split_sizes=split_sizes,
        volume_override=volume_override,
        master_seed=master_seed,
        min_depth_eval_benchmark=min_depth_eval_benchmark,
        fingerprint_cache_path=(
            eval_splits_dir / pipeline.EVAL_SPLIT_FINGERPRINT_CACHE_NAME
        ),
    )
    pipeline.write_eval_split_manifest(manifest, eval_splits_dir)


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
