"""Upload generated SFT datasets and benchmark files to Hugging Face Hub."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_llm.sft.files import iter_jsonl_artifacts
from chess_llm.sft.settings import SftDataSettings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_ROOT = _REPO_ROOT
SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)

DEFAULT_ORG = "Chess-Nut-Engine"
TRAINING_REPO = "chess-sft-data"
EVAL_REPO = "chess-sft-eval"

TIER_OUTPUT_DIR = SETTINGS.tier_output_dir
EVAL_SPLITS_DIR = SETTINGS.eval_splits_dir
BENCHMARK_DIR = SETTINGS.benchmark_dir

TIER_NAMES = {
    1: "Perception",
    2: "Rules",
    3: "Tactics",
    4: "Evaluation",
    5: "Openings",
    6: "Endgames",
    7: "Planning",
}

TASK_DESCRIPTIONS = {
    "1.1_fen_to_board": "Render a FEN string as a human-readable board diagram",
    "1.2_board_to_fen": "Convert a board diagram back to FEN notation",
    "1.3_piece_identification": "Identify which piece occupies a given square",
    "1.4_piece_counting": "Count pieces of a specific type/color on the board",
    "1.5_state_tracking": "Apply moves and report the resulting position",
    "1.6_square_lookup": "Read a single square's contents from FEN",
    "1.7_rank_lookup": "Read one compressed rank row from FEN",
    "1.8_move_square_edits": "Trace square lookups and rank edits for one move",
    "1.9_fen_assembly": "Apply one move and assemble the resulting full FEN",
    "1.19_multi_move_state_tracking": "Apply two to three moves and report the resulting FEN",
    "2.0_side_piece_inventory": "List side-to-move pieces and squares before move generation",
    "2.1_legal_move_gen": "List all legal moves for the side to move",
    "2.2_piece_specific_moves": "List legal moves for a specific piece",
    "2.3_move_legality_check": "Determine whether a move is legal",
    "2.4_check_detection": "Detect check, checkmate, or stalemate",
    "2.5_special_rules": "Handle castling, en passant, promotion, and 50-move rule",
    "2.10_ray_walk": "Walk each slider ray square by square to derive its moves",
    "2.11_legal_filter_trace": "Filter every piece's pseudo-legal moves into rejected and legal moves",
    "3.1_available_captures": "Find available capture moves",
    "3.2_threats": "Identify pieces that are threatening enemy pieces",
    "3.3_attacked_defended": "Count attackers and defenders of a queried square",
    "3.4_tactical_patterns": "Recognize tactical motifs",
    "3.5_hanging_pieces": "Find undefended pieces that can be captured",
    "3.6_hanging_piece_status": "Classify whether one piece is attacked, defended, and hanging",
    "3.7_hanging_piece_filter": "Audit attacked pieces and filter defended decoys from hanging pieces",
    "3.8_hanging_piece_claim_verification": "Verify and correct hanging-piece claims",
    "4.1_material_balance": "Count material and compute the balance",
    "4.2_position_evaluation": "Evaluate a position from engine-calibrated labels",
    "4.3_pawn_structure": "Analyze pawn structure",
    "5.1_opening_identification": "Name the opening from a position or line",
    "5.2_opening_continuation": "Suggest the next book move",
    "5.3_opening_principles": "Explain opening principles",
    "6.1_endgame_classification": "Classify the endgame material",
    "6.2_endgame_wdl": "Predict tablebase win/draw/loss",
    "6.3_endgame_best_move": "Find a tablebase-backed endgame move",
    "6.4_endgame_principles": "Explain endgame principles",
    "7.1_best_move_selection": "Select the best move from engine-evaluated positions",
    "7.2_puzzle_solving": "Solve a tactical puzzle",
    "7.3_move_consequence": "Predict the consequence of a candidate move",
    "7.8_candidate_ratings": "Rate five Stockfish MultiPV candidate moves with fixed grammar",
    "7.9_step_verification": "Audit a numbered chess trace and identify one broken step",
    "7.10_best_line_trace": "Emit a fixed-grammar Stockfish best-line trace",
}

SPLIT_DESCRIPTIONS = {
    "perception": "Board reading and representation",
    "rules": "Move legality and chess rules",
    "tactics": "Tactical motifs and puzzle labels",
    "evaluation": "Position assessment",
    "openings": "Opening knowledge with ECO holdout",
    "endgames": "Tablebase-backed endgame play",
    "planning": "Best move, puzzle, and consequence tasks",
    "chess960": "Fischer Random positions",
    "mate": "MATE move-choice examples",
}


@dataclass(frozen=True)
class UploadFileStat:
    """Stats for one file staged for dataset upload."""

    path: Path
    relative_path: Path
    rows: int
    size_mb: float
    task_id: str | None = None
    tier: int | None = None
    name: str | None = None


@dataclass(frozen=True)
class UploadSummary:
    """Summary returned by upload helpers and dry runs."""

    repo_id: str
    files: int
    rows: int
    size_mb: float
    dry_run: bool


def count_lines(path: Path) -> int:
    """Count non-empty JSONL rows."""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def file_size_mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 * 1024)


def tier_from_task(task_id: str) -> int:
    """Extract tier number from a task id like ``3.2_threats``."""
    return int(task_id.split(".", 1)[0])


def collect_training_file_stats(tier_root: str | Path) -> list[UploadFileStat]:
    """Collect stats for generated tier JSONL files."""
    root = Path(tier_root)
    if not root.exists():
        raise FileNotFoundError(f"training output dir not found: {root}")

    stats: list[UploadFileStat] = []
    for path in iter_jsonl_artifacts(root):
        task_id = path.stem
        stats.append(
            UploadFileStat(
                path=path,
                relative_path=path.relative_to(root),
                rows=count_lines(path),
                size_mb=file_size_mb(path),
                task_id=task_id,
                tier=tier_from_task(task_id),
            )
        )
    if not stats:
        raise ValueError(f"no JSONL files found in {root}")
    return stats


def collect_eval_file_stats(
    eval_dir: str | Path,
    benchmark_dir: str | Path,
) -> tuple[list[UploadFileStat], list[UploadFileStat]]:
    """Collect stats for eval split and benchmark JSONL files."""
    eval_root = Path(eval_dir)
    bench_root = Path(benchmark_dir)
    if not eval_root.exists():
        raise FileNotFoundError(f"eval splits dir not found: {eval_root}")
    if not bench_root.exists():
        raise FileNotFoundError(f"benchmark dir not found: {bench_root}")

    eval_stats = [
        UploadFileStat(
            path=path,
            relative_path=Path("eval_splits") / path.name,
            rows=count_lines(path),
            size_mb=file_size_mb(path),
            name=path.stem,
        )
        for path in sorted(eval_root.glob("*.jsonl"))
    ]
    bench_stats = [
        UploadFileStat(
            path=path,
            relative_path=Path("benchmark") / path.name,
            rows=count_lines(path),
            size_mb=file_size_mb(path),
            name=path.stem,
        )
        for path in sorted(bench_root.glob("*.jsonl"))
    ]
    if not bench_stats:
        raise ValueError(
            f"no benchmark JSONL files found in {bench_root}; refusing to "
            "publish an eval dataset without its frozen benchmark"
        )
    return eval_stats, bench_stats


def generate_training_card(file_stats: Sequence[UploadFileStat], org: str) -> str:
    """Generate a Hugging Face dataset card for training data."""
    total_rows = sum(stat.rows for stat in file_stats)
    total_mb = sum(stat.size_mb for stat in file_stats)
    repo_id = f"{org}/{TRAINING_REPO}"

    tiers: dict[int, list[UploadFileStat]] = {}
    for stat in file_stats:
        if stat.tier is not None:
            tiers.setdefault(stat.tier, []).append(stat)

    configs_yaml = "configs:\n"
    configs_yaml += "  - config_name: default\n"
    configs_yaml += "    data_files:\n"
    configs_yaml += '      - split: train\n        path: "tier*/*.jsonl"\n'
    for tier_num in sorted(tiers):
        tier_name = TIER_NAMES.get(tier_num, f"tier{tier_num}").lower()
        configs_yaml += f"  - config_name: tier{tier_num}_{tier_name}\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f'      - split: train\n        path: "tier{tier_num}/*.jsonl"\n'

    tier_summary = "| Tier | Category | Tasks | Examples | Size |\n"
    tier_summary += "|------|----------|-------|----------|------|\n"
    for tier_num in sorted(tiers):
        tier_stats = tiers[tier_num]
        tier_summary += (
            f"| {tier_num} | {TIER_NAMES.get(tier_num, 'Unknown')} "
            f"| {len(tier_stats)} | {sum(s.rows for s in tier_stats):,} "
            f"| {sum(s.size_mb for s in tier_stats):.1f} MB |\n"
        )

    file_listing = ""
    for tier_num in sorted(tiers):
        file_listing += f"\n### Tier {tier_num} - {TIER_NAMES.get(tier_num, 'Unknown')}\n\n"
        file_listing += "| File | Task | Examples | Size |\n"
        file_listing += "|------|------|----------|------|\n"
        for stat in sorted(tiers[tier_num], key=lambda item: item.task_id or ""):
            task_id = stat.task_id or stat.path.stem
            file_listing += (
                f"| `{stat.relative_path.as_posix()}` "
                f"| {TASK_DESCRIPTIONS.get(task_id, '')} "
                f"| {stat.rows:,} | {stat.size_mb:.1f} MB |\n"
            )

    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - chess
  - sft
  - instruction-tuning
  - reasoning
  - chess960
pretty_name: Chess SFT Training Data
{configs_yaml}---

# Chess SFT Training Data

A supervised fine-tuning dataset for teaching language models to reason about
chess. It covers board perception, legal move generation, tactics, evaluation,
openings, endgames, and planning.

| | |
|---|---|
| **Total examples** | {total_rows:,} |
| **Total size** | {total_mb:.1f} MB |
| **Format** | JSONL chat rows with `messages` |
| **Eval companion** | [{org}/{EVAL_REPO}](https://huggingface.co/datasets/{org}/{EVAL_REPO}) |

## Tier Overview

{tier_summary}

## Loading

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", streaming=True)
ds_tier1 = load_dataset("{repo_id}", "tier1_perception")
```

## Data Sources

Rows are generated from Lichess games, Lichess puzzles, Lichess openings,
Lichess position evaluations, MATE rows, Syzygy tablebases, Polyglot opening
books, and generated Chess960 positions. Eval and benchmark FENs are excluded
from training with a blocklist.

## Detailed File Listing
{file_listing}
"""


def generate_eval_card(
    eval_stats: Sequence[UploadFileStat],
    bench_stats: Sequence[UploadFileStat],
    org: str,
) -> str:
    """Generate a Hugging Face dataset card for eval and benchmark data."""
    total_eval = sum(stat.rows for stat in eval_stats)
    total_bench = sum(stat.rows for stat in bench_stats)
    repo_id = f"{org}/{EVAL_REPO}"

    configs_yaml = "configs:\n"
    configs_yaml += "  - config_name: eval_splits\n"
    configs_yaml += "    data_files:\n"
    configs_yaml += '      - split: test\n        path: "eval_splits/*.jsonl"\n'
    configs_yaml += "  - config_name: benchmark\n"
    configs_yaml += "    data_files:\n"
    configs_yaml += '      - split: test\n        path: "benchmark/*.jsonl"\n'
    for stat in sorted(eval_stats, key=lambda item: item.name or ""):
        configs_yaml += f"  - config_name: eval_{stat.name}\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f'      - split: test\n        path: "eval_splits/{stat.name}.jsonl"\n'
    for stat in sorted(bench_stats, key=lambda item: item.name or ""):
        configs_yaml += f"  - config_name: bench_{stat.name}\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f'      - split: test\n        path: "benchmark/{stat.name}.jsonl"\n'

    eval_listing = _split_listing(eval_stats)
    bench_listing = _split_listing(bench_stats)

    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - chess
  - sft
  - evaluation
  - benchmark
  - chess960
pretty_name: Chess SFT Eval and Benchmark
{configs_yaml}---

# Chess SFT Eval and Benchmark

Held-out evaluation splits and a frozen benchmark for
[{org}/{TRAINING_REPO}](https://huggingface.co/datasets/{org}/{TRAINING_REPO}).
Every FEN in these files is excluded from generated training data.

| | |
|---|---|
| **Eval examples** | {total_eval:,} |
| **Benchmark examples** | {total_bench:,} |
| **Format** | JSONL |

## Loading

```python
from datasets import load_dataset

eval_ds = load_dataset("{repo_id}", "eval_splits")
bench_ds = load_dataset("{repo_id}", "benchmark")
```

## Eval Splits

{eval_listing}

## Frozen Benchmark

{bench_listing}
"""


def stage_training_upload(
    tier_root: str | Path,
    staging_dir: str | Path,
    file_stats: Sequence[UploadFileStat],
    *,
    org: str,
) -> None:
    """Copy training JSONL files and write a dataset card into ``staging_dir``."""
    root = Path(tier_root)
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    for stat in file_stats:
        destination = staging / stat.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / stat.relative_path, destination)
    (staging / "README.md").write_text(
        generate_training_card(file_stats, org),
        encoding="utf-8",
    )


def stage_eval_upload(
    eval_dir: str | Path,
    benchmark_dir: str | Path,
    staging_dir: str | Path,
    eval_stats: Sequence[UploadFileStat],
    bench_stats: Sequence[UploadFileStat],
    *,
    org: str,
) -> None:
    """Copy eval/benchmark JSONL files and write a dataset card."""
    eval_root = Path(eval_dir)
    bench_root = Path(benchmark_dir)
    staging = Path(staging_dir)
    eval_dst = staging / "eval_splits"
    bench_dst = staging / "benchmark"
    eval_dst.mkdir(parents=True, exist_ok=True)
    bench_dst.mkdir(parents=True, exist_ok=True)

    for stat in eval_stats:
        shutil.copy2(eval_root / stat.path.name, eval_dst / stat.path.name)
    for stat in bench_stats:
        shutil.copy2(bench_root / stat.path.name, bench_dst / stat.path.name)
    manifest = bench_root / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, bench_dst / "manifest.json")
    (staging / "README.md").write_text(
        generate_eval_card(eval_stats, bench_stats, org),
        encoding="utf-8",
    )


def upload_training(
    org: str,
    *,
    dry_run: bool = False,
    tier_output_dir: str | Path = TIER_OUTPUT_DIR,
    api_factory: Callable[[], Any] | None = None,
) -> UploadSummary:
    """Upload generated training JSONL files to Hugging Face Hub."""
    repo_id = f"{org}/{TRAINING_REPO}"
    stats = collect_training_file_stats(tier_output_dir)
    summary = _summary(repo_id, stats, dry_run=dry_run)
    _log_summary("Training data", summary)

    if dry_run:
        for stat in stats:
            logger.info(
                "  %s - %s rows, %.1f MB",
                stat.relative_path.as_posix(),
                f"{stat.rows:,}",
                stat.size_mb,
            )
        logger.info("[DRY RUN] Would upload to %s", repo_id)
        return summary

    api = _build_api(api_factory)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage_training_upload(tier_output_dir, tmp, stats, org=org)
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload chess SFT training data ({summary.rows:,} examples)",
        )
    logger.info("Done: https://huggingface.co/datasets/%s", repo_id)
    return summary


def upload_eval(
    org: str,
    *,
    dry_run: bool = False,
    eval_dir: str | Path = EVAL_SPLITS_DIR,
    benchmark_dir: str | Path = BENCHMARK_DIR,
    api_factory: Callable[[], Any] | None = None,
) -> UploadSummary:
    """Upload eval splits and frozen benchmark files to Hugging Face Hub."""
    repo_id = f"{org}/{EVAL_REPO}"
    eval_stats, bench_stats = collect_eval_file_stats(eval_dir, benchmark_dir)
    stats = [*eval_stats, *bench_stats]
    summary = _summary(repo_id, stats, dry_run=dry_run)
    _log_summary("Eval and benchmark data", summary)

    if dry_run:
        for stat in eval_stats:
            logger.info("  eval_splits/%s.jsonl - %s rows", stat.name, f"{stat.rows:,}")
        for stat in bench_stats:
            logger.info("  benchmark/%s.jsonl - %s rows", stat.name, f"{stat.rows:,}")
        logger.info("[DRY RUN] Would upload to %s", repo_id)
        return summary

    api = _build_api(api_factory)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage_eval_upload(eval_dir, benchmark_dir, tmp, eval_stats, bench_stats, org=org)
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                "Upload chess SFT eval "
                f"({sum(stat.rows for stat in eval_stats):,} eval + "
                f"{sum(stat.rows for stat in bench_stats):,} benchmark)"
            ),
        )
    logger.info("Done: https://huggingface.co/datasets/%s", repo_id)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload chess SFT data to Hugging Face Hub",
    )
    parser.add_argument("--all", action="store_true", help="Upload both repositories")
    parser.add_argument("--training", action="store_true", help="Upload training data")
    parser.add_argument("--eval", action="store_true", help="Upload eval/benchmark data")
    parser.add_argument("--org", default=DEFAULT_ORG, help=f"HF org (default: {DEFAULT_ORG})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument(
        "--tier-output-dir",
        type=Path,
        default=TIER_OUTPUT_DIR,
        help="Generated tier output directory",
    )
    parser.add_argument(
        "--eval-splits-dir",
        type=Path,
        default=EVAL_SPLITS_DIR,
        help="Eval split directory",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Frozen benchmark directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not (args.all or args.training or args.eval):
        parser.print_help()
        return 1

    try:
        if args.all or args.training:
            upload_training(
                args.org,
                dry_run=args.dry_run,
                tier_output_dir=args.tier_output_dir,
            )
        if args.all or args.eval:
            upload_eval(
                args.org,
                dry_run=args.dry_run,
                eval_dir=args.eval_splits_dir,
                benchmark_dir=args.benchmark_dir,
            )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


def _split_listing(stats: Sequence[UploadFileStat]) -> str:
    listing = "| Split | Description | Examples | Size |\n"
    listing += "|-------|-------------|----------|------|\n"
    for stat in sorted(stats, key=lambda item: item.name or ""):
        name = stat.name or stat.path.stem
        listing += (
            f"| `{name}` | {SPLIT_DESCRIPTIONS.get(name, '')} "
            f"| {stat.rows:,} | {stat.size_mb:.1f} MB |\n"
        )
    return listing


def _build_api(api_factory: Callable[[], Any] | None) -> Any:
    if api_factory is not None:
        return api_factory()
    from huggingface_hub import HfApi

    return HfApi()


def _summary(
    repo_id: str,
    stats: Sequence[UploadFileStat],
    *,
    dry_run: bool,
) -> UploadSummary:
    return UploadSummary(
        repo_id=repo_id,
        files=len(stats),
        rows=sum(stat.rows for stat in stats),
        size_mb=sum(stat.size_mb for stat in stats),
        dry_run=dry_run,
    )


def _log_summary(label: str, summary: UploadSummary) -> None:
    logger.info(
        "%s: %d files, %s examples, %.1f MB",
        label,
        summary.files,
        f"{summary.rows:,}",
        summary.size_mb,
    )


__all__ = [
    "BENCHMARK_DIR",
    "DEFAULT_ORG",
    "EVAL_REPO",
    "EVAL_SPLITS_DIR",
    "SETTINGS",
    "TIER_NAMES",
    "TIER_OUTPUT_DIR",
    "TASK_DESCRIPTIONS",
    "TRAINING_REPO",
    "UploadFileStat",
    "UploadSummary",
    "build_arg_parser",
    "collect_eval_file_stats",
    "collect_training_file_stats",
    "count_lines",
    "file_size_mb",
    "generate_eval_card",
    "generate_training_card",
    "main",
    "stage_eval_upload",
    "stage_training_upload",
    "tier_from_task",
    "upload_eval",
    "upload_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
