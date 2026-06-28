#!/usr/bin/env python3
"""Upload chess SFT training data and eval splits to HuggingFace Hub.

Usage:
    python scripts/push_to_hub.py --all              # upload both repos
    python scripts/push_to_hub.py --training          # training data only
    python scripts/push_to_hub.py --eval              # eval + benchmark only
    python scripts/push_to_hub.py --org OTHER_ORG     # override org name
    python scripts/push_to_hub.py --all --dry-run     # show what would be uploaded
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    TIER_OUTPUT_DIR,
    EVAL_SPLITS_DIR,
    BENCHMARK_DIR,
    VOLUMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_ORG = "Chess-Nut-Engine"
TRAINING_REPO = "chess-sft-data"
EVAL_REPO = "chess-sft-eval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def tier_from_task(task_id: str) -> int:
    """Extract tier number from a task id like '3.2_threats'."""
    return int(task_id.split(".")[0])


TIER_NAMES = {
    1: "Perception",
    2: "Rules",
    3: "Tactics",
    4: "Evaluation",
    5: "Openings",
    6: "Endgames",
    7: "Planning",
}


# ---------------------------------------------------------------------------
# Dataset card generators
# ---------------------------------------------------------------------------

def generate_training_card(file_stats: list[dict], org: str) -> str:
    """Generate README.md dataset card for the training repo."""
    total_rows = sum(f["rows"] for f in file_stats)
    total_mb = sum(f["size_mb"] for f in file_stats)
    repo_id = f"{org}/{TRAINING_REPO}"

    # Build file listing grouped by tier
    tiers: dict[int, list[dict]] = {}
    for f in file_stats:
        tiers.setdefault(f["tier"], []).append(f)

    # Build YAML configs for dataset viewer
    configs_yaml = "configs:\n"
    configs_yaml += "  - config_name: default\n"
    configs_yaml += "    data_files:\n"
    configs_yaml += '      - split: train\n        path: "tier*/*.jsonl"\n'
    for tier_num in sorted(tiers):
        tier_name = TIER_NAMES[tier_num].lower()
        configs_yaml += f"  - config_name: tier{tier_num}_{tier_name}\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f'      - split: train\n        path: "tier{tier_num}/*.jsonl"\n'

    # Per-tier summary table
    tier_summary = "| Tier | Category | Tasks | Examples | Size |\n"
    tier_summary += "|------|----------|-------|----------|------|\n"
    for tier_num in sorted(tiers):
        tier_name = TIER_NAMES[tier_num]
        t_rows = sum(f["rows"] for f in tiers[tier_num])
        t_mb = sum(f["size_mb"] for f in tiers[tier_num])
        tier_summary += (
            f"| {tier_num} | {tier_name} | {len(tiers[tier_num])} "
            f"| {t_rows:,} | {t_mb:.0f} MB |\n"
        )

    # Detailed file listing grouped by tier
    file_listing = ""
    task_descriptions = {
        "1.1_fen_to_board": "Render a FEN string as a human-readable board diagram",
        "1.2_board_to_fen": "Convert a board diagram back to FEN notation",
        "1.3_piece_identification": "Identify which piece occupies a given square",
        "1.4_piece_counting": "Count pieces of a specific type/color on the board",
        "1.5_state_tracking": "Extract castling rights, en passant, side to move from FEN",
        "2.1_legal_move_gen": "List all legal moves for the side to move",
        "2.2_piece_specific_moves": "List legal moves for a specific piece on a given square",
        "2.3_move_legality_check": "Determine whether a given move is legal",
        "2.4_check_detection": "Detect if a king is in check, checkmate, or stalemate",
        "2.5_special_rules": "Handle castling, en passant, promotion, and 50-move rule",
        "3.1_available_captures": "Find all available capture moves",
        "3.2_threats": "Identify pieces that are threatening enemy pieces",
        "3.3_attacked_defended": "Determine which squares are attacked or defended",
        "3.4_tactical_patterns": "Recognize forks, pins, skewers, and discovered attacks",
        "3.5_hanging_pieces": "Find undefended pieces that can be captured for free",
        "4.1_material_balance": "Count material and compute the balance in centipawns",
        "4.2_position_evaluation": "Evaluate a position using Stockfish-calibrated assessments",
        "4.3_pawn_structure": "Analyze doubled, isolated, passed, and backward pawns",
        "5.1_opening_identification": "Name the opening from a position or move sequence",
        "5.2_opening_continuation": "Suggest the next book move in a known opening line",
        "5.3_opening_principles": "Explain opening principles relevant to the position",
        "6.1_endgame_classification": "Classify the type of endgame (e.g., KRK, KPK)",
        "6.2_endgame_wdl": "Predict Win/Draw/Loss using Syzygy tablebase probing",
        "6.3_endgame_best_move": "Find the best endgame move via DTZ tablebase lookup",
        "6.4_endgame_principles": "Explain relevant endgame principles for the position",
        "7.1_best_move_selection": "Select the best move from a Stockfish-evaluated position",
        "7.2_puzzle_solving": "Solve a tactical puzzle (from Lichess puzzles database)",
        "7.3_move_consequence": "Predict the evaluation change after a candidate move",
    }

    for tier_num in sorted(tiers):
        tier_name = TIER_NAMES[tier_num]
        file_listing += f"\n### Tier {tier_num} — {tier_name}\n\n"
        file_listing += "| File | Task | Examples | Size |\n"
        file_listing += "|------|------|----------|------|\n"
        for f in sorted(tiers[tier_num], key=lambda x: x["task_id"]):
            desc = task_descriptions.get(f["task_id"], "")
            file_listing += (
                f"| `tier{tier_num}/{f['task_id']}.jsonl` | {desc} "
                f"| {f['rows']:,} | {f['size_mb']:.1f} MB |\n"
            )

    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 1M<n<10M
tags:
  - chess
  - sft
  - instruction-tuning
  - reasoning
  - chess960
pretty_name: Chess SFT Training Data
{configs_yaml}---

# Chess SFT Training Data

A supervised fine-tuning dataset for teaching language models to reason about chess.
It covers **28 tasks** across **7 tiers** of increasing difficulty, from basic board
perception through tactical analysis to endgame play and strategic planning.

Every example uses standard chess conventions: positions are encoded in
[FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation), moves in
[UCI notation](https://en.wikipedia.org/wiki/Universal_Chess_Interface) (e.g. `e2e4`,
`g1f3`, `a7a8q` for promotion), and board diagrams use a consistent rank-file layout.
Approximately 10-20% of examples per tier use
[Chess960](https://en.wikipedia.org/wiki/Fischer_random_chess) starting positions.

| | |
|---|---|
| **Total examples** | {total_rows:,} |
| **Total size** | {total_mb:.1f} MB |
| **Format** | JSONL (chat-style `messages` with system / user / assistant turns) |
| **Eval companion** | [`{org}/{EVAL_REPO}`](https://huggingface.co/datasets/{org}/{EVAL_REPO}) |

## Tier Overview

{tier_summary}

## Example

Each line is a JSON object with the following schema:

```json
{{
  "task": "1.1_fen_to_board",
  "tier": 1,
  "fen": "r4rk1/p1q3pp/2p2b2/2P2p2/PB1P4/5N2/4Q1PP/R3R1K1 w - - 1 22",
  "is_chess960": false,
  "messages": [
    {{
      "role": "system",
      "content": "You are a chess reasoning engine. You understand chess positions in FEN notation and express all moves in UCI notation (e.g., e2e4, g1f3, a7a8q for promotion). When analyzing positions, think step by step."
    }},
    {{
      "role": "user",
      "content": "Position (FEN): r4rk1/p1q3pp/2p2b2/2P2p2/PB1P4/5N2/4Q1PP/R3R1K1 w - - 1 22\\nShow me the board."
    }},
    {{
      "role": "assistant",
      "content": "8 r . . . . r k .\\n7 p . q . . . p p\\n6 . . p . . b . .\\n5 . . P . . p . .\\n4 P B . P . . . .\\n3 . . . . . N . .\\n2 . . . . Q . P P\\n1 R . . . R . K .\\n  a b c d e f g h"
    }}
  ],
  "metadata": {{}}
}}
```

### Key fields

| Field | Description |
|-------|-------------|
| `task` | Task identifier (e.g. `3.2_threats`) — the first digit is the tier number |
| `tier` | Tier number (1-7) |
| `fen` | The chess position in FEN notation |
| `is_chess960` | Whether this example uses a Chess960 starting position |
| `messages` | Chat-format conversation (system + user + assistant) ready for SFT |
| `metadata` | Optional task-specific metadata (e.g. puzzle rating, eval depth) |

## Loading

```python
from datasets import load_dataset

# Load everything (default config)
ds = load_dataset("{repo_id}")

# Load a single tier
ds = load_dataset("{repo_id}", "tier3_tactics")

# Load with streaming (recommended for full dataset)
ds = load_dataset("{repo_id}", streaming=True)
```

## Data Sources

Training positions are sourced from:

- **[Lichess Standard Chess Games](https://huggingface.co/datasets/Lichess/standard-chess-games)** — real game positions (min Elo 1200)
- **[Lichess Chess Puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles)** — tactical puzzles with known solutions
- **[Lichess Chess Openings](https://huggingface.co/datasets/Lichess/chess-openings)** — ECO-classified opening positions
- **[Lichess Position Evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations)** — Stockfish evaluations (depth 20-40+)
- **[MATE Dataset](https://huggingface.co/datasets/OutFlankShu/MATE_DATASET)** — checkmate pattern positions
- **Syzygy Endgame Tablebases** — perfect endgame play (up to 6 pieces)
- **Polyglot Opening Books** — book moves for opening continuations
- **Chess960 Random Positions** — generated Fischer Random starting positions

All eval/benchmark FENs are excluded from training via a blocklist to prevent contamination.

## Detailed File Listing
{file_listing}

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
"""


def generate_eval_card(eval_stats: list[dict], bench_stats: list[dict], org: str) -> str:
    """Generate README.md dataset card for the eval repo."""
    total_eval = sum(f["rows"] for f in eval_stats)
    total_bench = sum(f["rows"] for f in bench_stats)
    repo_id = f"{org}/{EVAL_REPO}"

    split_descriptions = {
        "perception": "Board reading — FEN-to-board, piece identification, state tracking",
        "rules": "Move legality — legal move generation, check detection, special rules",
        "tactics": "Tactical patterns — captures, threats, pins, forks, hanging pieces",
        "evaluation": "Position assessment — material balance, Stockfish-calibrated evaluation",
        "openings": "Opening knowledge — identification, continuation, principles (ECO holdout)",
        "endgames": "Endgame play — classification, WDL prediction, tablebase best moves",
        "planning": "Strategic planning — best move selection, puzzle solving, move consequences",
        "chess960": "Fischer Random — all task types applied to Chess960 positions",
        "mate": "Checkmate patterns — forced mate detection and execution",
    }

    eval_listing = "| Split | Description | Examples | Size |\n"
    eval_listing += "|-------|-------------|----------|------|\n"
    for f in sorted(eval_stats, key=lambda x: x["name"]):
        desc = split_descriptions.get(f["name"], "")
        eval_listing += (
            f"| `{f['name']}` | {desc} "
            f"| {f['rows']:,} | {f['size_mb']:.1f} MB |\n"
        )

    bench_listing = "| Split | Examples | Size |\n|-------|----------|------|\n"
    for f in sorted(bench_stats, key=lambda x: x["name"]):
        bench_listing += (
            f"| `{f['name']}` "
            f"| {f['rows']:,} | {f['size_mb']:.1f} MB |\n"
        )

    # Build configs YAML for eval splits and benchmark
    configs_yaml = "configs:\n"
    configs_yaml += "  - config_name: eval_splits\n"
    configs_yaml += "    data_files:\n"
    configs_yaml += '      - split: test\n        path: "eval_splits/*.jsonl"\n'
    configs_yaml += "  - config_name: benchmark\n"
    configs_yaml += "    data_files:\n"
    configs_yaml += '      - split: test\n        path: "benchmark/*.jsonl"\n'
    # Per-split configs for granular viewing
    for f in sorted(eval_stats, key=lambda x: x["name"]):
        configs_yaml += f"  - config_name: eval_{f['name']}\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f'      - split: test\n        path: "eval_splits/{f["name"]}.jsonl"\n'
    for f in sorted(bench_stats, key=lambda x: x["name"]):
        configs_yaml += f"  - config_name: bench_{f['name']}\n"
        configs_yaml += "    data_files:\n"
        configs_yaml += f'      - split: test\n        path: "benchmark/{f["name"]}.jsonl"\n'

    return f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 10K<n<100K
tags:
  - chess
  - sft
  - evaluation
  - benchmark
  - chess960
pretty_name: Chess SFT Eval & Benchmark
{configs_yaml}---

# Chess SFT Eval & Benchmark

Held-out evaluation splits and a frozen benchmark for the
[Chess SFT training pipeline](https://huggingface.co/datasets/{org}/{TRAINING_REPO}).
Every FEN in these files is **excluded from training data** via a blocklist to guarantee
zero contamination.

| | |
|---|---|
| **Eval examples** | {total_eval:,} |
| **Benchmark examples** | {total_bench:,} |
| **Splits** | 9 (perception, rules, tactics, evaluation, openings, endgames, planning, chess960, mate) |
| **Format** | JSONL |
| **Training companion** | [`{org}/{TRAINING_REPO}`](https://huggingface.co/datasets/{org}/{TRAINING_REPO}) |

## How eval and benchmark differ

- **Eval splits** contain raw held-out positions with ground-truth labels (FENs, legal
  moves, puzzle solutions, etc.). Use these for flexible evaluation with custom metrics.
- **Frozen benchmark** is a deterministic, versioned subset of the eval splits
  (seed=42). Each row has a standardized `prompt` + `gold_answer` + `metric_type` format
  for reproducible scoring. The `manifest.json` file tracks version and split sizes.

## Eval Split Schema

Eval split rows vary by split but share a common core:

```json
{{
  "fen": "r2q1rk1/4P1pp/3p4/2pN4/...",
  "side": "black",
  "legal_moves": ["g8h8", "g8f7", ...],
  ...
}}
```

Fields are task-specific (e.g. `legal_moves` for rules, `puzzle_moves` for tactics).

## Benchmark Schema

Every benchmark row follows a uniform structure for automated scoring:

```json
{{
  "example_id": "perception_00000",
  "split": "perception",
  "task_type": "board_print",
  "fen": "qrkrn1bb/pp1p2pp/2p1np2/...",
  "prompt": "Position (FEN): ...\\nShow me the board.",
  "gold_answer": "8 q r k r n . b b\\n7 ...",
  "metric_type": "exact_match",
  "metadata": {{}}
}}
```

| Field | Description |
|-------|-------------|
| `example_id` | Unique identifier (`{{split}}_{{index}}`) |
| `split` | Which eval category this belongs to |
| `task_type` | Specific task within the split |
| `fen` | Chess position in FEN notation |
| `prompt` | The question to pose to the model |
| `gold_answer` | Ground truth answer for scoring |
| `metric_type` | How to score: `exact_match`, `set_match`, `f1`, or `numeric_tolerance` |
| `metadata` | Optional extra context (puzzle rating, eval depth, etc.) |

## Loading

```python
from datasets import load_dataset

# Load all eval splits
ds = load_dataset("{repo_id}", "eval_splits")

# Load all benchmark rows
ds = load_dataset("{repo_id}", "benchmark")

# Load a single split
ds = load_dataset("{repo_id}", "eval_tactics")
ds = load_dataset("{repo_id}", "bench_tactics")
```

## Eval Splits

{eval_listing}

## Frozen Benchmark

Deterministically sampled from eval splits (seed=42, version tracked in `manifest.json`).

{bench_listing}

## Decontamination

Eval/benchmark positions are held out from training through multiple mechanisms:

1. **FEN blocklist** — every FEN in these splits is on a blocklist checked during generation
2. **ECO holdout** — opening eval positions come from held-out ECO code families
3. **Depth filtering** — evaluation benchmark uses only depth-40+ Stockfish analyses

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
"""


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------

def upload_training(org: str, dry_run: bool = False) -> None:
    """Upload training data to HuggingFace."""
    repo_id = f"{org}/{TRAINING_REPO}"
    tier_root = Path(TIER_OUTPUT_DIR)

    if not tier_root.exists():
        logger.error("Training output dir not found: %s", tier_root)
        sys.exit(1)

    # Gather file stats
    file_stats = []
    jsonl_files = sorted(tier_root.rglob("*.jsonl"))
    if not jsonl_files:
        logger.error("No JSONL files found in %s", tier_root)
        sys.exit(1)

    for path in jsonl_files:
        task_id = path.stem
        rows = count_lines(path)
        file_stats.append({
            "task_id": task_id,
            "tier": tier_from_task(task_id),
            "rows": rows,
            "size_mb": file_size_mb(path),
            "path": path,
        })

    total_rows = sum(f["rows"] for f in file_stats)
    total_mb = sum(f["size_mb"] for f in file_stats)
    logger.info(
        "Training data: %d files, %s examples, %.1f MB",
        len(file_stats), f"{total_rows:,}", total_mb,
    )

    if dry_run:
        for f in file_stats:
            logger.info(
                "  tier%d/%s.jsonl — %s rows, %.1f MB",
                f["tier"], f["task_id"], f"{f['rows']:,}", f["size_mb"],
            )
        logger.info("[DRY RUN] Would upload to %s", repo_id)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Build staging directory with tier subdirs + README
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy tier directories
        for tier_num in range(1, 8):
            src = tier_root / f"tier{tier_num}"
            if src.exists():
                dst = tmp_path / f"tier{tier_num}"
                shutil.copytree(src, dst)

        # Write dataset card
        card = generate_training_card(file_stats, org)
        (tmp_path / "README.md").write_text(card, encoding="utf-8")

        logger.info("Uploading to %s ...", repo_id)
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload chess SFT training data ({total_rows:,} examples)",
        )

    logger.info("Done! https://huggingface.co/datasets/%s", repo_id)


def upload_eval(org: str, dry_run: bool = False) -> None:
    """Upload eval splits and benchmark to HuggingFace."""
    repo_id = f"{org}/{EVAL_REPO}"
    eval_dir = Path(EVAL_SPLITS_DIR)
    bench_dir = Path(BENCHMARK_DIR)

    if not eval_dir.exists():
        logger.error("Eval splits dir not found: %s", eval_dir)
        sys.exit(1)

    # Gather stats
    eval_stats = []
    for path in sorted(eval_dir.glob("*.jsonl")):
        eval_stats.append({
            "name": path.stem,
            "rows": count_lines(path),
            "size_mb": file_size_mb(path),
        })

    bench_stats = []
    for path in sorted(bench_dir.glob("*.jsonl")):
        bench_stats.append({
            "name": path.stem,
            "rows": count_lines(path),
            "size_mb": file_size_mb(path),
        })

    total_eval = sum(f["rows"] for f in eval_stats)
    total_bench = sum(f["rows"] for f in bench_stats)
    logger.info(
        "Eval: %d splits (%s examples), Benchmark: %d splits (%s examples)",
        len(eval_stats), f"{total_eval:,}",
        len(bench_stats), f"{total_bench:,}",
    )

    if dry_run:
        logger.info("Eval splits:")
        for f in eval_stats:
            logger.info("  %s.jsonl — %s rows", f["name"], f"{f['rows']:,}")
        logger.info("Benchmark:")
        for f in bench_stats:
            logger.info("  %s.jsonl — %s rows", f["name"], f"{f['rows']:,}")
        logger.info("[DRY RUN] Would upload to %s", repo_id)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy eval_splits (JSONL only, skip blocklist)
        eval_dst = tmp_path / "eval_splits"
        eval_dst.mkdir()
        for path in eval_dir.glob("*.jsonl"):
            shutil.copy2(path, eval_dst / path.name)

        # Copy benchmark (JSONL + manifest)
        bench_dst = tmp_path / "benchmark"
        bench_dst.mkdir()
        for path in bench_dir.glob("*.jsonl"):
            shutil.copy2(path, bench_dst / path.name)
        manifest = bench_dir / "manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, bench_dst / "manifest.json")

        # Write dataset card
        card = generate_eval_card(eval_stats, bench_stats, org)
        (tmp_path / "README.md").write_text(card, encoding="utf-8")

        logger.info("Uploading to %s ...", repo_id)
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload chess SFT eval ({total_eval:,} eval + {total_bench:,} benchmark)",
        )

    logger.info("Done! https://huggingface.co/datasets/%s", repo_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload chess SFT data to HuggingFace Hub"
    )
    parser.add_argument("--all", action="store_true", help="Upload both training and eval")
    parser.add_argument("--training", action="store_true", help="Upload training data only")
    parser.add_argument("--eval", action="store_true", help="Upload eval/benchmark only")
    parser.add_argument("--org", default=DEFAULT_ORG, help=f"HF org (default: {DEFAULT_ORG})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    if not (args.all or args.training or args.eval):
        parser.print_help()
        return 1

    if args.all or args.training:
        upload_training(args.org, dry_run=args.dry_run)

    if args.all or args.eval:
        upload_eval(args.org, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
