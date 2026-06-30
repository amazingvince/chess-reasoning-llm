"""Legacy wrapper for package-owned eval split helpers."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from config.settings import EVAL_SPLIT_SIZES as _CONFIG_EVAL_SPLIT_SIZES
    from config.settings import MASTER_SEED
except ModuleNotFoundError:
    from sft.make_data.config.settings import (
        EVAL_SPLIT_SIZES as _CONFIG_EVAL_SPLIT_SIZES,
    )
    from sft.make_data.config.settings import MASTER_SEED

try:
    from chess_llm.sft import eval_split as package_eval_split
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import eval_split as package_eval_split

EVAL_SPLIT_SIZES = dict(_CONFIG_EVAL_SPLIT_SIZES)

canonical_fen_key = package_eval_split.canonical_fen_key
build_blocklist = package_eval_split.build_blocklist
save_eval_splits = package_eval_split.save_eval_splits
load_blocklist = package_eval_split.load_blocklist
partition_eco_codes = package_eval_split.partition_eco_codes


def generate_all_eval_splits(
    sources: dict[str, list[dict]],
    seed: int = MASTER_SEED,
) -> dict[str, list[dict]]:
    """Create held-out eval splits using the monkeypatchable legacy sizes."""
    return package_eval_split.generate_all_eval_splits(
        sources,
        seed=seed,
        split_sizes=EVAL_SPLIT_SIZES,
    )


__all__ = [
    "EVAL_SPLIT_SIZES",
    "build_blocklist",
    "canonical_fen_key",
    "generate_all_eval_splits",
    "load_blocklist",
    "partition_eco_codes",
    "save_eval_splits",
]
