"""Compatibility wrapper for package-owned SFT data settings."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft.settings import SftDataSettings, apply_hf_cache_env
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.settings import SftDataSettings, apply_hf_cache_env


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = SftDataSettings.from_env(_PROJECT_ROOT)
apply_hf_cache_env(SETTINGS)

HF_CACHE_DIR = SETTINGS.hf_cache_dir
STOCKFISH_PATH = SETTINGS.stockfish_path
SYZYGY_PATH = SETTINGS.syzygy_path
POLYGLOT_DIR = SETTINGS.polyglot_dir
OUTPUT_DIR = SETTINGS.output_dir
POOL_DIR = SETTINGS.pool_dir
EVAL_SPLITS_DIR = SETTINGS.eval_splits_dir
ANNOTATIONS_DIR = SETTINGS.annotations_dir
TIER_OUTPUT_DIR = SETTINGS.tier_output_dir
BENCHMARK_DIR = SETTINGS.benchmark_dir

HF_DATASETS = dict(SETTINGS.hf_datasets)
VOLUMES = dict(SETTINGS.volumes)
MIN_ELO_GAMES = SETTINGS.min_elo_games
MIN_DEPTH_TRAINING = SETTINGS.min_depth_training
MIN_DEPTH_BESTMOVE = SETTINGS.min_depth_bestmove
MIN_DEPTH_EVAL_BENCHMARK = SETTINGS.min_depth_eval_benchmark
CHESS960_RATIOS = dict(SETTINGS.chess960_ratios)
EVAL_BUCKETS = list(SETTINGS.eval_buckets)
EVAL_SPLIT_SIZES = dict(SETTINGS.eval_split_sizes)
MASTER_SEED = SETTINGS.master_seed

__all__ = [
    "ANNOTATIONS_DIR",
    "BENCHMARK_DIR",
    "CHESS960_RATIOS",
    "EVAL_BUCKETS",
    "EVAL_SPLIT_SIZES",
    "EVAL_SPLITS_DIR",
    "HF_CACHE_DIR",
    "HF_DATASETS",
    "MASTER_SEED",
    "MIN_DEPTH_BESTMOVE",
    "MIN_DEPTH_EVAL_BENCHMARK",
    "MIN_DEPTH_TRAINING",
    "MIN_ELO_GAMES",
    "OUTPUT_DIR",
    "POLYGLOT_DIR",
    "POOL_DIR",
    "SETTINGS",
    "STOCKFISH_PATH",
    "SYZYGY_PATH",
    "TIER_OUTPUT_DIR",
    "VOLUMES",
]
