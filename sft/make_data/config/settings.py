"""
Global settings for the chess SFT data pipeline.

Paths, volume targets, depth filters, Chess960 mix ratios, eval buckets,
and reproducibility seeds.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace cache — redirect to E: drive BEFORE any HF import
# ---------------------------------------------------------------------------
HF_CACHE_DIR = "E:/hf_cache"
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # sft/make_data/

STOCKFISH_PATH: str = os.environ.get(
    "STOCKFISH_PATH",
    "C:/Users/amazi/Downloads/stockfish/stockfish/stockfish-windows-x86-64-avx512icl.exe",
)
SYZYGY_PATH: str = os.environ.get(
    "SYZYGY_PATH", str(_PROJECT_ROOT / "data" / "syzygy")
)
POLYGLOT_DIR: str = str(_PROJECT_ROOT / "polyglot_opening_books")

OUTPUT_DIR = Path(os.environ.get("CHESS_SFT_OUTPUT", "E:/chess_sft_data"))
POOL_DIR = OUTPUT_DIR / "pool"
EVAL_SPLITS_DIR = OUTPUT_DIR / "eval_splits"
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
TIER_OUTPUT_DIR = OUTPUT_DIR / "output"

# ---------------------------------------------------------------------------
# HuggingFace dataset IDs
# ---------------------------------------------------------------------------
HF_DATASETS: dict[str, str] = {
    "lichess_games": "Lichess/standard-chess-games",
    "lichess_puzzles": "Lichess/chess-puzzles",
    "lichess_openings": "Lichess/chess-openings",
    "lichess_evals": "Lichess/chess-position-evaluations",
    "mate": "OutFlankShu/MATE_DATASET",
}

# ---------------------------------------------------------------------------
# Volume targets per task  (task_id -> count)
# ---------------------------------------------------------------------------
VOLUMES: dict[str, int] = {
    # Tier 1 — Perception (~440K)
    "1.1_fen_to_board": 80_000,
    "1.2_board_to_fen": 80_000,
    "1.3_piece_identification": 100_000,
    "1.4_piece_counting": 80_000,
    "1.5_state_tracking": 100_000,
    # Tier 2 — Rules (~370K)
    "2.1_legal_move_gen": 100_000,
    "2.2_piece_specific_moves": 80_000,
    "2.3_move_legality_check": 80_000,
    "2.4_check_detection": 60_000,
    "2.5_special_rules": 50_000,
    # Tier 3 — Tactics (~260K)
    "3.1_available_captures": 60_000,
    "3.2_threats": 50_000,
    "3.3_attacked_defended": 60_000,
    "3.4_tactical_patterns": 50_000,
    "3.5_hanging_pieces": 40_000,
    # Tier 4 — Evaluation (~150K)
    "4.1_material_balance": 50_000,
    "4.2_position_evaluation": 60_000,
    "4.3_pawn_structure": 40_000,
    # Tier 5 — Openings (~10K; capped by ~3,630 Lichess openings)
    "5.1_opening_identification": 3_600,
    "5.2_opening_continuation": 3_600,
    "5.3_opening_principles": 3_600,
    # Tier 6 — Endgames (~140K)
    "6.1_endgame_classification": 30_000,
    "6.2_endgame_wdl": 40_000,
    "6.3_endgame_best_move": 40_000,
    "6.4_endgame_principles": 30_000,
    # Tier 7 — Planning (~170K)
    "7.1_best_move_selection": 80_000,
    "7.2_puzzle_solving": 50_000,
    "7.3_move_consequence": 40_000,
}

# ---------------------------------------------------------------------------
# Depth filters
# ---------------------------------------------------------------------------
MIN_DEPTH_TRAINING = 20
MIN_DEPTH_BESTMOVE = 30
MIN_DEPTH_EVAL_BENCHMARK = 40

# ---------------------------------------------------------------------------
# Chess960 mix ratios  (tier number -> fraction of examples that are Chess960)
# ---------------------------------------------------------------------------
CHESS960_RATIOS: dict[int, float] = {
    1: 0.20,
    2: 0.20,
    3: 0.15,
    4: 0.10,
    5: 0.05,
    6: 0.05,
    7: 0.10,
}

# ---------------------------------------------------------------------------
# Evaluation buckets  (low_cp, high_cp, label)
# ---------------------------------------------------------------------------
EVAL_BUCKETS: list[tuple[int, int, str]] = [
    (0, 50, "equal"),
    (50, 150, "slight edge"),
    (150, 300, "clear advantage"),
    (300, 600, "winning"),
    (600, 100_000, "decisive"),
]

# ---------------------------------------------------------------------------
# Eval split sizes  (split_name -> count)
# ---------------------------------------------------------------------------
EVAL_SPLIT_SIZES: dict[str, int] = {
    "perception": 2_000,
    "rules": 2_000,
    "tactics": 2_000,
    "evaluation": 1_500,
    "openings": 500,
    "endgames": 1_500,
    "planning": 2_000,
    "chess960": 500,
    "mate": 1_000,
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
MASTER_SEED = 42
