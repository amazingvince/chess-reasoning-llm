"""Configuration constants for SFT data generation."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


DEFAULT_OUTPUT_DIR = "chess_sft_data"
_PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_ROOT = Path(
    os.environ.get("XDG_CACHE_HOME")
    or os.environ.get("LOCALAPPDATA")
    or (Path.home() / ".cache")
)
DEFAULT_HF_CACHE_DIR = str(_DEFAULT_CACHE_ROOT / "chess_llm" / "huggingface")
DEFAULT_STOCKFISH_PATH = shutil.which("stockfish") or "stockfish"

DEFAULT_HF_DATASETS: dict[str, str] = {
    "lichess_games": "Lichess/standard-chess-games",
    "lichess_puzzles": "Lichess/chess-puzzles",
    "lichess_openings": "Lichess/chess-openings",
    "lichess_evals": "Lichess/chess-position-evaluations",
    "mate": "OutFlankShu/MATE_DATASET",
}

DEFAULT_VOLUME_LICHESS_GAME_DATA_FILES: tuple[str, ...] = (
    "data/year=2013/month=01/train-00000-of-00001.parquet",
)

DEFAULT_VOLUMES: dict[str, int] = {
    "1.1_fen_to_board": 80_000,
    "1.2_board_to_fen": 80_000,
    "1.3_piece_identification": 100_000,
    "1.4_piece_counting": 80_000,
    "1.5_state_tracking": 100_000,
    "1.6_square_lookup": 100_000,
    "1.7_rank_lookup": 80_000,
    "1.8_move_square_edits": 100_000,
    "1.9_fen_assembly": 100_000,
    "1.10_fen_row_application": 100_000,
    "1.11_square_coordinates": 100_000,
    "1.12_fen_rank_expansion": 100_000,
    "1.13_fen_rank_cell_edit": 100_000,
    "1.14_fen_board_edit": 100_000,
    "1.15_material_inventory": 50_000,
    "1.16_material_piece_counts": 50_000,
    "1.17_material_value_totals": 50_000,
    "1.18_material_balance_trace": 50_000,
    "1.19_multi_move_state_tracking": 40_000,
    "2.0_side_piece_inventory": 60_000,
    "2.1_legal_move_gen": 80_000,
    "2.2_piece_specific_moves": 80_000,
    "2.3_move_legality_check": 80_000,
    "2.4_check_detection": 60_000,
    "2.5_special_rules": 50_000,
    "2.6_piece_pseudo_legal_moves": 60_000,
    "2.7_piece_legal_filter": 60_000,
    "2.8_king_safety_filter": 60_000,
    "2.9_legal_moves_by_piece": 60_000,
    "2.10_ray_walk": 50_000,
    "2.11_legal_filter_trace": 40_000,
    "3.1_available_captures": 60_000,
    "3.2_threats": 50_000,
    "3.3_attacked_defended": 60_000,
    "3.4_tactical_patterns": 50_000,
    "3.5_hanging_pieces": 40_000,
    "4.1_material_balance": 50_000,
    "4.2_position_evaluation": 60_000,
    "4.3_pawn_structure": 40_000,
    "5.1_opening_identification": 10_000,
    "5.2_opening_continuation": 10_000,
    "5.3_opening_principles": 10_000,
    "6.1_endgame_classification": 30_000,
    "6.2_endgame_wdl": 40_000,
    "6.3_endgame_best_move": 40_000,
    "6.4_endgame_principles": 30_000,
    "7.1_best_move_selection": 80_000,
    "7.2_puzzle_solving": 50_000,
    "7.3_move_consequence": 40_000,
    "7.8_candidate_ratings": 40_000,
}

DEFAULT_CHESS960_RATIOS: dict[int, float] = {
    1: 0.20,
    2: 0.20,
    3: 0.15,
    4: 0.10,
    5: 0.05,
    6: 0.05,
    7: 0.10,
}

DEFAULT_EVAL_BUCKETS: list[tuple[int, int, str]] = [
    (0, 50, "equal"),
    (50, 150, "slight edge"),
    (150, 300, "clear advantage"),
    (300, 600, "winning"),
    (600, 100_000, "decisive"),
]

DEFAULT_EVAL_SPLIT_SIZES: dict[str, int] = {
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

DEFAULT_MIN_ELO_GAMES = 1200
DEFAULT_MIN_DEPTH_TRAINING = 20
DEFAULT_MIN_DEPTH_BESTMOVE = 30
DEFAULT_MIN_DEPTH_EVAL_BENCHMARK = 40
DEFAULT_MASTER_SEED = 42
DEFAULT_SELF_PLAY_MAX_POSITIONS = 150_000
DEFAULT_SELF_PLAY_RATIO = 0.25


@dataclass(frozen=True)
class SftDataSettings:
    """Resolved SFT data-generation settings."""

    project_root: Path
    hf_cache_dir: str
    stockfish_path: str
    syzygy_path: str
    polyglot_dir: str
    output_dir: Path
    pool_dir: Path
    eval_splits_dir: Path
    annotations_dir: Path
    tier_output_dir: Path
    benchmark_dir: Path
    self_play_dir: Path | None = None
    self_play_max_positions: int = DEFAULT_SELF_PLAY_MAX_POSITIONS
    self_play_ratio: float = DEFAULT_SELF_PLAY_RATIO
    hf_datasets: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HF_DATASETS))
    volumes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_VOLUMES))
    min_elo_games: int = DEFAULT_MIN_ELO_GAMES
    min_depth_training: int = DEFAULT_MIN_DEPTH_TRAINING
    min_depth_bestmove: int = DEFAULT_MIN_DEPTH_BESTMOVE
    min_depth_eval_benchmark: int = DEFAULT_MIN_DEPTH_EVAL_BENCHMARK
    chess960_ratios: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_CHESS960_RATIOS)
    )
    eval_buckets: list[tuple[int, int, str]] = field(
        default_factory=lambda: list(DEFAULT_EVAL_BUCKETS)
    )
    eval_split_sizes: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_EVAL_SPLIT_SIZES)
    )
    volume_lichess_game_data_files: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_VOLUME_LICHESS_GAME_DATA_FILES)
    )
    master_seed: int = DEFAULT_MASTER_SEED

    @classmethod
    def from_env(
        cls,
        project_root: str | Path,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "SftDataSettings":
        """Resolve settings from an environment mapping without mutating it."""
        values = os.environ if env is None else env
        root = Path(project_root)
        output_override = values.get("CHESS_SFT_OUTPUT")
        if output_override:
            output_dir = Path(output_override)
        else:
            # Anchor the relative default at the project root so runs from
            # different working directories share one data root.
            output_dir = Path(DEFAULT_OUTPUT_DIR)
            if not output_dir.is_absolute():
                output_dir = _PACKAGE_PROJECT_ROOT / output_dir
        return cls(
            project_root=root,
            hf_cache_dir=values.get("HF_HOME", DEFAULT_HF_CACHE_DIR),
            stockfish_path=values.get("STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH),
            syzygy_path=values.get("SYZYGY_PATH", str(root / "data" / "syzygy")),
            polyglot_dir=values.get(
                "POLYGLOT_DIR",
                str(root / "polyglot_opening_books"),
            ),
            output_dir=output_dir,
            pool_dir=output_dir / "pool",
            eval_splits_dir=output_dir / "eval_splits",
            annotations_dir=output_dir / "annotations",
            tier_output_dir=output_dir / "output",
            benchmark_dir=output_dir / "benchmark",
            self_play_dir=(
                Path(values["CHESS_SFT_SELF_PLAY_DIR"])
                if values.get("CHESS_SFT_SELF_PLAY_DIR")
                else output_dir / "self_play"
            ),
            self_play_max_positions=int(
                values.get(
                    "CHESS_SFT_SELF_PLAY_MAX_POSITIONS",
                    DEFAULT_SELF_PLAY_MAX_POSITIONS,
                )
            ),
            self_play_ratio=float(
                values.get("CHESS_SFT_SELF_PLAY_RATIO", DEFAULT_SELF_PLAY_RATIO)
            ),
        )


def apply_hf_cache_env(settings: SftDataSettings) -> None:
    """Apply the legacy HuggingFace cache default using ``setdefault``."""
    if settings.hf_cache_dir:
        os.environ.setdefault("HF_HOME", settings.hf_cache_dir)


__all__ = [
    "DEFAULT_CHESS960_RATIOS",
    "DEFAULT_EVAL_BUCKETS",
    "DEFAULT_EVAL_SPLIT_SIZES",
    "DEFAULT_HF_CACHE_DIR",
    "DEFAULT_HF_DATASETS",
    "DEFAULT_MASTER_SEED",
    "DEFAULT_MIN_DEPTH_BESTMOVE",
    "DEFAULT_MIN_DEPTH_EVAL_BENCHMARK",
    "DEFAULT_MIN_DEPTH_TRAINING",
    "DEFAULT_MIN_ELO_GAMES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SELF_PLAY_MAX_POSITIONS",
    "DEFAULT_SELF_PLAY_RATIO",
    "DEFAULT_STOCKFISH_PATH",
    "DEFAULT_VOLUMES",
    "DEFAULT_VOLUME_LICHESS_GAME_DATA_FILES",
    "SftDataSettings",
    "apply_hf_cache_env",
]
