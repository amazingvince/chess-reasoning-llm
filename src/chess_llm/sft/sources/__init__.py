"""Package-owned helpers for SFT data sources."""

from chess_llm.sft.sources.lichess_games import (
    extract_game_positions,
    game_phase,
    material_balance,
    stream_games,
)
from chess_llm.sft.sources.lichess_evals import (
    EVAL_PERSPECTIVE,
    normalize_pv_line,
    partition_evals,
    parse_lichess_eval_row,
    stream_evals,
)
from chess_llm.sft.sources.lichess_puzzles import load_puzzles, preprocess_puzzle
from chess_llm.sft.sources.lichess_openings import load_openings, parse_opening_row
from chess_llm.sft.sources.mate import load_mate, process_mate_row, validate_uci
from chess_llm.sft.sources.polyglot_books import (
    get_weighted_moves,
    load_book,
    scan_book_positions,
)
from chess_llm.sft.sources.syzygy_probing import (
    MATERIAL_CONFIGS,
    best_dtz_move,
    open_tablebase,
    probe,
    sample_endgame_positions,
)
from chess_llm.sft.sources.chess960 import (
    apply_random_moves,
    generate_all,
    generate_random,
    sample_chess960_positions,
)

__all__ = [
    "EVAL_PERSPECTIVE",
    "extract_game_positions",
    "apply_random_moves",
    "generate_all",
    "generate_random",
    "game_phase",
    "get_weighted_moves",
    "load_puzzles",
    "load_book",
    "load_mate",
    "load_openings",
    "material_balance",
    "normalize_pv_line",
    "partition_evals",
    "parse_lichess_eval_row",
    "parse_opening_row",
    "preprocess_puzzle",
    "process_mate_row",
    "probe",
    "sample_chess960_positions",
    "sample_endgame_positions",
    "scan_book_positions",
    "stream_games",
    "stream_evals",
    "validate_uci",
    "MATERIAL_CONFIGS",
    "best_dtz_move",
    "open_tablebase",
]
