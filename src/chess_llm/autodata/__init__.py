"""Self-guided data generation loop interfaces."""

from chess_llm.autodata.failure_buckets import (
    ILLEGAL_MOVE,
    LEGAL_UNSCORED,
    MISSING_FEN,
    PARSE_FAILURE,
)
from chess_llm.autodata.judge import judge_rollout
from chess_llm.autodata.rollouts import build_rollout
from chess_llm.autodata.stockfish_judge import judge_rollout_with_stockfish

__all__ = [
    "ILLEGAL_MOVE",
    "LEGAL_UNSCORED",
    "MISSING_FEN",
    "PARSE_FAILURE",
    "build_rollout",
    "judge_rollout",
    "judge_rollout_with_stockfish",
]
