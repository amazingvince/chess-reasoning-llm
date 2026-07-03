"""Self-guided data generation loop interfaces."""

from chess_llm.autodata.failure_buckets import (
    ILLEGAL_MOVE,
    INVALID_FEN,
    LEGAL_UNSCORED,
    MISSING_FEN,
    PARSE_FAILURE,
)
from chess_llm.autodata.judge import judge_rollout
from chess_llm.autodata.rollouts import build_rollout
from chess_llm.autodata.stockfish_judge import judge_rollout_with_stockfish

# Self-play exports resolve lazily: eager import would double-execute the
# module under ``python -m chess_llm.autodata.selfplay`` and drag the evals
# and SFT source modules into every autodata import.
_SELFPLAY_EXPORTS = frozenset(
    {
        "BatchMoveGenerator",
        "SelfPlayConfig",
        "SelfPlayResult",
        "run_self_play",
    }
)

__all__ = [
    "ILLEGAL_MOVE",
    "INVALID_FEN",
    "LEGAL_UNSCORED",
    "MISSING_FEN",
    "PARSE_FAILURE",
    "BatchMoveGenerator",
    "SelfPlayConfig",
    "SelfPlayResult",
    "build_rollout",
    "judge_rollout",
    "judge_rollout_with_stockfish",
    "run_self_play",
]


def __getattr__(name: str):
    if name in _SELFPLAY_EXPORTS:
        from chess_llm.autodata import selfplay

        return getattr(selfplay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
