"""Compatibility wrapper for package-owned Lichess position eval loading."""

from __future__ import annotations

try:
    from config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401
except ModuleNotFoundError:
    from sft.make_data.config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401

from chess_llm.sft.sources import lichess_evals as _package
from chess_llm.sft.sources.lichess_evals import (
    EVAL_PERSPECTIVE,
    _flush_batch,
    _init_dedup_db,
    normalize_pv_line,
    parse_lichess_eval_row,
    stream_evals,
)

board_from_raw = _package.board_from_raw
_normalize_pv_line = normalize_pv_line


def partition_evals(evals):
    """Delegate partitioning while preserving legacy monkeypatch behavior."""
    original = _package.board_from_raw
    _package.board_from_raw = board_from_raw
    try:
        return _package.partition_evals(evals)
    finally:
        _package.board_from_raw = original


__all__ = [
    "EVAL_PERSPECTIVE",
    "_flush_batch",
    "_init_dedup_db",
    "_normalize_pv_line",
    "board_from_raw",
    "normalize_pv_line",
    "parse_lichess_eval_row",
    "partition_evals",
    "stream_evals",
]
