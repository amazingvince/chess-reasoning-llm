"""Compatibility wrapper for package-owned Lichess puzzle loading."""

from __future__ import annotations

try:
    from config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401
except ModuleNotFoundError:
    from sft.make_data.config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401

from chess_llm.sft.sources.lichess_puzzles import load_puzzles, preprocess_puzzle

_preprocess_puzzle = preprocess_puzzle

__all__ = [
    "_preprocess_puzzle",
    "load_puzzles",
    "preprocess_puzzle",
]
