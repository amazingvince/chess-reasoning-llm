"""Compatibility wrapper for package-owned MATE dataset loading."""

from __future__ import annotations

try:
    from config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401
except ModuleNotFoundError:
    from sft.make_data.config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401

from chess_llm.sft.sources.mate import load_mate, process_mate_row, validate_uci

_process_row = process_mate_row
_validate_uci = validate_uci

__all__ = [
    "_process_row",
    "_validate_uci",
    "load_mate",
    "process_mate_row",
    "validate_uci",
]
