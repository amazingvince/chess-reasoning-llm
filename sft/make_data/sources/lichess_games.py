"""Compatibility wrapper for package-owned Lichess game loading."""

from __future__ import annotations

try:
    from config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401
except ModuleNotFoundError:
    from sft.make_data.config.settings import HF_DATASETS as _HF_DATASETS  # noqa: F401

from chess_llm.sft.sources.lichess_games import (
    extract_game_positions,
    game_phase,
    material_balance,
    stream_games,
)

extract_positions = extract_game_positions
_game_phase = game_phase
_material_balance = material_balance

__all__ = [
    "_game_phase",
    "_material_balance",
    "extract_positions",
    "extract_game_positions",
    "game_phase",
    "material_balance",
    "stream_games",
]
