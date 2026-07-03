"""Load harvested self-play position records for SFT source loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Contract produced by chess_llm.autodata.selfplay positions.jsonl rows:
# the lichess_games position shape plus self-play provenance extras.
POSITION_RECORD_KEYS: tuple[str, ...] = (
    "fen",
    "move_played_uci",
    "game_phase",
    "material_balance",
    "ply",
    "game_id",
    "mover",
    "model_id",
    "run_id",
)


def load_self_play_positions(self_play_dir: str | Path | None) -> list[dict]:
    """Return validated position records from ``<dir>/*/positions.jsonl``.

    A missing directory or zero harvested run files returns ``[]`` silently:
    Phase-A data generation runs before any self-play harvest exists.
    Malformed lines and rows missing contract keys are skipped with a warning.
    """
    if self_play_dir is None:
        return []
    root = Path(self_play_dir)
    if not root.is_dir():
        return []

    positions: list[dict] = []
    for path in sorted(root.glob("*/positions.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed self-play position at %s:%d: %s",
                    path,
                    line_number,
                    exc,
                )
                continue
            if not isinstance(row, dict) or any(
                key not in row for key in POSITION_RECORD_KEYS
            ):
                logger.warning(
                    "Skipping self-play position missing contract keys at %s:%d",
                    path,
                    line_number,
                )
                continue
            positions.append(row)
    return positions


__all__ = ["POSITION_RECORD_KEYS", "load_self_play_positions"]
