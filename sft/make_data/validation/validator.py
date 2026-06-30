"""Compatibility wrapper for package-owned SFT row validation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.formats.answers import (
        extract_uci_from_move_tag as _extract_uci_from_move_tag,
    )
    from chess_llm.sft.validation import (
        validate_example,
        validate_fen,
        validate_legal_moves,
        validate_move_legal,
        validate_state_tracking,
        validate_template_complete,
        validate_think_move_format,
    )
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.formats.answers import (
        extract_uci_from_move_tag as _extract_uci_from_move_tag,
    )
    from chess_llm.sft.validation import (
        validate_example,
        validate_fen,
        validate_legal_moves,
        validate_move_legal,
        validate_state_tracking,
        validate_template_complete,
        validate_think_move_format,
    )


__all__ = [
    "_extract_uci_from_move_tag",
    "validate_example",
    "validate_fen",
    "validate_legal_moves",
    "validate_move_legal",
    "validate_state_tracking",
    "validate_template_complete",
    "validate_think_move_format",
]
