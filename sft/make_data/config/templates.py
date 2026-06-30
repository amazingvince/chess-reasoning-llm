"""Compatibility alias for package-owned SFT prompt templates."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import templates as _templates
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import templates as _templates

sys.modules[__name__] = _templates
