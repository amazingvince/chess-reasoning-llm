"""Compatibility alias for package-owned Tier 5 generators."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft.generators import tier5_openings as _module
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.generators import tier5_openings as _module

sys.modules[__name__] = _module
