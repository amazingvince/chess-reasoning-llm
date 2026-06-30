"""Compatibility alias for package-owned generator base classes."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft.generators import base as _module
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.generators import base as _module

sys.modules[__name__] = _module
