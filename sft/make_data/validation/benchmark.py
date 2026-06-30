"""Compatibility alias for package-owned benchmark utilities."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.evals import benchmark as _benchmark
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.evals import benchmark as _benchmark
sys.modules[__name__] = _benchmark
