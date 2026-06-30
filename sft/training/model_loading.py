"""Compatibility wrapper for package-owned model-loading helpers."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.training import model_loading as _model_loading
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.training import model_loading as _model_loading

sys.modules[__name__] = _model_loading
globals().update(_model_loading.__dict__)
