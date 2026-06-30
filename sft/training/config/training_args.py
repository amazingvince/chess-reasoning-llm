"""Compatibility wrapper for package-owned SFTConfig construction."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_training_args_module():
    try:
        from chess_llm.training import training_args as module
    except ModuleNotFoundError:
        src_root = Path(__file__).resolve().parents[3] / "src"
        sys.path.insert(0, str(src_root))
        from chess_llm.training import training_args as module
    return module


_module = _load_training_args_module()
sys.modules[__name__] = _module
globals().update(_module.__dict__)
