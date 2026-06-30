#!/usr/bin/env python3
"""Compatibility wrapper for the package-owned training CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_train_module():
    try:
        from chess_llm.training import train as module
    except ModuleNotFoundError:
        src_root = Path(__file__).resolve().parents[2] / "src"
        sys.path.insert(0, str(src_root))
        from chess_llm.training import train as module
    return module


_module = _load_train_module()

if __name__ == "__main__":
    sys.exit(_module.main())

sys.modules[__name__] = _module
globals().update(_module.__dict__)
