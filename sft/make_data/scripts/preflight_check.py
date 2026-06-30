#!/usr/bin/env python3
"""Compatibility alias for package-owned SFT preflight checks."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import preflight as _preflight
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import preflight as _preflight

sys.modules[__name__] = _preflight

if __name__ == "__main__":
    raise SystemExit(_preflight.main())
