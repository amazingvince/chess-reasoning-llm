#!/usr/bin/env python3
"""Compatibility alias for the package-owned eval-split CLI."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import run_eval_split as _run_eval_split
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import run_eval_split as _run_eval_split
sys.modules[__name__] = _run_eval_split
if __name__ == "__main__":
    raise SystemExit(_run_eval_split.main())
