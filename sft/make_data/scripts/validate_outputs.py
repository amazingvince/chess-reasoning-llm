#!/usr/bin/env python3
"""Compatibility alias for package-owned SFT output validation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import validate_outputs as _validate_outputs
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import validate_outputs as _validate_outputs

sys.modules[__name__] = _validate_outputs

if __name__ == "__main__":
    raise SystemExit(_validate_outputs.main())
