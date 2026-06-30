#!/usr/bin/env python3
"""Compatibility alias for the package-owned Hugging Face upload CLI."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import hub_upload as _hub_upload
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import hub_upload as _hub_upload
sys.modules[__name__] = _hub_upload
if __name__ == "__main__":
    raise SystemExit(_hub_upload.main())
