#!/usr/bin/env python3
"""Compatibility alias for package-owned Polyglot archive extraction."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft import extract_polyglot_books as _extract_polyglot_books
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft import extract_polyglot_books as _extract_polyglot_books

sys.modules[__name__] = _extract_polyglot_books

if __name__ == "__main__":
    raise SystemExit(_extract_polyglot_books.main())
