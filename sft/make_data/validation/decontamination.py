"""Legacy wrapper for package-owned decontamination helpers."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from chess_llm.sft.decontamination import audit_output_files, check_no_contamination
except ModuleNotFoundError:
    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.sft.decontamination import audit_output_files, check_no_contamination

__all__ = ["audit_output_files", "check_no_contamination"]
