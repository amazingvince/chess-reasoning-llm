"""Single system prompt used across all SFT training examples."""

try:
    from chess_llm.formats.prompts import SYSTEM_PROMPT
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    _SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
    sys.path.insert(0, str(_SRC_ROOT))
    from chess_llm.formats.prompts import SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT"]
