import sys
from pathlib import Path

from chess_llm.formats.prompts import SYSTEM_PROMPT


def test_package_system_prompt_is_legacy_prompt_source_of_truth():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))
    from config.system_prompt import SYSTEM_PROMPT as legacy_prompt

    assert SYSTEM_PROMPT == legacy_prompt
    assert "UCI notation" in SYSTEM_PROMPT
