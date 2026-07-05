from chess_llm.formats.prompts import SYSTEM_PROMPT


def test_package_system_prompt_mentions_uci_notation():
    assert "UCI notation" in SYSTEM_PROMPT
