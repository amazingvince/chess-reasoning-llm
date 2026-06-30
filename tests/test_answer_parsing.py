from chess_llm.formats.answers import (
    extract_move,
    extract_uci_from_move_tag,
    parse_answer,
    validate_think_move_format,
)


def test_parse_answer_prefers_move_tag():
    parsed = parse_answer('<think>Candidate moves.</think>\n<move>e2e4</move>')

    assert parsed.move_uci == "e2e4"
    assert parsed.format_type == "move_tag"


def test_parse_answer_accepts_json_move_uci():
    parsed = parse_answer('{"move_uci": "g1f3", "confidence": 0.8}')

    assert parsed.move_uci == "g1f3"
    assert parsed.format_type == "json"


def test_parse_answer_accepts_bare_uci():
    parsed = parse_answer("a7a8q")

    assert parsed.move_uci == "a7a8q"
    assert parsed.format_type == "bare_uci"


def test_parse_answer_accepts_unambiguous_prose_uci():
    parsed = parse_answer("After checking the position, my final move is d2d4.")

    assert parsed.move_uci == "d2d4"
    assert parsed.format_type == "prose_uci"


def test_parse_answer_rejects_ambiguous_prose_moves():
    parsed = parse_answer("I considered e2e4 and d2d4, but the position is unclear.")

    assert parsed.move_uci is None
    assert parsed.format_type == "none"
    assert "ambiguous" in parsed.parse_error


def test_parse_answer_allows_flexible_move_tag_without_relaxing_strict_protocol():
    parsed = parse_answer("<think>Center.</think>\n<move> E2E4 </move>")

    assert parsed.move_uci == "e2e4"
    assert parsed.format_type == "move_tag"
    assert validate_think_move_format("<think>Center.</think>\n<move>e2e4</move>") is True
    assert validate_think_move_format("<think>Center.</think>\n<move> E2E4 </move>") is False


def test_strict_move_tag_extraction_is_for_protocol_metrics():
    assert extract_uci_from_move_tag("<think>x</think><move>a7a8q</move>") == "a7a8q"
    assert extract_uci_from_move_tag("<move> E2E4 </move>") is None
    assert extract_uci_from_move_tag("<MOVE>e2e4</MOVE>") is None


def test_extract_move_preserves_benchmark_first_uci_semantics():
    assert extract_move("<move>e2e4</move>") == "e2e4"
    assert extract_move("e2e4") == "e2e4"
    assert extract_move("I prefer e2e4 over d2d4.") == "e2e4"
    assert extract_move("no move") is None
