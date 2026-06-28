from chess_llm.formats.answers import parse_answer


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
