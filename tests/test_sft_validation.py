import chess

from chess_llm.formats import render_ascii_board
from chess_llm.sft import (
    build_sft_row,
    validate_example,
    validate_fen,
    validate_legal_moves,
    validate_move_legal,
    validate_state_tracking,
    validate_template_complete,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_CASTLE_FEN = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
IMPOSSIBLE_BOARD_FEN = "8/8/8/8/8/8/4P3/4K3 w - - 0 1"


def _starting_board_answer() -> str:
    return render_ascii_board(chess.Board(STARTING_FEN))


def test_package_sft_validation_exports_core_row_checks():
    board = chess.Board(STARTING_FEN)
    legal_moves = sorted(move.uci() for move in board.legal_moves)

    assert validate_fen(STARTING_FEN) is True
    assert validate_fen("8/8/8/8/8/8/8/8 w - - 0 1") is False
    assert validate_legal_moves(STARTING_FEN, legal_moves) is True
    assert validate_move_legal(STARTING_FEN, "e2e4") is True
    assert validate_template_complete("FEN: {fen}") is False


def test_package_sft_validation_rejects_duplicate_legal_moves():
    legal_moves = sorted(move.uci() for move in chess.Board(STARTING_FEN).legal_moves)

    assert validate_legal_moves(STARTING_FEN, legal_moves + [legal_moves[0]]) is False


def test_package_validate_example_accepts_grouped_legal_move_answer():
    answer = (
        "Side to move: white.\n"
        "Pieces to inspect: b1 white knight.\n"
        "Moves by piece:\n"
        "b1 white knight: b1a3 b1c3\n"
        "All legal moves: "
        + " ".join(sorted(move.uci() for move in chess.Board(STARTING_FEN).legal_moves))
    )
    row = build_sft_row(
        task="2.1_legal_move_gen",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content=answer,
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_accepts_compact_legal_move_answer():
    answer = (
        "Side to move: white.\n"
        "Legal moves: "
        + " ".join(sorted(move.uci() for move in chess.Board(STARTING_FEN).legal_moves))
    )
    row = build_sft_row(
        task="2.1_legal_move_gen",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content=answer,
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_accepts_side_piece_inventory_answer():
    row = build_sft_row(
        task="2.0_side_piece_inventory",
        tier=2,
        fen="8/8/8/8/8/8/4P3/R3K2k w - - 0 1",
        user_prompt="FEN: ...\nList side pieces.",
        assistant_content=(
            "Side to move: white.\n"
            "Pieces: a1 white rook; e1 white king; e2 white pawn."
        ),
        metadata={
            "side_to_move": "white",
            "expected_answer": (
                "Side to move: white.\n"
                "Pieces: a1 white rook; e1 white king; e2 white pawn."
            ),
        },
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_decontamination_ignores_hidden_cache_jsonl(tmp_path):
    from chess_llm.core.board import variant_fen_key
    from chess_llm.sft.decontamination import audit_output_files

    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content=_starting_board_answer(),
    )
    visible = tmp_path / "tier1" / "1.1_fen_to_board.jsonl"
    visible.parent.mkdir(parents=True)
    visible.write_text('{"fen": "8/8/8/8/8/8/8/8 w - - 0 1"}\n', encoding="utf-8")
    cache = tmp_path / ".training_cache" / "tier1" / "1.1_fen_to_board.stripped.jsonl"
    cache.parent.mkdir(parents=True)
    cache.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")

    report = audit_output_files(tmp_path, frozenset({variant_fen_key(STARTING_FEN)}))

    assert report == {}


def test_package_sft_validation_helpers_reject_invalid_but_parseable_boards():
    assert validate_fen(IMPOSSIBLE_BOARD_FEN) is False
    assert validate_move_legal(IMPOSSIBLE_BOARD_FEN, "e2e4") is False
    assert validate_legal_moves(IMPOSSIBLE_BOARD_FEN, ["e2e4"]) is False
    assert validate_state_tracking(
        IMPOSSIBLE_BOARD_FEN,
        ["e2e4"],
        "8/8/8/8/4P3/8/8/4K3 b - - 0 1",
    ) is False


def test_package_sft_validation_accepts_chess960_state_tracking():
    board = chess.Board(CHESS960_CASTLE_FEN, chess960=True)
    board.push(board.parse_uci("d1c1"))

    assert validate_state_tracking(
        CHESS960_CASTLE_FEN,
        ["d1c1"],
        board.fen(),
        chess960=True,
    ) is True


def test_package_validate_example_accepts_legal_tier7_row():
    row = build_sft_row(
        task="7.1_best_move_selection",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content="<think>Play for the center.</think>\n<move>e2e4</move>",
    )

    passed, errors = validate_example(row)

    assert passed is True
    assert errors == []


def test_package_validate_example_treats_chess960_id_metadata_as_chess960():
    row = build_sft_row(
        task="7.1_best_move_selection",
        tier=7,
        fen=CHESS960_CASTLE_FEN,
        user_prompt="FEN: ...",
        assistant_content="<think>Castle queenside.</think>\n<move>d1c1</move>",
        metadata={"chess960_id": 3, "target_move": "d1c1"},
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_illegal_tier7_move():
    row = build_sft_row(
        task="7.1_best_move_selection",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content="<think>Move the king.</think>\n<move>e1e3</move>",
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("not legal" in error.lower() for error in errors)


def test_package_validate_example_rejects_tier7_wrong_legal_target_move():
    row = build_sft_row(
        task="7.1_best_move_selection",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content="<think>Play another center pawn.</think>\n<move>d2d4</move>",
        metadata={"target_move": "e2e4"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("target move" in error.lower() for error in errors)


def test_package_validate_example_rejects_state_tracking_answer_mismatch():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result_fen = board.fen()
    row = build_sft_row(
        task="1.5_state_tracking",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMoves: e2e4",
        assistant_content="definitely not a fen",
        metadata={"result_fen": result_fen, "moves": "e2e4"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("assistant" in error.lower() and "result fen" in error.lower() for error in errors)


def test_package_validate_example_accepts_state_tracking_trace_with_result_fen():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result_fen = board.fen()
    row = build_sft_row(
        task="1.5_state_tracking",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMoves: e2e4",
        assistant_content=(
            "Move 1: white pawn moves from e2 to e4.\n"
            "Changed squares: e2 empty, e4 white pawn.\n"
            f"Result FEN: {result_fen}"
        ),
        metadata={"result_fen": result_fen, "moves": "e2e4"},
    )

    passed, errors = validate_example(row)

    assert passed is True
    assert errors == []


def test_package_validate_example_rejects_wrong_board_rendering():
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content="not the board",
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("board rendering" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_board_to_fen_answer():
    row = build_sft_row(
        task="1.2_board_to_fen",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Board: ...",
        assistant_content="8/8/8/8/8/8/8/8 w - - 0 1",
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("fen answer" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_piece_count_answer():
    row = build_sft_row(
        task="1.4_piece_counting",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nHow many queens are on the board?",
        assistant_content="White has 0 queen(s), black has 0 queen(s). Total: 0.",
        metadata={
            "count_kind": "piece_type",
            "piece": "queen",
            "expected_answer": "White has 1 queen(s), black has 1 queen(s). Total: 2.",
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("piece counting" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_square_lookup_answer():
    row = build_sft_row(
        task="1.6_square_lookup",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat is on e2?",
        assistant_content="e2=empty",
        metadata={"square": "e2", "expected_answer": "e2=white pawn"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("square lookup" in error.lower() for error in errors)


def test_package_validate_example_recomputes_square_lookup_from_fen():
    row = build_sft_row(
        task="1.6_square_lookup",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat is on e2?",
        assistant_content="e2=empty",
        metadata={"square": "e2", "expected_answer": "e2=empty"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("square lookup" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_rank_lookup_answer():
    row = build_sft_row(
        task="1.7_rank_lookup",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat is rank 1?",
        assistant_content="rank 1: 8",
        metadata={"rank": "1", "fen_rank_row": "RNBQKBNR", "expected_answer": "rank 1: RNBQKBNR"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("rank lookup" in error.lower() for error in errors)


def test_package_validate_example_recomputes_rank_lookup_from_fen():
    row = build_sft_row(
        task="1.7_rank_lookup",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat is rank 1?",
        assistant_content="rank 1: 8",
        metadata={"rank": "1", "fen_rank_row": "8", "expected_answer": "rank 1: 8"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("rank lookup" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_move_square_edits_answer():
    row = build_sft_row(
        task="1.8_move_square_edits",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content="Lookup: e2=empty; e4=empty.",
        metadata={
            "move": "e2e4",
            "expected_answer": (
                "Lookup: e2=white pawn; e4=empty.\n"
                "Squares: e2 white pawn->empty; e4 empty->white pawn.\n"
                "Ranks: rank 2 e P->1; rank 4 e 1->P."
            ),
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("move square edits" in error.lower() for error in errors)


def test_package_validate_example_recomputes_move_square_edits_from_fen():
    row = build_sft_row(
        task="1.8_move_square_edits",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content=(
            "Lookup: e2=empty; e4=empty.\n"
            "Squares: e2 empty->empty; e4 empty->empty.\n"
            "Ranks: rank 2 e 1->1; rank 4 e 1->1."
        ),
        metadata={
            "move": "e2e4",
            "expected_answer": (
                "Lookup: e2=empty; e4=empty.\n"
                "Squares: e2 empty->empty; e4 empty->empty.\n"
                "Ranks: rank 2 e 1->1; rank 4 e 1->1."
            ),
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("move square edits" in error.lower() for error in errors)


def _e2e4_fen_assembly_answer() -> str:
    result_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    return (
        "Move 1: white pawn e2e4.\n"
        "Lookup: e2=white pawn; e4=empty.\n"
        "Squares: e2 white pawn->empty; e4 empty->white pawn.\n"
        "Ranks: rank 2 e P->1; rank 4 e 1->P.\n"
        f"Result FEN: {result_fen}"
    )


def test_package_validate_example_accepts_fen_assembly_trace():
    result_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    answer = _e2e4_fen_assembly_answer()
    row = build_sft_row(
        task="1.9_fen_assembly",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content=answer,
        metadata={
            "move": "e2e4",
            "result_fen": result_fen,
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_wrong_fen_assembly_answer():
    result_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    answer = _e2e4_fen_assembly_answer()
    row = build_sft_row(
        task="1.9_fen_assembly",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content=answer.replace(result_fen, STARTING_FEN),
        metadata={
            "move": "e2e4",
            "result_fen": result_fen,
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("fen assembly" in error.lower() for error in errors)


def test_package_validate_example_recomputes_fen_assembly_metadata_from_fen():
    wrong_answer = (
        "Move 1: white pawn e2e4.\n"
        "Lookup: e2=empty; e4=empty.\n"
        "Squares: e2 empty->empty; e4 empty->empty.\n"
        "Ranks: rank 2 e 1->1; rank 4 e 1->1.\n"
        f"Result FEN: {STARTING_FEN}"
    )
    row = build_sft_row(
        task="1.9_fen_assembly",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content=wrong_answer,
        metadata={
            "move": "e2e4",
            "result_fen": STARTING_FEN,
            "expected_answer": wrong_answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("fen assembly" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_piece_specific_moves():
    row = build_sft_row(
        task="2.2_piece_specific_moves",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat legal moves does the piece on e2 have?",
        assistant_content="e2e3",
        metadata={"source_square": "e2", "expected_moves": "e2e3 e2e4"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("piece-specific" in error.lower() for error in errors)


def test_package_validate_example_accepts_piece_specific_no_moves():
    row = build_sft_row(
        task="2.2_piece_specific_moves",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat legal moves does the piece on e1 have?",
        assistant_content="No legal moves.",
        metadata={"source_square": "e1", "expected_moves": "No legal moves."},
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_wrong_side_piece_inventory():
    row = build_sft_row(
        task="2.0_side_piece_inventory",
        tier=2,
        fen="8/8/8/8/8/8/4P3/R3K2k w - - 0 1",
        user_prompt="FEN: ...\nList side pieces.",
        assistant_content="Side to move: white.\nPieces: e1 white king.",
        metadata={
            "side_to_move": "white",
            "expected_answer": "Side to move: white.\nPieces: e1 white king.",
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("side piece inventory" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_check_detection_answer():
    row = build_sft_row(
        task="2.4_check_detection",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs the king in check?",
        assistant_content="Check.",
        metadata={"state_label": "normal"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("check detection" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_special_rules_answer():
    row = build_sft_row(
        task="2.5_special_rules",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs en passant possible in this position?",
        assistant_content="En passant possible: e5d6.",
        metadata={"expected_answer": "No en passant is possible in this position."},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("special rules" in error.lower() for error in errors)


def test_package_validate_example_rejects_state_tracking_missing_metadata():
    row = build_sft_row(
        task="1.5_state_tracking",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMoves: e2e4",
        assistant_content=STARTING_FEN,
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("state tracking metadata" in error.lower() for error in errors)


def test_package_validate_example_rejects_unknown_move_legality_answer():
    row = build_sft_row(
        task="2.3_move_legality_check",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs e1e3 legal?",
        assistant_content="maybe",
        metadata={"tested_move": "e1e3"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("yes/no" in error.lower() for error in errors)


def test_package_validate_example_accepts_reason_labeled_move_legality_answer():
    answer = "No, illegal. Reason: empty source."
    row = build_sft_row(
        task="2.3_move_legality_check",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs a3a4 legal?",
        assistant_content=answer,
        metadata={
            "tested_move": "a3a4",
            "expected_is_legal": False,
            "legality_reason_label": "empty_source",
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_missing_move_legality_reason_label():
    row = build_sft_row(
        task="2.3_move_legality_check",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs a3a4 legal?",
        assistant_content="No, illegal. Reason: empty source.",
        metadata={"tested_move": "a3a4"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legality_reason_label" in error for error in errors)


def test_package_validate_example_rejects_wrong_move_legality_reason_label():
    answer = "No, illegal. Reason: wrong side piece."
    row = build_sft_row(
        task="2.3_move_legality_check",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs a3a4 legal?",
        assistant_content=answer,
        metadata={
            "tested_move": "a3a4",
            "expected_is_legal": False,
            "legality_reason_label": "wrong_side_piece",
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legality reason" in error.lower() for error in errors)


def test_package_validate_example_rejects_missing_tested_move_metadata():
    row = build_sft_row(
        task="2.3_move_legality_check",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nIs e2e4 legal?",
        assistant_content="Yes, the move is legal.",
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("tested_move" in error for error in errors)


def test_package_validate_example_reports_malformed_row_schema_without_crashing():
    passed, errors = validate_example(
        {
            "task": "1.1_fen_to_board",
            "tier": "one",
            "fen": STARTING_FEN,
            "messages": [
                {"content": "missing role"},
                {"role": "assistant"},
                "not a message",
            ],
            "metadata": [],
        }
    )

    assert passed is False
    assert any("tier" in error.lower() for error in errors)
    assert any("metadata" in error.lower() for error in errors)
    assert any("message 0" in error.lower() and "role" in error.lower() for error in errors)
    assert any("message 1" in error.lower() and "content" in error.lower() for error in errors)
    assert any("message 2" in error.lower() and "object" in error.lower() for error in errors)


def test_package_validate_example_rejects_non_object_rows_without_crashing():
    passed, errors = validate_example(["not", "a", "row"])

    assert passed is False
    assert errors == ["Example row must be a JSON object"]
