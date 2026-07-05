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


def test_package_validate_example_rejects_stale_legal_move_list_before_trailing_text():
    moves = " ".join(sorted(move.uci() for move in chess.Board(STARTING_FEN).legal_moves))
    answer = (
        "Side to move: white.\n"
        f"Legal moves: {moves}\n"
        "unlabelled trailing line"
    )
    row = build_sft_row(
        task="2.1_legal_move_gen",
        tier=2,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content=answer,
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legal move list" in error.lower() for error in errors)


def test_package_validate_example_accepts_terminal_legal_move_answer_conventions():
    checkmate_fen = "R6k/8/7K/8/8/8/8/8 b - - 0 1"
    for no_move_text in ("none", "No legal moves available."):
        row = build_sft_row(
            task="2.1_legal_move_gen",
            tier=2,
            fen=checkmate_fen,
            user_prompt="FEN: ...",
            assistant_content=f"Side to move: black.\nLegal moves: {no_move_text}",
        )

        passed, errors = validate_example(row)

        assert passed is True, errors


def test_package_validate_example_accepts_piece_identification_answers():
    square_row = build_sft_row(
        task="1.3_piece_identification",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat piece is on e2?",
        assistant_content="white pawn",
        metadata={
            "query_kind": "square_piece",
            "square": "e2",
            "expected_answer": "white pawn",
        },
    )

    passed, errors = validate_example(square_row)

    assert passed is True, errors

    locate_row = build_sft_row(
        task="1.3_piece_identification",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhere are the white rooks?",
        assistant_content="a1 h1",
        metadata={
            "query_kind": "locate_pieces",
            "color": "white",
            "piece": "rook",
            "expected_answer": "a1 h1",
        },
    )

    passed, errors = validate_example(locate_row)

    assert passed is True, errors


def test_package_validate_example_recomputes_piece_identification_from_fen():
    row = build_sft_row(
        task="1.3_piece_identification",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhat piece is on e2?",
        assistant_content="empty",
        metadata={
            "query_kind": "square_piece",
            "square": "e2",
            "expected_answer": "empty",
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("piece identification" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_locate_pieces_answer():
    row = build_sft_row(
        task="1.3_piece_identification",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nWhere are the white rooks?",
        assistant_content="a1",
        metadata={
            "query_kind": "locate_pieces",
            "color": "white",
            "piece": "rook",
            "expected_answer": "a1",
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("piece identification" in error.lower() for error in errors)


def _e2e4_fen_row_application_answer() -> str:
    result_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    return (
        "Rows: rank 2 PPPPPPPP->PPPP1PPP; rank 4 8->4P3.\n"
        f"Result FEN: {result_fen}"
    )


def test_package_validate_example_accepts_fen_row_application_answer():
    result_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    answer = _e2e4_fen_row_application_answer()
    row = build_sft_row(
        task="1.10_fen_row_application",
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


def test_package_validate_example_rejects_wrong_fen_row_application_answer():
    result_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    answer = _e2e4_fen_row_application_answer()
    row = build_sft_row(
        task="1.10_fen_row_application",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content=answer.replace("PPPP1PPP", "PPPPPPPP"),
        metadata={
            "move": "e2e4",
            "result_fen": result_fen,
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("fen row application" in error.lower() for error in errors)


def test_package_validate_example_rejects_fen_row_application_missing_metadata():
    row = build_sft_row(
        task="1.10_fen_row_application",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nMove: e2e4",
        assistant_content="Rows: rank 2 PPPPPPPP->PPPP1PPP.",
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("fen row application" in error.lower() for error in errors)


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


def test_package_validate_example_accepts_history_best_move_only_row():
    row = build_sft_row(
        task="7.11_history_best_move",
        tier=7,
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        user_prompt="Moves so far: e2e4\nFEN: ...",
        assistant_content="<move>e7e5</move>",
        metadata={"target_move": "e7e5"},
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


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


def test_package_validate_example_accepts_candidate_ratings_fixed_grammar():
    answer = "\n".join(
        [
            "Candidate e2e4: +42cp; Bucket: equal",
            "Candidate d2d4: +15cp; Bucket: equal",
            "Candidate g1f3: +5cp; Bucket: equal",
            "Candidate c2c4: -20cp; Bucket: equal",
            "Candidate b1c3: -80cp; Bucket: slight edge",
            "Best: e2e4",
        ]
    )
    row = build_sft_row(
        task="7.8_candidate_ratings",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nRate candidates e2e4 d2d4 g1f3 c2c4 b1c3.",
        assistant_content=answer,
        metadata={
            "target_move": "e2e4",
            "candidate_ratings": [
                {"uci": "e2e4", "cp": 42},
                {"uci": "d2d4", "cp": 15},
                {"uci": "g1f3", "cp": 5},
                {"uci": "c2c4", "cp": -20},
                {"uci": "b1c3", "cp": -80},
            ],
        },
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_candidate_ratings_wrong_best_and_illegal_move():
    row = build_sft_row(
        task="7.8_candidate_ratings",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content="\n".join(
            [
                "Candidate e2e5: +42cp; Bucket: equal",
                "Candidate d2d4: +15cp; Bucket: equal",
                "Candidate g1f3: +5cp; Bucket: equal",
                "Candidate c2c4: -20cp; Bucket: equal",
                "Candidate b1c3: -80cp; Bucket: slight edge",
                "Best: d2d4",
            ]
        ),
        metadata={
            "target_move": "e2e4",
            "candidate_ratings": [
                {"uci": "e2e4", "cp": 42},
                {"uci": "d2d4", "cp": 15},
                {"uci": "g1f3", "cp": 5},
                {"uci": "c2c4", "cp": -20},
                {"uci": "b1c3", "cp": -80},
            ],
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("candidate ratings" in error.lower() for error in errors)
    assert any("not legal" in error.lower() for error in errors)
    assert any("best" in error.lower() for error in errors)


def test_package_validate_example_accepts_best_line_trace_fixed_grammar():
    answer = "\n".join(
        [
            "<think>",
            "Root: e2e4",
            "Eval: +42cp; Bucket: equal",
            "PV: e2e4 e7e5 g1f3 b8c6",
            "Best: e2e4",
            "</think><move>e2e4</move>",
        ]
    )
    row = build_sft_row(
        task="7.10_best_line_trace",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nEmit the fixed grammar engine best line trace.",
        assistant_content=answer,
        metadata={
            "target_move": "e2e4",
            "pv": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_best_line_trace_wrong_or_illegal_pv():
    answer = "\n".join(
        [
            "<think>",
            "Root: e2e4",
            "Eval: +42cp; Bucket: equal",
            "PV: e2e4 e7e6 e2e4",
            "Best: e2e4",
            "</think><move>e2e4</move>",
        ]
    )
    row = build_sft_row(
        task="7.10_best_line_trace",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...",
        assistant_content=answer,
        metadata={
            "target_move": "e2e4",
            "pv": ["e2e4", "e7e5", "g1f3"],
            "expected_answer": answer,
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("best-line trace" in error.lower() for error in errors)
    assert any("pv" in error.lower() for error in errors)


def test_package_validate_example_accepts_step_verification_fixed_grammar():
    row = build_sft_row(
        task="7.9_step_verification",
        tier=7,
        fen=STARTING_FEN,
        user_prompt=(
            "FEN: ...\nTrace to verify:\n"
            "1. Candidate e2e4: +42cp; Bucket: equal\n"
            "2. Best: d2d4"
        ),
        assistant_content="\n".join(
            [
                "Verdict: broken",
                "Faulty line: 2",
                "Error type: wrong_best",
                "Correction: Best should be e2e4.",
            ]
        ),
        metadata={
            "verification_verdict": "broken",
            "faulty_line": 2,
            "error_type": "wrong_best",
            "correction": "Best should be e2e4.",
        },
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


def test_package_validate_example_rejects_step_verification_mismatched_metadata():
    row = build_sft_row(
        task="7.9_step_verification",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nTrace to verify:\n1. Candidate e2e4: +42cp; Bucket: equal",
        assistant_content="\n".join(
            [
                "Verdict: sound",
                "Faulty line: none",
                "Error type: none",
                "Correction: none",
            ]
        ),
        metadata={
            "verification_verdict": "broken",
            "faulty_line": 1,
            "error_type": "wrong_bucket",
            "correction": "Candidate e2e4 bucket should be equal.",
        },
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("step verification" in error.lower() for error in errors)


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


def test_package_validate_example_rejects_wrong_material_trace_answer():
    fen = "8/8/8/8/8/8/6p1/4K2k w - - 0 1"
    row = build_sft_row(
        task="1.18_material_balance_trace",
        tier=1,
        fen=fen,
        user_prompt="FEN: ...\nTrace material.",
        assistant_content=(
            "Inventory: white king=e1; queen=none; rook=none; bishop=none; "
            "knight=none; pawn=none | black king=h1; queen=none; rook=none; "
            "bishop=none; knight=none; pawn=g2\n"
            "Counts: white king=1; queen=0; rook=0; bishop=0; knight=0; pawn=0 | "
            "black king=1; queen=0; rook=0; bishop=0; knight=0; pawn=1\n"
            "Values: white total=0; black total=0\n"
            "Final: Material is equal."
        ),
        metadata={"expected_answer": "wrong"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("material decomposition" in error.lower() for error in errors)


def test_package_validate_example_rejects_old_material_trace_final_label():
    fen = "8/8/8/8/8/8/6p1/4K2k w - - 0 1"
    row = build_sft_row(
        task="1.18_material_balance_trace",
        tier=1,
        fen=fen,
        user_prompt="FEN: ...\nTrace material.",
        assistant_content=(
            "Inventory: white king=e1; queen=none; rook=none; bishop=none; "
            "knight=none; pawn=none | black king=h1; queen=none; rook=none; "
            "bishop=none; knight=none; pawn=g2\n"
            "Counts: white king=1; queen=0; rook=0; bishop=0; knight=0; pawn=0 | "
            "black king=1; queen=0; rook=0; bishop=0; knight=0; pawn=1\n"
            "Values: white total=0; black total=1\n"
            "Final: Black is up 1 point(s) of material."
        ),
        metadata={"expected_answer": "wrong"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("material decomposition" in error.lower() for error in errors)


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


def test_package_validate_example_rejects_wrong_piece_legal_filter_trace():
    fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    row = build_sft_row(
        task="2.7_piece_legal_filter",
        tier=2,
        fen=fen,
        user_prompt="FEN: ...\nFilter pseudo moves from e2.",
        assistant_content=(
            "Pseudo-legal from e2: e2a2 e2b2 e2c2 e2d2 e2e3 e2e4 e2e5 e2e6 "
            "e2e7 e2e8 e2f2 e2g2 e2h2.\n"
            "Legal: e2a2 e2e3.\n"
            "Rejected: none."
        ),
        metadata={"source_square": "e2", "expected_answer": "wrong"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legal decomposition" in error.lower() for error in errors)


def test_package_validate_example_rejects_piece_legal_filter_without_rejection_reasons():
    fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    row = build_sft_row(
        task="2.7_piece_legal_filter",
        tier=2,
        fen=fen,
        user_prompt="FEN: ...\nFilter pseudo moves from e2.",
        assistant_content=(
            "Pseudo-legal from e2: "
            "e2a2 e2b2 e2c2 e2d2 e2e3 e2e4 e2e5 e2e6 e2e7 e2e8 e2f2 e2g2 e2h2.\n"
            "Legal: e2e3 e2e4 e2e5 e2e6 e2e7 e2e8.\n"
            "Rejected: e2a2 e2b2 e2c2 e2d2 e2f2 e2g2 e2h2."
        ),
        metadata={"source_square": "e2", "expected_answer": "wrong"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legal decomposition" in error.lower() for error in errors)


def test_package_validate_example_rejects_wrong_ray_walk_answer():
    fen = "7k/3p4/8/8/3RN3/8/8/7K w - - 0 1"
    corrupted = (
        "Piece: d4 white rook.\n"
        "Ray N: d5 empty; d6 empty; d7 empty; edge.\n"
        "Ray E: e4 white knight: own piece (stop, excluded).\n"
        "Ray S: d3 empty; d2 empty; d1 empty; edge.\n"
        "Ray W: c4 empty; b4 empty; a4 empty; edge.\n"
        "Moves from rays: d4a4 d4b4 d4c4 d4d1 d4d2 d4d3 d4d5 d4d6 d4d7 d4d8"
    )
    row = build_sft_row(
        task="2.10_ray_walk",
        tier=2,
        fen=fen,
        user_prompt="FEN: ...\nWalk each ray for the slider on d4.",
        assistant_content=corrupted,
        metadata={"source_square": "d4", "expected_answer": corrupted},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("ray walk" in error.lower() for error in errors)


def test_package_validate_example_rejects_ray_walk_on_non_slider_square():
    fen = "7k/3p4/8/8/3RN3/8/8/7K w - - 0 1"
    row = build_sft_row(
        task="2.10_ray_walk",
        tier=2,
        fen=fen,
        user_prompt="FEN: ...\nWalk each ray for the slider on e4.",
        assistant_content="Piece: e4 white knight.",
        metadata={"source_square": "e4", "expected_answer": "Piece: e4 white knight."},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("side-to-move slider" in error for error in errors)


def test_package_validate_example_rejects_wrong_legal_filter_trace_answer():
    fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    corrupted = (
        "Side to move: white.\n"
        "Pieces: e1 white king; e2 white rook.\n"
        "Filter by piece:\n"
        "e1 white king: pseudo-legal e1d1 e1d2 e1f1 e1f2 | rejected none | "
        "legal e1d1 e1d2 e1f1 e1f2\n"
        "e2 white rook: pseudo-legal e2e3 | rejected none | legal e2e3\n"
        "All legal moves: e1d1 e1d2 e1f1 e1f2 e2e3"
    )
    row = build_sft_row(
        task="2.11_legal_filter_trace",
        tier=2,
        fen=fen,
        user_prompt="FEN: ...\nTrace pseudo-legal -> rejected -> legal.",
        assistant_content=corrupted,
        metadata={"expected_answer": corrupted},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legal filter trace" in error.lower() for error in errors)


def test_package_validate_example_rejects_legal_filter_trace_wrong_move_metadata():
    fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    from chess_llm.core.legality import format_legal_filter_trace_answer

    answer = format_legal_filter_trace_answer(chess.Board(fen))
    assert answer is not None
    row = build_sft_row(
        task="2.11_legal_filter_trace",
        tier=2,
        fen=fen,
        user_prompt="FEN: ...\nTrace pseudo-legal -> rejected -> legal.",
        assistant_content=answer,
        metadata={"expected_answer": answer, "legal_moves": "e2e3"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("legal_moves metadata" in error for error in errors)


def test_package_validate_example_rejects_multi_move_state_tracking_wrong_answer():
    board = chess.Board(STARTING_FEN)
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    result_fen = board.fen()
    row = build_sft_row(
        task="1.19_multi_move_state_tracking",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nApply e2e4 e7e5.",
        assistant_content=f"Result FEN: {STARTING_FEN}",
        metadata={"result_fen": result_fen, "moves": "e2e4 e7e5"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("state tracking" in error.lower() for error in errors)


def test_package_validate_example_rejects_single_ply_multi_move_state_tracking():
    board = chess.Board(STARTING_FEN)
    board.push_uci("e2e4")
    result_fen = board.fen()
    row = build_sft_row(
        task="1.19_multi_move_state_tracking",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nApply e2e4.",
        assistant_content=f"Result FEN: {result_fen}",
        metadata={"result_fen": result_fen, "moves": "e2e4"},
    )

    passed, errors = validate_example(row)

    assert passed is False
    assert any("at least 2 moves" in error for error in errors)


def test_package_validate_example_accepts_multi_move_state_tracking_answer():
    board = chess.Board(STARTING_FEN)
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    result_fen = board.fen()
    row = build_sft_row(
        task="1.19_multi_move_state_tracking",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nApply e2e4 e7e5.",
        assistant_content=f"Result FEN: {result_fen}",
        metadata={"result_fen": result_fen, "moves": "e2e4 e7e5"},
    )

    passed, errors = validate_example(row)

    assert passed is True, errors


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
        user_prompt="FEN: ...\nWhat is the check state: check, checkmate, stalemate, or normal?",
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
