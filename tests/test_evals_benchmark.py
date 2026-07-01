import importlib
import json
import sys
from pathlib import Path
from random import Random

import chess
from chess_llm.evals import benchmark as packaged
from chess_llm.formats.answers import validate_think_move_format


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS960_CASTLE_FEN = "bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1"
STALEMATE_FEN = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"


def _example(
    *,
    example_id: str = "planning_00000",
    task_type: str = "best_move",
    gold_answer: str = "e2e4",
    metric_type: str = "move_extraction",
    metadata: dict | None = None,
) -> packaged.BenchmarkExample:
    return packaged.BenchmarkExample(
        example_id=example_id,
        split="planning",
        task_type=task_type,
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer=gold_answer,
        metric_type=metric_type,
        metadata=metadata or {},
    )


def test_package_benchmark_import_does_not_pull_legacy_modules():
    for name in list(sys.modules):
        if (
            name.startswith("validation")
            or name.startswith("generators")
            or name == "config"
            or name.startswith("config.")
        ):
            sys.modules.pop(name, None)

    importlib.reload(packaged)

    assert "validation.benchmark" not in sys.modules
    assert "validation.eval_harness" not in sys.modules
    assert "generators.base" not in sys.modules
    assert "config" not in sys.modules


def test_package_freezes_benchmark_without_legacy_modules():
    for name in list(sys.modules):
        if (
            name.startswith("validation")
            or name.startswith("generators")
            or name == "config"
            or name.startswith("config.")
        ):
            sys.modules.pop(name, None)

    importlib.reload(packaged)

    raw = {"fen": STARTING_FEN}
    legal_gold = packaged.derive_gold_answer("legal_moves", raw, Random(42))
    examples = packaged.freeze_split("rules", [raw], seed=42)
    expected_moves = " ".join(sorted(m.uci() for m in chess.Board(STARTING_FEN).legal_moves))

    assert legal_gold == f"Side to move: white.\nLegal moves: {expected_moves}"
    assert examples[0].task_type == "legal_moves"
    assert examples[0].metric_type == "uci_set_jaccard"
    assert examples[0].gold_answer == legal_gold
    assert "validation.benchmark" not in sys.modules
    assert "validation.eval_harness" not in sys.modules
    assert not any(name.startswith("generators.") for name in sys.modules)


def test_package_freeze_rejects_illegal_best_move_gold():
    examples = packaged.freeze_split(
        "planning",
        [{"fen": STARTING_FEN, "best_move": "e2e5"}],
        seed=42,
    )

    assert examples == []


def test_package_freeze_rejects_invalid_binary_choice_gold():
    examples = packaged.freeze_split(
        "mate",
        [{
            "fen": STARTING_FEN,
            "move_a": "e2e4",
            "move_b": "d2d4",
            "better_move": "e2e5",
        }],
        seed=42,
    )

    assert examples == []


def test_package_freeze_rejects_missing_opening_labels():
    examples = packaged.freeze_split(
        "openings",
        [{"fen": STARTING_FEN}],
        seed=42,
    )

    assert examples == []


def test_package_freeze_uses_explicit_no_legal_moves_for_stalemate():
    examples = packaged.freeze_split(
        "rules",
        [{"fen": STALEMATE_FEN}],
        seed=42,
    )

    assert len(examples) == 1
    assert examples[0].task_type == "legal_moves"
    assert examples[0].gold_answer == "Side to move: black.\nLegal moves: No legal moves available."
    assert packaged.validate_oracle(examples) == []


def test_package_board_to_fen_prompt_includes_full_fen_state():
    examples = packaged.freeze_split(
        "perception",
        [
            {"fen": STARTING_FEN},
            {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 7 12"},
        ],
        seed=42,
    )
    board_to_fen = next(example for example in examples if example.task_type == "board_to_fen")

    assert "State needed for full FEN:" in board_to_fen.prompt
    assert "Side to move: white" in board_to_fen.prompt
    assert "Castling rights: KQkq" in board_to_fen.prompt
    assert "En passant: e6" in board_to_fen.prompt
    assert "Halfmove clock: 7" in board_to_fen.prompt
    assert "Fullmove number: 12" in board_to_fen.prompt


def test_package_fen_assembly_prompt_includes_full_fen_state():
    raw_examples = [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - -"} for _ in range(12)]

    examples = packaged.freeze_split("perception", raw_examples, seed=42)
    fen_assembly = next(example for example in examples if example.task_type == "fen_assembly")

    assert "Starting FEN: 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" in fen_assembly.prompt
    assert "State needed for full FEN:" in fen_assembly.prompt
    assert "Side to move: white" in fen_assembly.prompt
    assert "Halfmove clock: 0" in fen_assembly.prompt
    assert "Fullmove number: 1" in fen_assembly.prompt


def test_package_fen_row_application_prompt_and_gold_use_full_rank_rewrites():
    raw_examples = [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"} for _ in range(13)]

    examples = packaged.freeze_split("perception", raw_examples, seed=1)
    row_application = next(
        example for example in examples if example.task_type == "fen_row_application"
    )

    assert "Starting FEN: 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" in row_application.prompt
    assert "Move:" in row_application.prompt
    assert "State needed for full FEN:" in row_application.prompt
    lines = row_application.gold_answer.splitlines()
    assert lines[0].startswith("Rows: rank ")
    assert "->" in lines[0]
    assert lines[1].startswith("Result FEN: ")
    assert "Lookup:" not in row_application.gold_answer
    assert "Squares:" not in row_application.gold_answer
    assert row_application.metric_type == "fen_exact_match"


def test_package_state_tracking_prompt_includes_full_fen_state():
    raw_examples = [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - -"} for _ in range(14)]

    examples = packaged.freeze_split("perception", raw_examples, seed=42)
    state_tracking = next(example for example in examples if example.task_type == "state_tracking")

    assert "Starting FEN: 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" in state_tracking.prompt
    assert "State needed for full FEN:" in state_tracking.prompt
    assert "Side to move: white" in state_tracking.prompt
    assert "Halfmove clock: 0" in state_tracking.prompt
    assert "Fullmove number: 1" in state_tracking.prompt


def test_package_state_tracking_benchmark_gold_matches_answer_contract():
    raw_examples = [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"} for _ in range(14)]

    examples = packaged.freeze_split("perception", raw_examples, seed=42)
    state_tracking = next(example for example in examples if example.task_type == "state_tracking")

    assert state_tracking.gold_answer.startswith("Result FEN: ")
    assert packaged.score_prediction(state_tracking, state_tracking.gold_answer)["primary"] == 1.0


def test_package_format_sensitive_benchmark_prompts_include_answer_contracts():
    perception_examples = packaged.freeze_split(
        "perception",
        [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"} for _ in range(14)],
        seed=42,
    )
    rules_examples = packaged.freeze_split(
        "rules",
        [{"fen": STARTING_FEN} for _ in range(7)],
        seed=42,
    )
    by_task = {
        example.task_type: example
        for example in [*perception_examples, *rules_examples]
    }

    assert (
        "Answer format: return exactly one complete six-field FEN"
        in by_task["board_to_fen"].prompt
    )
    assert (
        "Answer format: for a square query, return only the piece name or empty"
        in by_task["piece_id"].prompt
    )
    assert (
        'Answer format: return exactly "<square>=<piece>" or "<square>=empty".'
        in by_task["square_lookup"].prompt
    )
    assert (
        'Answer format: return exactly "rank N: <compressed FEN row>".'
        in by_task["rank_lookup"].prompt
    )
    assert (
        'Answer format: return exactly three lines starting "Lookup:", "Squares:", and "Ranks:"'
        in by_task["move_square_edits"].prompt
    )
    assert (
        'final line "Result FEN: <complete six-field FEN>"'
        in by_task["fen_assembly"].prompt
    )
    assert (
        'first line starts "Rows:" and final line is "Result FEN: <complete six-field FEN>"'
        in by_task["fen_row_application"].prompt
    )
    assert (
        'Answer format: return exactly "Result FEN: <complete six-field FEN>".'
        in by_task["state_tracking"].prompt
    )
    assert (
        'Answer format: return exactly two lines: "Side to move: <white|black>."'
        in by_task["legal_moves"].prompt
    )


def test_package_state_tracking_benchmark_uses_one_ply_by_default():
    raw = {"fen": STARTING_FEN}

    moves, _result_fen = packaged._derive_state_tracking(STARTING_FEN, Random(1))

    assert len(moves.split()) == 1


def test_package_board_to_fen_gold_canonicalizes_to_complete_fen():
    raw = {"fen": "4k3/8/8/8/8/8/4K3/8 w - -"}

    assert packaged.derive_gold_answer("board_to_fen", raw, Random(42)) == (
        "4k3/8/8/8/8/8/4K3/8 w - - 0 1"
    )


def test_package_mechanics_benchmark_derives_square_rank_and_edit_tasks():
    raw_examples = [{"fen": STARTING_FEN} for _ in range(12)]

    examples = packaged.freeze_split("perception", raw_examples, seed=42)
    by_task = {example.task_type: example for example in examples}

    assert "square_lookup" in by_task
    assert "rank_lookup" in by_task
    assert "move_square_edits" in by_task
    assert "fen_assembly" in by_task

    square = by_task["square_lookup"]
    assert square.metric_type == "text_exact_match"
    assert square.metadata["diagnostic"] is True
    assert square.metadata["hard_gate"] is False
    assert square.gold_answer.startswith(square.metadata["square"] + "=")
    assert square.metadata["square"] in square.prompt

    rank = by_task["rank_lookup"]
    assert rank.metric_type == "text_exact_match"
    assert rank.gold_answer == f"rank {rank.metadata['rank']}: {rank.metadata['fen_rank_row']}"
    assert f"rank {rank.metadata['rank']}" in rank.prompt

    edits = by_task["move_square_edits"]
    assert edits.metric_type == "text_exact_match"
    assert edits.metadata["move"] in edits.prompt
    assert edits.gold_answer.startswith("Lookup: ")
    assert "\nSquares: " in edits.gold_answer
    assert "\nRanks: " in edits.gold_answer
    assert "Result FEN:" not in edits.gold_answer

    assembly = by_task["fen_assembly"]
    assert assembly.metric_type == "fen_exact_match"
    assert assembly.metadata["diagnostic"] is True
    assert assembly.metadata["hard_gate"] is False
    assert assembly.metadata["move"] in assembly.prompt
    assert assembly.gold_answer.startswith("Move 1: ")
    assert "\nLookup: " in assembly.gold_answer
    assert "\nSquares: " in assembly.gold_answer
    assert "\nRanks: " in assembly.gold_answer
    assert assembly.gold_answer.count("Result FEN:") == 1


def test_package_atomic_fen_edit_benchmark_derives_prerequisite_tasks():
    raw_examples = [{"fen": STARTING_FEN} for _ in range(14)]

    examples = packaged.freeze_split("perception", raw_examples, seed=42)
    by_task = {example.task_type: example for example in examples}

    assert "square_coordinates" in by_task
    assert "fen_rank_expansion" in by_task
    assert "fen_rank_cell_edit" in by_task
    assert "fen_board_edit" in by_task

    square = by_task["square_coordinates"]
    assert square.metric_type == "text_exact_match"
    assert square.metadata["diagnostic"] is True
    assert square.metadata["hard_gate"] is False
    assert square.metadata["square"] in square.prompt
    assert square.gold_answer.startswith(square.metadata["square"] + ": ")
    assert "fen_row_from_top=" in square.gold_answer

    expansion = by_task["fen_rank_expansion"]
    assert expansion.metric_type == "text_exact_match"
    assert expansion.metadata["fen_rank_row"] in expansion.prompt
    assert expansion.gold_answer.startswith(f"rank {expansion.metadata['rank']}: ")
    assert "a=" in expansion.gold_answer
    assert "h=" in expansion.gold_answer

    cell_edit = by_task["fen_rank_cell_edit"]
    assert cell_edit.metric_type == "text_exact_match"
    assert cell_edit.metadata["file"] in cell_edit.prompt
    assert cell_edit.metadata["before_row"] in cell_edit.prompt
    assert cell_edit.gold_answer == (
        f"rank {cell_edit.metadata['rank']}: "
        f"{cell_edit.metadata['before_row']} -> {cell_edit.metadata['after_row']}"
    )

    board_edit = by_task["fen_board_edit"]
    assert board_edit.metric_type == "text_exact_match"
    assert board_edit.metadata["board_fen_before"] in board_edit.prompt
    assert board_edit.metadata["edit_text"] in board_edit.prompt
    assert board_edit.gold_answer == (
        f"Result board FEN: {board_edit.metadata['board_fen_after']}"
    )


def test_package_rules_benchmark_derives_move_mechanics_tasks():
    raw_examples = [{"fen": STARTING_FEN} for _ in range(7)]

    examples = packaged.freeze_split("rules", raw_examples, seed=42)
    by_task = {example.task_type: example for example in examples}

    assert "side_piece_inventory" in by_task
    assert "piece_legal_moves" in by_task
    assert "legality_check" in by_task

    inventory = by_task["side_piece_inventory"]
    assert inventory.metric_type == "side_piece_inventory"
    assert inventory.metadata["diagnostic"] is True
    assert inventory.metadata["hard_gate"] is False
    assert inventory.gold_answer.startswith("Side to move: white.\nPieces: ")
    assert "e1 white king" in inventory.gold_answer

    piece_moves = by_task["piece_legal_moves"]
    assert piece_moves.metric_type == "uci_set_jaccard"
    assert piece_moves.metadata["diagnostic"] is True
    assert piece_moves.metadata["hard_gate"] is False
    assert piece_moves.metadata["source_square"] in piece_moves.prompt
    assert packaged.score_prediction(piece_moves, piece_moves.gold_answer)["primary"] == 1.0

    legality = by_task["legality_check"]
    assert legality.metric_type == "legality_check"
    assert "Reason:" in legality.gold_answer
    assert legality.metadata["legality_reason_label"]
    assert legality.metadata["move"] in legality.prompt
    assert packaged.score_prediction(legality, legality.gold_answer)["primary"] == 1.0
    assert packaged.score_prediction(legality, legality.gold_answer)["legality_reason_accuracy"] == 1.0


def test_package_benchmark_derives_material_decomposition_diagnostics():
    raw_examples = [{"fen": "8/8/8/8/8/8/6p1/4K2k w - - 0 1"} for _ in range(18)]

    examples = packaged.freeze_split("perception", raw_examples, seed=42)
    by_task = {example.task_type: example for example in examples}

    for task_type in (
        "material_inventory",
        "material_piece_counts",
        "material_value_totals",
        "material_balance_trace",
    ):
        example = by_task[task_type]
        assert example.metric_type == "text_exact_match"
        assert example.metadata["diagnostic"] is True
        assert example.metadata["hard_gate"] is False
        assert packaged.score_prediction(example, example.gold_answer)["primary"] == 1.0

    assert by_task["material_balance_trace"].gold_answer.endswith(
        "Final: Black is up 1 point(s) of material."
    )


def test_package_benchmark_derives_legal_decomposition_diagnostics():
    raw_examples = [{"fen": "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"} for _ in range(11)]

    examples = packaged.freeze_split("rules", raw_examples, seed=42)
    by_task = {example.task_type: example for example in examples}

    for task_type in (
        "piece_pseudo_legal_moves",
        "piece_legal_filter",
        "king_safety_filter",
        "legal_moves_by_piece",
    ):
        example = by_task[task_type]
        assert example.metric_type == "text_exact_match"
        assert example.metadata["diagnostic"] is True
        assert example.metadata["hard_gate"] is False
        assert packaged.score_prediction(example, example.gold_answer)["primary"] == 1.0

    assert "Rejected:" in by_task["piece_legal_filter"].gold_answer


def test_package_piece_legal_moves_benchmark_can_score_no_move_piece():
    raw = {"fen": STARTING_FEN}

    gold = packaged.derive_gold_answer("piece_legal_moves", raw, Random(1))

    assert raw["_piece_legal_moves_square"] == "e1"
    assert gold == "No legal moves."
    example = packaged.BenchmarkExample(
        example_id="rules_00000",
        split="rules",
        task_type="piece_legal_moves",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer=gold,
        metric_type="uci_set_jaccard",
        metadata={"source_square": "e1"},
    )
    assert packaged.score_prediction(example, "No legal moves from e1.")["primary"] == 1.0


def test_package_side_piece_inventory_metric_gives_partial_square_piece_credit():
    example = _example(
        task_type="side_piece_inventory",
        gold_answer=(
            "Side to move: white.\n"
            "Pieces: a1 white rook; e1 white king; e2 white pawn."
        ),
        metric_type="side_piece_inventory",
    )

    scores = packaged.score_prediction(
        example,
        (
            "Side to move: white.\n"
            "Pieces: a1 white rook; e1 white king; d2 white pawn."
        ),
    )

    assert round(scores["primary"], 3) == 0.667


def test_package_side_piece_inventory_scoring_overrides_stale_exact_metric_type():
    example = _example(
        task_type="side_piece_inventory",
        gold_answer=(
            "Side to move: white.\n"
            "Pieces: a1 white rook; e1 white king; e2 white pawn."
        ),
        metric_type="text_exact_match",
    )

    scores = packaged.score_prediction(
        example,
        (
            "Side to move: white.\n"
            "Pieces: a1 white rook; e1 white king; d2 white pawn."
        ),
    )

    assert round(scores["primary"], 3) == 0.667


def test_package_legality_check_scores_binary_answer_and_reason_separately():
    example = _example(
        task_type="legality_check",
        gold_answer="No, illegal. Reason: empty source.",
        metric_type="exact_match",
        metadata={"legality_reason_label": "empty_source"},
    )

    wrong_reason = packaged.score_prediction(
        example,
        "No, illegal. Reason: wrong side piece.",
    )
    wrong_binary = packaged.score_prediction(
        example,
        "Yes, legal. Reason: legal.",
    )

    assert wrong_reason["primary"] == 1.0
    assert wrong_reason["legality_reason_accuracy"] == 0.0
    assert wrong_binary["primary"] == 0.0
    assert wrong_binary["legality_reason_accuracy"] == 0.0


def test_package_score_split_aggregates_legality_reason_accuracy():
    examples = [
        _example(
            example_id="rules_00000",
            task_type="legality_check",
            gold_answer="No, illegal. Reason: empty source.",
            metric_type="legality_check",
            metadata={"legality_reason_label": "empty_source"},
        ),
        _example(
            example_id="rules_00001",
            task_type="legality_check",
            gold_answer="Yes, legal. Reason: legal.",
            metric_type="legality_check",
            metadata={"legality_reason_label": "legal"},
        ),
    ]

    metrics = packaged.score_split(
        examples,
        {
            "rules_00000": "No, illegal. Reason: wrong side piece.",
            "rules_00001": "Yes, legal. Reason: legal.",
        },
    )

    assert metrics["legality_check"] == 1.0
    assert metrics["legality_check_legality_reason_accuracy"] == 0.5


def test_package_text_exact_match_preserves_case_for_fen_mechanics():
    example = _example(
        task_type="rank_lookup",
        gold_answer="rank 1: RNBQKBNR",
        metric_type="text_exact_match",
    )

    assert packaged.score_prediction(example, "rank 1: RNBQKBNR")["primary"] == 1.0
    assert packaged.score_prediction(example, "rank 1: rnbqkbnr")["primary"] == 0.0


def test_package_fen_exact_match_preserves_piece_case():
    example = _example(
        task_type="board_to_fen",
        gold_answer="4k3/8/8/8/8/8/4K3/8 w - - 0 1",
        metric_type="fen_exact_match",
    )

    scores = packaged.score_prediction(
        example,
        "4K3/8/8/8/8/8/4k3/8 w - - 0 1",
    )

    assert scores["primary"] == 0.0


def test_package_fen_exact_match_rejects_missing_clock_fields():
    example = _example(
        task_type="fen_assembly",
        gold_answer="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        metric_type="fen_exact_match",
    )

    scores = packaged.score_prediction(
        example,
        "Result FEN: 4k3/8/8/8/8/8/4P3/4K3 w - -",
    )

    assert scores["primary"] == 0.0
    assert scores["fen_first4"] == 1.0


def test_package_state_tracking_trace_scores_by_result_fen():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result_fen = board.fen()
    example = _example(
        task_type="state_tracking",
        gold_answer=result_fen,
        metric_type="fen_exact_match",
    )

    scores = packaged.score_prediction(
        example,
        (
            "Move 1: white pawn moves from e2 to e4.\n"
            "Changed squares: e2 empty, e4 white pawn.\n"
            f"Result FEN: {result_fen}"
        ),
    )

    assert scores["primary"] == 1.0


def test_package_fen_assembly_trace_scores_by_result_fen():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result_fen = board.fen()
    example = _example(
        task_type="fen_assembly",
        gold_answer=(
            "Move 1: white pawn e2e4.\n"
            "Lookup: e2=white pawn; e4=empty.\n"
            "Squares: e2 white pawn->empty; e4 empty->white pawn.\n"
            "Ranks: rank 2 e P->1; rank 4 e 1->P.\n"
            f"Result FEN: {result_fen}"
        ),
        metric_type="fen_exact_match",
    )

    scores = packaged.score_prediction(example, f"Result FEN: {result_fen}")

    assert scores["primary"] == 1.0


def test_package_fen_exact_match_prefers_explicit_result_fen_over_echoed_context():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result_fen = board.fen()
    example = _example(
        task_type="state_tracking",
        gold_answer=result_fen,
        metric_type="fen_exact_match",
    )

    scores = packaged.score_prediction(
        example,
        f"Starting FEN: {STARTING_FEN}\nResult FEN: {result_fen}",
    )

    assert scores["primary"] == 1.0


def test_package_board_exact_match_preserves_piece_case():
    example = _example(
        task_type="board_print",
        gold_answer="8 k . . . . . . .",
        metric_type="board_exact_match",
    )

    scores = packaged.score_prediction(example, "8 K . . . . . . .")

    assert scores["primary"] == 0.0


def test_package_uci_set_jaccard_parses_prose_and_punctuation():
    example = _example(
        task_type="legal_moves",
        gold_answer="d2d4 e2e4",
        metric_type="uci_set_jaccard",
    )

    scores = packaged.score_prediction(example, "Legal moves: e2e4, d2d4.")

    assert scores["primary"] == 1.0


def test_package_uci_set_jaccard_accepts_no_move_variants():
    example = _example(
        task_type="captures",
        gold_answer="No captures available.",
        metric_type="uci_set_jaccard",
    )

    scores = packaged.score_prediction(example, "There are no captures.")

    assert scores["primary"] == 1.0


def test_package_uci_set_jaccard_rejects_blank_no_move_prediction():
    example = _example(
        task_type="legal_moves",
        gold_answer="No legal moves available.",
        metric_type="uci_set_jaccard",
    )

    scores = packaged.score_prediction(example, "")

    assert scores["primary"] == 0.0


def test_package_check_state_metric_accepts_semantic_labels():
    example = _example(
        task_type="check_detection",
        gold_answer="Normal position -- no check, checkmate, or stalemate.",
        metric_type="check_state",
    )

    assert packaged.score_prediction(example, "Normal")["primary"] == 1.0
    assert packaged.score_prediction(example, "none")["primary"] == 1.0
    assert packaged.score_prediction(example, "Check.")["primary"] == 0.0


def test_package_check_detection_exact_metric_uses_semantic_labels_for_legacy_benchmarks():
    example = _example(
        task_type="check_detection",
        gold_answer="Normal position -- no check, checkmate, or stalemate.",
        metric_type="exact_match",
    )

    assert packaged.score_prediction(example, "Normal")["primary"] == 1.0


def test_package_material_count_metric_scores_structured_counts_without_exact_prose():
    gold = (
        "White (1 pieces): 1 king. "
        "Black (2 pieces): 1 king, 1 pawn. "
        "Black is up 1 point(s) of material."
    )
    example = _example(
        task_type="material_count",
        gold_answer=gold,
        metric_type="material_count",
    )

    prediction = (
        "Inventory:\n"
        "White: king=e1.\n"
        "Black: king=h1; pawns=g2.\n"
        "Counts:\n"
        "White has 1 king and 0 pawns. "
        "Black has 1 king and 1 pawn. "
        "Black is up 1 point."
    )

    assert packaged.score_prediction(example, prediction)["primary"] == 1.0
    assert packaged.score_prediction(
        example,
        "White has 1 king. Black has 1 king. Material is equal.",
    )["primary"] == 0.0


def test_package_special_rules_gold_lists_exact_promotion_moves():
    raw = {"fen": "7k/P7/8/8/8/8/8/4K3 w - - 0 1"}

    gold = packaged.derive_gold_answer("special_rules", raw, Random(42))

    assert "a7a8q" in gold
    assert "a7a8r" in gold
    assert "a7a8b" in gold
    assert "a7a8n" in gold
    assert "Promotion possible from:" not in gold


def test_package_special_rules_metric_accepts_equivalent_no_special_answer():
    example = _example(
        task_type="special_rules",
        gold_answer="No special moves available.",
        metric_type="exact_match",
    )

    assert packaged.score_prediction(
        example,
        "No castling is available for the side to move.",
    )["primary"] == 1.0
    assert packaged.score_prediction(
        example,
        "Castling available: kingside.",
    )["primary"] == 0.0


def test_package_special_rules_metric_accepts_descriptive_promotion_answer():
    gold = "Promotion possible: a7a8b a7a8n a7a8q a7a8r."
    example = _example(
        task_type="special_rules",
        gold_answer=gold,
        metric_type="exact_match",
    )

    prediction = (
        "The pawn on a7 can promote with: "
        "a7a8q (promote to queen on a8), "
        "a7a8r (promote to rook on a8), "
        "a7a8b (promote to bishop on a8), "
        "a7a8n (promote to knight on a8)."
    )

    assert packaged.score_prediction(example, prediction)["primary"] == 1.0
    assert packaged.score_prediction(
        example,
        "No promotion is available for the side to move.",
    )["primary"] == 0.0


def test_package_freeze_and_save_writes_manifest_and_jsonl(tmp_path):
    manifest = packaged.freeze_and_save(
        {"rules": [{"fen": STARTING_FEN}]},
        tmp_path,
        seed=42,
        version="unit-v1",
        strict_coverage=False,
    )

    rows = packaged.load_benchmark(tmp_path / "rules.jsonl")

    assert manifest["version"] == "unit-v1"
    assert manifest["splits"] == {"rules": 1}
    assert rows[0].example_id == "rules_00000"
    assert rows[0].task_type == "legal_moves"


def test_benchmark_example_jsonl_round_trip(tmp_path):
    path = tmp_path / "planning.jsonl"
    original = [
        _example(metadata={"source": "unit"}),
        _example(example_id="planning_00001", gold_answer="d2d4"),
    ]

    packaged.save_benchmark(original, path)
    loaded = packaged.load_benchmark(path)

    assert [example.to_dict() for example in loaded] == [
        example.to_dict() for example in original
    ]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == original[0].to_dict()


def test_package_load_benchmark_stops_after_max_examples(tmp_path):
    path = tmp_path / "planning.jsonl"
    first = _example().to_dict()
    path.write_text(
        json.dumps(first) + "\n"
        "{not valid json after the requested cap}\n",
        encoding="utf-8",
    )

    loaded = packaged.load_benchmark(path, max_examples=1)

    assert [example.example_id for example in loaded] == ["planning_00000"]


def test_package_load_benchmark_applies_task_filter_before_max_examples(tmp_path):
    path = tmp_path / "rules.jsonl"
    out_of_scope = _example(
        example_id="rules_00000",
        task_type="captures",
    ).to_dict()
    in_scope = _example(
        example_id="rules_00001",
        task_type="legal_moves",
    ).to_dict()
    path.write_text(
        json.dumps(out_of_scope) + "\n"
        + json.dumps(in_scope) + "\n"
        "{not valid json after the requested cap}\n",
        encoding="utf-8",
    )

    loaded = packaged.load_benchmark(
        path,
        max_examples=1,
        task_types={"legal_moves"},
    )

    assert [example.example_id for example in loaded] == ["rules_00001"]


def test_package_scoring_metrics_cover_move_protocol_and_acpl():
    example = _example()
    prediction = "<think>control center</think><move>e2e4</move>"

    scores = packaged.score_prediction(example, prediction)
    aggregate = packaged.score_split(
        [example],
        {"planning_00000": prediction},
        acpl_scores={"planning_00000": 12.0},
    )

    assert scores["primary"] == 1.0
    assert scores["format_compliance"] == 1.0
    assert scores["legal_move"] == 1.0
    assert aggregate["best_move"] == 1.0
    assert aggregate["best_move_acpl"] == 12.0
    assert aggregate["acpl"] == 12.0
    assert aggregate["overall"] == 1.0


def test_package_scoring_treats_chess960_id_as_chess960():
    example = packaged.BenchmarkExample(
        example_id="chess960_00000",
        split="chess960",
        task_type="best_move",
        fen=CHESS960_CASTLE_FEN,
        prompt="FEN: ...",
        gold_answer="f1g1",
        metric_type="move_extraction",
        metadata={"chess960_id": 3},
    )

    scores = packaged.score_prediction(
        example,
        "<think>castle kingside</think><move>f1g1</move>",
    )

    assert scores["primary"] == 1.0
    assert scores["legal_move"] == 1.0


def test_package_score_split_reports_fen_first4_match():
    board = chess.Board(STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    gold = board.fen()
    examples = [
        _example(
            example_id="perception_00000",
            task_type="fen_assembly",
            gold_answer=gold,
            metric_type="fen_exact_match",
        ),
        _example(
            example_id="perception_00001",
            task_type="fen_assembly",
            gold_answer=gold,
            metric_type="fen_exact_match",
        ),
    ]

    metrics = packaged.score_split(
        examples,
        {
            "perception_00000": "Result FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 9 9",
            "perception_00001": "Result FEN: rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        },
    )

    assert metrics["fen_assembly"] == 0.0
    assert metrics["fen_assembly_fen_first4"] == 0.5


def test_package_legal_move_rate_counts_malformed_move_tag_as_illegal():
    example = _example()
    scores = packaged.score_prediction(
        example,
        "<think>center</think><move> e2e4 </move>",
    )

    assert scores["primary"] == 1.0
    assert scores["format_compliance"] == 0.0
    assert scores["legal_move"] == 0.0


def test_package_score_split_includes_malformed_move_tag_in_legal_rate():
    examples = [
        _example(example_id="planning_00000"),
        _example(example_id="planning_00001"),
    ]

    metrics = packaged.score_split(
        examples,
        {
            "planning_00000": "<think>x</think><move>e2e4</move>",
            "planning_00001": "<think>x</think><move> e2e4 </move>",
        },
    )

    assert metrics["best_move_legal_move"] == 0.5


def test_package_score_split_aggregates_chess960_id_legal_move():
    example = packaged.BenchmarkExample(
        example_id="chess960_00000",
        split="chess960",
        task_type="best_move",
        fen=CHESS960_CASTLE_FEN,
        prompt="FEN: ...",
        gold_answer="f1g1",
        metric_type="move_extraction",
        metadata={"chess960_id": 3},
    )

    metrics = packaged.score_split(
        [example],
        {"chess960_00000": "<think>castle</think><move>f1g1</move>"},
    )

    assert metrics["best_move_legal_move"] == 1.0


def test_package_extract_move_accepts_tag_bare_and_prose():
    assert packaged.extract_move("<move>e2e4</move>") == "e2e4"
    assert packaged.extract_move("e2e4") == "e2e4"
    assert packaged.extract_move("I would play e2e4 here.") == "e2e4"
    assert packaged.extract_move("no move") is None


def test_package_benchmark_uses_shared_strict_protocol_helpers():
    assert packaged.validate_think_move_format is validate_think_move_format
    assert packaged.format_compliance("<think>x</think><move>e2e4</move>") == 1.0
    assert packaged.format_compliance("<think>x</think><move> E2E4 </move>") == 0.0
    assert packaged.legal_move_rate("e2e4", STARTING_FEN) is None
    assert packaged.extract_move("I prefer e2e4 over d2d4.") == "e2e4"


def test_package_binary_choice_handles_negated_first_candidate():
    example = _example(
        task_type="binary_choice",
        gold_answer="d2d4",
        metric_type="move_choice",
        metadata={"move_a": "e2e4", "move_b": "d2d4"},
    )

    scores = packaged.score_prediction(example, "e2e4 is not best; d2d4 is better.")

    assert scores["primary"] == 1.0


def test_package_binary_choice_rejects_prompt_echo_without_choice():
    example = _example(
        task_type="binary_choice",
        gold_answer="e2e4",
        metric_type="move_choice",
        metadata={"move_a": "e2e4", "move_b": "d2d4"},
    )

    scores = packaged.score_prediction(example, "Compare moves e2e4 and d2d4.")

    assert scores["primary"] == 0.0


def test_package_eval_bucket_requires_correct_side():
    assert packaged.eval_bucket_accuracy("Black has a slight edge.", "White has a slight edge.") == 0.0
    assert packaged.eval_bucket_accuracy("White has a slight edge.", "White has a slight edge.") == 1.0


def test_package_and_legacy_score_prediction_parity():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))
    from validation import benchmark as legacy

    assert legacy is packaged
    assert sys.modules["validation.benchmark"] is packaged
    assert legacy.BenchmarkExample is packaged.BenchmarkExample
    assert legacy.score_prediction is packaged.score_prediction
    assert legacy.score_split is packaged.score_split
    assert legacy.load_benchmark is packaged.load_benchmark
    assert legacy._extract_move is packaged._extract_move

    assert legacy.derive_gold_answer is packaged.derive_gold_answer
    assert legacy.freeze_split is packaged.freeze_split
    assert legacy.freeze_and_save is packaged.freeze_and_save

    examples = [
        _example(),
        _example(
            example_id="mate_00000",
            task_type="binary_choice",
            gold_answer="d2d4",
            metric_type="move_choice",
            metadata={"move_a": "e2e4", "move_b": "d2d4"},
        ),
        _example(
            example_id="evaluation_00000",
            task_type="eval_bucket",
            gold_answer="White has a slight edge.",
            metric_type="eval_bucket",
        ),
    ]
    predictions = {
        "planning_00000": "<think>center</think><move>e2e4</move>",
        "mate_00000": "e2e4 is not best; d2d4 is better.",
        "evaluation_00000": "White has a slight edge.",
    }

    for example in examples:
        legacy_example = legacy.BenchmarkExample.from_dict(example.to_dict())
        assert packaged.score_prediction(example, predictions[example.example_id]) == legacy.score_prediction(
            legacy_example,
            predictions[example.example_id],
        )


def test_legacy_benchmark_required_exports_are_package_objects():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))
    legacy = importlib.import_module("validation.benchmark")
    required_names = [
        "BenchmarkExample",
        "_render_prompt",
        "derive_gold_answer",
        "freeze_split",
        "freeze_and_save",
        "validate_oracle",
        "save_benchmark",
        "load_benchmark",
        "score_prediction",
        "score_split",
        "exact_match",
        "jaccard_similarity",
        "eval_bucket_accuracy",
        "format_compliance",
        "legal_move_rate",
        "move_extraction_match",
        "move_choice_match",
        "pass_at_k",
        "threat_f1",
        "continuation_rank",
        "centipawn_loss",
        "_extract_move",
        "extract_move",
        "normalize_prediction",
        "validate_think_move_format",
        "SPLIT_TASK_TYPES",
        "CANONICAL_PROMPTS",
        "TASK_METRIC_TYPE",
        "_eval_bucket_key",
        "_is_negated",
    ]

    for name in required_names:
        assert getattr(legacy, name) is getattr(packaged, name)


def test_legacy_benchmark_import_spellings_share_package_module():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    short_legacy = importlib.import_module("validation.benchmark")
    package_legacy = importlib.import_module("sft.make_data.validation.benchmark")

    assert short_legacy is packaged
    assert package_legacy is packaged
    assert sys.modules["validation.benchmark"] is packaged
    assert sys.modules["sft.make_data.validation.benchmark"] is packaged


def test_legacy_benchmark_alias_monkeypatches_package_globals(monkeypatch):
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))
    legacy = importlib.import_module("validation.benchmark")
    calls: list[tuple[str, str]] = []

    def fake_derive_gold_answer(task_type, raw, rng):
        calls.append((task_type, raw["fen"]))
        return "patched-gold"

    monkeypatch.setattr(legacy, "derive_gold_answer", fake_derive_gold_answer)

    examples = legacy.freeze_split(
        "planning",
        [{"fen": STARTING_FEN, "best_move": "e2e4"}],
        seed=42,
    )

    assert examples[0].gold_answer == "patched-gold"
    assert calls == [("best_move", STARTING_FEN)]
