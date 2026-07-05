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


def test_package_freeze_planning_prompts_require_think_move_contract():
    examples = packaged.freeze_split(
        "planning",
        [
            {"fen": STARTING_FEN, "best_move": "e2e4"},
            {
                "fen": STARTING_FEN,
                "puzzle_id": "puzzle-1",
                "solution_first_move": "d2d4",
            },
        ],
        seed=42,
    )

    assert [example.task_type for example in examples] == ["best_move", "puzzle_solve"]
    for example in examples:
        assert 'Answer format: return exactly "<think>...</think><move><uci></move>"' in example.prompt
        assert "final lowercase UCI move" in example.prompt


def test_package_freeze_preserves_puzzle_rating_and_themes():
    examples = packaged.freeze_split(
        "planning",
        [
            {
                "fen": STARTING_FEN,
                "puzzle_id": "puzzle-1",
                "solution_first_move": "e2e4",
                "rating": 1420,
                "themes": ["fork", "pin"],
            }
        ],
        seed=42,
    )

    assert len(examples) == 1
    assert examples[0].task_type == "puzzle_solve"
    assert examples[0].metadata["rating"] == 1420
    assert examples[0].metadata["themes"] == ["fork", "pin"]


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
    assert examples[0].gold_answer == "Side to move: black.\nLegal moves: none"
    assert packaged.validate_oracle(examples) == []


def test_package_terminal_legal_moves_scoring_accepts_none_and_legacy_phrasing():
    rng = Random(42)
    gold = packaged.derive_gold_answer("legal_moves", {"fen": STALEMATE_FEN}, rng)

    assert gold == "Side to move: black.\nLegal moves: none"

    example = _example(
        task_type="legal_moves",
        gold_answer=gold,
        metric_type="uci_set_jaccard",
    )
    assert packaged.score_prediction(
        example, "Side to move: black.\nLegal moves: none"
    )["primary"] == 1.0
    assert packaged.score_prediction(example, "No legal moves available.")["primary"] == 1.0
    assert packaged.score_prediction(example, "Legal moves: e7e5")["primary"] == 0.0
    assert packaged.score_prediction(example, "")["primary"] == 0.0


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


def test_package_state_tracking_skips_terminal_positions():
    rng = Random(42)

    assert packaged._derive_state_tracking(STALEMATE_FEN, rng) == ("", "")
    assert packaged.derive_gold_answer("state_tracking", {"fen": STALEMATE_FEN}, rng) == ""


def test_package_freeze_resamples_terminal_positions_away_from_state_tracking():
    examples = packaged.freeze_split(
        "perception",
        [{"fen": STALEMATE_FEN} for _ in range(14)],
        seed=42,
    )

    assert examples
    assert all(example.task_type != "state_tracking" for example in examples)
    assert all("After the moves ," not in example.prompt for example in examples)


def test_package_state_tracking_gold_fen_is_canonical():
    gold = packaged.derive_gold_answer(
        "state_tracking",
        {"fen": "4k3/8/8/8/8/8/4P3/4K3 w - -"},
        Random(42),
    )

    assert gold.startswith("Result FEN: ")
    result_fen = gold[len("Result FEN: "):]
    assert len(result_fen.split()) == 6
    assert chess.Board(result_fen).fen() == result_fen


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
        "Balance: Black is up 1 point(s) of material."
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
    assert "pinned_piece_exposes_king" in by_task["piece_legal_filter"].gold_answer
    assert "rejection reasons" in packaged.CANONICAL_PROMPTS["piece_legal_filter"]


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


def test_package_legal_moves_by_piece_reports_partial_diagnostics():
    gold = (
        "Side to move: white.\n"
        "Pieces: a1 white rook; e1 white king.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1b1\n"
        "e1 white king: e1d1\n"
        "All legal moves: a1a2 a1b1 e1d1"
    )
    prediction = (
        "Side to move: white.\n"
        "Pieces: a1 white rook; e1 white king.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1a3\n"
        "e1 white king: no legal moves"
    )
    example = _example(
        task_type="legal_moves_by_piece",
        gold_answer=gold,
        metric_type="text_exact_match",
    )

    scores = packaged.score_prediction(example, prediction)

    assert scores["primary"] == 0.0
    assert scores["section_completeness"] == 0.75
    assert scores["all_legal_line_present"] == 0.0
    assert scores["piece_inventory_accuracy"] == 1.0
    assert round(scores["all_moves_precision"], 3) == 0.5
    assert round(scores["all_moves_recall"], 3) == 0.333
    assert round(scores["all_moves_jaccard"], 3) == 0.25
    assert round(scores["per_piece_group_jaccard"], 3) == 0.167
    assert scores["illegal_extra_count"] == 1.0
    assert scores["missing_move_count"] == 2.0


def test_package_score_split_aggregates_legal_moves_by_piece_diagnostics():
    gold = (
        "Side to move: white.\n"
        "Pieces: a1 white rook.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1b1\n"
        "All legal moves: a1a2 a1b1"
    )
    example = _example(
        example_id="rules_00000",
        task_type="legal_moves_by_piece",
        gold_answer=gold,
        metric_type="text_exact_match",
    )

    metrics = packaged.score_split(
        [example],
        {
            "rules_00000": (
                "Side to move: white.\n"
                "Pieces: a1 white rook.\n"
                "Moves by piece:\n"
                "a1 white rook: a1a2 a1a3\n"
                "All legal moves: a1a2 a1a3"
            )
        },
    )

    assert metrics["legal_moves_by_piece"] == 0.0
    assert metrics["legal_moves_by_piece_all_legal_line_present"] == 1.0
    assert round(metrics["legal_moves_by_piece_all_moves_jaccard"], 3) == 0.333
    assert metrics["legal_moves_by_piece_illegal_extra_count"] == 1.0
    assert metrics["legal_moves_by_piece_missing_move_count"] == 1.0


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


def test_package_uci_set_jaccard_ignores_fen_rows_that_look_like_uci():
    fen_with_uci_like_row = "b2b4/8/8/8/7k/8/8/K7 w - - 0 1"
    example = _example(
        task_type="legal_moves",
        gold_answer="d2d4 e2e4",
        metric_type="uci_set_jaccard",
    )

    scores = packaged.score_prediction(
        example,
        f"FEN: {fen_with_uci_like_row}\nLegal moves: e2e4 d2d4",
    )

    assert scores["primary"] == 1.0


def test_package_extract_move_strips_fen_strings_before_prose_fallback():
    fen_with_uci_like_row = "b2b4/8/8/8/7k/8/8/K7 w - - 0 1"

    assert packaged.extract_move(
        f"The position is {fen_with_uci_like_row} and I recommend g1f3"
    ) == "g1f3"
    assert packaged.extract_move(fen_with_uci_like_row) is None


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


def test_package_opening_name_scores_family_match_with_exact_telemetry():
    example = _example(
        task_type="opening_name",
        gold_answer="English Opening: Symmetrical Variation, Botvinnik System Reversed (ECO: A37)",
        metric_type="exact_match",
    )
    example.split = "openings"

    scores = packaged.score_prediction(
        example,
        "English Opening: Symmetrical Variation, Full Symmetry Line (ECO: A38)",
    )

    assert scores["primary"] == 1.0
    assert scores["exact_match"] == 0.0
    assert scores["eco_exact"] == 0.0
    assert scores["eco_decade"] == 1.0
    assert scores["name_family"] == 1.0


def test_package_opening_name_rejects_unrelated_family_and_eco():
    example = _example(
        task_type="opening_name",
        gold_answer="French Defense: Winawer Variation, Advance Variation (ECO: C17)",
        metric_type="exact_match",
    )
    example.split = "openings"

    scores = packaged.score_prediction(
        example,
        "Queen's Gambit Declined: Albin Countergambit, Modern Line (ECO: D08)",
    )

    assert scores["primary"] == 0.0
    assert scores["exact_match"] == 0.0
    assert scores["eco_decade"] == 0.0
    assert scores["name_family"] == 0.0


def test_package_tactical_patterns_scores_best_move_with_exact_telemetry():
    example = _example(
        task_type="tactical_patterns",
        gold_answer="The tactic is mate in 2. Best move: f7e7",
        metric_type="exact_match",
    )
    example.split = "tactics"

    scores = packaged.score_prediction(
        example,
        "The tactic is tactical. Best move: f7e7",
    )

    assert scores["primary"] == 1.0
    assert scores["best_move_match"] == 1.0
    assert scores["exact_match"] == 0.0


def test_package_tactical_patterns_rejects_wrong_best_move():
    example = _example(
        task_type="tactical_patterns",
        gold_answer="The tactic is pin. Best move: c4d5",
        metric_type="exact_match",
    )
    example.split = "tactics"

    scores = packaged.score_prediction(
        example,
        "The tactic is pin. Best move: c4f7",
    )

    assert scores["primary"] == 0.0
    assert scores["best_move_match"] == 0.0
    assert scores["exact_match"] == 0.0


def test_package_check_state_metric_accepts_semantic_labels():
    example = _example(
        task_type="check_detection",
        gold_answer="Normal position -- no check, checkmate, or stalemate.",
        metric_type="check_state",
    )

    assert packaged.score_prediction(example, "Normal")["primary"] == 1.0
    assert packaged.score_prediction(example, "none")["primary"] == 1.0
    assert packaged.score_prediction(example, "Check.")["primary"] == 0.0


def test_package_check_state_handles_negated_checkmate_mentions():
    gold_check = "Check."
    gold_normal = "Normal position -- no check, checkmate, or stalemate."
    gold_stalemate = "Stalemate."
    negated_mate_prediction = "no checkmate here, but the king is in check"

    assert packaged.check_state_accuracy(negated_mate_prediction, gold_check) == 1.0
    assert packaged.check_state_accuracy(negated_mate_prediction, gold_normal) == 0.0
    assert packaged.check_state_accuracy("no check", gold_normal) == 1.0
    assert packaged.check_state_accuracy("stalemate, not checkmate", gold_stalemate) == 1.0
    assert packaged.check_state_accuracy("stalemate, not checkmate", "Checkmate.") == 0.0


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


def test_package_centipawn_loss_clamps_tail_values():
    assert packaged.ACPL_CP_LOSS_CLAMP == 1000.0
    assert packaged.ACPL_INVALID_MOVE_PENALTY == 1000.0
    assert packaged.centipawn_loss(12000.0, -5000.0) == 1000.0
    assert packaged.centipawn_loss(120.0, 20.0) == 100.0
    assert packaged.centipawn_loss(-20.0, 120.0) == 0.0


def test_package_score_split_reports_acpl_distribution_stats():
    examples = [_example(example_id=f"planning_{index:05d}") for index in range(5)]
    prediction = "<think>control center</think><move>e2e4</move>"

    aggregate = packaged.score_split(
        examples,
        {example.example_id: prediction for example in examples},
        acpl_scores={
            examples[0].example_id: 0.0,
            examples[1].example_id: 10.0,
            examples[2].example_id: 100.0,
            examples[3].example_id: 400.0,
            examples[4].example_id: 1000.0,
        },
    )

    assert aggregate["best_move_acpl"] == 302.0
    assert aggregate["best_move_acpl_median"] == 100.0
    assert aggregate["best_move_acpl_p90"] == 1000.0
    assert aggregate["best_move_acpl_p95"] == 1000.0
    assert aggregate["acpl"] == 302.0
    assert aggregate["acpl_median"] == 100.0
    assert aggregate["acpl_p90"] == 1000.0
    assert aggregate["acpl_p95"] == 1000.0


def test_package_score_split_reports_wpd_diagnostics():
    example = _example()
    prediction = "<think>control center</think><move>e2e4</move>"

    aggregate = packaged.score_split(
        [example],
        {"planning_00000": prediction},
        wpd_scores={
            "planning_00000": {
                "wpd": 0.125,
                "best_expectation": 0.62,
                "predicted_expectation": 0.495,
                "reward": 0.875,
                "multipv_hit": True,
                "postmove_hit": False,
                "cache_hit": True,
            }
        },
    )

    assert aggregate["best_move_wpd"] == 0.125
    assert aggregate["best_move_best_expectation"] == 0.62
    assert aggregate["best_move_predicted_expectation"] == 0.495
    assert aggregate["best_move_reward"] == 0.875
    assert aggregate["best_move_multipv_hit_rate"] == 1.0
    assert aggregate["best_move_postmove_hit_rate"] == 0.0
    assert aggregate["best_move_cache_hit_rate"] == 1.0
    assert aggregate["wpd"] == 0.125


def test_package_score_split_reports_wpd_reward_std_per_prompt():
    first = _example(example_id="planning_00000")
    second = _example(example_id="planning_00001")
    prediction = "<think>control center</think><move>e2e4</move>"

    aggregate = packaged.score_split(
        [first, second],
        {
            first.example_id: prediction,
            second.example_id: prediction,
        },
        wpd_scores={
            first.example_id: [
                {"reward": 1.0, "wpd": 0.0},
                {"reward": 0.4, "wpd": 0.6},
            ],
            second.example_id: [
                {"reward": 0.25, "wpd": 0.75},
                {"reward": 0.25, "wpd": 0.75},
            ],
        },
    )

    assert aggregate["best_move_reward"] == 0.475
    assert round(aggregate["best_move_reward_std_per_prompt"], 6) == 0.15
    assert round(aggregate["reward_std_per_prompt"], 6) == 0.15


def test_package_candidate_ratings_gold_scoring_and_freeze():
    raw = {
        "fen": STARTING_FEN,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42},
            {"uci": "d2d4", "cp": 15},
            {"uci": "g1f3", "cp": 5},
            {"uci": "c2c4", "cp": -20},
            {"uci": "b1c3", "cp": -80},
        ],
    }
    gold = packaged.derive_gold_answer("candidate_ratings", dict(raw), Random(42))
    examples = packaged.freeze_split("planning", [dict(raw)], seed=42)

    assert gold.splitlines() == [
        "Candidate e2e4: +42cp; Bucket: equal",
        "Candidate d2d4: +15cp; Bucket: equal",
        "Candidate g1f3: +5cp; Bucket: equal",
        "Candidate c2c4: -20cp; Bucket: equal",
        "Candidate b1c3: -80cp; Bucket: slight edge",
        "Best: e2e4",
    ]
    assert examples[0].task_type == "candidate_ratings"
    assert examples[0].metric_type == "candidate_ratings"
    assert "e2e4 d2d4 g1f3 c2c4 b1c3" in examples[0].prompt
    assert examples[0].gold_answer == gold

    scores = packaged.score_prediction(examples[0], gold)
    assert scores["primary"] == 1.0
    assert scores["candidate_set_jaccard"] == 1.0
    assert scores["best_move_match"] == 1.0
    assert scores["cp_bucket_accuracy"] == 1.0
    assert scores["cp_bucket_mae"] == 0.0


def test_package_candidate_ratings_scores_partial_bucket_and_best_errors():
    gold = "\n".join(
        [
            "Candidate e2e4: +42cp; Bucket: equal",
            "Candidate d2d4: +15cp; Bucket: equal",
            "Candidate g1f3: +5cp; Bucket: equal",
            "Candidate c2c4: -20cp; Bucket: equal",
            "Candidate b1c3: -80cp; Bucket: slight edge",
            "Best: e2e4",
        ]
    )
    prediction = "\n".join(
        [
            "Candidate e2e4: +42cp; Bucket: equal",
            "Candidate d2d4: +15cp; Bucket: equal",
            "Candidate g1f3: +5cp; Bucket: slight edge",
            "Candidate c2c4: -20cp; Bucket: equal",
            "Candidate b1a3: -120cp; Bucket: slight edge",
            "Best: d2d4",
        ]
    )
    example = _example(
        task_type="candidate_ratings",
        gold_answer=gold,
        metric_type="candidate_ratings",
    )

    scores = packaged.score_prediction(example, prediction)

    assert scores["primary"] < 1.0
    assert round(scores["candidate_set_jaccard"], 3) == 0.667
    assert scores["best_move_match"] == 0.0
    assert scores["cp_bucket_accuracy"] == 0.8
    assert scores["cp_bucket_mae"] == 0.2


def test_package_best_line_trace_freeze_uses_fixed_grammar_gold():
    raw = {
        "fen": STARTING_FEN,
        "best_line_trace": True,
        "multipv_depth": 18,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42, "pv_line": "e2e4 e7e5 g1f3 b8c6"},
            {"uci": "d2d4", "cp": 15, "pv_line": "d2d4 d7d5"},
            {"uci": "g1f3", "cp": 5, "pv_line": "g1f3 d7d5"},
            {"uci": "c2c4", "cp": -20, "pv_line": "c2c4 e7e5"},
            {"uci": "b1c3", "cp": -80, "pv_line": "b1c3 d7d5"},
        ],
    }
    expected = "\n".join(
        [
            "<think>",
            "Root: e2e4",
            "Eval: +42cp; Bucket: equal",
            "PV: e2e4 e7e5 g1f3 b8c6",
            "Best: e2e4",
            "</think><move>e2e4</move>",
        ]
    )

    gold = packaged.derive_gold_answer("best_line_trace", dict(raw), Random(42))
    examples = packaged.freeze_split("planning", [dict(raw)], seed=42)

    assert gold == expected
    assert examples[0].task_type == "best_line_trace"
    assert examples[0].metric_type == "best_line_trace"
    assert "engine best line" in examples[0].prompt.lower()
    assert examples[0].gold_answer == expected

    scores = packaged.score_prediction(examples[0], expected)
    assert scores["primary"] == 1.0
    assert scores["format_compliance"] == 1.0
    assert scores["legal_move"] == 1.0
    assert scores["trace_referenced_move_accuracy"] == 1.0
    assert scores["trace_line_depth"] == 4.0
    assert scores["trace_conclusion_move_match"] == 1.0


def test_package_step_verification_scores_verdict_line_and_error_type():
    gold = "\n".join(
        [
            "Verdict: broken",
            "Faulty line: 6",
            "Error type: wrong_best",
            "Correction: Best should be e2e4.",
        ]
    )
    prediction = "\n".join(
        [
            "Verdict: broken",
            "Faulty line: 4",
            "Error type: wrong_bucket",
            "Correction: Best should be e2e4.",
        ]
    )
    example = _example(
        task_type="step_verification",
        gold_answer=gold,
        metric_type="step_verification",
    )

    scores = packaged.score_prediction(example, prediction)

    assert scores["primary"] == 0.5
    assert scores["verdict_accuracy"] == 1.0
    assert scores["faulty_line_accuracy"] == 0.0
    assert scores["error_type_accuracy"] == 0.0
    assert scores["correction_match"] == 1.0


def test_package_step_verification_freeze_uses_fixed_grammar_gold():
    raw = {
        "fen": STARTING_FEN,
        "verification_trace": "1. Candidate e2e4: +42cp; Bucket: equal\n2. Best: d2d4",
        "verification_verdict": "broken",
        "faulty_line": 2,
        "error_type": "wrong_best",
        "correction": "Best should be e2e4.",
    }

    gold = packaged.derive_gold_answer("step_verification", dict(raw), Random(42))
    examples = packaged.freeze_split("planning", [dict(raw)], seed=42)

    assert gold == "\n".join(
        [
            "Verdict: broken",
            "Faulty line: 2",
            "Error type: wrong_best",
            "Correction: Best should be e2e4.",
        ]
    )
    assert examples[0].task_type == "step_verification"
    assert examples[0].metric_type == "step_verification"
    assert examples[0].metadata["diagnostic"] is True
    assert "1. Candidate e2e4" in examples[0].prompt
    assert examples[0].gold_answer == gold


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


def test_package_cp_bucket_gold_uses_grammatical_side_attributed_text():
    rng = Random(42)
    raw = {"fen": STARTING_FEN}

    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": 0}, rng) == (
        "The position is equal."
    )
    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": -30}, rng) == (
        "The position is equal."
    )
    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": 49}, rng) == (
        "The position is equal."
    )
    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": 100}, rng) == (
        "White has a slight edge."
    )
    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": -200}, rng) == (
        "Black has a clear advantage."
    )
    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": 400}, rng) == (
        "White is winning."
    )
    assert packaged.derive_gold_answer("eval_bucket", {**raw, "cp": -700}, rng) == (
        "Black has a decisive advantage."
    )


def test_package_cp_bucket_clamps_out_of_range_cp_to_last_bucket():
    rng = Random(42)

    assert packaged.derive_gold_answer(
        "eval_bucket", {"fen": STARTING_FEN, "cp": 250_000}, rng
    ) == "White has a decisive advantage."
    assert packaged.derive_gold_answer(
        "eval_bucket", {"fen": STARTING_FEN, "cp": -250_000}, rng
    ) == "Black has a decisive advantage."


def test_package_eval_bucket_gold_answers_self_score_as_oracle():
    rng = Random(42)
    examples = []
    for index, cp in enumerate((0, 100, -200, 400, -700, 250_000)):
        gold = packaged.derive_gold_answer("eval_bucket", {"fen": STARTING_FEN, "cp": cp}, rng)
        examples.append(
            _example(
                example_id=f"evaluation_{index:05d}",
                task_type="eval_bucket",
                gold_answer=gold,
                metric_type="eval_bucket",
            )
        )

    assert packaged.validate_oracle(examples) == []


def test_package_eval_bucket_scoring_ignores_side_for_equal_only():
    gold_equal = "The position is equal."

    assert packaged.eval_bucket_accuracy("The position is equal.", gold_equal) == 1.0
    assert packaged.eval_bucket_accuracy("White has an equal position.", gold_equal) == 1.0
    assert packaged.eval_bucket_accuracy("White is winning.", "White is winning.") == 1.0
    assert packaged.eval_bucket_accuracy("Black is winning.", "White is winning.") == 0.0
    assert packaged.eval_bucket_accuracy("White has a clear advantage.", "White is winning.") == 0.0
    assert packaged.eval_bucket_accuracy(
        "Black has a decisive advantage.", "Black has a decisive advantage."
    ) == 1.0


PINNED_ROOK_FEN = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"


def test_package_new_diagnostic_task_gold_answers_self_score_primary():
    rng = Random(42)
    cases = [
        ("ray_walk", {"fen": PINNED_ROOK_FEN}),
        ("legal_filter_trace", {"fen": PINNED_ROOK_FEN}),
        ("multi_state_tracking", {"fen": STARTING_FEN}),
    ]

    for task_type, raw in cases:
        gold = packaged.derive_gold_answer(task_type, raw, rng)
        assert gold, task_type
        example = _example(
            task_type=task_type,
            gold_answer=gold,
            metric_type=packaged.TASK_METRIC_TYPE[task_type],
        )
        assert packaged.score_prediction(example, gold)["primary"] == 1.0, task_type


def test_package_multi_state_tracking_benchmark_uses_two_to_three_plies():
    moves, result_fen = packaged._derive_multi_state_tracking(STARTING_FEN, Random(1))

    assert len(moves.split()) in (2, 3)
    assert result_fen
    assert chess.Board(result_fen).fen() == result_fen
    # Positions where fewer than 2 plies are playable are rejected.
    assert packaged._derive_multi_state_tracking(STALEMATE_FEN, Random(1)) == ("", "")
    assert packaged.derive_gold_answer(
        "multi_state_tracking", {"fen": STALEMATE_FEN}, Random(1)
    ) == ""


def test_package_freeze_includes_new_diagnostic_tasks_with_filled_metadata():
    rules_examples = packaged.freeze_split(
        "rules",
        [{"fen": PINNED_ROOK_FEN} for _ in range(13)],
        seed=42,
    )
    by_task = {example.task_type: example for example in rules_examples}

    ray = by_task["ray_walk"]
    assert ray.metric_type == "text_exact_match"
    assert ray.metadata["diagnostic"] is True
    assert ray.metadata["hard_gate"] is False
    # First side-to-move slider in a1..h8 order is the e2 rook.
    assert ray.metadata["source_square"] == "e2"
    assert "slider on e2" in ray.prompt
    assert "{source_square}" not in ray.prompt
    assert ray.gold_answer.startswith("Piece: e2 white rook.")
    assert ray.gold_answer.splitlines()[-1].startswith("Moves from rays: ")

    trace = by_task["legal_filter_trace"]
    assert trace.metric_type == "text_exact_match"
    assert trace.metadata["diagnostic"] is True
    assert trace.metadata["hard_gate"] is False
    assert "Filter by piece:" in trace.gold_answer
    assert trace.gold_answer.splitlines()[-1].startswith("All legal moves: ")

    perception_examples = packaged.freeze_split(
        "perception",
        [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"} for _ in range(19)],
        seed=42,
    )
    multi = next(
        example
        for example in perception_examples
        if example.task_type == "multi_state_tracking"
    )
    assert multi.metric_type == "fen_exact_match"
    assert multi.metadata["diagnostic"] is True
    assert multi.metadata["hard_gate"] is False
    assert len(multi.metadata["moves"].split()) in (2, 3)
    assert multi.metadata["moves"] in multi.prompt
    assert "{moves}" not in multi.prompt
    assert "State needed for full FEN:" in multi.prompt
    assert packaged.validate_oracle([ray, trace, multi]) == []


def test_package_legal_filter_trace_derivation_caps_fall_back_at_freeze():
    # The starting position has 16 side-to-move pieces (> the 12-piece cap).
    assert packaged._derive_legal_filter_trace(STARTING_FEN) == ""

    examples = packaged.freeze_split(
        "rules",
        [{"fen": STARTING_FEN} for _ in range(13)],
        seed=42,
    )

    assert examples
    assert all(example.task_type != "legal_filter_trace" for example in examples)


def test_package_piece_legal_filter_near_miss_earns_partial_set_f1():
    gold = (
        "Pseudo-legal from e2: e2d2 e2e3 e2e4.\n"
        "Legal: e2e3 e2e4.\n"
        "Rejected: e2d2 pinned_piece_exposes_king."
    )
    example = _example(
        task_type="piece_legal_filter",
        gold_answer=gold,
        metric_type="text_exact_match",
    )

    near_miss = (
        "Pseudo-legal from e2: e2d2 e2e3 e2e4.\n"
        "Legal: e2e3.\n"
        "Rejected: e2d2 pinned_piece_exposes_king."
    )
    scores = packaged.score_prediction(example, near_miss)

    assert scores["primary"] == 0.0
    assert 0.0 < scores["set_f1"] < 1.0
    assert packaged.score_prediction(example, gold) == {"primary": 1.0, "set_f1": 1.0}


def test_package_legal_filter_trace_near_miss_earns_partial_final_jaccard():
    gold = packaged.derive_gold_answer(
        "legal_filter_trace",
        {"fen": PINNED_ROOK_FEN},
        Random(42),
    )
    example = _example(
        task_type="legal_filter_trace",
        gold_answer=gold,
        metric_type="text_exact_match",
    )

    near_miss = (
        gold.rsplit("All legal moves:", 1)[0] + "All legal moves: e1d1 e2e3 e2e4"
    )
    scores = packaged.score_prediction(example, near_miss)

    assert scores["primary"] == 0.0
    assert scores["final_jaccard"] == 0.3
    assert packaged.score_prediction(example, gold)["final_jaccard"] == 1.0
    # Degenerate predictions without the final marker skip the secondary.
    assert packaged.score_prediction(example, "garbage")["final_jaccard"] is None


def test_package_legal_moves_by_piece_reports_set_f1_secondary():
    gold = (
        "Side to move: white.\n"
        "Pieces: a1 white rook; e1 white king.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1b1\n"
        "e1 white king: e1d1\n"
        "All legal moves: a1a2 a1b1 e1d1"
    )
    prediction = (
        "Side to move: white.\n"
        "Pieces: a1 white rook; e1 white king.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1a3\n"
        "e1 white king: no legal moves"
    )
    example = _example(
        task_type="legal_moves_by_piece",
        gold_answer=gold,
        metric_type="text_exact_match",
    )

    scores = packaged.score_prediction(example, prediction)

    # F1 from all-moves precision 0.5 and recall 1/3.
    assert round(scores["set_f1"], 3) == 0.4
    assert packaged.score_prediction(example, gold)["set_f1"] == 1.0


def test_package_hanging_pieces_reports_set_f1_secondary():
    gold = "Hanging pieces: white pawn on d4, black queen on f3."
    example = _example(
        task_type="hanging_pieces",
        gold_answer=gold,
        metric_type="exact_match",
    )

    scores = packaged.score_prediction(
        example,
        "Hanging pieces: white pawn on d4, black pawn on c6.",
    )

    assert scores["primary"] == 0.0
    assert round(scores["set_f1"], 3) == 0.5
    assert packaged.score_prediction(example, gold) == {"primary": 1.0, "set_f1": 1.0}
    assert (
        packaged.score_prediction(
            example,
            "No hanging pieces — all attacked pieces are defended.",
        )["set_f1"]
        == 0.0
    )


def test_package_hanging_piece_claim_verification_scores_fixed_fields():
    gold = (
        "Verdict: incorrect\n"
        "Attacked: yes\n"
        "Defended: yes\n"
        "Hanging: no\n"
        "Correction: white queen on e2 is not hanging."
    )
    example = _example(
        task_type="hanging_piece_claim_verification",
        gold_answer=gold,
        metric_type="hanging_piece_claim_verification",
    )

    scores = packaged.score_prediction(
        example,
        (
            "Verdict: incorrect\n"
            "Attacked: yes\n"
            "Defended: no\n"
            "Hanging: yes\n"
            "Correction: white queen on e2 is hanging."
        ),
    )

    assert scores["primary"] == 0.4
    assert scores["verdict_accuracy"] == 1.0
    assert scores["attacked_accuracy"] == 1.0
    assert scores["defended_accuracy"] == 0.0
    assert scores["hanging_accuracy"] == 0.0
    assert scores["correction_match"] == 0.0
    assert packaged.score_prediction(example, gold)["primary"] == 1.0


def test_package_hanging_piece_claim_verification_freezes_as_diagnostic():
    rows = [
        {"fen": "4k3/8/8/8/4r3/8/4Q3/K7 w - - 0 1"}
        for _ in range(5)
    ]

    examples = packaged.freeze_split("tactics", rows, seed=42)
    verifier = [
        example
        for example in examples
        if example.task_type == "hanging_piece_claim_verification"
    ]

    assert verifier
    example = verifier[0]
    assert example.metadata["diagnostic"] is True
    assert example.metadata["hard_gate"] is False
    assert "Claim to verify:" in example.prompt
    assert "Answer format: return exactly five lines" in example.prompt
    assert '"Verdict: correct|incorrect"' in example.prompt
    assert "verification_claim" in example.metadata
    assert packaged.score_prediction(example, example.gold_answer)["primary"] == 1.0


def test_package_score_split_reports_new_composite_secondary_names():
    trace_gold = packaged.derive_gold_answer(
        "legal_filter_trace",
        {"fen": PINNED_ROOK_FEN},
        Random(42),
    )
    filter_gold = (
        "Pseudo-legal from e2: e2d2 e2e3.\n"
        "Legal: e2e3.\n"
        "Rejected: e2d2 pinned_piece_exposes_king."
    )
    examples = [
        _example(
            example_id="rules_00000",
            task_type="piece_legal_filter",
            gold_answer=filter_gold,
            metric_type="text_exact_match",
        ),
        _example(
            example_id="rules_00001",
            task_type="legal_filter_trace",
            gold_answer=trace_gold,
            metric_type="text_exact_match",
        ),
        _example(
            example_id="tactics_00000",
            task_type="hanging_pieces",
            gold_answer="Hanging pieces: white pawn on d4.",
            metric_type="exact_match",
        ),
        _example(
            example_id="tactics_00001",
            task_type="hanging_piece_claim_verification",
            gold_answer=(
                "Verdict: correct\n"
                "Attacked: yes\n"
                "Defended: no\n"
                "Hanging: yes\n"
                "Correction: none"
            ),
            metric_type="hanging_piece_claim_verification",
        ),
    ]

    metrics = packaged.score_split(
        examples,
        {
            "rules_00000": filter_gold,
            "rules_00001": trace_gold,
            "tactics_00000": "Hanging pieces: white pawn on d4.",
            "tactics_00001": (
                "Verdict: correct\n"
                "Attacked: yes\n"
                "Defended: no\n"
                "Hanging: yes\n"
                "Correction: none"
            ),
        },
    )

    assert metrics["piece_legal_filter_set_f1"] == 1.0
    assert metrics["legal_filter_trace_final_jaccard"] == 1.0
    assert metrics["hanging_pieces_set_f1"] == 1.0
    assert metrics["hanging_piece_claim_verification_verdict_accuracy"] == 1.0
    assert metrics["hanging_piece_claim_verification_correction_match"] == 1.0
