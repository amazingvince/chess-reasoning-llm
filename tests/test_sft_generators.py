from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path
from random import Random


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _user_prompt(row: dict) -> str:
    return row["messages"][1]["content"]


def _clear_legacy_generator_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "config.templates"
            or module_name.startswith("generators")
            or module_name.startswith("sft.make_data.generators")
        ):
            sys.modules.pop(module_name, None)


def test_package_templates_import_without_legacy_modules():
    _clear_legacy_generator_modules()

    templates = importlib.import_module("chess_llm.sft.templates")

    assert "1.1_fen_to_board" in templates.TEMPLATES
    assert templates.select_template("1.1_fen_to_board", Random(1))
    assert "config.templates" not in sys.modules


def test_package_generators_import_without_legacy_modules():
    _clear_legacy_generator_modules()

    tier1 = importlib.import_module("chess_llm.sft.generators.tier1_perception")
    tier7 = importlib.import_module("chess_llm.sft.generators.tier7_planning")

    assert tier1.FENToBoard.__module__ == "chess_llm.sft.generators.tier1_perception"
    assert tier7.BestMoveSelection.__module__ == "chess_llm.sft.generators.tier7_planning"
    assert "generators.base" not in sys.modules
    assert "config.templates" not in sys.modules


def test_package_piece_counting_board_piece_template_counts_requested_piece(monkeypatch):
    from chess_llm.sft.generators.tier1_perception import PieceCounting

    class ChooseQueen:
        def choice(self, values):
            if "queen" in values:
                return "queen"
            if "white" in values:
                return "white"
            return values[0]

    tpl = "Board:\n{board}\nCount the {piece}s on the board."
    monkeypatch.setattr(
        "chess_llm.sft.generators.tier1_perception.select_template",
        lambda _task_id, _rng: tpl,
    )

    gen = PieceCounting(
        config={
            "fen_pool": [
                {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
            ],
            "volume_override": 1,
            "piece_counting_include_partial": True,
        },
        rng=ChooseQueen(),
    )

    row = next(gen.generate())

    assert "Count the queens" in row["messages"][1]["content"]
    assert row["messages"][2]["content"] == (
        "White has 1 queen(s), black has 1 queen(s). Total: 2."
    )
    assert row["metadata"]["count_kind"] == "piece_type"
    assert row["metadata"]["piece"] == "queen"


def test_package_piece_counting_defaults_to_full_material_targets():
    from chess_llm.sft.generators.tier1_perception import PieceCounting

    fen = "8/8/8/8/8/8/6p1/4K2k w - - 0 1"
    gen = PieceCounting(
        config={
            "fen_pool": [{"fen": fen} for _ in range(24)],
            "volume_override": 24,
        },
        rng=Random(3),
    )

    rows = list(gen.generate())

    assert len(rows) == 24
    assert {row["metadata"]["count_kind"] for row in rows} == {"full_material"}
    assert all("Inventory:" not in row["messages"][2]["content"] for row in rows)
    assert all("White (" in row["messages"][2]["content"] for row in rows)
    assert all("Black (" in row["messages"][2]["content"] for row in rows)


def test_package_piece_counting_full_material_teaches_compact_count(monkeypatch):
    from chess_llm.sft.generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nWhat is the material count for both sides?"
    monkeypatch.setattr(
        "chess_llm.sft.generators.tier1_perception.select_template",
        lambda _task_id, _rng: tpl,
    )
    fen = "8/8/8/8/8/8/6p1/4K2k w - - 0 1"
    gen = PieceCounting(
        config={"fen_pool": [{"fen": fen}], "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert answer == (
        "White (1 pieces): 1 king. "
        "Black (2 pieces): 1 king, 1 pawn. "
        "Black is up 1 point(s) of material."
    )
    assert "Inventory:" not in answer
    assert row["metadata"]["white_counts"] == {
        "king": 1,
        "queen": 0,
        "rook": 0,
        "bishop": 0,
        "knight": 0,
        "pawn": 0,
    }
    assert row["metadata"]["black_counts"] == {
        "king": 1,
        "queen": 0,
        "rook": 0,
        "bishop": 0,
        "knight": 0,
        "pawn": 1,
    }
    assert row["metadata"]["material_balance"] == -1
    assert row["metadata"]["material_count_answer"] == (
        "White (1 pieces): 1 king. "
        "Black (2 pieces): 1 king, 1 pawn. "
        "Black is up 1 point(s) of material."
    )


def test_package_piece_identification_square_queries_balance_empty_and_occupied():
    from chess_llm.sft.generators.tier1_perception import PieceIdentification

    sparse_king_board = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"
    gen = PieceIdentification(
        config={
            "fen_pool": [{"fen": sparse_king_board} for _ in range(80)],
            "volume_override": 80,
        },
        rng=Random(5),
    )

    square_answers = [
        row["messages"][2]["content"]
        for row in gen.generate()
        if "piece is on" in row["messages"][1]["content"].lower()
        or "occupies" in row["messages"][1]["content"].lower()
        or "what is on" in row["messages"][1]["content"].lower()
    ]

    assert square_answers
    empty_count = sum(answer == "empty" for answer in square_answers)
    occupied_count = len(square_answers) - empty_count

    assert empty_count / len(square_answers) >= 0.35
    assert occupied_count / len(square_answers) >= 0.35


def test_package_square_lookup_generator_teaches_fen_square_mapping():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import SquareLookup

    fen = "8/8/8/8/8/Q7/8/4K2k w - - 0 1"
    gen = SquareLookup(
        config={
            "fen_pool": [{"fen": fen} for _ in range(12)],
            "volume_override": 12,
        },
        rng=Random(7),
    )

    rows = list(gen.generate())

    assert len(rows) == 12
    assert all(row["task"] == "1.6_square_lookup" for row in rows)
    answers = [row["messages"][2]["content"] for row in rows]
    assert any(answer.endswith("=empty") for answer in answers)
    assert any(answer.endswith("=white queen") for answer in answers)
    for row in rows:
        square = row["metadata"]["square"]
        expected = row["metadata"]["expected_answer"]
        assert square in row["messages"][1]["content"]
        assert fen in row["messages"][1]["content"]
        assert row["messages"][2]["content"] == expected
        passed, errors = validate_example(row)
        assert passed is True, errors


def test_package_format_sensitive_generators_append_answer_contracts():
    from chess_llm.sft.generators.tier1_perception import (
        BoardToFEN,
        FENAssembly,
        FENRowApplication,
        PieceIdentification,
        SquareLookup,
        StateTracking,
    )
    from chess_llm.sft.generators.tier2_rules import LegalMoveGen

    position = "8/8/8/8/8/Q7/8/4K2k w - - 0 1"
    game_position = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"

    board_to_fen = next(
        BoardToFEN(
            config={"fen_pool": [{"fen": position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    piece_id = next(
        PieceIdentification(
            config={"fen_pool": [{"fen": position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    square_lookup = next(
        SquareLookup(
            config={"fen_pool": [{"fen": position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    state_tracking = next(
        StateTracking(
            config={"game_positions": [{"fen": game_position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    fen_assembly = next(
        FENAssembly(
            config={"game_positions": [{"fen": game_position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    fen_row_application = next(
        FENRowApplication(
            config={"game_positions": [{"fen": game_position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    legal_moves = next(
        LegalMoveGen(
            config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )

    assert (
        "Answer format: return exactly one complete six-field FEN"
        in _user_prompt(board_to_fen)
    )
    assert (
        "Answer format: for a square query, return only the piece name or empty"
        in _user_prompt(piece_id)
    )
    assert (
        'Answer format: return exactly "<square>=<piece>" or "<square>=empty".'
        in _user_prompt(square_lookup)
    )
    assert (
        'Answer format: return exactly "Result FEN: <complete six-field FEN>".'
        in _user_prompt(state_tracking)
    )
    assert (
        'final line "Result FEN: <complete six-field FEN>"'
        in _user_prompt(fen_assembly)
    )
    assert (
        'first line starts "Rows:" and final line is "Result FEN: <complete six-field FEN>"'
        in _user_prompt(fen_row_application)
    )
    assert (
        'Answer format: return exactly two lines: "Side to move: <white|black>."'
        in _user_prompt(legal_moves)
    )


def test_package_rank_lookup_generator_teaches_fen_rank_rows():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import RankLookup

    fen = "8/8/8/8/8/Q7/8/4K2k w - - 0 1"
    gen = RankLookup(
        config={
            "fen_pool": [{"fen": fen} for _ in range(8)],
            "volume_override": 8,
        },
        rng=Random(3),
    )

    rows = list(gen.generate())

    assert len(rows) == 8
    assert all(row["task"] == "1.7_rank_lookup" for row in rows)
    for row in rows:
        rank = row["metadata"]["rank"]
        expected = f"rank {rank}: {row['metadata']['fen_rank_row']}"
        assert f"rank {rank}" in row["messages"][1]["content"].lower()
        assert row["messages"][2]["content"] == expected
        passed, errors = validate_example(row)
        assert passed is True, errors


def test_package_square_coordinates_generator_teaches_fen_row_and_file_mapping():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import SquareCoordinates

    fen = "7k/8/8/8/8/5Q2/8/4K3 w - - 0 1"
    gen = SquareCoordinates(
        config={
            "fen_pool": [{"fen": fen} for _ in range(16)],
            "volume_override": 16,
        },
        rng=Random(11),
    )

    rows = list(gen.generate())

    assert len(rows) == 16
    assert all(row["task"] == "1.11_square_coordinates" for row in rows)
    for row in rows:
        square = row["metadata"]["square"]
        expected = row["metadata"]["expected_answer"]
        assert expected == row["messages"][2]["content"]
        assert square in row["messages"][1]["content"]
        assert "fen_row_from_top=" in expected
        assert "file_index=" in expected
        passed, errors = validate_example(row)
        assert passed is True, errors


def test_package_fen_rank_expansion_generator_teaches_file_cells():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import FENRankExpansion

    fen = "7k/8/8/8/8/5Q2/8/4K3 w - - 0 1"
    gen = FENRankExpansion(
        config={
            "fen_pool": [{"fen": fen} for _ in range(8)],
            "volume_override": 8,
        },
        rng=Random(13),
    )

    rows = list(gen.generate())

    assert len(rows) == 8
    assert all(row["task"] == "1.12_fen_rank_expansion" for row in rows)
    assert any("f=Q" in row["messages"][2]["content"] for row in rows)
    for row in rows:
        rank = row["metadata"]["rank"]
        row_text = row["metadata"]["fen_rank_row"]
        assert f"rank {rank}" in row["messages"][1]["content"].lower()
        assert row_text in row["messages"][1]["content"]
        assert row["metadata"]["expected_answer"] == row["messages"][2]["content"]
        passed, errors = validate_example(row)
        assert passed is True, errors


def test_package_fen_rank_cell_edit_generator_teaches_single_cell_rewrites():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import FENRankCellEdit

    gen = FENRankCellEdit(
        config={
            "game_positions": [{"fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1"}],
            "volume_override": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert row["task"] == "1.13_fen_rank_cell_edit"
    assert row["metadata"]["file"] in row["messages"][1]["content"]
    assert row["metadata"]["before_row"] in row["messages"][1]["content"]
    assert row["metadata"]["after_row"] in answer
    assert answer == row["metadata"]["expected_answer"]
    assert answer == (
        f"rank {row['metadata']['rank']}: "
        f"{row['metadata']['before_row']} -> {row['metadata']['after_row']}"
    )
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_fen_board_edit_generator_teaches_changed_square_assembly():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import FENBoardEdit

    gen = FENBoardEdit(
        config={
            "game_positions": [{"fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1"}],
            "volume_override": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert row["task"] == "1.14_fen_board_edit"
    assert row["metadata"]["board_fen_before"] in row["messages"][1]["content"]
    assert row["metadata"]["edit_text"] in row["messages"][1]["content"]
    assert answer == f"Result board FEN: {row['metadata']['board_fen_after']}"
    assert row["metadata"]["expected_answer"] == answer
    assert "Result FEN:" not in answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_atomic_fen_edit_tasks_are_registered_for_tier1_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import (
        FENBoardEdit,
        FENRankCellEdit,
        FENRankExpansion,
        SquareCoordinates,
    )
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import TEMPLATES

    tier1_generators = pipeline.TIER_GENERATORS[1]

    for generator, task_id in [
        (SquareCoordinates, "1.11_square_coordinates"),
        (FENRankExpansion, "1.12_fen_rank_expansion"),
        (FENRankCellEdit, "1.13_fen_rank_cell_edit"),
        (FENBoardEdit, "1.14_fen_board_edit"),
    ]:
        assert generator in tier1_generators
        assert DEFAULT_VOLUMES[task_id] > 0
        assert TEMPLATES[task_id]


def test_package_move_square_edits_generator_teaches_single_move_edits():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import MoveSquareEdits

    fen = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
    gen = MoveSquareEdits(
        config={
            "fen_pool": [{"fen": fen}],
            "volume_override": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert row["task"] == "1.8_move_square_edits"
    assert row["metadata"]["move"]
    assert row["metadata"]["expected_answer"] == answer
    assert answer.startswith("Lookup: ")
    assert "\nSquares: " in answer
    assert "\nRanks: " in answer
    assert "Result FEN:" not in answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_fen_assembly_generator_teaches_explicit_fen_rewrite():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import FENAssembly

    fen = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
    gen = FENAssembly(
        config={
            "game_positions": [{"fen": fen}],
            "volume_override": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert row["task"] == "1.9_fen_assembly"
    assert row["metadata"]["move"]
    assert row["metadata"]["result_fen"]
    assert row["metadata"]["expected_answer"] == answer
    assert answer.startswith("Move 1: ")
    assert "\nLookup: " in answer
    assert "\nSquares: " in answer
    assert "\nRanks: " in answer
    assert answer.count("Result FEN:") == 1
    assert answer.endswith(f"Result FEN: {row['metadata']['result_fen']}")
    assert "Result placement:" not in answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_fen_assembly_generator_uses_complete_start_fen_state():
    from chess_llm.sft.generators.tier1_perception import FENAssembly

    gen = FENAssembly(
        config={
            "game_positions": [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - -"}],
            "volume_override": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    prompt = row["messages"][1]["content"]

    assert row["fen"] == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" in prompt
    assert "State needed for full FEN:" in prompt
    assert "Side to move: white" in prompt
    assert "Halfmove clock: 0" in prompt
    assert "Fullmove number: 1" in prompt


def test_package_fen_assembly_task_is_registered_for_tier1_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import FENAssembly
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import TEMPLATES

    tier1_generators = pipeline.TIER_GENERATORS[1]

    assert FENAssembly in tier1_generators
    assert DEFAULT_VOLUMES["1.9_fen_assembly"] > 0
    assert TEMPLATES["1.9_fen_assembly"]


def test_package_fen_row_application_generator_teaches_full_rank_rewrites():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import FENRowApplication

    gen = FENRowApplication(
        config={
            "game_positions": [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"}],
            "volume_override": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert row["task"] == "1.10_fen_row_application"
    assert row["metadata"]["move"]
    assert row["metadata"]["result_fen"]
    assert row["metadata"]["rank_updates"]
    lines = answer.splitlines()
    assert lines[0].startswith("Rows: rank ")
    assert lines[1] == f"Result FEN: {row['metadata']['result_fen']}"
    for update in row["metadata"]["rank_updates"]:
        rewrite = f"rank {update['rank']} {update['before']}->{update['after']}"
        assert rewrite in lines[0]
    assert "Lookup:" not in answer
    assert "Squares:" not in answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_fen_row_application_task_is_registered_for_tier1_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import FENRowApplication
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import TEMPLATES

    tier1_generators = pipeline.TIER_GENERATORS[1]

    assert FENRowApplication in tier1_generators
    assert DEFAULT_VOLUMES["1.10_fen_row_application"] > 0
    assert TEMPLATES["1.10_fen_row_application"]


def test_package_piece_specific_templates_request_uci_moves():
    from chess_llm.sft.templates import TEMPLATES

    prompts = " ".join(TEMPLATES["2.2_piece_specific_moves"]).lower()

    assert "which squares can" not in prompts
    assert "where can" not in prompts
    assert "uci" in prompts


def test_package_side_piece_inventory_generator_lists_side_to_move_pieces():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import SidePieceInventory

    fen = "8/8/8/8/8/8/4P3/R3K2k w - - 0 1"
    gen = SidePieceInventory(
        config={"fen_pool": [{"fen": fen}], "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())

    assert row["task"] == "2.0_side_piece_inventory"
    assert row["metadata"]["side_to_move"] == "white"
    assert row["metadata"]["side_piece_inventory"] == [
        {"square": "a1", "piece": "white rook"},
        {"square": "e1", "piece": "white king"},
        {"square": "e2", "piece": "white pawn"},
    ]
    assert row["messages"][2]["content"] == (
        "Side to move: white.\n"
        "Pieces: a1 white rook; e1 white king; e2 white pawn."
    )
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_piece_specific_generator_includes_blocked_no_move_pieces():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import PieceSpecificMoves

    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    gen = PieceSpecificMoves(
        config={"fen_pool": [{"fen": starting_fen}], "volume_override": 1},
        rng=Random(1),
    )

    row = next(gen.generate())

    assert row["metadata"]["source_square"] == "e1"
    assert row["metadata"]["piece"] == "king"
    assert row["metadata"]["expected_moves"] == "No legal moves."
    assert row["messages"][2]["content"] == "No legal moves."
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_rules_mechanics_tasks_are_registered_for_tier2_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import SidePieceInventory
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import TEMPLATES

    tier2_generators = pipeline.TIER_GENERATORS[2]

    assert SidePieceInventory in tier2_generators
    assert DEFAULT_VOLUMES["2.0_side_piece_inventory"] > 0
    assert TEMPLATES["2.0_side_piece_inventory"]


def test_package_state_tracking_generator_defaults_to_one_ply():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    gen = StateTracking(
        config={
            "game_positions": [
                {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
                for _ in range(20)
            ],
            "volume_override": 20,
        },
        rng=Random(11),
    )

    rows = list(gen.generate())

    assert rows
    assert all(int(row["metadata"]["n_moves"]) == 1 for row in rows)


def test_package_state_tracking_samples_game_positions_in_shuffled_order():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    positions = [
        {
            "fen": fen,
            "metadata": {"label": label},
        }
        for label, fen in [
            ("first", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("second", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            ("third", "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
            ("fourth", "rnbqkbnr/ppp2ppp/3pp3/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"),
            ("fifth", "rnbqkbnr/ppp2ppp/3pp3/8/2PPP3/8/PP3PPP/RNBQKBNR b KQkq - 0 3"),
        ]
    ]
    gen = StateTracking(
        config={"game_positions": positions, "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())

    assert row["metadata"]["label"] == "third"


def test_package_state_tracking_falls_back_to_fen_pool_when_game_positions_empty():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    fen_pool = [
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
        for _ in range(6)
    ]
    gen = StateTracking(
        config={"game_positions": [], "fen_pool": fen_pool, "volume_override": 6},
        rng=Random(13),
    )

    rows = list(gen.generate())

    assert len(rows) == 6
    assert all(row["task"] == "1.5_state_tracking" for row in rows)
    assert all(row["metadata"]["moves"] for row in rows)


def test_package_state_tracking_falls_back_when_game_positions_are_blocked():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    blocked_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_pool = [
        {"fen": fen}
        for fen in [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
            "rnbqkbnr/ppp2ppp/3pp3/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        ]
    ]
    gen = StateTracking(
        config={
            "game_positions": [{"fen": blocked_fen} for _ in range(3)],
            "fen_pool": fen_pool,
            "volume_override": 3,
        },
        blocklist=frozenset({blocked_fen}),
        rng=Random(17),
    )

    rows = list(gen.generate())

    assert len(rows) == 3
    assert {row["fen"] for row in rows}.isdisjoint({blocked_fen})


def test_package_state_tracking_generator_teaches_result_fen_only():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    gen = StateTracking(
        config={
            "game_positions": [
                {"fen": "8/8/8/8/8/8/4P3/4K2k w - - 0 1"},
            ],
            "volume_override": 1,
            "state_tracking_min_plies": 1,
            "state_tracking_max_plies": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert answer == f"Result FEN: {row['metadata']['result_fen']}"
    assert "Move 1:" not in answer
    assert "Lookup:" not in answer
    assert "Squares:" not in answer
    assert "Ranks:" not in answer
    assert "Changed squares:" not in answer
    assert "Result placement:" not in answer
    assert row["metadata"]["move_details"][0]["move"] == row["metadata"]["moves"]
    assert row["metadata"]["move_details"][0]["changed_squares"]
    assert row["metadata"]["move_details"][0]["rank_updates"]


def test_package_state_tracking_generator_uses_complete_start_fen_state():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    gen = StateTracking(
        config={
            "game_positions": [{"fen": "4k3/8/8/8/8/8/4P3/4K3 w - -"}],
            "volume_override": 1,
            "state_tracking_min_plies": 1,
            "state_tracking_max_plies": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    prompt = row["messages"][1]["content"]

    assert row["fen"] == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" in prompt
    assert "State needed for full FEN:" in prompt
    assert "Side to move: white" in prompt
    assert "Halfmove clock: 0" in prompt
    assert "Fullmove number: 1" in prompt


def test_package_state_tracking_generator_preserves_non_default_fen_counters():
    from chess_llm.sft.generators.tier1_perception import StateTracking

    gen = StateTracking(
        config={
            "game_positions": [
                {"fen": "4k3/8/8/8/8/8/4P3/4K3 b - - 17 23"},
            ],
            "volume_override": 1,
            "state_tracking_min_plies": 1,
            "state_tracking_max_plies": 1,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    prompt = row["messages"][1]["content"]
    answer = row["messages"][2]["content"]

    assert row["fen"] == "4k3/8/8/8/8/8/4P3/4K3 b - - 17 23"
    assert "Halfmove clock: 17" in prompt
    assert "Fullmove number: 23" in prompt
    assert row["metadata"]["result_fen"].endswith("18 24")
    assert answer.endswith(f"Result FEN: {row['metadata']['result_fen']}")


def test_package_state_tracking_move_detail_orders_origin_before_destination():
    import chess

    from chess_llm.sft.generators.tier1_perception import _state_tracking_move_detail

    before = chess.Board("7k/1p6/8/8/8/8/8/4K3 b - - 0 1")
    move = chess.Move.from_uci("b7b6")
    after = before.copy(stack=False)
    after.push(move)

    detail = _state_tracking_move_detail(before, move, after, 1)

    assert [item["square"] for item in detail["changed_squares"]] == ["b7", "b6"]


def test_package_state_tracking_move_detail_records_fen_rank_update():
    import chess

    from chess_llm.sft.generators.tier1_perception import _state_tracking_move_detail

    before = chess.Board("rnbq1rk1/ppp1ppbp/5np1/8/2BP4/2N2N2/PPP3PP/R1B1QRK1 b - - 5 8")
    move = chess.Move.from_uci("f8e8")
    after = before.copy(stack=False)
    after.push(move)

    detail = _state_tracking_move_detail(before, move, after, 1)

    assert detail["rank_updates"] == [
        {"rank": "8", "before": "rnbq1rk1", "after": "rnbqr1k1"}
    ]


def test_package_state_tracking_formatter_uses_square_lookup_and_single_result_fen():
    import chess

    from chess_llm.sft.generators.tier1_perception import (
        _format_state_tracking_answer,
        _state_tracking_move_detail,
    )

    before = chess.Board("8/8/8/8/8/8/8/3QK2k w - - 0 1")
    move = chess.Move.from_uci("d1f3")
    after = before.copy(stack=False)
    after.push(move)

    answer = _format_state_tracking_answer(
        [_state_tracking_move_detail(before, move, after, 1)],
        after.fen(),
    )

    assert answer.splitlines() == [
        "Move 1: white queen d1f3.",
        "Lookup: d1=white queen; f3=empty.",
        "Squares: d1 white queen->empty; f3 empty->white queen.",
        "Ranks: rank 1 d Q->1; rank 3 f 1->Q.",
        f"Result FEN: {after.fen()}",
    ]
    assert answer.count("Result FEN:") == 1
    assert "Result placement:" not in answer


def test_package_board_to_fen_generator_canonicalizes_to_complete_fen():
    from chess_llm.sft.generators.tier1_perception import BoardToFEN

    gen = BoardToFEN(
        config={
            "fen_pool": [{"fen": "4k3/8/8/8/8/8/4K3/8 w - -"}],
            "volume_override": 1,
        },
        rng=Random(19),
    )

    row = next(gen.generate())

    assert row["fen"] == "4k3/8/8/8/8/8/4K3/8 w - - 0 1"
    assert row["messages"][2]["content"] == "4k3/8/8/8/8/8/4K3/8 w - - 0 1"


def test_package_legal_move_generator_samples_fen_pool_in_shuffled_order():
    from chess_llm.sft.generators.tier2_rules import LegalMoveGen

    pool = [
        {
            "fen": fen,
            "metadata": {"label": label},
        }
        for label, fen in [
            ("first", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("second", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            ("third", "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
            ("fourth", "rnbqkbnr/ppp2ppp/3pp3/8/3PPPP2/8/PPP3PP/RNBQKBNR b KQkq - 0 3"),
            ("fifth", "rnbqkbnr/ppp2ppp/3pp3/8/2PPP3/8/PP3PPP/RNBQKBNR b KQkq - 0 3"),
        ]
    ]
    gen = LegalMoveGen(config={"fen_pool": pool, "volume_override": 1}, rng=Random(0))

    row = next(gen.generate())

    assert row["metadata"]["label"] == "third"


def test_package_legal_move_generator_records_move_set_metadata():
    from chess_llm.sft.generators.tier2_rules import LegalMoveGen

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    gen = LegalMoveGen(
        config={"fen_pool": [{"fen": fen}], "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())

    assert row["metadata"]["legal_move_count"] == 20
    assert row["metadata"]["side_to_move"] == "white"
    assert row["metadata"]["in_check"] is False
    assert row["metadata"]["legal_moves_by_piece"]["b1"] == ["b1a3", "b1c3"]
    assert row["metadata"]["expected_answer"] == row["messages"][2]["content"]
    assert row["messages"][2]["content"].startswith("Side to move: white.")
    assert "Pieces to inspect:" not in row["messages"][2]["content"]
    assert "Moves by piece:" not in row["messages"][2]["content"]
    assert "Legal moves:" in row["messages"][2]["content"]
    assert len(row["messages"][2]["content"].splitlines()) == 2


def test_package_move_legality_generator_samples_fen_pool_in_shuffled_order():
    from chess_llm.sft.generators.tier2_rules import MoveLegalityCheck

    pool = [
        {
            "fen": fen,
            "metadata": {"label": label},
        }
        for label, fen in [
            ("first", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("second", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            ("third", "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"),
            ("fourth", "rnbqkbnr/ppp2ppp/3pp3/8/3PPPP2/8/PPP3PP/RNBQKBNR b KQkq - 0 3"),
            ("fifth", "rnbqkbnr/ppp2ppp/3pp3/8/2PPP3/8/PP3PPP/RNBQKBNR b KQkq - 0 3"),
        ]
    ]
    gen = MoveLegalityCheck(
        config={"fen_pool": pool, "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())

    assert row["metadata"]["label"] == "third"


def test_package_move_legality_generator_records_expected_label_metadata():
    from chess_llm.sft.generators.tier2_rules import MoveLegalityCheck

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    gen = MoveLegalityCheck(
        config={"fen_pool": [{"fen": fen}], "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())

    assert "tested_move" in row["metadata"]
    assert isinstance(row["metadata"]["expected_is_legal"], bool)
    assert row["metadata"]["legality_reason_label"] in {
        "legal",
        "empty_source",
        "wrong_side_piece",
        "own_piece_destination",
        "illegal_piece_movement_or_blocked_path",
        "missing_or_invalid_promotion",
        "does_not_resolve_check",
        "pinned_piece_exposes_king",
        "king_would_be_in_check",
    }
    assert row["metadata"]["expected_answer"] == row["messages"][2]["content"]
    assert "Reason:" in row["messages"][2]["content"]


def test_package_move_legality_classifier_explains_illegal_move_reason():
    import chess

    from chess_llm.core.legality import classify_move_legality

    cases = [
        (STARTING_FEN, "a3a4", "empty_source"),
        (STARTING_FEN, "a7a6", "wrong_side_piece"),
        (STARTING_FEN, "e1d1", "own_piece_destination"),
        (STARTING_FEN, "e2e5", "illegal_piece_movement_or_blocked_path"),
        ("k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1", "e2d2", "pinned_piece_exposes_king"),
        ("4r2k/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a2", "does_not_resolve_check"),
    ]

    for fen, move_uci, expected_reason in cases:
        result = classify_move_legality(chess.Board(fen), move_uci)

        assert result.is_legal is False
        assert result.reason_label == expected_reason


def test_package_check_detection_includes_rare_states_from_normal_pool():
    from chess_llm.sft.generators.tier2_rules import CheckDetection

    gen = CheckDetection(
        config={
            "fen_pool": [
                {"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"}
                for _ in range(20)
            ],
            "volume_override": 12,
        },
        rng=Random(3),
    )

    rows = list(gen.generate())
    labels = Counter(row["metadata"]["state_label"] for row in rows)

    assert len(rows) == 12
    assert labels["check"] > 0
    assert labels["checkmate"] > 0
    assert labels["stalemate"] > 0
    assert labels["normal"] > 0


def test_package_check_detection_keeps_normal_state_majority():
    from chess_llm.sft.generators.tier2_rules import CheckDetection

    gen = CheckDetection(
        config={
            "fen_pool": [
                {"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"}
                for _ in range(30)
            ],
            "volume_override": 20,
        },
        rng=Random(13),
    )

    labels = Counter(row["metadata"]["state_label"] for row in gen.generate())

    assert labels["normal"] >= 10
    assert labels["check"] > 0
    assert labels["checkmate"] > 0
    assert labels["stalemate"] > 0


def test_package_special_rules_includes_positive_rule_buckets_from_normal_pool():
    from chess_llm.sft.generators.tier2_rules import SpecialRules

    gen = SpecialRules(
        config={
            "fen_pool": [
                {"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"}
                for _ in range(20)
            ],
            "volume_override": 12,
        },
        rng=Random(5),
    )

    rows = list(gen.generate())
    buckets = Counter(row["metadata"]["special_rule_bucket"] for row in rows)

    assert len(rows) == 12
    assert buckets["castling"] > 0
    assert buckets["en_passant"] > 0
    assert buckets["promotion"] > 0
    assert buckets["none"] > 0


def test_package_special_rules_keeps_no_special_majority():
    from chess_llm.sft.generators.tier2_rules import SpecialRules

    gen = SpecialRules(
        config={
            "fen_pool": [
                {"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"}
                for _ in range(30)
            ],
            "volume_override": 20,
        },
        rng=Random(17),
    )

    buckets = Counter(row["metadata"]["special_rule_bucket"] for row in gen.generate())

    assert buckets["none"] >= 10
    assert buckets["castling"] > 0
    assert buckets["en_passant"] > 0
    assert buckets["promotion"] > 0


def test_package_special_rules_generic_promotion_answer_uses_exact_uci_moves(monkeypatch):
    from chess_llm.sft.generators.tier2_rules import SpecialRules

    monkeypatch.setattr(
        "chess_llm.sft.generators.tier2_rules.select_template",
        lambda _task_id, _rng: (
            "Given FEN: {fen}\n"
            "List any special moves available (castling, en passant, promotion)."
        ),
    )
    gen = SpecialRules(
        config={
            "fen_pool": [
                {"fen": "7k/P7/8/8/8/8/8/4K3 w - - 0 1"},
            ],
            "volume_override": 20,
        },
        rng=Random(7),
    )

    row = next(
        row
        for row in gen.generate()
        if row["metadata"]["special_rule_bucket"] == "promotion"
    )
    answer = row["messages"][2]["content"]
    promotion_moves = row["metadata"]["promotion_moves"].split()

    assert promotion_moves
    for move in promotion_moves:
        assert move in answer
    assert "Promotion possible from:" not in answer


def test_pipeline_registry_uses_package_generators():
    from chess_llm.sft import pipeline

    assert pipeline.TIER_GENERATORS[1][0].__module__.startswith(
        "chess_llm.sft.generators."
    )
    assert pipeline.TIER_GENERATORS[7][0].__module__.startswith(
        "chess_llm.sft.generators."
    )


def test_legacy_generator_imports_alias_package_modules(monkeypatch):
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    _clear_legacy_generator_modules()

    package_tier1 = importlib.import_module("chess_llm.sft.generators.tier1_perception")
    legacy_tier1 = importlib.import_module("generators.tier1_perception")

    assert legacy_tier1 is package_tier1
