from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path
from random import Random

import chess


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


def test_package_material_decomposition_generators_emit_structured_traces():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import (
        MaterialBalanceTrace,
        MaterialInventory,
        MaterialPieceCounts,
        MaterialValueTotals,
    )

    fen = "8/8/8/8/8/8/6p1/4K2k w - - 0 1"
    config = {"fen_pool": [{"fen": fen} for _ in range(4)], "volume_override": 1}
    expected = {
        "1.15_material_inventory": (
            "White inventory: king=e1; queen=none; rook=none; bishop=none; "
            "knight=none; pawn=none.\n"
            "Black inventory: king=h1; queen=none; rook=none; bishop=none; "
            "knight=none; pawn=g2."
        ),
        "1.16_material_piece_counts": (
            "White counts: king=1; queen=0; rook=0; bishop=0; knight=0; pawn=0.\n"
            "Black counts: king=1; queen=0; rook=0; bishop=0; knight=0; pawn=1."
        ),
        "1.17_material_value_totals": (
            "White values: king=0; queen=0; rook=0; bishop=0; knight=0; pawn=0; total=0.\n"
            "Black values: king=0; queen=0; rook=0; bishop=0; knight=0; pawn=1; total=1."
        ),
        "1.18_material_balance_trace": (
            "Inventory: white king=e1; queen=none; rook=none; bishop=none; "
            "knight=none; pawn=none | black king=h1; queen=none; rook=none; "
            "bishop=none; knight=none; pawn=g2\n"
            "Counts: white king=1; queen=0; rook=0; bishop=0; knight=0; pawn=0 | "
            "black king=1; queen=0; rook=0; bishop=0; knight=0; pawn=1\n"
            "Values: white total=0; black total=1\n"
            "Balance: Black is up 1 point(s) of material."
        ),
    }

    for generator_cls in (
        MaterialInventory,
        MaterialPieceCounts,
        MaterialValueTotals,
        MaterialBalanceTrace,
    ):
        row = next(generator_cls(config=config, rng=Random(0)).generate())

        assert row["messages"][2]["content"] == expected[row["task"]]
        assert row["metadata"]["expected_answer"] == expected[row["task"]]
        assert row["metadata"]["example_identity"]
        passed, errors = validate_example(row)
        assert passed is True, errors


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
        MaterialBalanceTrace,
        PieceIdentification,
        SquareLookup,
        StateTracking,
    )
    from chess_llm.sft.generators.tier2_rules import LegalMoveGen, PieceLegalFilter

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
    material_trace = next(
        MaterialBalanceTrace(
            config={"fen_pool": [{"fen": position}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    legal_filter = next(
        PieceLegalFilter(
            config={
                "fen_pool": [{"fen": "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"}],
                "volume_override": 1,
            },
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
    assert (
        '"Counts:", "Values:", and "Balance:".'
        in _user_prompt(material_trace)
    )
    assert (
        'Rejected entries must be "<uci> <reason_label>" separated by semicolons, '
        'or "none" when no moves are rejected.'
        in _user_prompt(legal_filter)
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
        MaterialBalanceTrace,
        MaterialInventory,
        MaterialPieceCounts,
        MaterialValueTotals,
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
        (MaterialInventory, "1.15_material_inventory"),
        (MaterialPieceCounts, "1.16_material_piece_counts"),
        (MaterialValueTotals, "1.17_material_value_totals"),
        (MaterialBalanceTrace, "1.18_material_balance_trace"),
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


def test_package_move_result_generators_skip_blocklisted_result_fens():
    import chess

    from chess_llm.core.board import variant_fen_key
    from chess_llm.sft.decontamination import row_contaminated_fens
    from chess_llm.sft.generators.tier1_perception import (
        FENAssembly,
        FENRowApplication,
        MultiMoveStateTracking,
        StateTracking,
    )

    contaminated_start = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"
    clean_start = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    board = chess.Board(contaminated_start)
    blocked_results = set()
    for move in board.legal_moves:
        next_board = board.copy(stack=False)
        next_board.push(move)
        blocked_results.add(variant_fen_key(next_board.fen()))
    blocklist = frozenset(blocked_results)

    cases = [
        (StateTracking, {}),
        (MultiMoveStateTracking, {
            "multi_state_tracking_min_plies": 1,
            "multi_state_tracking_max_plies": 1,
        }),
        (FENAssembly, {}),
        (FENRowApplication, {}),
    ]

    for gen_cls, extra_config in cases:
        config = {
            "game_positions": [
                {"fen": contaminated_start},
            ],
            "fen_pool": [{"fen": clean_start}],
            "volume_override": 1,
            **extra_config,
        }
        gen = gen_cls(config=config, blocklist=blocklist, rng=Random(1))
        row = next(gen.generate())

        assert row["fen"] == clean_start
        assert row_contaminated_fens(row, blocklist) == []


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


def test_package_legal_decomposition_generators_emit_structured_traces():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import (
        KingSafetyFilter,
        LegalMovesByPiece,
        PieceLegalFilter,
        PiecePseudoLegalMoves,
    )

    quiet_rook_fen = "7k/8/8/8/8/8/8/R3K3 w - - 0 1"
    pinned_rook_fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    checked_fen = "4r2k/8/8/8/8/8/8/R3K3 w - - 0 1"

    pseudo = next(
        PiecePseudoLegalMoves(
            config={"fen_pool": [{"fen": quiet_rook_fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    assert pseudo["task"] == "2.6_piece_pseudo_legal_moves"
    assert pseudo["metadata"]["source_square"] == "a1"
    assert pseudo["messages"][2]["content"] == (
        "Pseudo-legal moves from a1: "
        "a1a2 a1a3 a1a4 a1a5 a1a6 a1a7 a1a8 a1b1 a1c1 a1d1"
    )

    legal_filter = next(
        PieceLegalFilter(
            config={"fen_pool": [{"fen": pinned_rook_fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    assert legal_filter["task"] == "2.7_piece_legal_filter"
    assert legal_filter["metadata"]["source_square"] == "e2"
    assert legal_filter["messages"][2]["content"] == (
        "Pseudo-legal from e2: "
        "e2a2 e2b2 e2c2 e2d2 e2e3 e2e4 e2e5 e2e6 e2e7 e2e8 e2f2 e2g2 e2h2.\n"
        "Legal: e2e3 e2e4 e2e5 e2e6 e2e7 e2e8.\n"
        "Rejected: e2a2 pinned_piece_exposes_king; e2b2 pinned_piece_exposes_king; "
        "e2c2 pinned_piece_exposes_king; e2d2 pinned_piece_exposes_king; "
        "e2f2 pinned_piece_exposes_king; e2g2 pinned_piece_exposes_king; "
        "e2h2 pinned_piece_exposes_king."
    )

    # The >=50% rejection floor defers rejection-free rows until a
    # contrast row has been emitted, so lead with the pinned position.
    quiet_rows = list(
        PieceLegalFilter(
            config={
                "fen_pool": [{"fen": pinned_rook_fen}, {"fen": quiet_rook_fen}],
                "volume_override": 2,
            },
            rng=Random(0),
        ).generate()
    )
    quiet_filter = quiet_rows[1]
    assert quiet_filter["task"] == "2.7_piece_legal_filter"
    assert quiet_filter["fen"] == quiet_rook_fen
    assert quiet_filter["metadata"]["rejection_category"] == "no_rejection"
    assert quiet_filter["metadata"]["rejected_count"] == 0
    assert quiet_filter["metadata"]["rejected_moves"] == "none"
    assert quiet_filter["messages"][2]["content"].endswith("Rejected: none.")

    safety = next(
        KingSafetyFilter(
            config={"fen_pool": [{"fen": checked_fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    assert safety["task"] == "2.8_king_safety_filter"
    assert safety["metadata"]["tested_move"] == "a1a2"
    assert safety["messages"][2]["content"] == (
        "Move: a1a2.\n"
        "Pseudo-legal: yes.\n"
        "King safe after move: no.\n"
        "Final: illegal; does_not_resolve_check."
    )

    grouped = next(
        LegalMovesByPiece(
            config={"fen_pool": [{"fen": quiet_rook_fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    assert grouped["task"] == "2.9_legal_moves_by_piece"
    assert grouped["messages"][2]["content"] == (
        "Side to move: white.\n"
        "Pieces: a1 white rook; e1 white king.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1a3 a1a4 a1a5 a1a6 a1a7 a1a8 a1b1 a1c1 a1d1\n"
        "e1 white king: e1d1 e1d2 e1e2 e1f1 e1f2\n"
        "All legal moves: a1a2 a1a3 a1a4 a1a5 a1a6 a1a7 a1a8 a1b1 a1c1 a1d1 "
        "e1d1 e1d2 e1e2 e1f1 e1f2"
    )

    for row in (pseudo, legal_filter, quiet_filter, safety, grouped):
        assert row["metadata"]["expected_answer"] == row["messages"][2]["content"]
        assert row["metadata"]["example_identity"]
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
    from chess_llm.sft.generators import (
        KingSafetyFilter,
        LegalMovesByPiece,
        PieceLegalFilter,
        PiecePseudoLegalMoves,
    )
    from chess_llm.sft.generators import SidePieceInventory
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import TEMPLATES

    tier2_generators = pipeline.TIER_GENERATORS[2]

    assert SidePieceInventory in tier2_generators
    assert PiecePseudoLegalMoves in tier2_generators
    assert PieceLegalFilter in tier2_generators
    assert KingSafetyFilter in tier2_generators
    assert LegalMovesByPiece in tier2_generators
    assert DEFAULT_VOLUMES["2.0_side_piece_inventory"] > 0
    assert TEMPLATES["2.0_side_piece_inventory"]
    for task_id in (
        "2.6_piece_pseudo_legal_moves",
        "2.7_piece_legal_filter",
        "2.8_king_safety_filter",
        "2.9_legal_moves_by_piece",
    ):
        assert DEFAULT_VOLUMES[task_id] > 0
        assert TEMPLATES[task_id]


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


def test_package_board_to_fen_teaches_legal_en_passant_convention():
    from chess_llm.sft.generators.tier1_perception import BoardToFEN

    fen_with_phantom_ep = (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )
    gen = BoardToFEN(
        config={"fen_pool": [{"fen": fen_with_phantom_ep}], "volume_override": 1},
        rng=Random(0),
    )

    row = next(gen.generate())

    assert row["messages"][2]["content"] == (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    assert row["fen"] == row["messages"][2]["content"]


def test_package_tier1_pool_generators_sample_fen_pool_in_shuffled_order():
    from chess_llm.sft.generators.tier1_perception import BoardToFEN, RankLookup

    pool = [
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

    for generator_cls in (BoardToFEN, RankLookup):
        row = next(
            generator_cls(
                config={"fen_pool": list(pool), "volume_override": 1},
                rng=Random(0),
            ).generate()
        )
        assert row["metadata"]["label"] == "third"


class _ForceCastleRng:
    """Fake rng whose choice() prefers the e1g1 castling move."""

    def choice(self, values):
        for value in values:
            if getattr(value, "uci", None) is not None and value.uci() == "e1g1":
                return value
        return values[0]


def test_package_fen_rank_cell_edit_castling_teaches_true_post_move_row():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import FENRankCellEdit

    gen = FENRankCellEdit(
        config={
            "game_positions": [{"fen": "4k3/8/8/8/8/8/8/4K2R w K - 0 1"}],
            "volume_override": 1,
        },
        rng=_ForceCastleRng(),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert row["metadata"]["move"] == "e1g1"
    assert row["metadata"]["before_row"] == "4K2R"
    assert row["metadata"]["after_row"] == "5RK1"
    assert answer == "rank 1: 4K2R -> 5RK1"
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_piece_counting_full_material_identity_ignores_color_and_piece(monkeypatch):
    from chess_llm.sft.generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nWhat is the material count for both sides?"
    monkeypatch.setattr(
        "chess_llm.sft.generators.tier1_perception.select_template",
        lambda _task_id, _rng: tpl,
    )

    rows = [
        next(
            PieceCounting(
                config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
                rng=Random(seed),
            ).generate()
        )
        for seed in (0, 99)
    ]

    assert rows[0]["metadata"]["example_identity"] == rows[1]["metadata"]["example_identity"]
    for row in rows:
        assert "color" not in row["metadata"]
        assert "piece" not in row["metadata"]


def test_package_tier5_variant_identities_are_distinct():
    from chess_llm.sft.generators.tier5_openings import OpeningIdentification
    from chess_llm.sft.identity import TASK_IDENTITY_FIELDS

    for task_id in (
        "5.1_opening_identification",
        "5.2_opening_continuation",
        "5.3_opening_principles",
    ):
        assert TASK_IDENTITY_FIELDS[task_id] == ("variant", "cycle")

    opening = {
        "fen": STARTING_FEN,
        "name": "Test Opening",
        "eco": "C20",
        "uci_moves": ["e2e4"],
    }
    gen = OpeningIdentification(
        config={"openings": [opening], "volume_override": 5},
        rng=Random(0),
    )

    rows = list(gen.generate())
    identities = {row["metadata"]["example_identity"] for row in rows}

    assert len(rows) == 5
    assert len(identities) == 5


def test_package_tier5_opening_identification_cycles_small_sources_to_target():
    from chess_llm.sft.generators.tier5_openings import OpeningIdentification

    opening = {
        "fen": STARTING_FEN,
        "name": "Test Opening",
        "eco": "C20",
        "uci_moves": ["e2e4"],
    }
    gen = OpeningIdentification(
        config={"openings": [opening], "volume_override": 6},
        rng=Random(0),
    )

    rows = list(gen.generate())

    assert len(rows) == 6
    assert [row["metadata"]["cycle"] for row in rows] == [0, 0, 0, 0, 0, 1]
    assert len({row["metadata"]["example_identity"] for row in rows}) == 6


def test_package_tier5_continuation_and_principles_cycle_to_target():
    from chess_llm.sft.generators.tier5_openings import (
        OpeningContinuation,
        OpeningPrinciples,
    )

    opening = {
        "fen": STARTING_FEN,
        "name": "Test Opening",
        "eco": "C20",
        "uci_moves": ["e2e4"],
    }

    continuation_rows = list(
        OpeningContinuation(
            config={
                "openings": [opening],
                "book_moves": {STARTING_FEN: [("e2e4", 2), ("d2d4", 1)]},
                "volume_override": 5,
            },
            rng=Random(0),
        ).generate()
    )
    principles_rows = list(
        OpeningPrinciples(
            config={"openings": [opening], "volume_override": 6},
            rng=Random(0),
        ).generate()
    )

    assert len(continuation_rows) == 5
    assert [row["metadata"]["cycle"] for row in continuation_rows] == [0, 0, 0, 0, 1]
    assert len({row["metadata"]["example_identity"] for row in continuation_rows}) == 5
    assert len(principles_rows) == 6
    assert [row["metadata"]["cycle"] for row in principles_rows] == [0, 0, 0, 0, 0, 1]
    assert len({row["metadata"]["example_identity"] for row in principles_rows}) == 6


def test_package_cp_to_bucket_implements_shared_contract():
    from chess_llm.sft.generators.tier4_evaluation import _cp_to_bucket

    assert _cp_to_bucket(0) == "The position is equal."
    assert _cp_to_bucket(30) == "The position is equal."
    assert _cp_to_bucket(-49) == "The position is equal."
    assert _cp_to_bucket(50) == "White has a slight edge."
    assert _cp_to_bucket(-149) == "Black has a slight edge."
    assert _cp_to_bucket(150) == "White has a clear advantage."
    assert _cp_to_bucket(-299) == "Black has a clear advantage."
    assert _cp_to_bucket(300) == "White is winning."
    assert _cp_to_bucket(-599) == "Black is winning."
    assert _cp_to_bucket(600) == "White has a decisive advantage."
    # Values beyond the last bucket bound clamp to the last bucket.
    assert _cp_to_bucket(100_000) == "White has a decisive advantage."
    assert _cp_to_bucket(-250_000) == "Black has a decisive advantage."


def test_package_position_evaluation_answers_use_shared_bucket_sentences():
    from chess_llm.sft.generators.tier4_evaluation import PositionEvaluation

    evals = [
        {"fen": STARTING_FEN, "cp": 0},
        {"fen": STARTING_FEN, "cp": -75},
        {"fen": STARTING_FEN, "cp": 700},
    ]
    rows = list(
        PositionEvaluation(
            config={"position_evals": evals, "volume_override": 3},
            rng=Random(0),
        ).generate()
    )

    assert [row["messages"][2]["content"] for row in rows] == [
        "The position is equal.",
        "Black has a slight edge.",
        "White has a decisive advantage.",
    ]


def test_package_kqkp_endgame_principle_is_chess_accurate():
    from chess_llm.sft.generators.tier6_endgames import _ENDGAME_PRINCIPLES

    principles = _ENDGAME_PRINCIPLES["KQKP"]

    assert (
        "Queen vs pawn on the 7th: a bishop (c/f) or rook (a/h) pawn can draw; "
        "center and knight pawns lose."
        in principles
    )
    assert all("bishop/center pawn may draw" not in text for text in principles)


def test_package_tactical_patterns_filter_and_humanize_themes():
    from chess_llm.sft.generators.tier3_tactics import TacticalPatterns

    puzzles = [
        {
            "fen": STARTING_FEN,
            "themes": [
                "short",
                "crushing",
                "masterVsMaster",
                "discoveredAttack",
                "mateIn2",
            ],
            "solution_first_move": "e2e4",
            "rating": 1500,
        },
        {
            "fen": STARTING_FEN,
            "themes": ["short", "long", "masterVsMaster"],
            "solution_first_move": "d2d4",
            "rating": 1500,
        },
    ]
    rows = list(
        TacticalPatterns(
            config={"puzzles": puzzles, "volume_override": 2},
            rng=Random(0),
        ).generate()
    )

    assert rows[0]["messages"][2]["content"] == (
        "The tactic is discovered attack, mate in 2. Best move: e2e4"
    )
    assert rows[1]["messages"][2]["content"] == (
        "The tactic is tactical. Best move: d2d4"
    )


def test_package_attacked_defended_uses_fixed_count_grammar(monkeypatch):
    from chess_llm.sft.generators.tier3_tactics import AttackedDefended

    class ChooseE2:
        def choice(self, _items):
            return chess.E2

    monkeypatch.setattr(
        "chess_llm.sft.generators.tier3_tactics.select_template",
        lambda _task_id, _rng: "FEN: {fen}\nAnalyze attackers and defenders of {square}.",
    )

    rows = list(
        AttackedDefended(
            config={
                "fen_pool": ["4k3/8/8/8/4r3/8/4Q3/4K3 w - - 0 1"],
                "volume_override": 1,
            },
            rng=ChooseE2(),
        ).generate()
    )

    assert len(rows) == 1
    assert rows[0]["messages"][2]["content"] == (
        "Square: e2\n"
        "Occupant: white queen\n"
        "White attackers (1): king on e1\n"
        "Black attackers (1): rook on e4\n"
        "Defenders (1): king on e1"
    )
    assert "White attackers" in rows[0]["messages"][1]["content"]
    assert rows[0]["metadata"]["query_square"] == "e2"
    assert rows[0]["metadata"]["white_attacker_count"] == 1
    assert rows[0]["metadata"]["black_attacker_count"] == 1
    assert rows[0]["metadata"]["defender_count"] == 1


def test_package_attacked_defended_registered_with_fixed_contract():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators.tier3_tactics import AttackedDefended
    from chess_llm.sft.hub_upload import TASK_DESCRIPTIONS
    from chess_llm.sft.identity import TASK_IDENTITY_FIELDS
    from chess_llm.sft.templates import ANSWER_CONTRACTS

    assert AttackedDefended in pipeline.TIER_GENERATORS[3]
    assert "White attackers" in ANSWER_CONTRACTS["3.3_attacked_defended"]
    assert TASK_IDENTITY_FIELDS["3.3_attacked_defended"] == ("query_square",)
    assert TASK_DESCRIPTIONS["3.3_attacked_defended"]


def test_package_hanging_piece_status_teaches_attacked_defended_conjunction(monkeypatch):
    from chess_llm.sft.generators.tier3_tactics import HangingPieceStatus

    tpl = (
        "FEN: {fen}\nFor the {piece_description}, decide whether it is "
        "attacked, defended, and hanging."
    )
    monkeypatch.setattr(
        "chess_llm.sft.generators.tier3_tactics.select_template",
        lambda _task_id, _rng: tpl,
    )
    rows = list(
        HangingPieceStatus(
            config={
                "fen_pool": [
                    {"fen": "4k3/8/8/8/4r3/8/4Q3/K7 w - - 0 1"},
                    {"fen": "4k3/8/8/8/4r3/8/4Q3/4K3 w - - 0 1"},
                    {"fen": STARTING_FEN},
                ],
                "volume_override": 3,
            },
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 3
    answers = [row["messages"][2]["content"] for row in rows]
    assert answers[0] == (
        "Piece: white queen on e2\n"
        "Attacked: yes\n"
        "Defended: no\n"
        "Hanging: yes"
    )
    assert answers[1] == (
        "Piece: white queen on e2\n"
        "Attacked: yes\n"
        "Defended: yes\n"
        "Hanging: no"
    )
    assert answers[2].endswith("Hanging: no")
    assert "For the white queen on e2" in rows[0]["messages"][1]["content"]
    assert rows[0]["metadata"]["query_square"] == "e2"
    assert rows[0]["metadata"]["hanging_status"] == "hanging"


def test_package_hanging_piece_status_registered_for_tier3_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators.tier3_tactics import HangingPieceStatus
    from chess_llm.sft.hub_upload import TASK_DESCRIPTIONS
    from chess_llm.sft.identity import TASK_IDENTITY_FIELDS
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES

    assert HangingPieceStatus in pipeline.TIER_GENERATORS[3]
    assert DEFAULT_VOLUMES["3.6_hanging_piece_status"] > 0
    assert TEMPLATES["3.6_hanging_piece_status"]
    assert "Attacked:" in ANSWER_CONTRACTS["3.6_hanging_piece_status"]
    assert TASK_IDENTITY_FIELDS["3.6_hanging_piece_status"] == ("query_square",)
    assert TASK_DESCRIPTIONS["3.6_hanging_piece_status"]


def test_package_hanging_piece_filter_audits_attacked_pieces(monkeypatch):
    from chess_llm.sft.generators.tier3_tactics import HangingPieceFilter

    tpl = "FEN: {fen}\nAudit attacked pieces, then list hanging pieces."
    monkeypatch.setattr(
        "chess_llm.sft.generators.tier3_tactics.select_template",
        lambda _task_id, _rng: tpl,
    )
    rows = list(
        HangingPieceFilter(
            config={
                "fen_pool": [
                    "4r3/P1p5/8/1P6/2R1P1p1/4K1kp/8/8 w - - 1 55",
                    "6k1/8/5n2/8/4r3/8/4Q3/4R1K1 w - - 0 1",
                ],
                "volume_override": 2,
            },
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 2
    assert rows[0]["messages"][2]["content"] == (
        "Attacked white pawn on e4: Defended: yes; Hanging: no\n"
        "Attacked black pawn on c7: Defended: no; Hanging: yes\n"
        "Hanging pieces: black pawn on c7."
    )
    assert rows[1]["messages"][2]["content"] == (
        "Attacked white queen on e2: Defended: yes; Hanging: no\n"
        "Attacked black rook on e4: Defended: yes; Hanging: no\n"
        "No hanging pieces \u2014 all attacked pieces are defended."
    )
    assert rows[0]["metadata"]["attacked_piece_count"] == 2
    assert rows[0]["metadata"]["hanging_count"] == 1
    assert rows[0]["metadata"]["attacked_defended_count"] == 1


def test_package_hanging_piece_filter_registered_for_tier3_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators.tier3_tactics import HangingPieceFilter
    from chess_llm.sft.hub_upload import TASK_DESCRIPTIONS
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES

    assert HangingPieceFilter in pipeline.TIER_GENERATORS[3]
    assert DEFAULT_VOLUMES["3.7_hanging_piece_filter"] > 0
    assert TEMPLATES["3.7_hanging_piece_filter"]
    assert "Defended:" in ANSWER_CONTRACTS["3.7_hanging_piece_filter"]
    assert TASK_DESCRIPTIONS["3.7_hanging_piece_filter"]


def test_package_hanging_piece_claim_verification_balances_claim_types(monkeypatch):
    from chess_llm.sft.generators.tier3_tactics import HangingPieceClaimVerification

    tpl = "FEN: {fen}\nClaim to verify: {verification_claim}"
    monkeypatch.setattr(
        "chess_llm.sft.generators.tier3_tactics.select_template",
        lambda _task_id, _rng: tpl,
    )
    rows = list(
        HangingPieceClaimVerification(
            config={
                "fen_pool": [
                    "4k3/8/8/8/4r3/8/4Q3/K7 w - - 0 1",
                    "4k3/8/8/8/4r3/8/4Q3/4K3 w - - 0 1",
                    "4k3/8/8/8/4r3/8/4Q3/K7 w - - 0 1",
                ],
                "volume_override": 3,
            },
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 3
    assert "Claim to verify: white queen on e2 is hanging." in rows[0]["messages"][1]["content"]
    assert rows[0]["messages"][2]["content"] == (
        "Verdict: correct\n"
        "Attacked: yes\n"
        "Defended: no\n"
        "Hanging: yes\n"
        "Correction: none"
    )
    assert rows[1]["metadata"]["corruption_kind"] == "defended_decoy_claim"
    assert rows[1]["messages"][2]["content"] == (
        "Verdict: incorrect\n"
        "Attacked: yes\n"
        "Defended: yes\n"
        "Hanging: no\n"
        "Correction: white queen on e2 is not hanging."
    )
    assert rows[2]["metadata"]["corruption_kind"] == "missed_hanging_claim"
    assert rows[2]["messages"][2]["content"].startswith("Verdict: incorrect")


def test_package_hanging_piece_claim_verification_registered_for_tier3_generation():
    from chess_llm.evals.benchmark import DIAGNOSTIC_TASK_TYPES, TASK_METRIC_TYPE
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators.tier3_tactics import HangingPieceClaimVerification
    from chess_llm.sft.hub_upload import TASK_DESCRIPTIONS
    from chess_llm.sft.identity import TASK_IDENTITY_FIELDS
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES
    from chess_llm.training.phase_gate import DIAGNOSTIC_METRICS

    assert HangingPieceClaimVerification in pipeline.TIER_GENERATORS[3]
    assert DEFAULT_VOLUMES["3.8_hanging_piece_claim_verification"] > 0
    assert TEMPLATES["3.8_hanging_piece_claim_verification"]
    assert "Verdict:" in ANSWER_CONTRACTS["3.8_hanging_piece_claim_verification"]
    assert TASK_IDENTITY_FIELDS["3.8_hanging_piece_claim_verification"] == (
        "verification_claim",
        "corruption_kind",
    )
    assert TASK_DESCRIPTIONS["3.8_hanging_piece_claim_verification"]
    assert TASK_METRIC_TYPE["hanging_piece_claim_verification"] == (
        "hanging_piece_claim_verification"
    )
    assert "hanging_piece_claim_verification" in DIAGNOSTIC_TASK_TYPES
    assert "hanging_piece_claim_verification" in DIAGNOSTIC_METRICS


def test_package_mate_trace_claims_check_only_when_true():
    import chess

    from chess_llm.sft.generators.reasoning_traces import generate_tactical_trace

    quiet = generate_tactical_trace(
        chess.Board(STARTING_FEN), "e2e4", ["mateIn2"], "", Random(0)
    )
    assert "delivers check" not in quiet
    assert "tightens the mating net" in quiet

    checking = generate_tactical_trace(
        chess.Board("k7/8/8/8/8/8/1Q6/4K3 w - - 0 1"),
        "b2b8",
        ["mateIn1"],
        "",
        Random(0),
    )
    assert "delivers check from b8" in checking


def test_package_best_move_mate_traces_only_for_the_mating_side():
    from chess_llm.sft.generators.tier7_planning import BestMoveSelection

    black_defends_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    evals = [
        {"fen": STARTING_FEN, "best_move": "e2e4", "mate": 2, "depth": 30},
        {"fen": black_defends_fen, "best_move": "e7e5", "mate": 2, "depth": 30},
    ]
    rows = list(
        BestMoveSelection(
            config={"best_move_evals": evals, "volume_override": 2},
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 2
    assert "mating pattern" in rows[0]["messages"][2]["content"]
    assert "mating pattern" not in rows[1]["messages"][2]["content"]


def test_package_best_move_selection_excludes_mate_pairwise_rows():
    from chess_llm.sft.generators.tier7_planning import BestMoveSelection

    mate_rows = [
        {
            "fen": STARTING_FEN,
            "better_move": "e2e4",
            "move_a": "e2e4",
            "move_b": "d2d4",
            "strategy": "control the center",
            "tactic": "central pawn push",
        }
    ]

    only_mate = BestMoveSelection(
        config={"best_move_evals": [], "mate_rows": mate_rows, "volume_override": 3},
        rng=Random(0),
    )
    assert list(only_mate.generate()) == []

    evals = [{"fen": STARTING_FEN, "best_move": "e2e4", "cp": 20, "depth": 30}]
    mixed = BestMoveSelection(
        config={
            "best_move_evals": evals,
            "mate_rows": mate_rows,
            "volume_override": 5,
        },
        rng=Random(0),
    )
    rows = list(mixed.generate())
    assert rows
    assert all(row["metadata"]["source"] == "lichess_evals" for row in rows)


def test_package_candidate_ratings_generator_uses_true_multipv_rows_only():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier7_planning import CandidateRatings

    true_multipv = {
        "fen": STARTING_FEN,
        "multipv_depth": 18,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42},
            {"uci": "d2d4", "cp": 15},
            {"uci": "g1f3", "cp": 5},
            {"uci": "c2c4", "cp": -20},
            {"uci": "b1c3", "cp": -80},
        ],
    }
    pseudo_candidates = {
        "fen": STARTING_FEN,
        "best_move": "e2e4",
        "candidate_moves": ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3"],
    }

    rows = list(
        CandidateRatings(
            config={
                "candidate_rating_evals": [pseudo_candidates, true_multipv],
                "volume_override": 2,
            },
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 1
    row = rows[0]
    answer = row["messages"][2]["content"]
    assert row["task"] == "7.8_candidate_ratings"
    assert answer.splitlines() == [
        "Candidate e2e4: +42cp; Bucket: equal",
        "Candidate d2d4: +15cp; Bucket: equal",
        "Candidate g1f3: +5cp; Bucket: equal",
        "Candidate c2c4: -20cp; Bucket: equal",
        "Candidate b1c3: -80cp; Bucket: slight edge",
        "Best: e2e4",
    ]
    assert "e2e4 d2d4 g1f3 c2c4 b1c3" in row["messages"][1]["content"]
    assert row["metadata"]["source"] == "stockfish_multipv"
    assert row["metadata"]["target_move"] == "e2e4"
    assert row["metadata"]["candidate_moves"] == "e2e4 d2d4 g1f3 c2c4 b1c3"
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_candidate_ratings_registered_for_tier7_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import CandidateRatings
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES

    assert CandidateRatings in pipeline.TIER_GENERATORS[7]
    assert DEFAULT_VOLUMES["7.8_candidate_ratings"] > 0
    assert TEMPLATES["7.8_candidate_ratings"]
    assert "Candidate <uci>" in ANSWER_CONTRACTS["7.8_candidate_ratings"]


def test_package_best_line_trace_generator_uses_true_multipv_pv():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier7_planning import BestLineTrace

    candidate_row = {
        "fen": STARTING_FEN,
        "multipv_depth": 18,
        "multipv_k": 5,
        "pv_len": 4,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42, "pv_line": "e2e4 e7e5 g1f3 b8c6"},
            {"uci": "d2d4", "cp": 15, "pv_line": "d2d4 d7d5 c2c4"},
            {"uci": "g1f3", "cp": 5, "pv_line": "g1f3 d7d5"},
            {"uci": "c2c4", "cp": -20, "pv_line": "c2c4 e7e5"},
            {"uci": "b1c3", "cp": -80, "pv_line": "b1c3 d7d5"},
        ],
    }

    rows = list(
        BestLineTrace(
            config={"candidate_rating_evals": [candidate_row], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 1
    row = rows[0]
    answer = row["messages"][2]["content"]
    assert row["task"] == "7.10_best_line_trace"
    assert answer == "\n".join(
        [
            "<think>",
            "Root: e2e4",
            "Eval: +42cp; Bucket: equal",
            "PV: e2e4 e7e5 g1f3 b8c6",
            "Best: e2e4",
            "</think><move>e2e4</move>",
        ]
    )
    assert "engine best line" in row["messages"][1]["content"].lower()
    assert row["metadata"]["source"] == "stockfish_multipv"
    assert row["metadata"]["source_task"] == "7.8_candidate_ratings"
    assert row["metadata"]["target_move"] == "e2e4"
    assert row["metadata"]["pv"] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert row["metadata"]["expected_answer"] == answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_best_line_trace_generator_skips_invalid_pv():
    from chess_llm.sft.generators.tier7_planning import BestLineTrace

    invalid_row = {
        "fen": STARTING_FEN,
        "multipv_depth": 18,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42, "pv_line": "e2e5 e7e5"},
            {"uci": "d2d4", "cp": 15, "pv_line": "d2d4 d7d5"},
            {"uci": "g1f3", "cp": 5, "pv_line": "g1f3 d7d5"},
            {"uci": "c2c4", "cp": -20, "pv_line": "c2c4 e7e5"},
            {"uci": "b1c3", "cp": -80, "pv_line": "b1c3 d7d5"},
        ],
    }

    rows = list(
        BestLineTrace(
            config={"candidate_rating_evals": [invalid_row], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )

    assert rows == []


def test_package_best_line_trace_registered_for_tier7_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import BestLineTrace
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES

    assert BestLineTrace in pipeline.TIER_GENERATORS[7]
    assert DEFAULT_VOLUMES["7.10_best_line_trace"] > 0
    assert TEMPLATES["7.10_best_line_trace"]
    assert "<think>" in ANSWER_CONTRACTS["7.10_best_line_trace"]


def test_package_step_verification_generator_corrupts_candidate_rating_trace():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier7_verification import StepVerification

    candidate_row = {
        "fen": STARTING_FEN,
        "multipv_depth": 18,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42},
            {"uci": "d2d4", "cp": 15},
            {"uci": "g1f3", "cp": 5},
            {"uci": "c2c4", "cp": -20},
            {"uci": "b1c3", "cp": -80},
        ],
    }

    rows = list(
        StepVerification(
            config={"candidate_rating_evals": [candidate_row], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 1
    row = rows[0]
    answer = row["messages"][2]["content"]
    assert row["task"] == "7.9_step_verification"
    prompt = row["messages"][1]["content"]
    assert "numbered trace" in prompt or "Trace to verify:" in prompt
    assert "6. Best: d2d4" in prompt
    assert answer.splitlines() == [
        "Verdict: broken",
        "Faulty line: 6",
        "Error type: wrong_best",
        "Correction: Best should be e2e4.",
    ]
    assert row["metadata"]["source_task"] == "7.8_candidate_ratings"
    assert row["metadata"]["verification_verdict"] == "broken"
    assert row["metadata"]["faulty_line"] == 6
    assert row["metadata"]["error_type"] == "wrong_best"
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_step_verification_generator_includes_sound_cases():
    from chess_llm.sft.generators.tier7_verification import StepVerification

    candidate_row = {
        "fen": STARTING_FEN,
        "multipv_depth": 18,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42},
            {"uci": "d2d4", "cp": 15},
            {"uci": "g1f3", "cp": 5},
            {"uci": "c2c4", "cp": -20},
            {"uci": "b1c3", "cp": -80},
        ],
    }

    rows = list(
        StepVerification(
            config={"candidate_rating_evals": [candidate_row], "volume_override": 5},
            rng=Random(0),
        ).generate()
    )

    assert len(rows) == 5
    assert rows[-1]["metadata"]["verification_verdict"] == "sound"
    assert rows[-1]["metadata"]["faulty_line"] == "none"
    assert rows[-1]["messages"][2]["content"].splitlines() == [
        "Verdict: sound",
        "Faulty line: none",
        "Error type: none",
        "Correction: none",
    ]


def test_package_step_verification_generator_varies_candidate_corruptions():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier7_verification import StepVerification

    candidate_row = {
        "fen": STARTING_FEN,
        "multipv_depth": 18,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42},
            {"uci": "d2d4", "cp": 180},
            {"uci": "g1f3", "cp": 5},
            {"uci": "c2c4", "cp": -20},
            {"uci": "b1c3", "cp": -80},
        ],
    }

    rows = list(
        StepVerification(
            config={"candidate_rating_evals": [candidate_row], "volume_override": 5},
            rng=Random(0),
        ).generate()
    )

    assert [row["metadata"]["error_type"] for row in rows] == [
        "wrong_best",
        "wrong_bucket",
        "wrong_eval",
        "malformed_line",
        "none",
    ]
    assert [row["metadata"]["difficulty"] for row in rows] == [
        "hard",
        "hard",
        "hard",
        "easy",
        "sound",
    ]
    assert "Bucket: decisive" in rows[1]["messages"][1]["content"]
    assert "Candidate e2e4: -42cp" in rows[2]["messages"][1]["content"]
    assert "Candidate e2e4 +42cp; Bucket: equal" in rows[3]["messages"][1]["content"]
    assert all(validate_example(row)[0] for row in rows)


def test_package_step_verification_generator_uses_deterministic_trace_sources():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier7_verification import StepVerification

    fen = "7k/8/8/8/8/8/8/R3K3 w - - 0 1"
    rows = list(
        StepVerification(
            config={"fen_pool": [{"fen": fen}], "volume_override": 6},
            rng=Random(0),
        ).generate()
    )

    assert [row["metadata"]["source_task"] for row in rows] == [
        "1.18_material_balance_trace",
        "1.18_material_balance_trace",
        "2.10_ray_walk",
        "2.10_ray_walk",
        "2.11_legal_filter_trace",
        "2.11_legal_filter_trace",
    ]
    assert [row["metadata"]["verification_verdict"] for row in rows] == [
        "broken",
        "sound",
        "broken",
        "sound",
        "broken",
        "sound",
    ]
    assert rows[0]["metadata"]["error_type"] == "wrong_eval"
    assert rows[2]["metadata"]["error_type"] == "wrong_move_set"
    assert rows[4]["metadata"]["error_type"] == "wrong_move_set"
    assert all(validate_example(row)[0] for row in rows)


def test_package_step_verification_registered_for_tier7_generation():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import StepVerification
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES

    assert StepVerification in pipeline.TIER_GENERATORS[7]
    assert DEFAULT_VOLUMES["7.9_step_verification"] > 0
    assert TEMPLATES["7.9_step_verification"]
    assert "Verdict:" in ANSWER_CONTRACTS["7.9_step_verification"]


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


def test_package_move_legality_classifier_only_flags_promotion_for_pseudo_legal_shapes():
    import chess

    from chess_llm.core.legality import classify_move_legality

    cases = [
        # Push onto an occupied promotion square is a movement defect.
        ("r6k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8", "illegal_piece_movement_or_blocked_path"),
        # Diagonal move to an EMPTY promotion square is a movement defect.
        ("7k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8", "illegal_piece_movement_or_blocked_path"),
        # Push onto an own-piece promotion square is an own-destination defect.
        ("N6k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8", "own_piece_destination"),
        # A genuinely promotable push without a suffix is a promotion defect.
        ("7k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8", "missing_or_invalid_promotion"),
        # A capture-promotion without a suffix is a promotion defect.
        ("1r5k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8", "missing_or_invalid_promotion"),
    ]

    for fen, move_uci, expected_reason in cases:
        board = chess.Board(fen)
        result = classify_move_legality(board, move_uci)

        assert result.is_legal is False, (fen, move_uci)
        assert result.reason_label == expected_reason, (fen, move_uci)


def test_package_move_legality_classifier_labels_castling_king_safety():
    import chess

    from chess_llm.core.legality import classify_move_legality

    # Castling through an attacked transit square is a king-safety defect.
    through_check = classify_move_legality(
        chess.Board("4k3/8/8/8/8/8/5r2/4K2R w K - 0 1"),
        "e1g1",
    )
    assert through_check.is_legal is False
    assert through_check.reason_label == "king_would_be_in_check"

    # Castling out of check is a king-safety defect too.
    out_of_check = classify_move_legality(
        chess.Board("4k3/8/8/8/8/8/4r3/4K2R w K - 0 1"),
        "e1g1",
    )
    assert out_of_check.reason_label == "king_would_be_in_check"

    # A blocked castling path stays a movement/blocked-path defect.
    blocked_path = classify_move_legality(
        chess.Board("4k3/8/8/8/8/8/8/4KB1R w K - 0 1"),
        "e1g1",
    )
    assert blocked_path.reason_label == "illegal_piece_movement_or_blocked_path"

    # Castling without rights stays a movement defect.
    no_rights = classify_move_legality(
        chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1"),
        "e1g1",
    )
    assert no_rights.reason_label == "illegal_piece_movement_or_blocked_path"


def test_package_select_bucketed_rows_fills_from_unique_rows_only():
    from chess_llm.sft.generators.tier2_rules import _select_bucketed_rows

    buckets = {
        "a": [{"fen": f"a{i}"} for i in range(3)],
        "b": [{"fen": f"b{i}"} for i in range(2)],
    }
    quotas = {"a": 5, "b": 5}

    selected = _select_bucketed_rows(buckets, quotas, 10, Random(0))
    fens = [row["fen"] for _bucket, row in selected]

    assert len(selected) == 5
    assert len(set(fens)) == 5


def test_package_select_bucketed_rows_backfills_from_other_buckets():
    from chess_llm.sft.generators.tier2_rules import _select_bucketed_rows

    buckets = {
        "a": [{"fen": f"a{i}"} for i in range(4)],
        "b": [{"fen": "b0"}],
    }
    quotas = {"a": 1, "b": 4}

    selected = _select_bucketed_rows(buckets, quotas, 5, Random(1))
    fens = sorted(row["fen"] for _bucket, row in selected)

    assert fens == ["a0", "a1", "a2", "a3", "b0"]


def test_package_check_detection_templates_match_state_answer_contract():
    from chess_llm.sft.templates import TEMPLATES

    templates = TEMPLATES["2.4_check_detection"]

    assert "FEN: {fen}\nIs the king in check?" not in templates
    assert "FEN: {fen}\nIs the side to move in check?" not in templates
    assert (
        "FEN: {fen}\nWhat is the check state: check, checkmate, stalemate, or normal?"
        in templates
    )
    assert (
        "FEN: {fen}\nWhat is the check state for the side to move: "
        "check, checkmate, stalemate, or normal?"
        in templates
    )


def test_package_move_legality_negatives_include_king_safety_share():
    from chess_llm.sft.generators.tier2_rules import MoveLegalityCheck

    pinned_fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    gen = MoveLegalityCheck(
        config={
            "fen_pool": [{"fen": pinned_fen} for _ in range(60)],
            "volume_override": 60,
        },
        rng=Random(0),
    )

    rows = list(gen.generate())
    negatives = [
        row for row in rows if row["metadata"]["expected_is_legal"] is False
    ]
    positives = [row for row in rows if row["metadata"]["expected_is_legal"] is True]

    assert negatives
    assert all(
        row["metadata"]["negative_category"] in {"king_safety", "other_illegal"}
        for row in negatives
    )
    king_safety = [
        row
        for row in negatives
        if row["metadata"]["negative_category"] == "king_safety"
    ]
    assert len(king_safety) / len(negatives) >= 0.25
    assert all("negative_category" not in row["metadata"] for row in positives)


def test_package_legal_move_generator_emits_none_for_terminal_positions():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import LegalMoveGen

    checkmate_fen = "R6k/8/7K/8/8/8/8/8 b - - 0 1"
    stalemate_fen = "k7/8/1QK5/8/8/8/8/8 b - - 0 1"
    for fen in (checkmate_fen, stalemate_fen):
        row = next(
            LegalMoveGen(
                config={"fen_pool": [{"fen": fen}], "volume_override": 1},
                rng=Random(0),
            ).generate()
        )
        answer = row["messages"][2]["content"]

        assert answer == "Side to move: black.\nLegal moves: none"
        assert row["metadata"]["legal_move_count"] == 0
        passed, errors = validate_example(row)
        assert passed is True, errors


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


def test_package_ray_walk_generator_walks_blockers_captures_and_edges():
    import chess

    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import RayWalk

    fen = "7k/3p4/8/8/3RN3/8/8/7K w - - 0 1"
    row = next(
        RayWalk(
            config={"fen_pool": [{"fen": fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    answer = row["messages"][2]["content"]

    assert row["task"] == "2.10_ray_walk"
    assert row["metadata"]["source_square"] == "d4"
    assert row["metadata"]["piece"] == "rook"
    assert row["metadata"]["ray_move_count"] == 9
    assert row["metadata"]["blocked_ray_count"] == 2
    assert answer == (
        "Piece: d4 white rook.\n"
        "Ray N: d5 empty; d6 empty; d7 black pawn: capture (stop, included).\n"
        "Ray E: e4 white knight: own piece (stop, excluded).\n"
        "Ray S: d3 empty; d2 empty; d1 empty; edge.\n"
        "Ray W: c4 empty; b4 empty; a4 empty; edge.\n"
        "Moves from rays: d4a4 d4b4 d4c4 d4d1 d4d2 d4d3 d4d5 d4d6 d4d7"
    )
    board = chess.Board(fen)
    pseudo = sorted(
        move.uci()
        for move in board.pseudo_legal_moves
        if move.from_square == chess.D4
    )
    assert answer.splitlines()[-1] == "Moves from rays: " + " ".join(pseudo)
    assert row["metadata"]["expected_answer"] == answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_ray_walk_generator_emits_none_for_fully_blocked_slider():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import RayWalk

    fen = "7k/8/8/8/8/8/P7/RN5K w - - 0 1"
    row = next(
        RayWalk(
            config={"fen_pool": [{"fen": fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )

    assert row["messages"][2]["content"].splitlines() == [
        "Piece: a1 white rook.",
        "Ray N: a2 white pawn: own piece (stop, excluded).",
        "Ray E: b1 white knight: own piece (stop, excluded).",
        "Ray S: edge.",
        "Ray W: edge.",
        "Moves from rays: none",
    ]
    assert row["metadata"]["ray_move_count"] == 0
    assert row["metadata"]["blocked_ray_count"] == 2
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_ray_walk_generator_skips_sliderless_positions_and_keeps_piece_floors():
    from chess_llm.sft.generators.tier2_rules import RayWalk

    sliderless_fen = "7k/8/8/8/8/8/P7/7K w - - 0 1"
    slider_fen = "7k/8/8/8/2B5/1Q6/8/R6K w - - 0 1"

    row = next(
        RayWalk(
            config={
                "fen_pool": [{"fen": sliderless_fen} for _ in range(3)]
                + [{"fen": slider_fen}],
                "volume_override": 1,
            },
            rng=Random(0),
        ).generate()
    )
    assert row["fen"] == slider_fen

    gen = RayWalk(
        config={
            "fen_pool": [{"fen": slider_fen} for _ in range(40)],
            "volume_override": 40,
        },
        rng=Random(0),
    )
    counts = Counter(r["metadata"]["piece"] for r in gen.generate())
    total = sum(counts.values())

    assert total == 40
    assert set(counts) <= {"rook", "bishop", "queen"}
    assert counts["rook"] / total >= 0.30
    assert counts["bishop"] / total >= 0.25
    assert counts["queen"] / total >= 0.25


def test_package_piece_legal_filter_meets_rejection_floor_with_rejection_free_pool():
    from chess_llm.sft.generators.tier2_rules import (
        PieceLegalFilter,
        _synthetic_pin_check_fens,
    )

    quiet_rook_fen = "7k/8/8/8/8/8/8/R3K3 w - - 0 1"
    gen = PieceLegalFilter(
        config={
            "fen_pool": [{"fen": quiet_rook_fen} for _ in range(20)],
            "volume_override": 10,
        },
        rng=Random(0),
    )

    rows = list(gen.generate())
    categories = Counter(row["metadata"]["rejection_category"] for row in rows)

    assert len(rows) == 10
    assert categories["has_rejection"] / len(rows) >= 0.5
    for row in rows:
        if row["metadata"]["rejection_category"] == "has_rejection":
            assert row["metadata"]["rejected_count"] > 0
        else:
            assert row["metadata"]["rejected_count"] == 0

    labels = {label for label, _fen in _synthetic_pin_check_fens(10)}
    assert labels == {
        "pinned_piece_exposes_king",
        "does_not_resolve_check",
        "king_would_be_in_check",
    }


def test_package_legal_filter_trace_generator_composes_filter_lines():
    import chess

    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import (
        LegalFilterTrace,
        _format_legal_moves_by_piece_answer,
    )

    pinned_rook_fen = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    row = next(
        LegalFilterTrace(
            config={"fen_pool": [{"fen": pinned_rook_fen}], "volume_override": 1},
            rng=Random(0),
        ).generate()
    )
    answer = row["messages"][2]["content"]

    assert row["task"] == "2.11_legal_filter_trace"
    assert answer == (
        "Side to move: white.\n"
        "Pieces: e1 white king; e2 white rook.\n"
        "Filter by piece:\n"
        "e1 white king: pseudo-legal e1d1 e1d2 e1f1 e1f2 | rejected none | "
        "legal e1d1 e1d2 e1f1 e1f2\n"
        "e2 white rook: pseudo-legal e2a2 e2b2 e2c2 e2d2 e2e3 e2e4 e2e5 e2e6 "
        "e2e7 e2e8 e2f2 e2g2 e2h2 | rejected e2a2 pinned_piece_exposes_king; "
        "e2b2 pinned_piece_exposes_king; e2c2 pinned_piece_exposes_king; "
        "e2d2 pinned_piece_exposes_king; e2f2 pinned_piece_exposes_king; "
        "e2g2 pinned_piece_exposes_king; e2h2 pinned_piece_exposes_king | "
        "legal e2e3 e2e4 e2e5 e2e6 e2e7 e2e8\n"
        "All legal moves: e1d1 e1d2 e1f1 e1f2 e2e3 e2e4 e2e5 e2e6 e2e7 e2e8"
    )
    grouped_answer, _grouped, _final = _format_legal_moves_by_piece_answer(
        chess.Board(pinned_rook_fen)
    )
    # Header and final lines are byte-identical to 2.9's grouped format.
    assert answer.splitlines()[0] == grouped_answer.splitlines()[0]
    assert answer.splitlines()[1] == grouped_answer.splitlines()[1]
    assert answer.splitlines()[-1] == grouped_answer.splitlines()[-1]
    assert row["metadata"]["rejection_category"] == "has_rejection"
    assert row["metadata"]["rejected_total"] == 7
    assert row["metadata"]["legal_move_count"] == 10
    assert row["metadata"]["legal_moves_by_piece"]["e2"] == [
        "e2e3", "e2e4", "e2e5", "e2e6", "e2e7", "e2e8",
    ]
    assert row["metadata"]["expected_answer"] == answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_legal_filter_trace_generator_respects_caps_and_rejection_floor():
    import chess

    from chess_llm.core.legality import format_legal_filter_trace_answer
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier2_rules import LegalFilterTrace

    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # 16 side-to-move pieces exceed the 12-piece cap.
    assert format_legal_filter_trace_answer(chess.Board(starting_fen)) is None

    gen = LegalFilterTrace(
        config={
            "fen_pool": [{"fen": starting_fen} for _ in range(4)],
            "volume_override": 4,
        },
        rng=Random(0),
    )
    rows = list(gen.generate())
    categories = Counter(row["metadata"]["rejection_category"] for row in rows)

    assert len(rows) == 4
    assert all(row["fen"] != starting_fen for row in rows)
    assert all(len(row["messages"][2]["content"]) <= 1200 for row in rows)
    assert all(row["metadata"]["legal_move_count"] > 0 for row in rows)
    assert categories["has_rejection"] / len(rows) >= 0.4
    for row in rows:
        passed, errors = validate_example(row)
        assert passed is True, errors


def test_package_legal_filter_trace_synthetic_fallback_scales_to_training_volume():
    import chess

    from chess_llm.core.legality import format_legal_filter_trace_answer
    from chess_llm.sft.generators.tier2_rules import (
        LegalFilterTrace,
        _synthetic_pin_check_fens,
    )

    fallback_rows = _synthetic_pin_check_fens(500)

    assert len(fallback_rows) == 500
    assert len({fen for _label, fen in fallback_rows}) == 500
    assert {label for label, _fen in fallback_rows} == {
        "pinned_piece_exposes_king",
        "does_not_resolve_check",
        "king_would_be_in_check",
    }
    assert all(
        format_legal_filter_trace_answer(chess.Board(fen)) is not None
        for _label, fen in fallback_rows
    )
    assert all(
        any(chess.Board(fen).legal_moves)
        for _label, fen in fallback_rows
    )

    gen = LegalFilterTrace(config={"fen_pool": [], "volume_override": 200}, rng=Random(0))
    rows = list(gen.generate())

    assert len(rows) == 200
    assert len({row["metadata"]["example_identity"] for row in rows}) == 200


def test_package_multi_move_state_tracking_uses_two_to_three_plies():
    from chess_llm.sft.generators.tier1_perception import MultiMoveStateTracking

    gen = MultiMoveStateTracking(
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
    assert all(row["task"] == "1.19_multi_move_state_tracking" for row in rows)
    assert all(int(row["metadata"]["n_moves"]) in (2, 3) for row in rows)
    assert {int(row["metadata"]["n_moves"]) for row in rows} == {2, 3}


def test_package_multi_move_state_tracking_honors_config_override_and_bare_answer():
    from chess_llm.sft import validate_example
    from chess_llm.sft.generators.tier1_perception import MultiMoveStateTracking

    gen = MultiMoveStateTracking(
        config={
            "game_positions": [{"fen": STARTING_FEN}],
            "volume_override": 1,
            "multi_state_tracking_min_plies": 4,
            "multi_state_tracking_max_plies": 4,
        },
        rng=Random(1),
    )

    row = next(gen.generate())
    answer = row["messages"][2]["content"]

    assert int(row["metadata"]["n_moves"]) == 4
    assert len(row["metadata"]["moves"].split()) == 4
    assert answer == f"Result FEN: {row['metadata']['result_fen']}"
    assert "Move 1:" not in answer
    assert "Lookup:" not in answer
    assert "Squares:" not in answer
    assert "Ranks:" not in answer
    passed, errors = validate_example(row)
    assert passed is True, errors


def test_package_new_curriculum_tasks_are_registered():
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators import (
        LegalFilterTrace,
        MultiMoveStateTracking,
        RayWalk,
    )
    from chess_llm.sft.identity import TASK_IDENTITY_FIELDS
    from chess_llm.sft.settings import DEFAULT_VOLUMES
    from chess_llm.sft.templates import ANSWER_CONTRACTS, TEMPLATES

    assert MultiMoveStateTracking in pipeline.TIER_GENERATORS[1]
    assert RayWalk in pipeline.TIER_GENERATORS[2]
    assert LegalFilterTrace in pipeline.TIER_GENERATORS[2]
    for task_id, alias in (
        ("1.19_multi_move_state_tracking", "multi_state_tracking"),
        ("2.10_ray_walk", "ray_walk"),
        ("2.11_legal_filter_trace", "legal_filter_trace"),
    ):
        assert DEFAULT_VOLUMES[task_id] > 0
        assert TEMPLATES[task_id]
        assert ANSWER_CONTRACTS[task_id] == ANSWER_CONTRACTS[alias]
    assert TEMPLATES["1.19_multi_move_state_tracking"] == TEMPLATES["1.5_state_tracking"]
    assert TASK_IDENTITY_FIELDS["1.19_multi_move_state_tracking"] == ("moves",)
    assert TASK_IDENTITY_FIELDS["2.10_ray_walk"] == ("source_square",)
    assert TASK_IDENTITY_FIELDS["2.11_legal_filter_trace"] == ()


def test_pipeline_registry_uses_package_generators():
    from chess_llm.sft import pipeline

    assert pipeline.TIER_GENERATORS[1][0].__module__.startswith(
        "chess_llm.sft.generators."
    )
    assert pipeline.TIER_GENERATORS[7][0].__module__.startswith(
        "chess_llm.sft.generators."
    )
