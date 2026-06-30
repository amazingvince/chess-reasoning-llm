"""Tests for generators: tier1, tier2, tier3, tier5, tier7."""

import re
from random import Random

import chess
import pytest

from conftest import (
    CHECKMATE_FEN,
    EN_PASSANT_FEN,
    KRK_FEN,
    PROMOTION_FEN,
    STARTING_FEN,
)


# ── helpers ──────────────────────────────────────────────────────────

# Position after 1.e4 d5 — white e4 pawn attacked by black d5 pawn
AFTER_E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
NO_CASTLING_FEN = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"


@pytest.mark.parametrize(
    "raw",
    [
        {"fen": STARTING_FEN},
        {"fen": "not a fen", "board": "custom"},
        {"fen": STARTING_FEN, "board": "custom"},
        {
            "fen": "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1",
            "is_chess960": True,
        },
        {"fen": NO_CASTLING_FEN},
        {"fen": AFTER_E4},
    ],
)
def test_task_generator_template_context_matches_package_context(raw):
    from chess_llm.sft import build_template_context
    from generators.tier2_rules import LegalMoveGen

    gen = LegalMoveGen(config={}, rng=Random(42))

    assert gen.build_template_context(raw) == build_template_context(raw)


def test_fen_to_board_uses_package_ascii_renderer(monkeypatch):
    from chess_llm.formats import render_ascii_board
    from generators.tier1_perception import FENToBoard

    tpl = "FEN: {fen}\nShow me the board."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = FENToBoard(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    expected = render_ascii_board(chess.Board(STARTING_FEN))

    assert ex["messages"][1]["content"] == f"FEN: {STARTING_FEN}\nShow me the board."
    assert ex["messages"][2]["content"] == expected


# ── Tier 1: PieceCounting ────────────────────────────────────────────


def test_fen_to_board_accepts_chess960_fen(monkeypatch):
    from chess_llm.formats import render_ascii_board
    from generators.tier1_perception import FENToBoard

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    tpl = "FEN: {fen}\nShow me the board."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = FENToBoard(
        config={
            "fen_pool": [{"fen": chess960_fen, "is_chess960": True}],
            "volume_override": 1,
        },
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    expected = render_ascii_board(chess.Board(chess960_fen, chess960=True))

    assert ex["is_chess960"] is True
    assert ex["messages"][2]["content"] == expected


def test_board_to_fen_prompt_includes_state_needed_for_full_fen(monkeypatch):
    from generators.tier1_perception import BoardToFEN

    tpl = "Board:\n{board}\nProduce the FEN string."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = BoardToFEN(
        config={"fen_pool": [{"fen": AFTER_E4}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    user_prompt = ex["messages"][1]["content"]

    assert "Side to move: black" in user_prompt
    assert "Castling rights: KQkq" in user_prompt
    assert "En passant: e3" in user_prompt
    assert "Halfmove clock: 0" in user_prompt
    assert "Fullmove number: 1" in user_prompt
    assert ex["messages"][2]["content"] == AFTER_E4
    assert "{" not in user_prompt


def test_generator_blocklist_accepts_chess960_rows():
    from chess_llm.core.board import canonical_fen_key
    from generators.tier1_perception import FENToBoard

    chess960_row = {
        "fen": "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1",
        "is_chess960": True,
        "chess960_id": 3,
    }
    blocklist = frozenset(
        [canonical_fen_key(chess960_row["fen"], chess960=True)]
    )

    gen = FENToBoard(config={}, blocklist=blocklist, rng=Random(42))

    assert gen.is_blocked(chess960_row) is True


def test_piece_identification_square_board_template(monkeypatch):
    """PieceIdentification can render board-based square prompts."""
    from generators import tier1_perception

    monkeypatch.setitem(
        tier1_perception.TEMPLATES,
        "1.3_piece_identification",
        ["Board:\n{board}\nWhat piece is on {square}?"],
    )

    gen = tier1_perception.PieceIdentification(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(1),
    )
    ex = list(gen.generate())[0]
    user_prompt = ex["messages"][1]["content"]

    assert "Board:" in user_prompt
    assert "8 r n b q k b n r" in user_prompt
    assert "FEN:" not in user_prompt
    assert "{" not in user_prompt


def test_piece_counting_color_template(monkeypatch):
    """Force 'How many pieces does {color}' -> answer mentions only that color."""
    from generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nHow many pieces does {color} have?"
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    assert "piece(s):" in answer


def test_piece_counting_piece_type_template(monkeypatch):
    """Force 'how many {piece}s' -> answer counts only that piece type."""
    from generators.tier1_perception import PieceCounting

    tpl = "In this position, how many {piece}s are on the board?\nFEN: {fen}"
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    # Specific piece type -> "White has X ...(s), black has Y ...(s)."
    # or generic "piece" -> full material count
    assert "White" in answer or "white" in answer


def test_piece_counting_minor_pieces_template(monkeypatch):
    """Force 'minor pieces' -> answer mentions knights + bishops."""
    from generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nHow many minor pieces does {color} have?"
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    assert "minor piece(s)" in answer
    assert "knight(s)" in answer
    assert "bishop(s)" in answer


def test_piece_counting_board_template(monkeypatch):
    """Board-based counting prompts render the derived ASCII board."""
    from generators.tier1_perception import PieceCounting

    tpl = "Board:\n{board}\nHow many pieces does {color} have?"
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    user_prompt = ex["messages"][1]["content"]

    assert "Board:" in user_prompt
    assert "8 r n b q k b n r" in user_prompt
    assert "{" not in user_prompt


def test_piece_counting_example_structure(monkeypatch):
    """Generated example has correct task, tier, messages structure."""
    from generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nCount all pieces and pawns for each side."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]

    assert ex["task"] == "1.4_piece_counting"
    assert ex["tier"] == 1
    assert len(ex["messages"]) == 3
    assert ex["messages"][0]["role"] == "system"
    assert ex["messages"][1]["role"] == "user"
    assert ex["messages"][2]["role"] == "assistant"
    # No unfilled placeholders
    assert "{" not in ex["messages"][1]["content"]


# ── Tier 1: StateTracking ────────────────────────────────────────────


def test_state_tracking_valid_metadata(monkeypatch):
    from generators.tier1_perception import StateTracking

    tpl = "Starting FEN: {fen}\nAfter the moves {moves}, what is the resulting position?"
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = StateTracking(
        config={"game_positions": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    ex = examples[0]
    metadata = ex["metadata"]
    assert "result_fen" in metadata
    assert "moves" in metadata
    answer = ex["messages"][2]["content"]
    assert "Lookup:" in answer
    assert "Squares:" in answer
    assert "Ranks:" in answer
    assert "Result placement:" not in answer
    assert answer.splitlines()[-1] == f"Result FEN: {metadata['result_fen']}"


def test_fen_assembly_legacy_import_valid_metadata(monkeypatch):
    from generators.tier1_perception import FENAssembly
    from validation.validator import validate_example

    tpl = "Starting FEN: {fen}\nMove: {move}\nAssemble the resulting full FEN."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = FENAssembly(
        config={"game_positions": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    ex = examples[0]
    answer = ex["messages"][2]["content"]
    metadata = ex["metadata"]
    assert ex["task"] == "1.9_fen_assembly"
    assert metadata["expected_answer"] == answer
    assert "\nLookup: " in answer
    assert "\nSquares: " in answer
    assert "\nRanks: " in answer
    assert answer.splitlines()[-1] == f"Result FEN: {metadata['result_fen']}"
    passed, errors = validate_example(ex)
    assert passed is True, errors


def test_state_tracking_uses_chess960_legal_moves(monkeypatch):
    """Chess960 state tracking must be able to sample Chess960 castling."""
    from generators.tier1_perception import StateTracking
    from validation.validator import validate_example

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    tpl = "Starting FEN: {fen}\nAfter the moves {moves}, what is the resulting position?"
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    class PreferChess960Castle:
        def randint(self, _low, _high):
            return 1

        def choice(self, values):
            for value in values:
                if value.uci() == "d1c1":
                    return value
            return values[0]

    gen = StateTracking(
        config={
            "game_positions": [{"fen": chess960_fen, "is_chess960": True}],
            "volume_override": 1,
        },
        rng=PreferChess960Castle(),
    )
    ex = list(gen.generate())[0]

    assert ex["metadata"]["moves"] == "d1c1"
    passed, errors = validate_example(ex)
    assert passed is True, errors


# ── Tier 2: LegalMoveGen ─────────────────────────────────────────────


def test_legal_move_gen_correct_moves(monkeypatch):
    from generators.tier2_rules import LegalMoveGen

    tpl = "FEN: {fen}\nList all legal moves."
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = LegalMoveGen(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    answer = examples[0]["messages"][2]["content"]

    board = chess.Board(STARTING_FEN)
    expected = sorted(m.uci() for m in board.legal_moves)
    assert answer.strip().split() == expected


def test_legal_move_gen_board_state_template(monkeypatch):
    """Board/state placeholders render for legal-move prompts."""
    from generators.tier2_rules import LegalMoveGen

    tpl = (
        "Board:\n{board}\n"
        "Side to move: {side_to_move}\n"
        "Castling rights: {castling_rights}\n"
        "En passant: {en_passant_square}\n"
        "List all legal moves."
    )
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = LegalMoveGen(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    user_prompt = ex["messages"][1]["content"]

    assert "Board:" in user_prompt
    assert "8 r n b q k b n r" in user_prompt
    assert "Side to move: white" in user_prompt
    assert "Castling rights: KQkq" in user_prompt
    assert "En passant: none" in user_prompt
    assert "{" not in user_prompt


# ── Tier 2: SpecialRules ─────────────────────────────────────────────


def test_legal_move_gen_accepts_metadata_only_chess960_marker(monkeypatch):
    from generators.tier2_rules import LegalMoveGen

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    tpl = "FEN: {fen}\nList all legal moves."
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = LegalMoveGen(
        config={
            "fen_pool": [
                {"fen": chess960_fen, "metadata": {"chess960_id": 3}},
            ],
            "volume_override": 1,
        },
        rng=Random(42),
    )

    examples = list(gen.generate())

    assert len(examples) == 1
    assert examples[0]["is_chess960"] is True
    assert "d1c1" in examples[0]["messages"][2]["content"]


def test_special_rules_promotion(monkeypatch):
    """Promotion FEN + 'promotion options' -> lists Q/R/B/N."""
    from generators.tier2_rules import SpecialRules

    tpl = "FEN: {fen}\nWhat promotion options are available for the pawn on {square}?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": PROMOTION_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"].lower()
    assert "queen" in answer
    assert "rook" in answer
    assert "bishop" in answer
    assert "knight" in answer


def test_special_rules_skips_promotion_template_without_promotion(monkeypatch):
    """Promotion-specific prompts must not render an empty pawn square."""
    from generators.tier2_rules import SpecialRules

    tpl = "FEN: {fen}\nWhat promotion options are available for the pawn on {square}?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )

    assert list(gen.generate()) == []


def test_special_rules_castling_requires_legal_castle(monkeypatch):
    """Starting FEN has rights but blocked pieces, so castling is unavailable."""
    from generators.tier2_rules import SpecialRules

    tpl = "FEN: {fen}\nCan the side to move castle? If so, which side(s)?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert examples[0]["messages"][2]["content"] == (
        "No castling is available for the side to move."
    )


def test_special_rules_castling_available_when_legal(monkeypatch):
    """Open castling lanes with rights should report legal castling sides."""
    from generators.tier2_rules import SpecialRules

    tpl = "FEN: {fen}\nCan the side to move castle? If so, which side(s)?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    gen = SpecialRules(
        config={"fen_pool": [{"fen": fen}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())

    assert len(examples) == 1
    assert examples[0]["messages"][2]["content"] == (
        "Castling available: kingside, queenside."
    )


def test_special_rules_en_passant(monkeypatch):
    """EP FEN -> answer mentions en passant."""
    from generators.tier2_rules import SpecialRules

    tpl = "FEN: {fen}\nIs en passant possible in this position?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": EN_PASSANT_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "en passant" in examples[0]["messages"][2]["content"].lower()


def test_special_rules_en_passant_prompt_without_ep_answers_no(monkeypatch):
    """An en-passant question should not be answered with unrelated castling info."""
    from generators.tier2_rules import SpecialRules

    tpl = "FEN: {fen}\nIs en passant possible in this position?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    answer = examples[0]["messages"][2]["content"].lower()
    assert "no en passant" in answer
    assert "castling available" not in answer


def test_special_rules_no_special(monkeypatch):
    """KRK position -> 'No special moves available'."""
    from generators.tier2_rules import SpecialRules

    tpl = "Given FEN: {fen}\nList any special moves available (castling, en passant, promotion)."
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": KRK_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "no special" in examples[0]["messages"][2]["content"].lower()


def test_special_rules_example_structure(monkeypatch):
    from generators.tier2_rules import SpecialRules

    tpl = "Given FEN: {fen}\nList any special moves available (castling, en passant, promotion)."
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = SpecialRules(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    assert ex["task"] == "2.5_special_rules"
    assert ex["tier"] == 2


# ── Tier 2: CheckDetection ───────────────────────────────────────────


def test_check_detection_checkmate(monkeypatch):
    from generators.tier2_rules import CheckDetection

    tpl = "FEN: {fen}\nIs the king in check?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = CheckDetection(
        config={"fen_pool": [{"fen": CHECKMATE_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "checkmate" in examples[0]["messages"][2]["content"].lower()


def test_check_detection_normal(monkeypatch):
    from generators.tier2_rules import CheckDetection

    tpl = "FEN: {fen}\nIs the king in check?"
    monkeypatch.setattr(
        "generators.tier2_rules.select_template", lambda tid, rng: tpl
    )

    gen = CheckDetection(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "normal" in examples[0]["messages"][2]["content"].lower()


# ── Tier 3: Threats ──────────────────────────────────────────────────


def test_threats_under_attack(monkeypatch):
    """'under attack' -> lists same-color pieces attacked by opponent."""
    from generators.tier3_tactics import Threats

    tpl = "FEN: {fen}\nWhich {color} pieces are under attack?"
    monkeypatch.setattr(
        "generators.tier3_tactics.select_template", lambda tid, rng: tpl
    )

    gen = Threats(
        config={"fen_pool": [{"fen": AFTER_E4_D5}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"].lower()
    # White e4 pawn is under attack by black d5 pawn
    assert "under attack" in answer
    assert "pawn" in answer
    assert "e4" in answer


def test_threats_threatening(monkeypatch):
    """Generic/threatening template -> lists opponent pieces attacked by color."""
    from generators.tier3_tactics import Threats

    tpl = "FEN: {fen}\nWhat pieces are {color} threatening?"
    monkeypatch.setattr(
        "generators.tier3_tactics.select_template", lambda tid, rng: tpl
    )

    gen = Threats(
        config={"fen_pool": [{"fen": AFTER_E4_D5}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"].lower()
    # White e4 pawn threatens black d5 pawn
    assert "threatens" in answer
    assert "pawn" in answer
    assert "d5" in answer


def test_threats_example_structure(monkeypatch):
    from generators.tier3_tactics import Threats

    tpl = "In this position, identify all threats.\nFEN: {fen}"
    monkeypatch.setattr(
        "generators.tier3_tactics.select_template", lambda tid, rng: tpl
    )

    gen = Threats(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 1},
        rng=Random(42),
    )
    ex = list(gen.generate())[0]
    assert ex["task"] == "3.2_threats"
    assert ex["tier"] == 3
    assert len(ex["messages"]) == 3


# ── Tier 3: AvailableCaptures ────────────────────────────────────────


def test_available_captures(monkeypatch):
    """After 1.e4 d5, exd5 is available."""
    from generators.tier3_tactics import AvailableCaptures

    tpl = "FEN: {fen}\nList all capture moves available."
    monkeypatch.setattr(
        "generators.tier3_tactics.select_template", lambda tid, rng: tpl
    )

    gen = AvailableCaptures(
        config={"fen_pool": [{"fen": AFTER_E4_D5}], "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1
    assert "e4d5" in examples[0]["messages"][2]["content"]


# ── Tier 7: MoveConsequence ──────────────────────────────────────────


def test_move_consequence_format(monkeypatch):
    """Output matches strict <think>/<move> regex."""
    from generators.tier7_planning import MoveConsequence

    tpl = "FEN: {fen}\nIf {move} is played, what happens next?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    evals = [{
        "fen": STARTING_FEN, "best_move": "e2e4",
        "pv_line": "e2e4 e7e5 g1f3", "cp": 30, "mate": None, "depth": 25,
    }]

    gen = MoveConsequence(
        config={"consequence_evals": evals, "volume_override": 1},
        rng=Random(42),
    )
    example = list(gen.generate())[0]
    answer = example["messages"][2]["content"]
    assert re.match(
        r"^<think>.*</think>\s*<move>.*</move>\s*$", answer, re.DOTALL
    )
    assert example["metadata"]["target_move"] == "e2e4"


def test_move_consequence_no_trailing_content(monkeypatch):
    from generators.tier7_planning import MoveConsequence

    tpl = "FEN: {fen}\nIf {move} is played, what happens next?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    evals = [{
        "fen": STARTING_FEN, "best_move": "e2e4",
        "pv_line": "e2e4 e7e5 g1f3", "cp": 30, "mate": None, "depth": 25,
    }]

    gen = MoveConsequence(
        config={"consequence_evals": evals, "volume_override": 1},
        rng=Random(42),
    )
    answer = list(gen.generate())[0]["messages"][2]["content"]
    after_move = answer.split("</move>")[-1]
    assert after_move.strip() == ""


def test_move_consequence_pv_inside_think(monkeypatch):
    from generators.tier7_planning import MoveConsequence

    tpl = "FEN: {fen}\nIf {move} is played, what happens next?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    evals = [{
        "fen": STARTING_FEN, "best_move": "e2e4",
        "pv_line": "e2e4 e7e5 g1f3 b8c6", "cp": 30, "mate": None, "depth": 25,
    }]

    gen = MoveConsequence(
        config={"consequence_evals": evals, "volume_override": 1},
        rng=Random(42),
    )
    answer = list(gen.generate())[0]["messages"][2]["content"]
    think_content = answer.split("<think>")[1].split("</think>")[0]
    assert "Expected line:" in think_content


# ── Tier 7: BestMoveSelection ────────────────────────────────────────


def test_best_move_selection_budget(monkeypatch):
    """8 evals + 4 MATE rows, target=10 -> MATE budget=2, eval budget=8."""
    from generators.tier7_planning import BestMoveSelection

    tpl = "FEN: {fen}\nWhat is the best move?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    evals = [
        {
            "fen": STARTING_FEN, "best_move": "e2e4",
            "pv_line": "e2e4 e7e5", "cp": 30, "mate": None, "depth": 35,
        }
        for _ in range(8)
    ]
    mate_rows = [
        {
            "fen": STARTING_FEN, "better_move": "e2e4",
            "move_a": "e2e4", "move_b": "d2d4",
            "strategy": "control center", "tactic": "",
        }
        for _ in range(4)
    ]

    gen = BestMoveSelection(
        config={
            "best_move_evals": evals,
            "mate_rows": mate_rows,
            "volume_override": 10,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 10

    sources = [ex["metadata"].get("source") for ex in examples]
    assert "lichess_evals" in sources
    assert "mate_dataset" in sources


def test_best_move_selection_tier7_format(monkeypatch):
    from generators.tier7_planning import BestMoveSelection
    from validation.validator import validate_think_move_format

    tpl = "FEN: {fen}\nWhat is the best move?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    evals = [{
        "fen": STARTING_FEN, "best_move": "e2e4",
        "pv_line": "e2e4 e7e5", "cp": 30, "mate": None, "depth": 35,
    }]

    gen = BestMoveSelection(
        config={"best_move_evals": evals, "volume_override": 1},
        rng=Random(42),
    )
    example = list(gen.generate())[0]
    answer = example["messages"][2]["content"]
    assert validate_think_move_format(answer)
    assert example["metadata"]["target_move"] == "e2e4"


# ── Cross-cutting: blocklist & volume ─────────────────────────────────


def test_best_move_selection_preserves_chess960_eval_rows(monkeypatch):
    from generators.tier7_planning import BestMoveSelection

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    tpl = "FEN: {fen}\nWhat is the best move?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    evals = [{
        "fen": chess960_fen,
        "best_move": "d1c1",
        "pv_line": "d1c1",
        "cp": 30,
        "mate": None,
        "depth": 35,
        "is_chess960": True,
        "chess960_id": 3,
    }]

    gen = BestMoveSelection(
        config={"best_move_evals": evals, "volume_override": 1},
        rng=Random(42),
    )
    example = list(gen.generate())[0]

    assert example["is_chess960"] is True
    assert example["metadata"]["chess960_id"] == 3
    assert example["metadata"]["target_move"] == "d1c1"


def test_blocklist_excludes_fen(monkeypatch):
    from generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nCount all pieces and pawns for each side."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={"fen_pool": [{"fen": STARTING_FEN}], "volume_override": 10},
        blocklist=frozenset([STARTING_FEN]),
        rng=Random(42),
    )
    assert list(gen.generate()) == []


def test_volume_override(monkeypatch):
    from generators.tier1_perception import PieceCounting

    tpl = "FEN: {fen}\nCount all pieces and pawns for each side."
    monkeypatch.setattr(
        "generators.tier1_perception.select_template", lambda tid, rng: tpl
    )

    gen = PieceCounting(
        config={
            "fen_pool": [{"fen": STARTING_FEN}] * 100,
            "volume_override": 3,
        },
        rng=Random(42),
    )
    assert len(list(gen.generate())) == 3


# ── Tier 5: OpeningContinuation ──────────────────────────────────────


def test_opening_continuation_with_book(monkeypatch):
    """When Polyglot book data is available, answer shows weighted moves."""
    from generators.tier5_openings import OpeningContinuation

    tpl = "FEN: {fen}\nWhat are the main continuation moves in this opening?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{"fen": STARTING_FEN, "name": "Starting Position", "uci_moves": []}]
    book_moves = {STARTING_FEN: [("e2e4", 100), ("d2d4", 80), ("g1f3", 30)]}

    gen = OpeningContinuation(
        config={
            "openings": openings,
            "book_moves": book_moves,
            "volume_override": 1,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    assert "Top continuations:" in answer
    assert "e2e4" in answer
    assert "%" in answer


def test_opening_continuation_fallback_skips(monkeypatch):
    """Without Polyglot data, the position is skipped — no arbitrary noise."""
    from generators.tier5_openings import OpeningContinuation

    tpl = "FEN: {fen}\nWhat are the main continuation moves in this opening?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{"fen": STARTING_FEN, "name": "Amar Opening", "uci_moves": ["g1h3"]}]
    book_moves = {}  # No Polyglot data

    gen = OpeningContinuation(
        config={
            "openings": openings,
            "book_moves": book_moves,
            "volume_override": 1,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert examples == [], "Should skip positions without book data"


def test_opening_identification(monkeypatch):
    from generators.tier5_openings import OpeningIdentification

    tpl = "FEN: {fen}\nWhat opening is this?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{
        "fen": STARTING_FEN,
        "name": "Italian Game",
        "eco": "C50",
        "uci_moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    }]

    gen = OpeningIdentification(
        config={"openings": openings, "volume_override": 1},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 1

    answer = examples[0]["messages"][2]["content"]
    assert "Italian Game" in answer
    assert "C50" in answer


def test_opening_identification_multi_variant(monkeypatch):
    """volume_override=5, 1 opening with full data -> 5 distinct variants."""
    from generators.tier5_openings import OpeningIdentification

    tpl = "FEN: {fen}\nWhat opening is this?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{
        "fen": STARTING_FEN,
        "name": "Italian Game",
        "eco": "C50",
        "eco_volume": "C",
        "uci_moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    }]

    gen = OpeningIdentification(
        config={"openings": openings, "volume_override": 5},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 5

    variants = [ex["metadata"]["variant"] for ex in examples]
    assert len(set(variants)) == 5, f"Expected 5 distinct variants, got {set(variants)}"


def test_opening_continuation_multi_variant(monkeypatch):
    """volume_override=4, 1 opening with 3+ book moves -> 4 variants."""
    from generators.tier5_openings import OpeningContinuation

    tpl = "FEN: {fen}\nWhat are the main continuation moves?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{"fen": STARTING_FEN, "name": "Starting Position", "uci_moves": []}]
    book_moves = {STARTING_FEN: [("e2e4", 100), ("d2d4", 80), ("g1f3", 30)]}

    gen = OpeningContinuation(
        config={
            "openings": openings,
            "book_moves": book_moves,
            "volume_override": 4,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 4

    variants = [ex["metadata"]["variant"] for ex in examples]
    assert len(set(variants)) == 4


def test_opening_continuation_alternatives_skipped(monkeypatch):
    """1 opening with exactly 1 book move -> 'alternatives' variant skipped."""
    from generators.tier5_openings import OpeningContinuation

    tpl = "FEN: {fen}\nWhat are the main continuation moves?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{"fen": STARTING_FEN, "name": "Odd Opening", "uci_moves": []}]
    book_moves = {STARTING_FEN: [("e2e4", 100)]}

    gen = OpeningContinuation(
        config={
            "openings": openings,
            "book_moves": book_moves,
            "volume_override": 4,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    # top_n, best_single, with_context -> 3; alternatives skipped (only 1 move)
    assert len(examples) == 3

    variants = {ex["metadata"]["variant"] for ex in examples}
    assert "alternatives" not in variants


def test_opening_principles_multi_variant(monkeypatch):
    """volume_override=5, 1 opening -> 5 distinct variants."""
    from generators.tier5_openings import OpeningPrinciples

    tpl = "FEN: {fen}\nWhat are the key ideas?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    openings = [{
        "fen": STARTING_FEN,
        "name": "Starting Position",
        "eco": "A00",
        "eco_volume": "A",
    }]

    gen = OpeningPrinciples(
        config={"openings": openings, "volume_override": 5},
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 5

    variants = [ex["metadata"]["variant"] for ex in examples]
    assert len(set(variants)) == 5


def test_opening_principles_pawn_structure(monkeypatch):
    """Opening with known doubled pawns -> pawn_structure answer mentions 'doubled'."""
    from generators.tier5_openings import OpeningPrinciples

    tpl = "FEN: {fen}\nWhat are the key ideas?"
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    # Position with doubled white e-pawns
    doubled_fen = "rnbqkbnr/pppp1ppp/8/8/4P3/4P3/PPPP2PP/RNBQKBNR b KQkq - 0 2"
    openings = [{
        "fen": doubled_fen,
        "name": "Weird Line",
        "eco": "C00",
        "eco_volume": "C",
    }]

    gen = OpeningPrinciples(
        config={"openings": openings, "volume_override": 5},
        rng=Random(42),
    )
    examples = list(gen.generate())
    pawn_examples = [
        ex for ex in examples if ex["metadata"]["variant"] == "pawn_structure"
    ]
    assert len(pawn_examples) == 1
    assert "doubled" in pawn_examples[0]["messages"][2]["content"].lower()


def test_opening_principles_preserves_chess960_metadata(monkeypatch):
    from generators.tier5_openings import OpeningPrinciples

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    tpl = "Board:\n{board}\nFEN: {fen}\nDescribe the character of this opening position."
    monkeypatch.setattr(
        "generators.tier5_openings.select_template", lambda tid, rng: tpl
    )

    gen = OpeningPrinciples(
        config={
            "openings": [
                {
                    "fen": chess960_fen,
                    "name": "Chess960 Start",
                    "eco": "A00",
                    "is_chess960": True,
                    "chess960_id": 3,
                }
            ],
            "volume_override": 1,
        },
        rng=Random(42),
    )
    example = list(gen.generate())[0]

    assert example["is_chess960"] is True
    assert example["metadata"]["chess960_id"] == 3
    assert "8 b q r k r n n b" in example["messages"][1]["content"]


@pytest.mark.parametrize(
    "generator_name",
    ["EndgameClassification", "EndgamePrinciples"],
)
def test_endgame_generators_preserve_chess960_metadata(monkeypatch, generator_name):
    import generators.tier6_endgames as tier6

    chess960_fen = "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1"
    tpl = "FEN: {fen}\nWhat endgame information applies?"
    monkeypatch.setattr(
        "generators.tier6_endgames.select_template", lambda tid, rng: tpl
    )

    generator_cls = getattr(tier6, generator_name)
    gen = generator_cls(
        config={
            "endgame_positions": [
                {
                    "fen": chess960_fen,
                    "metadata": {"chess960_id": 3},
                    "source": "test",
                }
            ],
            "volume_override": 1,
        },
        rng=Random(42),
    )
    example = list(gen.generate())[0]

    assert example["is_chess960"] is True
    assert example["metadata"]["chess960_id"] == 3


# ── Tier 7: MATE dataset path ────────────────────────────────────────


def test_best_move_selection_mate_only(monkeypatch):
    """When only MATE rows are provided (no evals), they fill the budget."""
    from generators.tier7_planning import BestMoveSelection
    from validation.validator import validate_think_move_format

    tpl = "FEN: {fen}\nWhat is the best move?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    mate_rows = [
        {
            "fen": STARTING_FEN, "better_move": "e2e4",
            "move_a": "e2e4", "move_b": "d2d4",
            "strategy": "control center", "tactic": "central pawn push",
        }
        for _ in range(5)
    ]

    gen = BestMoveSelection(
        config={
            "best_move_evals": [],
            "mate_rows": mate_rows,
            "volume_override": 3,
        },
        rng=Random(42),
    )
    examples = list(gen.generate())
    assert len(examples) == 3

    for ex in examples:
        assert ex["metadata"]["source"] == "mate_dataset"
        assert validate_think_move_format(ex["messages"][2]["content"])


def test_best_move_selection_uses_parser_preserved_mate_annotations(monkeypatch):
    from chess_llm.sft.sources.mate import process_mate_row
    from generators.tier7_planning import BestMoveSelection

    tpl = "FEN: {fen}\nWhat is the best move?"
    monkeypatch.setattr(
        "generators.tier7_planning.select_template", lambda tid, rng: tpl
    )

    parsed = process_mate_row(
        {
            "input": f'The FEN of the given chess board is "{STARTING_FEN}". '
            "Which move is better? MoveA:e2e4 MoveB:d2d4 ",
            "output": "MoveA:e2e4",
            "strategy": "control the center",
            "tactic": "central pawn push",
        }
    )
    assert parsed is not None

    gen = BestMoveSelection(
        config={
            "best_move_evals": [],
            "mate_rows": [parsed],
            "volume_override": 1,
        },
        rng=Random(42),
    )
    example = list(gen.generate())[0]
    answer = example["messages"][2]["content"]

    assert example["metadata"]["strategy"] == "control the center"
    assert example["metadata"]["tactic"] == "central pawn push"
    assert "control the center" in answer
    assert "central pawn push" in answer
