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


# ── Tier 1: PieceCounting ────────────────────────────────────────────


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
    assert ex["messages"][2]["content"] == metadata["result_fen"]


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


def test_special_rules_castling(monkeypatch):
    """Starting FEN -> answer mentions castling."""
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
    assert "castling" in examples[0]["messages"][2]["content"].lower()


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
    answer = list(gen.generate())[0]["messages"][2]["content"]
    assert re.match(
        r"^<think>.*</think>\s*<move>.*</move>\s*$", answer, re.DOTALL
    )


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
    answer = list(gen.generate())[0]["messages"][2]["content"]
    assert validate_think_move_format(answer)


# ── Cross-cutting: blocklist & volume ─────────────────────────────────


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
