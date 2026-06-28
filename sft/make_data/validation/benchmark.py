"""Frozen benchmark schema, gold answer derivation, and scoring.

Provides:
- ``BenchmarkExample`` — canonical schema for frozen benchmark examples.
- ``CANONICAL_PROMPTS`` — one fixed prompt per task type (deterministic).
- ``derive_gold_answer`` — derives ground-truth answer from raw source data.
- Metric functions — exact_match, jaccard, eval_bucket, format/legal, ACPL.
- ``freeze_split`` / ``score_prediction`` / ``score_split`` — freeze and score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from random import Random

import chess

from generators.base import _board_to_ascii
from validation.eval_harness import (
    answer_legal_moves,
    answer_material_balance,
    answer_position_eval,
    answer_pawn_structure,
    answer_endgame_classification,
    answer_endgame_wdl,
)
from validation.validator import (
    validate_think_move_format,
    _extract_uci_from_move_tag,
    validate_move_legal,
)

logger = logging.getLogger(__name__)


# Shared piece name/value lookup (matches training generators)
_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkExample:
    """A single frozen benchmark example."""

    example_id: str       # "{split}_{index:05d}"
    split: str            # one of 9 split names
    task_type: str        # sub-task (e.g. "legal_moves", "eval_bucket")
    fen: str
    prompt: str           # canonical rendered prompt (deterministic)
    gold_answer: str      # ground-truth answer
    metric_type: str      # "exact_match", "jaccard", "eval_bucket", etc.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkExample:
        return cls(**d)


# ---------------------------------------------------------------------------
# Split -> task type mapping
# ---------------------------------------------------------------------------

SPLIT_TASK_TYPES: dict[str, list[str]] = {
    "perception":  ["board_print", "board_to_fen", "piece_id", "material_count", "state_tracking"],
    "rules":       ["legal_moves", "check_detection", "captures", "special_rules", "legality_check"],
    "tactics":     ["capture_id", "hanging_pieces", "threats", "tactical_patterns"],
    "evaluation":  ["material_balance", "eval_bucket", "pawn_structure"],
    "openings":    ["opening_name", "opening_continuation"],
    "endgames":    ["endgame_classification", "endgame_wdl", "endgame_best_move"],
    "planning":    ["best_move", "puzzle_solve"],  # source-detected, not round-robin
    "chess960":    ["legal_moves_960", "check_detection_960", "castling_rules_960"],
    "mate":        ["binary_choice"],
}

# ---------------------------------------------------------------------------
# Canonical prompts (one fixed prompt per task, no randomization)
# ---------------------------------------------------------------------------

CANONICAL_PROMPTS: dict[str, str] = {
    "board_print":            "Position (FEN): {fen}\nShow me the board.",
    "board_to_fen":           "Here is the current board:\n{board}\nWrite the FEN for this position.",
    "piece_id":               "FEN: {fen}\nWhat piece is on {square}?",
    "material_count":         "FEN: {fen}\nWhat is the material count for both sides?",
    "state_tracking":         "Starting FEN: {fen}\nAfter the moves {moves}, what is the resulting position?",
    "legal_moves":            "FEN: {fen}\nList all legal moves.",
    "check_detection":        "FEN: {fen}\nDetect the game state: check, checkmate, stalemate, or none.",
    "captures":               "FEN: {fen}\nList all capture moves available.",
    "special_rules":          "Given FEN: {fen}\nList any special moves available (castling, en passant, promotion).",
    "legality_check":         "FEN: {fen}\nIs the move {move} legal?",
    "capture_id":             "FEN: {fen}\nList all capture moves available.",
    "hanging_pieces":         "FEN: {fen}\nIdentify all hanging (undefended attacked) pieces.",
    "threats":                "FEN: {fen}\nWhat pieces is {color} threatening?",
    "tactical_patterns":      "FEN: {fen}\nFind the best tactical move.",
    "material_balance":       "FEN: {fen}\nWhat is the material balance?",
    "eval_bucket":            "FEN: {fen}\nEvaluate this position. Who is better?",
    "pawn_structure":         "FEN: {fen}\nAnalyze the pawn structure.",
    "opening_name":           "FEN: {fen}\nWhat opening is this?",
    "opening_continuation":   "FEN: {fen}\nWhat are the main continuation moves?",
    "endgame_classification": "FEN: {fen}\nWhat type of endgame is this?",
    "endgame_wdl":            "FEN: {fen}\nIs this endgame a win, draw, or loss for the side to move?",
    "endgame_best_move":      "FEN: {fen}\nWhat is the best move in this endgame?",
    "best_move":              "FEN: {fen}\nWhat is the best move?",
    "puzzle_solve":           "FEN: {fen}\nSolve this puzzle. Find the winning move.",
    "legal_moves_960":        "FEN: {fen}\nList all legal moves.",
    "check_detection_960":    "FEN: {fen}\nDetect the game state: check, checkmate, stalemate, or none.",
    "castling_rules_960":     "FEN: {fen}\nWhat castling options are available?",
    "binary_choice":          "FEN: {fen}\nCompare moves {move_a} and {move_b}. Which is better?",
}

# Task type -> metric type mapping
TASK_METRIC_TYPE: dict[str, str] = {
    "board_print":            "exact_match",
    "board_to_fen":           "exact_match",
    "piece_id":               "exact_match",
    "material_count":         "exact_match",
    "state_tracking":         "exact_match",
    "legal_moves":            "jaccard",
    "check_detection":        "exact_match",
    "captures":               "jaccard",
    "special_rules":          "exact_match",
    "legality_check":         "exact_match",
    "capture_id":             "jaccard",
    "hanging_pieces":         "exact_match",
    "threats":                "threat_f1",
    "tactical_patterns":      "exact_match",
    "material_balance":       "exact_match",
    "eval_bucket":            "eval_bucket",
    "pawn_structure":         "exact_match",
    "opening_name":           "exact_match",
    "opening_continuation":   "continuation_rank",
    "endgame_classification": "exact_match",
    "endgame_wdl":            "exact_match",
    "endgame_best_move":      "exact_match",
    "best_move":              "move_extraction",
    "puzzle_solve":           "move_extraction",
    "legal_moves_960":        "jaccard",
    "check_detection_960":    "exact_match",
    "castling_rules_960":     "exact_match",
    "binary_choice":          "move_choice",
}


# ---------------------------------------------------------------------------
# Gold answer derivation helpers — formats match training generators
# ---------------------------------------------------------------------------


def _derive_check_detection(fen: str, chess960: bool = False) -> str:
    """Check detection matching CheckDetection generator (task 2.4) format."""
    board = chess.Board(fen, chess960=chess960)
    if board.is_checkmate():
        return "Checkmate."
    if board.is_stalemate():
        return "Stalemate."
    if board.is_check():
        return "Check."
    return "Normal position -- no check, checkmate, or stalemate."


def _derive_captures(fen: str) -> str:
    """Captures matching AvailableCaptures generator (task 3.1) format."""
    board = chess.Board(fen)
    captures = [m.uci() for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return "No captures available."
    return " ".join(sorted(captures))


def _derive_material_count(fen: str) -> str:
    """Material count matching PieceCounting generator (generic branch)."""
    board = chess.Board(fen)
    white_counts: dict[str, int] = {}
    black_counts: dict[str, int] = {}
    white_total = 0
    black_total = 0
    white_mat = 0
    black_mat = 0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        name = _PIECE_NAMES[piece.piece_type]
        val = _PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            white_counts[name] = white_counts.get(name, 0) + 1
            white_total += 1
            white_mat += val
        else:
            black_counts[name] = black_counts.get(name, 0) + 1
            black_total += 1
            black_mat += val

    w_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(white_counts.items())]
    b_parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in sorted(black_counts.items())]
    balance = white_mat - black_mat
    if balance > 0:
        balance_str = f"White is up {balance} point(s) of material."
    elif balance < 0:
        balance_str = f"Black is up {abs(balance)} point(s) of material."
    else:
        balance_str = "Material is equal."

    return (
        f"White ({white_total} pieces): {', '.join(w_parts)}. "
        f"Black ({black_total} pieces): {', '.join(b_parts)}. "
        f"{balance_str}"
    )


def _derive_special_rules(fen: str, chess960: bool = False) -> str:
    """Special rules matching SpecialRules generator format."""
    board = chess.Board(fen, chess960=chess960)
    parts: list[str] = []

    # Castling (check for side-to-move rights)
    castling: list[str] = []
    if board.has_kingside_castling_rights(board.turn):
        castling.append("kingside")
    if board.has_queenside_castling_rights(board.turn):
        castling.append("queenside")
    if castling:
        parts.append(f"Castling available: {', '.join(castling)}.")

    # En passant
    if board.ep_square is not None:
        ep_moves = []
        for move in board.legal_moves:
            if move.to_square == board.ep_square and board.is_en_passant(move):
                ep_moves.append(move.uci())
        if ep_moves:
            parts.append(f"En passant possible: {' '.join(ep_moves)}.")

    # Promotions
    promo_squares: set[str] = set()
    for move in board.legal_moves:
        if move.promotion:
            promo_squares.add(chess.square_name(move.from_square))
    if promo_squares:
        parts.append(
            f"Promotion possible from: {' '.join(sorted(promo_squares))}. "
            f"Each pawn can promote to queen, rook, bishop, or knight."
        )

    return " ".join(parts) if parts else "No special moves available."


def _derive_hanging_pieces(fen: str) -> str:
    """Hanging pieces matching HangingPieces generator format."""
    board = chess.Board(fen)
    hanging: list[str] = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue

        enemy_color = not piece.color
        friendly_color = piece.color

        is_attacked = board.is_attacked_by(enemy_color, sq)
        is_defended = board.is_attacked_by(friendly_color, sq)

        if is_attacked and not is_defended:
            color_name = "white" if piece.color == chess.WHITE else "black"
            hanging.append(
                f"{color_name} {_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
            )

    if hanging:
        return f"Hanging pieces: {', '.join(hanging)}."
    return "No hanging pieces \u2014 all attacked pieces are defended."


def _derive_threats(fen: str) -> tuple[str, str]:
    """Threats matching Threats generator format. Returns (answer, color_name)."""
    board = chess.Board(fen)
    color = board.turn
    color_name = "white" if color == chess.WHITE else "black"
    opp_color = not color

    threatened: list[str] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == opp_color:
            if board.is_attacked_by(color, sq):
                threatened.append(
                    f"{_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
                )

    if threatened:
        answer = f"{color_name.capitalize()} threatens: {', '.join(threatened)}."
    else:
        answer = f"{color_name.capitalize()} has no immediate threats."
    return answer, color_name


def _derive_state_tracking(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    """Generate random moves and result FEN. Returns (moves_str, result_fen)."""
    board = chess.Board(fen, chess960=chess960)
    n_moves = rng.randint(1, 8)
    moves_played: list[str] = []
    for _ in range(n_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        moves_played.append(move.uci())
        board.push(move)
    if not moves_played:
        return "", fen
    return " ".join(moves_played), board.fen()


def _derive_legality_check(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    """Pick a legal or illegal move. Returns (move_uci, answer)."""
    board = chess.Board(fen, chess960=chess960)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "", ""

    if rng.random() < 0.5:
        move = rng.choice(legal_moves)
        return move.uci(), "Yes, the move is legal."

    legal_set = {m.uci() for m in legal_moves}
    for _ in range(50):
        from_sq = rng.choice(chess.SQUARES)
        to_sq = rng.choice(chess.SQUARES)
        if from_sq == to_sq:
            continue
        uci = chess.square_name(from_sq) + chess.square_name(to_sq)
        if uci not in legal_set:
            return uci, "No, the move is not legal."
    # Fallback: legal move
    move = rng.choice(legal_moves)
    return move.uci(), "Yes, the move is legal."


def _derive_opening_continuation(raw: dict) -> str:
    """Opening continuation matching OpeningContinuation generator format."""
    book_moves = raw.get("book_moves", [])
    if not book_moves:
        return ""
    parts: list[str] = []
    total_weight = sum(w for _, w in book_moves)
    for uci, weight in book_moves[:5]:
        pct = weight / total_weight * 100 if total_weight > 0 else 0
        parts.append(f"{uci} ({pct:.0f}%)")
    return "Top continuations: " + ", ".join(parts) + "."


def _derive_castling_rules(fen: str, chess960: bool = False) -> str:
    """Castling availability for Chess960 positions."""
    board = chess.Board(fen, chess960=chess960)
    castling: list[str] = []
    if board.has_kingside_castling_rights(board.turn):
        castling.append("kingside")
    if board.has_queenside_castling_rights(board.turn):
        castling.append("queenside")
    if castling:
        return f"Castling available: {', '.join(castling)}."
    return "No castling available."


def derive_gold_answer(
    task_type: str, raw: dict, rng: Random,
) -> str:
    """Derive the ground-truth answer for a benchmark example.

    Returns ``""`` on any chess-library exception (bad FEN, etc.) so that
    ``freeze_split()``'s skip logic handles it uniformly.
    """
    try:
        return _derive_gold_answer_inner(task_type, raw, rng)
    except (ValueError, TypeError, IndexError) as exc:
        logger.debug("derive_gold_answer(%r) failed: %s", task_type, exc)
        return ""


def _derive_gold_answer_inner(
    task_type: str, raw: dict, rng: Random,
) -> str:
    """Inner implementation — may raise on bad FEN / data."""
    fen = raw.get("fen", "")
    is_960 = bool(raw.get("is_chess960") or task_type.endswith("_960"))

    if task_type == "board_print":
        board = chess.Board(fen, chess960=is_960)
        return _board_to_ascii(board)

    if task_type == "board_to_fen":
        return fen

    if task_type == "piece_id":
        board = chess.Board(fen, chess960=is_960)
        occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
        sq = rng.choice(occupied)
        piece = board.piece_at(sq)
        color_name = "white" if piece.color == chess.WHITE else "black"
        piece_name = chess.piece_name(piece.piece_type)
        raw["_piece_id_square"] = chess.square_name(sq)
        return f"{color_name} {piece_name}"

    if task_type == "material_count":
        return _derive_material_count(fen)

    if task_type == "state_tracking":
        moves_str, result_fen = _derive_state_tracking(fen, rng, chess960=is_960)
        raw["_state_tracking_moves"] = moves_str
        raw["_state_tracking_result"] = result_fen
        return result_fen

    if task_type in ("legal_moves", "legal_moves_960"):
        return answer_legal_moves(fen, chess960=is_960)

    if task_type in ("check_detection", "check_detection_960"):
        return _derive_check_detection(fen, chess960=is_960)

    if task_type in ("captures", "capture_id"):
        return _derive_captures(fen)

    if task_type == "special_rules":
        return _derive_special_rules(fen, chess960=is_960)

    if task_type == "legality_check":
        move_uci, answer = _derive_legality_check(fen, rng, chess960=is_960)
        raw["_legality_check_move"] = move_uci
        return answer

    if task_type == "hanging_pieces":
        return _derive_hanging_pieces(fen)

    if task_type == "threats":
        answer, color_name = _derive_threats(fen)
        raw["_threats_color"] = color_name
        return answer

    if task_type == "tactical_patterns":
        themes = raw.get("themes", [])
        solution = raw.get("solution_first_move", "")
        if not solution:
            return ""
        theme_str = ", ".join(themes) if themes else "tactical"
        return f"The tactic is {theme_str}. Best move: {solution}"

    if task_type == "material_balance":
        return answer_material_balance(fen)

    if task_type == "eval_bucket":
        cp = raw.get("cp")
        mate = raw.get("mate")
        return answer_position_eval(cp, mate)

    if task_type == "pawn_structure":
        return answer_pawn_structure(fen)

    if task_type == "opening_name":
        name = raw.get("name", "")
        eco = raw.get("eco", "")
        return f"{name} (ECO: {eco})"

    if task_type == "opening_continuation":
        return _derive_opening_continuation(raw)

    if task_type == "endgame_classification":
        material = raw.get("material")
        return answer_endgame_classification(fen, material)

    if task_type == "endgame_wdl":
        wdl = raw.get("wdl")
        if wdl is not None:
            return answer_endgame_wdl(wdl)
        return ""

    if task_type == "endgame_best_move":
        return raw.get("best_move", "")

    if task_type == "best_move":
        return raw.get("best_move", "")

    if task_type == "puzzle_solve":
        return raw.get("solution_first_move", "")

    if task_type == "castling_rules_960":
        return _derive_castling_rules(fen, chess960=True)

    if task_type == "binary_choice":
        return raw.get("better_move", "")

    return ""


def _render_prompt(task_type: str, raw: dict) -> str:
    """Render the canonical prompt for a task type."""
    template = CANONICAL_PROMPTS[task_type]
    ctx: dict = dict(raw)
    fen = ctx.get("fen", "")
    is_960 = bool(ctx.get("is_chess960") or task_type.endswith("_960"))

    if task_type == "board_to_fen" and fen:
        try:
            board = chess.Board(fen, chess960=is_960)
            ctx.setdefault("board", _board_to_ascii(board))
        except (ValueError, TypeError):
            ctx.setdefault("board", "")

    if task_type == "piece_id":
        ctx.setdefault("square", raw.get("_piece_id_square", "e1"))

    if task_type == "state_tracking":
        ctx.setdefault("moves", raw.get("_state_tracking_moves", ""))

    if task_type == "legality_check":
        ctx.setdefault("move", raw.get("_legality_check_move", ""))

    if task_type == "threats":
        ctx.setdefault("color", raw.get("_threats_color", "white"))

    if task_type == "binary_choice":
        ctx.setdefault("move_a", raw.get("move_a", ""))
        ctx.setdefault("move_b", raw.get("move_b", ""))

    try:
        return template.format(**ctx)
    except KeyError:
        return template


# ---------------------------------------------------------------------------
# Metric functions (all pure)
# ---------------------------------------------------------------------------


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized prediction == normalized gold, else 0.0."""
    return 1.0 if prediction.strip().lower() == gold.strip().lower() else 0.0


def jaccard_similarity(prediction: str, gold: str) -> float:
    """Jaccard of word sets: |intersection| / |union|."""
    pred_set = set(prediction.strip().lower().split())
    gold_set = set(gold.strip().lower().split())
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    intersection = pred_set & gold_set
    union = pred_set | gold_set
    return len(intersection) / len(union)


def eval_bucket_accuracy(prediction: str, gold: str) -> float:
    """1.0 if prediction contains the gold bucket keyword."""
    gold_lower = gold.strip().lower()
    pred_lower = prediction.strip().lower()
    for keyword in ("equal", "slight", "clear advantage", "winning",
                    "decisive", "forced mate"):
        if keyword in gold_lower and keyword in pred_lower:
            return 1.0
    return 0.0


def normalize_prediction(prediction: str) -> str:
    """Strip non-answer wrappers from a model prediction.

    This keeps benchmark scoring robust against chat-template markers and
    chain-of-thought blocks while preserving the model's final answer text.
    """
    import re as _re

    text = prediction or ""
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    text = text.replace("<|im_end|>", " ")
    text = text.replace("<|endoftext|>", " ")
    text = text.replace("<|assistant|>", " ")
    text = text.replace("<|user|>", " ")
    text = _re.sub(r"<\|im_start\|>\s*assistant", " ", text, flags=_re.IGNORECASE)
    text = _re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=_re.IGNORECASE)
    return text.strip()


def format_compliance(prediction: str) -> float:
    """1.0 if valid <think>...</think><move>UCI</move> format."""
    return 1.0 if validate_think_move_format(prediction) else 0.0


def legal_move_rate(
    prediction: str,
    fen: str,
    chess960: bool = False,
) -> float | None:
    """1.0 if move in <move> tag is legal in fen, None if no tag present.

    Per the benchmark plan, legal move rate is "of all outputs that
    contain a ``<move>`` tag".  Returning ``None`` for tag-less outputs
    lets callers exclude them from the denominator.
    """
    uci = _extract_uci_from_move_tag(normalize_prediction(prediction))
    if uci is None:
        return None
    return 1.0 if validate_move_legal(fen, uci, chess960=chess960) else 0.0


def move_extraction_match(prediction: str, gold_move: str) -> float:
    """1.0 if the UCI move extracted from <move> tag matches gold.

    Tier 7 training outputs ``<think>...</think><move>UCI</move>`` but
    the frozen gold answer is bare UCI.  This metric extracts the move
    from the tag before comparing.  Falls back to plain exact match if
    no ``<move>`` tag is found (handles bare UCI predictions too).
    """
    uci = _extract_move(prediction)
    if uci is None:
        return 0.0
    return 1.0 if uci.strip().lower() == gold_move.strip().lower() else 0.0


_NEGATION_RE = None  # lazy-compiled


def _is_negated(text: str, move_start: int, move: str) -> bool:
    """Check if *move* at *move_start* is negated by surrounding context.

    Checks two zones:

    1. **Prefix** (30 chars before): negation words like "not", "avoid",
       "rather than", etc. that precede the move.
    2. **Subject-verb** (20 chars after): patterns like "is not", "is
       worse", "is inferior" where the move is the grammatical subject.

    Does NOT flag bare "not"/"never" in the suffix — those typically
    negate the *next* clause (e.g. "d2d4, not e2e4").
    """
    global _NEGATION_RE
    if _NEGATION_RE is None:
        import re
        _NEGATION_RE = re.compile(
            r"\b(?:not|don'?t|never|avoid|reject|worse|inferior|"
            r"rather\s+than|instead\s+of)\b"
        )

    # 1. Prefix check
    window_before = text[max(0, move_start - 30):move_start]
    if _NEGATION_RE.search(window_before):
        return True

    # 2. Subject-verb check: "MOVE is not/worse/inferior"
    #    Dismissal check: "MOVE? No" / "MOVE? No."
    import re as _re
    suffix = text[move_start + len(move):move_start + len(move) + 25]
    if _re.match(
        r"\s+is\s+(?:not|worse|inferior|bad|wrong|weaker)",
        suffix,
    ):
        return True
    if _re.match(r"[?!]\s*no\b", suffix):
        return True

    return False


def move_choice_match(
    prediction: str,
    gold_move: str,
    candidates: tuple[str, str] | None = None,
) -> float:
    """1.0 if the model chose the gold move in a binary-choice task.

    Strategy when *candidates* are provided:

    1. If only one candidate appears, it is the choice.
    2. If both first appear close together joined by "and"/"or" (an
       enumeration preamble like "Between X and Y"), skip past the
       preamble and take the first candidate mentioned afterward.
    3. If the first-mentioned candidate is surrounded by negation
       cues (not, don't, avoid, rather than, instead of, worse,
       inferior, reject, never), the *other* candidate is the choice.
    4. Otherwise the first candidate mentioned is the choice.

    This correctly handles all common patterns::

        "d2d4"                                      → d2d4
        "d2d4 because ..."                          → d2d4
        "Between e2e4 and d2d4, d2d4 is better"    → d2d4
        "d2d4 is better than e2e4"                  → d2d4
        "I choose d2d4, not e2e4"                   → d2d4
        "I choose d2d4 over e2e4 because e2e4 ..."  → d2d4
        "Not e2e4; choose d2d4"                     → d2d4
        "e2e4 is not best; d2d4 is better"          → d2d4
        "Rather than e2e4, play d2d4"               → d2d4
    """
    import re
    gold = gold_move.strip().lower()
    pred = normalize_prediction(prediction).strip().lower()

    if candidates:
        ca, cb = candidates[0].strip().lower(), candidates[1].strip().lower()

        # Find all positions of each candidate
        pos_a = [m.start() for m in re.finditer(re.escape(ca), pred)]
        pos_b = [m.start() for m in re.finditer(re.escape(cb), pred)]

        if not pos_a and not pos_b:
            return 0.0
        if pos_a and not pos_b:
            return 1.0 if ca == gold else 0.0
        if pos_b and not pos_a:
            return 1.0 if cb == gold else 0.0

        # Both present — check for enumeration preamble
        first_a, first_b = pos_a[0], pos_b[0]
        earlier = min(first_a, first_b)
        later = max(first_a, first_b)
        earlier_move = ca if first_a <= first_b else cb
        later_move = cb if first_a <= first_b else ca
        between = pred[earlier + len(earlier_move):later]

        is_preamble = (
            (later - earlier) < 25
            and re.search(r"\b(?:and|or)\b", between)
        )

        if is_preamble:
            # Skip past the preamble — find first candidate after both
            skip = later + len(later_move)
            next_a = next((p for p in pos_a if p >= skip), -1)
            next_b = next((p for p in pos_b if p >= skip), -1)
            if next_a >= 0 and (next_b < 0 or next_a <= next_b):
                return 1.0 if ca == gold else 0.0
            if next_b >= 0:
                return 1.0 if cb == gold else 0.0
            # Nothing after preamble — fall through to negation check

        # Check if the first-mentioned candidate is negated
        first_move = earlier_move
        first_pos = earlier
        if _is_negated(pred, first_pos, first_move):
            # First candidate is rejected → the other is the choice
            chosen = later_move
        else:
            chosen = first_move

        return 1.0 if chosen == gold else 0.0

    # Fallback (no candidates): first UCI token is the answer
    tokens = re.findall(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", pred)
    if not tokens:
        return 0.0
    return 1.0 if tokens[0] == gold else 0.0


def _extract_move(prediction: str) -> str | None:
    """Extract the UCI move from a prediction.

    Checks ``<move>`` tag first, then falls back to treating the entire
    stripped prediction as a bare UCI string.  This keeps ``pass_at_k``
    and ``compute_acpl`` consistent with ``move_extraction_match``.
    """
    import re as _re

    text = normalize_prediction(prediction)
    uci = _extract_uci_from_move_tag(text)
    if uci is not None:
        return uci
    # Bare UCI fallback — must match the standard UCI pattern
    bare = text.strip().lower()
    if _re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", bare):
        return bare
    match = _re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", bare)
    if match:
        return match.group(1)
    return None


def pass_at_k(predictions: list[str], gold_move: str) -> float:
    """1.0 if any prediction's move matches gold_move.

    Extracts the move from ``<move>`` tags first, falls back to bare UCI.
    """
    gold = gold_move.strip().lower()
    for pred in predictions:
        uci = _extract_move(pred)
        if uci and uci == gold:
            return 1.0
    return 0.0


def centipawn_loss(gold_cp: float, predicted_cp: float) -> float:
    """Centipawn loss for a single position. Lower is better.

    Both ``gold_cp`` and ``predicted_cp`` are from the same side's
    perspective (the side to move in the original position).
    """
    return max(0.0, gold_cp - predicted_cp)


def threat_f1(prediction: str, gold: str) -> float:
    """F1 score for threat detection based on parsed threat sets.

    Both *prediction* and *gold* follow the Threats generator format::

        "White threatens: knight on e4, pawn on d5."
        "White has no immediate threats."
    """
    def _parse(text: str) -> set[str]:
        t = text.strip().lower()
        # "has no immediate threats" → empty set
        if "no immediate threats" in t or "no " in t and "threat" in t:
            return set()
        # Strip prefix: "white threatens: ..." → "..."
        for marker in ("threatens:", "pieces under attack:"):
            if marker in t:
                t = t.split(marker, 1)[1]
        # Split by comma, clean up
        items = set()
        for item in t.split(","):
            item = item.strip().rstrip(".")
            if item:
                items.add(item)
        return items

    pred_set = _parse(prediction)
    gold_set = _parse(gold)

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def continuation_rank(prediction: str, gold: str) -> float:
    """Mean reciprocal rank of predicted moves in the gold continuation list.

    *gold* follows the OpeningContinuation format::

        "Top continuations: e2e4 (50%), d2d4 (30%), g1f3 (20%)."

    Returns the reciprocal rank (1/rank) of the first predicted UCI move
    found in the gold list. 1.0 = top move, 0.5 = second, etc. 0.0 = miss.
    """
    import re as _re

    gold_moves = _re.findall(r"([a-h][1-8][a-h][1-8][qrbn]?)\s*\(\d+%\)", gold)
    if not gold_moves:
        return 0.0

    pred_moves = _re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", prediction.lower())
    if not pred_moves:
        return 0.0

    for pm in pred_moves:
        for rank, gm in enumerate(gold_moves, 1):
            if pm == gm:
                return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Freeze & Score
# ---------------------------------------------------------------------------

# Keys to preserve in frozen metadata from raw dicts
_META_KEYS = frozenset({
    "cp", "mate", "wdl", "material", "eco", "name",
    "best_move", "solution_first_move", "better_move",
    "move_a", "move_b", "puzzle_id", "source", "themes",
    "book_moves", "depth", "is_chess960", "chess960_id",
})


def freeze_split(
    split_name: str,
    raw_examples: list[dict],
    seed: int,
) -> list[BenchmarkExample]:
    """Create frozen benchmark examples from raw eval split data.

    Uses round-robin task assignment (except ``planning`` which is
    source-detected).
    """
    rng = Random(seed)
    task_types = SPLIT_TASK_TYPES.get(split_name, [])
    if not task_types:
        logger.warning("No task types defined for split %r", split_name)
        return []

    examples: list[BenchmarkExample] = []
    skipped = 0

    for i, raw in enumerate(raw_examples):
        # Task type assignment
        if split_name == "planning":
            task_type = "puzzle_solve" if raw.get("puzzle_id") else "best_move"
            candidates = [task_type]
        else:
            primary = task_types[i % len(task_types)]
            # Build fallback order: primary first, then remaining task types
            candidates = [primary] + [t for t in task_types if t != primary]

        # Try each candidate task type until one produces a gold answer
        gold = ""
        task_type = candidates[0]
        for candidate in candidates:
            gold = derive_gold_answer(candidate, raw, rng)
            if gold:
                task_type = candidate
                break

        if not gold:
            skipped += 1
            continue

        # Render canonical prompt
        prompt = _render_prompt(task_type, raw)

        # Metric type
        metric = TASK_METRIC_TYPE.get(task_type, "exact_match")

        example_id = f"{split_name}_{len(examples):05d}"

        # Preserve raw fields + derived private keys in metadata
        meta: dict = {}
        for key in raw:
            if key in _META_KEYS or key.startswith("_"):
                meta[key] = raw[key]

        examples.append(BenchmarkExample(
            example_id=example_id,
            split=split_name,
            task_type=task_type,
            fen=raw.get("fen", ""),
            prompt=prompt,
            gold_answer=gold,
            metric_type=metric,
            metadata=meta,
        ))

    if skipped:
        logger.warning(
            "Split %r: %d / %d examples skipped (no viable task type). "
            "Frozen %d examples vs %d raw.",
            split_name, skipped, len(raw_examples),
            len(examples), len(raw_examples),
        )

    # Warn about task-mix drift: any planned task type with zero coverage
    actual_counts: dict[str, int] = {}
    for ex in examples:
        actual_counts[ex.task_type] = actual_counts.get(ex.task_type, 0) + 1
    for tt in task_types:
        if actual_counts.get(tt, 0) == 0:
            logger.warning(
                "Split %r: task type %r has 0 examples (all fell back to "
                "other types). Metric coverage for this task is missing.",
                split_name, tt,
            )

    return examples


def score_prediction(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, float | None]:
    """Score a single prediction against its benchmark example.

    Returns dict with ``primary`` metric and optional ``secondary`` metrics.
    """
    raw_prediction = prediction
    prediction = normalize_prediction(prediction)
    metric = example.metric_type
    gold = example.gold_answer
    scores: dict[str, float | None] = {}

    if metric == "move_extraction":
        scores["primary"] = move_extraction_match(prediction, gold)
    elif metric == "move_choice":
        meta = example.metadata
        candidates = None
        if meta.get("move_a") and meta.get("move_b"):
            candidates = (meta["move_a"], meta["move_b"])
        scores["primary"] = move_choice_match(prediction, gold, candidates)
    elif metric == "exact_match":
        scores["primary"] = exact_match(prediction, gold)
    elif metric == "jaccard":
        scores["primary"] = jaccard_similarity(prediction, gold)
    elif metric == "eval_bucket":
        scores["primary"] = eval_bucket_accuracy(prediction, gold)
    elif metric == "threat_f1":
        scores["primary"] = threat_f1(prediction, gold)
    elif metric == "continuation_rank":
        scores["primary"] = continuation_rank(prediction, gold)
    else:
        scores["primary"] = exact_match(prediction, gold)

    # Secondary metrics for Tier 7 move-prediction tasks (think/move format)
    if example.task_type in ("best_move", "puzzle_solve"):
        scores["format_compliance"] = format_compliance(raw_prediction)
        scores["legal_move"] = legal_move_rate(
            raw_prediction,
            example.fen,
            chess960=bool(example.metadata.get("is_chess960")),
        )

    return scores


def score_split(
    examples: list[BenchmarkExample],
    predictions: dict[str, str],
    acpl_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Aggregate metrics for one split.

    Parameters
    ----------
    examples : list[BenchmarkExample]
    predictions : dict[str, str]
        Mapping of example_id -> prediction string.
    acpl_scores : dict[str, float] | None
        Optional mapping of example_id -> centipawn loss (from engine eval).

    Returns
    -------
    dict[str, float]
        Aggregated metrics keyed by ``"{task_type}_{metric_name}"``.
    """
    from collections import defaultdict

    task_scores: dict[str, list[float]] = defaultdict(list)
    task_secondary: dict[str, list[float]] = defaultdict(list)
    task_acpl: dict[str, list[float]] = defaultdict(list)

    for ex in examples:
        pred = predictions.get(ex.example_id, "")
        scores = score_prediction(ex, pred)

        primary = scores.get("primary", 0.0)
        task_scores[ex.task_type].append(primary)

        for key in ("format_compliance", "legal_move"):
            if key in scores and scores[key] is not None:
                task_secondary[f"{ex.task_type}_{key}"].append(scores[key])

        # ACPL if available
        if acpl_scores and ex.example_id in acpl_scores:
            task_acpl[ex.task_type].append(acpl_scores[ex.example_id])

    result: dict[str, float] = {}
    for task_type, vals in task_scores.items():
        if vals:
            result[task_type] = sum(vals) / len(vals)

    for key, vals in task_secondary.items():
        if vals:
            result[key] = sum(vals) / len(vals)

    # ACPL per task type
    for task_type, vals in task_acpl.items():
        if vals:
            result[f"{task_type}_acpl"] = sum(vals) / len(vals)

    # Overall ACPL across all move-prediction tasks
    all_acpl: list[float] = []
    for vals in task_acpl.values():
        all_acpl.extend(vals)
    if all_acpl:
        result["acpl"] = sum(all_acpl) / len(all_acpl)

    # Overall split score
    all_primary: list[float] = []
    for vals in task_scores.values():
        all_primary.extend(vals)
    if all_primary:
        result["overall"] = sum(all_primary) / len(all_primary)

    return result


# ---------------------------------------------------------------------------
# Oracle validation
# ---------------------------------------------------------------------------


def validate_oracle(examples: list[BenchmarkExample]) -> list[str]:
    """Verify every gold answer scores 1.0 on its primary metric.

    Feeds each example's ``gold_answer`` as the prediction and checks
    that ``score_prediction`` returns a perfect primary score.  Returns
    a list of failure descriptions (empty = all pass).
    """
    failures: list[str] = []
    for ex in examples:
        scores = score_prediction(ex, ex.gold_answer)
        primary = scores.get("primary", 0.0)
        if primary != 1.0:
            failures.append(
                f"{ex.example_id} ({ex.task_type}, metric={ex.metric_type}): "
                f"gold self-score={primary}"
            )
    return failures


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def save_benchmark(examples: list[BenchmarkExample], path: str) -> None:
    """Save benchmark examples to JSONL."""
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_benchmark(path: str) -> list[BenchmarkExample]:
    """Load benchmark examples from JSONL."""
    examples: list[BenchmarkExample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(BenchmarkExample.from_dict(json.loads(line)))
    return examples


def freeze_and_save(
    splits: dict[str, list[dict]],
    output_dir: str,
    seed: int,
    version: str = "chess-sft-eval-v1",
    clean: bool = True,
    strict_coverage: bool = True,
) -> dict:
    """Freeze eval splits into canonical benchmark JSONL + manifest.

    Shared implementation used by ``run_pipeline``, ``run_eval_split``,
    and ``freeze_benchmark`` scripts.

    Parameters
    ----------
    clean : bool
        If ``True`` (default), remove all existing ``*.jsonl`` files in
        *output_dir* before writing.  Set to ``False`` for partial
        refreezes (e.g. ``--split rules``) so other splits are kept and
        the manifest is merged with the existing one.
    strict_coverage : bool
        If ``True`` (default), raise ``ValueError`` when any planned
        task type has 0 frozen examples. Set to ``False`` for backfill
        refreezes from disk where some splits may have incomplete data.

    Returns the manifest dict.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if clean:
        for old_jsonl in out.glob("*.jsonl"):
            old_jsonl.unlink()

    # For partial freezes, start from the existing manifest so other
    # splits' counts are preserved.
    manifest_path = out / "manifest.json"
    if not clean and manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        # Only update the timestamp; preserve the original version/seed
        # so untouched split files stay consistent with global metadata.
        manifest["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Use the manifest's seed for freeze_split so regenerated splits
        # produce identical gold answers (some are seed-dependent).
        seed = manifest.get("seed", seed)
    else:
        manifest = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "splits": {},
        }

    total = 0
    all_frozen: list[BenchmarkExample] = []
    for split_name, raw_examples in splits.items():
        if not raw_examples:
            continue
        frozen = freeze_split(split_name, raw_examples, seed=seed)
        out_path = out / f"{split_name}.jsonl"
        save_benchmark(frozen, str(out_path))
        manifest["splits"][split_name] = len(frozen)
        total += len(frozen)
        all_frozen.extend(frozen)
        logger.info("Froze benchmark split %r: %d examples", split_name, len(frozen))

    # Oracle validation: every gold answer must self-score 1.0
    if all_frozen:
        failures = validate_oracle(all_frozen)
        if failures:
            logger.error(
                "Oracle validation FAILED — %d / %d examples do not "
                "self-score 1.0:",
                len(failures), len(all_frozen),
            )
            for f in failures[:10]:
                logger.error("  %s", f)
            raise ValueError(
                f"Oracle validation failed: {len(failures)} examples "
                f"have gold answers that don't self-score 1.0"
            )
        logger.info("Oracle validation passed: %d examples", len(all_frozen))

    # Task-coverage check: every planned task type must have >= 1 example
    coverage_gaps: list[str] = []
    for split_name in splits:
        planned = SPLIT_TASK_TYPES.get(split_name, [])
        actual = {
            ex.task_type for ex in all_frozen if ex.split == split_name
        }
        for tt in planned:
            if tt not in actual:
                coverage_gaps.append(f"{split_name}/{tt}")
    if coverage_gaps:
        msg = (
            f"Task coverage gaps: {', '.join(coverage_gaps)}. "
            f"Ensure source data covers all planned task types."
        )
        if strict_coverage:
            logger.error(
                "Task coverage gaps — %d task type(s) have 0 frozen examples: %s",
                len(coverage_gaps), ", ".join(coverage_gaps),
            )
            raise ValueError(msg)
        else:
            logger.warning(
                "Task coverage gaps (non-fatal) — %d task type(s) have "
                "0 frozen examples: %s",
                len(coverage_gaps), ", ".join(coverage_gaps),
            )

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Frozen benchmark: %d total examples, manifest at %s", total, manifest_path)

    return manifest
