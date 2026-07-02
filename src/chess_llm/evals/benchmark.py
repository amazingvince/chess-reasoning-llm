"""Frozen benchmark schema, loading, and scoring primitives."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import chess

from chess_llm.core.legality import (
    classify_move_legality,
    format_legality_answer,
    legality_binary_accuracy,
    legality_reason_accuracy,
    random_illegal_move_with_reason,
)
from chess_llm.core.board import is_legal_move
from chess_llm.formats.answers import (
    extract_move as _extract_move_from_answer_text,
    extract_uci_from_move_tag as _extract_uci_from_move_tag,
    validate_think_move_format,
)
from chess_llm.formats.board import render_ascii_board
from chess_llm.sft.context import build_template_context, raw_is_chess960
from chess_llm.sft.settings import DEFAULT_EVAL_BUCKETS
from chess_llm.sft.templates import append_answer_contract

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkExample:
    """A single frozen benchmark example."""

    example_id: str
    split: str
    task_type: str
    fen: str
    prompt: str
    gold_answer: str
    metric_type: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BenchmarkExample":
        return cls(**payload)


_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
_LEGAL_MOVES_BY_PIECE_LINE_RE = re.compile(
    r"^\s*([a-h][1-8])\s+"
    r"(white|black)\s+"
    r"(king|queen|rook|bishop|knight|pawn):\s*"
    r"(.*?)\s*$",
    re.IGNORECASE,
)
_ANY_MOVE_TAG_RE = re.compile(r"<move\b[^>]*>.*?</move>", re.DOTALL | re.IGNORECASE)
_NEGATION_RE: re.Pattern[str] | None = None
_FEN_RE = re.compile(
    r"((?:[prnbqkPRNBQK1-8]+/){7}[prnbqkPRNBQK1-8]+"
    r"\s+[wb]\s+[-A-HKQabcdefghkq]+\s+(?:-|[a-h][36])"
    r"(?:\s+\d+\s+\d+)?)"
)

DIAGNOSTIC_TASK_TYPES = frozenset({
    "square_lookup",
    "rank_lookup",
    "square_coordinates",
    "fen_rank_expansion",
    "fen_rank_cell_edit",
    "fen_board_edit",
    "move_square_edits",
    "fen_assembly",
    "fen_row_application",
    "material_inventory",
    "material_piece_counts",
    "material_value_totals",
    "material_balance_trace",
    "side_piece_inventory",
    "piece_legal_moves",
    "piece_pseudo_legal_moves",
    "piece_legal_filter",
    "king_safety_filter",
    "legal_moves_by_piece",
})
FULL_FEN_STATE_PROMPT_TASK_TYPES = frozenset({
    "board_to_fen",
    "fen_assembly",
    "fen_row_application",
    "state_tracking",
})
CANONICAL_START_FEN_PROMPT_TASK_TYPES = frozenset({
    "fen_assembly",
    "fen_row_application",
    "state_tracking",
})

SPLIT_TASK_TYPES: dict[str, list[str]] = {
    "perception": [
        "board_print",
        "board_to_fen",
        "piece_id",
        "material_count",
        "square_lookup",
        "rank_lookup",
        "square_coordinates",
        "fen_rank_expansion",
        "fen_rank_cell_edit",
        "fen_board_edit",
        "move_square_edits",
        "fen_assembly",
        "fen_row_application",
        "state_tracking",
        "material_inventory",
        "material_piece_counts",
        "material_value_totals",
        "material_balance_trace",
    ],
    "rules": [
        "legal_moves",
        "side_piece_inventory",
        "piece_legal_moves",
        "check_detection",
        "captures",
        "special_rules",
        "legality_check",
        "piece_pseudo_legal_moves",
        "piece_legal_filter",
        "king_safety_filter",
        "legal_moves_by_piece",
    ],
    "tactics": ["capture_id", "hanging_pieces", "threats", "tactical_patterns"],
    "evaluation": ["material_balance", "eval_bucket", "pawn_structure"],
    "openings": ["opening_name", "opening_continuation"],
    "endgames": ["endgame_classification", "endgame_wdl", "endgame_best_move"],
    "planning": ["best_move", "puzzle_solve"],
    "chess960": ["legal_moves_960", "check_detection_960", "castling_rules_960"],
    "mate": ["binary_choice"],
}

CANONICAL_PROMPTS: dict[str, str] = {
    "board_print": "Position (FEN): {fen}\nShow me the board.",
    "board_to_fen": "Here is the current board:\n{board}\nWrite the FEN for this position.",
    "piece_id": "FEN: {fen}\nWhat piece is on {square}?",
    "material_count": "FEN: {fen}\nWhat is the material count for both sides?",
    "material_inventory": "FEN: {fen}\nList the material inventory by side and piece type.",
    "material_piece_counts": "FEN: {fen}\nCount each piece type for both sides.",
    "material_value_totals": "FEN: {fen}\nConvert the piece counts into material value totals.",
    "material_balance_trace": "FEN: {fen}\nTrace inventory, counts, values, and material balance.",
    "square_lookup": "FEN: {fen}\nWhat is on {square}?",
    "rank_lookup": "FEN: {fen}\nWhat is the compressed FEN row for rank {rank}?",
    "square_coordinates": "FEN: {fen}\nFor square {square}, give the FEN row-from-top and file index.",
    "fen_rank_expansion": "FEN: {fen}\nCompressed rank {rank} row: {fen_rank_row}\nExpand it into file cells a through h.",
    "fen_rank_cell_edit": "Rank {rank} row before: {before_row}\nFile {file} changes from {before_fen} to {after_fen}. Rewrite the compressed row.",
    "fen_board_edit": "Board FEN before: {board_fen_before}\nApply square edits: {edit_text}\nReturn the resulting board FEN only.",
    "move_square_edits": "Starting FEN: {fen}\nMove: {move}\nList the square lookups and rank edits.",
    "fen_assembly": "Starting FEN: {fen}\nMove: {move}\nUse square lookups and rank edits to assemble the resulting full FEN.",
    "fen_row_application": "Starting FEN: {fen}\nMove: {move}\nRewrite the affected compressed FEN rank rows and give Result FEN.",
    "state_tracking": "Starting FEN: {fen}\nAfter the moves {moves}, what is the resulting position?",
    "legal_moves": "FEN: {fen}\nList all legal moves.",
    "side_piece_inventory": "FEN: {fen}\nList the side-to-move pieces and their squares.",
    "piece_legal_moves": "FEN: {fen}\nWhat legal moves does the piece on {source_square} have?",
    "piece_pseudo_legal_moves": "FEN: {fen}\nList pseudo-legal moves from {source_square}.",
    "piece_legal_filter": "FEN: {fen}\nFor the piece on {source_square}, split pseudo-legal moves into legal and rejected moves with rejection reasons.",
    "king_safety_filter": "FEN: {fen}\nFor move {move}, decide whether king safety allows it.",
    "legal_moves_by_piece": "FEN: {fen}\nGroup all legal moves by side-to-move piece.",
    "check_detection": "FEN: {fen}\nDetect the game state: check, checkmate, stalemate, or none.",
    "captures": "FEN: {fen}\nList all capture moves available.",
    "special_rules": "Given FEN: {fen}\nList any special moves available (castling, en passant, promotion).",
    "legality_check": "FEN: {fen}\nIs the move {move} legal?",
    "capture_id": "FEN: {fen}\nList all capture moves available.",
    "hanging_pieces": "FEN: {fen}\nIdentify all hanging (undefended attacked) pieces.",
    "threats": "FEN: {fen}\nWhat pieces is {color} threatening?",
    "tactical_patterns": "FEN: {fen}\nFind the best tactical move.",
    "material_balance": "FEN: {fen}\nWhat is the material balance?",
    "eval_bucket": "FEN: {fen}\nEvaluate this position. Who is better?",
    "pawn_structure": "FEN: {fen}\nAnalyze the pawn structure.",
    "opening_name": "FEN: {fen}\nWhat opening is this?",
    "opening_continuation": "FEN: {fen}\nWhat are the main continuation moves?",
    "endgame_classification": "FEN: {fen}\nWhat type of endgame is this?",
    "endgame_wdl": "FEN: {fen}\nIs this endgame a win, draw, or loss for the side to move?",
    "endgame_best_move": "FEN: {fen}\nWhat is the best move in this endgame?",
    "best_move": "FEN: {fen}\nWhat is the best move?",
    "puzzle_solve": "FEN: {fen}\nSolve this puzzle. Find the winning move.",
    "legal_moves_960": "FEN: {fen}\nList all legal moves.",
    "check_detection_960": "FEN: {fen}\nDetect the game state: check, checkmate, stalemate, or none.",
    "castling_rules_960": "FEN: {fen}\nWhat castling options are available?",
    "binary_choice": "FEN: {fen}\nCompare moves {move_a} and {move_b}. Which is better?",
}

TASK_METRIC_TYPE: dict[str, str] = {
    "board_print": "board_exact_match",
    "board_to_fen": "fen_exact_match",
    "piece_id": "exact_match",
    "material_count": "material_count",
    "material_inventory": "text_exact_match",
    "material_piece_counts": "text_exact_match",
    "material_value_totals": "text_exact_match",
    "material_balance_trace": "text_exact_match",
    "square_lookup": "text_exact_match",
    "rank_lookup": "text_exact_match",
    "square_coordinates": "text_exact_match",
    "fen_rank_expansion": "text_exact_match",
    "fen_rank_cell_edit": "text_exact_match",
    "fen_board_edit": "text_exact_match",
    "move_square_edits": "text_exact_match",
    "fen_assembly": "fen_exact_match",
    "fen_row_application": "fen_exact_match",
    "state_tracking": "fen_exact_match",
    "legal_moves": "uci_set_jaccard",
    "side_piece_inventory": "side_piece_inventory",
    "piece_legal_moves": "uci_set_jaccard",
    "piece_pseudo_legal_moves": "text_exact_match",
    "piece_legal_filter": "text_exact_match",
    "king_safety_filter": "text_exact_match",
    "legal_moves_by_piece": "text_exact_match",
    "check_detection": "check_state",
    "captures": "uci_set_jaccard",
    "special_rules": "special_rules",
    "legality_check": "legality_check",
    "capture_id": "uci_set_jaccard",
    "hanging_pieces": "exact_match",
    "threats": "threat_f1",
    "tactical_patterns": "exact_match",
    "material_balance": "exact_match",
    "eval_bucket": "eval_bucket",
    "pawn_structure": "exact_match",
    "opening_name": "exact_match",
    "opening_continuation": "continuation_rank",
    "endgame_classification": "exact_match",
    "endgame_wdl": "exact_match",
    "endgame_best_move": "exact_match",
    "best_move": "move_extraction",
    "puzzle_solve": "move_extraction",
    "legal_moves_960": "uci_set_jaccard",
    "check_detection_960": "check_state",
    "castling_rules_960": "exact_match",
    "binary_choice": "move_choice",
}

_META_KEYS = frozenset({
    "cp", "mate", "wdl", "material", "eco", "name",
    "best_move", "solution_first_move", "better_move",
    "move_a", "move_b", "puzzle_id", "source", "themes",
    "book_moves", "depth", "is_chess960", "chess960_id",
    "diagnostic", "hard_gate", "source_square", "side_to_move",
    "side_piece_inventory", "legality_reason_label", "move",
    "square", "rank", "file", "fen_rank_row", "before_row",
    "after_row", "before_fen", "after_fen", "board_fen_before",
    "board_fen_after", "edit_text",
})

_NO_MOVE_RE = re.compile(
    r"\b(?:no|none|zero)\b.*\b(?:legal\s+)?(?:moves?|captures?)\b"
    r"|(?:legal\s+)?(?:moves?|captures?)\s+available\s*:\s*(?:none|no)\b",
    re.IGNORECASE,
)

_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}
_PIECE_ORDER = (
    chess.KING,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
)
_WDL_LABELS = {
    2: "Win for the side to move.",
    1: "Cursed win (win but 50-move rule may prevent it).",
    0: "Draw with best play.",
    -1: "Blessed loss (loss but 50-move rule saves it).",
    -2: "Loss for the side to move.",
}
_STATE_TRACKING_MAX_PLIES = 1


def _append_full_fen_state(user_text: str, context: dict) -> str:
    """Append non-board FEN fields so full-FEN prompts are answerable."""
    state_lines = [
        "State needed for full FEN:",
        f"Side to move: {context['side_to_move']}",
        f"Castling rights: {context['castling_rights']}",
        f"En passant: {context['en_passant_square']}",
        f"Halfmove clock: {context['halfmove_clock']}",
        f"Fullmove number: {context['fullmove_number']}",
    ]
    return f"{user_text}\n" + "\n".join(state_lines)


def _canonical_start_fen_for_prompt(fen: str, *, chess960: bool = False) -> str:
    """Return a complete FEN for prompts that ask the model to edit state."""
    try:
        board = _board_from_fen(fen, chess960=chess960)
    except (ValueError, TypeError):
        return fen
    if not board.is_valid():
        return fen
    return board.fen()


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized prediction == normalized gold, else 0.0."""
    return 1.0 if prediction.strip().lower() == gold.strip().lower() else 0.0


_MATERIAL_PIECE_NAMES = ("king", "queen", "rook", "bishop", "knight", "pawn")
_MATERIAL_COUNT_RE = re.compile(
    r"\b(\d+)\s+"
    r"(king|kings|queen|queens|rook|rooks|bishop|bishops|knight|knights|pawn|pawns)\b",
    re.IGNORECASE,
)
_SIDE_INVENTORY_RE = re.compile(
    r"\b([a-h][1-8])\s+"
    r"(white|black)\s+"
    r"(king|queen|rook|bishop|knight|pawn)\b",
    re.IGNORECASE,
)


def _singular_piece_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith("s"):
        lowered = lowered[:-1]
    return lowered


def _material_color_chunks(text: str, color: str) -> list[str]:
    pattern = re.compile(r"\b(white|black)\b", re.IGNORECASE)
    matches = list(pattern.finditer(text or ""))
    chunks: list[str] = []
    for index, match in enumerate(matches):
        if match.group(1).lower() != color:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunks.append(text[match.start():end])
    return chunks


def _extract_material_counts(text: str, color: str) -> dict[str, int] | None:
    chunks = _material_color_chunks(text, color)
    counts = {piece_name: 0 for piece_name in _MATERIAL_PIECE_NAMES}
    found = False
    for chunk in chunks:
        for count_text, piece_name in _MATERIAL_COUNT_RE.findall(chunk):
            counts[_singular_piece_name(piece_name)] = int(count_text)
            found = True
    return counts if found else None


def _material_value(counts: dict[str, int]) -> int:
    values = {
        "king": 0,
        "queen": 9,
        "rook": 5,
        "bishop": 3,
        "knight": 3,
        "pawn": 1,
    }
    return sum(counts[piece_name] * values[piece_name] for piece_name in values)


def _material_inventory(text: str) -> dict[str, dict[str, int]] | None:
    white = _extract_material_counts(text, "white")
    black = _extract_material_counts(text, "black")
    if white is None or black is None:
        return None
    return {"white": white, "black": black}


def material_count_accuracy(prediction: str, gold: str) -> float:
    """Score material count by numeric piece inventory instead of prose shape."""
    pred = _material_inventory(prediction)
    expected = _material_inventory(gold)
    if pred is None or expected is None:
        return exact_match(prediction, gold)
    return 1.0 if pred == expected else 0.0


def material_piece_accuracy(prediction: str, gold: str) -> float | None:
    """Fraction of color/piece count cells that match."""
    pred = _material_inventory(prediction)
    expected = _material_inventory(gold)
    if pred is None or expected is None:
        return None
    total = 0
    matched = 0
    for color in ("white", "black"):
        for piece_name in _MATERIAL_PIECE_NAMES:
            total += 1
            if pred[color][piece_name] == expected[color][piece_name]:
                matched += 1
    return matched / total if total else None


def _extract_side_piece_inventory_items(text: str) -> set[tuple[str, str, str]]:
    return {
        (square.lower(), color.lower(), piece.lower())
        for square, color, piece in _SIDE_INVENTORY_RE.findall(text or "")
    }


def side_piece_inventory_accuracy(prediction: str, gold: str) -> float:
    """Score side-piece inventory by square/color/piece tuples."""
    pred_items = _extract_side_piece_inventory_items(prediction)
    gold_items = _extract_side_piece_inventory_items(gold)
    if not pred_items and not gold_items:
        return 1.0
    if not pred_items or not gold_items:
        return 0.0
    true_positive = len(pred_items & gold_items)
    precision = true_positive / len(pred_items)
    recall = true_positive / len(gold_items)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _move_set_overlap_scores(predicted: set[str], gold: set[str]) -> dict[str, float]:
    true_positive = len(predicted & gold)
    extra = len(predicted - gold)
    missing = len(gold - predicted)
    precision = true_positive / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / len(gold) if gold else (1.0 if not predicted else 0.0)
    union_size = len(predicted | gold)
    jaccard = true_positive / union_size if union_size else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "jaccard": jaccard,
        "illegal_extra_count": float(extra),
        "missing_move_count": float(missing),
    }


def _uci_set(text: str) -> set[str]:
    return set(_UCI_RE.findall((text or "").lower()))


def _all_legal_moves_line_set(text: str) -> set[str] | None:
    match = re.search(r"(?im)^\s*All legal moves:\s*(.*?)\s*$", text or "")
    if match is None:
        return None
    return _uci_set(match.group(1))


def _answer_move_set(text: str) -> set[str]:
    final_line_moves = _all_legal_moves_line_set(text)
    if final_line_moves is not None:
        return final_line_moves
    return _uci_set(text)


def _legal_moves_by_piece_groups(text: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for line in (text or "").splitlines():
        match = _LEGAL_MOVES_BY_PIECE_LINE_RE.match(line)
        if match is None:
            continue
        square = match.group(1).lower()
        moves = _uci_set(match.group(4))
        groups.setdefault(square, set()).update(moves)
    return groups


def _set_jaccard(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    return len(predicted & gold) / len(predicted | gold)


def _per_piece_group_jaccard(prediction: str, gold: str) -> float:
    pred_groups = _legal_moves_by_piece_groups(prediction)
    gold_groups = _legal_moves_by_piece_groups(gold)
    squares = sorted(set(pred_groups) | set(gold_groups))
    if not squares:
        return 1.0 if prediction.strip() == gold.strip() else 0.0
    scores = []
    for square in squares:
        if square not in pred_groups or square not in gold_groups:
            scores.append(0.0)
        else:
            scores.append(_set_jaccard(pred_groups[square], gold_groups[square]))
    return sum(scores) / len(scores)


def legal_moves_by_piece_diagnostics(prediction: str, gold: str) -> dict[str, float]:
    """Diagnostic partial metrics for the grouped legal-move task."""
    lowered = (prediction or "").lower()
    section_hits = [
        "side to move:" in lowered,
        "pieces:" in lowered,
        "moves by piece:" in lowered,
        "all legal moves:" in lowered,
    ]
    move_scores = _move_set_overlap_scores(
        _answer_move_set(prediction),
        _answer_move_set(gold),
    )
    return {
        "section_completeness": sum(1 for hit in section_hits if hit) / len(section_hits),
        "all_legal_line_present": 1.0 if section_hits[-1] else 0.0,
        "piece_inventory_accuracy": side_piece_inventory_accuracy(prediction, gold),
        "all_moves_precision": move_scores["precision"],
        "all_moves_recall": move_scores["recall"],
        "all_moves_jaccard": move_scores["jaccard"],
        "per_piece_group_jaccard": _per_piece_group_jaccard(prediction, gold),
        "illegal_extra_count": move_scores["illegal_extra_count"],
        "missing_move_count": move_scores["missing_move_count"],
    }


def text_exact_match(prediction: str, gold: str) -> float:
    """Case-sensitive exact match for compact FEN mechanics text."""
    return 1.0 if prediction.strip() == gold.strip() else 0.0


def board_exact_match(prediction: str, gold: str) -> float:
    """Case-sensitive exact match for board diagrams."""
    return 1.0 if prediction.strip() == gold.strip() else 0.0


def _extract_canonical_fen(
    text: str,
    *,
    chess960: bool = False,
    require_six_fields: bool = False,
) -> str | None:
    """Extract and canonicalize a 4-field or 6-field FEN from model text."""
    stripped = (text or "").strip().strip("`")
    if not stripped:
        return None

    for candidate in _iter_fen_candidates(stripped):
        if require_six_fields and len(candidate.split()) != 6:
            continue
        try:
            board = chess.Board(candidate, chess960=chess960)
        except (ValueError, TypeError):
            continue
        if board.is_valid():
            return board.fen()
    return None


def _iter_fen_candidates(text: str) -> list[str]:
    """Return FEN-like substrings, preferring explicit Result FEN lines."""
    stripped = (text or "").strip().strip("`")
    result_fen_candidates: list[str] = []
    for line in stripped.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        if re.match(r"^result\s+fen\s*:", line, flags=re.IGNORECASE):
            result_fen_candidates.append(line.split(":", 1)[1].strip().rstrip("."))

    candidates = [*result_fen_candidates, stripped]
    for line in stripped.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        candidates.append(line.rstrip("."))
    candidates.extend(match.group(1) for match in _FEN_RE.finditer(stripped))
    return candidates


def _extract_fen_candidate(
    text: str,
    *,
    chess960: bool = False,
    require_six_fields: bool = False,
) -> str | None:
    """Extract the first syntactically parseable FEN, even if the board is invalid."""
    for candidate in _iter_fen_candidates(text):
        if require_six_fields and len(candidate.split()) != 6:
            continue
        try:
            board = chess.Board(candidate, chess960=chess960)
        except (ValueError, TypeError):
            continue
        return board.fen()
    return None


def fen_exact_match(
    prediction: str,
    gold: str,
    *,
    chess960: bool = False,
) -> float:
    """1.0 when extracted FENs match after canonicalization."""
    pred_fen = _extract_canonical_fen(
        prediction,
        chess960=chess960,
        require_six_fields=True,
    )
    gold_fen = _extract_canonical_fen(
        gold,
        chess960=chess960,
        require_six_fields=True,
    )
    if pred_fen is None or gold_fen is None:
        return 0.0
    return 1.0 if pred_fen == gold_fen else 0.0


def fen_first4_match(
    prediction: str,
    gold: str,
    *,
    chess960: bool = False,
) -> float:
    """1.0 when board/turn/castling/en-passant fields match."""
    pred_fen = _extract_canonical_fen(prediction, chess960=chess960)
    gold_fen = _extract_canonical_fen(gold, chess960=chess960)
    if pred_fen is None or gold_fen is None:
        return 0.0
    return 1.0 if pred_fen.split()[:4] == gold_fen.split()[:4] else 0.0


def jaccard_similarity(prediction: str, gold: str) -> float:
    """Jaccard of word sets."""
    pred_set = set(prediction.strip().lower().split())
    gold_set = set(gold.strip().lower().split())
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(pred_set | gold_set)


def uci_set_jaccard(prediction: str, gold: str) -> float:
    """Jaccard similarity over extracted UCI move sets."""
    pred_set = set(_UCI_RE.findall(prediction.lower()))
    gold_set = set(_UCI_RE.findall(gold.lower()))
    if not gold_set and not gold.strip():
        return 1.0 if _is_no_move_answer(prediction) else 0.0
    if not gold_set and _is_no_move_answer(gold):
        return 1.0 if _is_no_move_answer(prediction) else 0.0
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    return len(pred_set & gold_set) / len(pred_set | gold_set)


def _is_no_move_answer(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return _NO_MOVE_RE.search(normalized) is not None


def check_state_accuracy(prediction: str, gold: str) -> float:
    """1.0 if check/checkmate/stalemate/normal labels match semantically."""
    gold_key = _check_state_key(gold)
    pred_key = _check_state_key(prediction)
    return 1.0 if gold_key is not None and pred_key == gold_key else 0.0


def _check_state_key(text: str) -> str | None:
    lower = (text or "").strip().lower()
    if not lower:
        return None
    if (
        "normal" in lower
        or lower in {"none", "no", "no check"}
        or "no check" in lower
    ):
        return "normal"
    if "checkmate" in lower:
        return "checkmate"
    if "stalemate" in lower:
        return "stalemate"
    if "check" in lower:
        return "check"
    return None


def _special_rule_sentences(text: str, keyword_pattern: str) -> list[str]:
    sentence_re = re.compile(
        r"[^.!?\n]*(?:" + keyword_pattern + r")[^.!?\n]*",
        re.IGNORECASE,
    )
    return [match.group(0).strip().lower() for match in sentence_re.finditer(text or "")]


def _is_negative_special_sentence(sentence: str) -> bool:
    return re.search(r"\b(?:no|not|none|without|unavailable)\b", sentence) is not None


def _special_rules_signature(text: str) -> dict[str, set[str]]:
    castling: set[str] = set()
    en_passant: set[str] = set()
    promotions: set[str] = set()

    for sentence in _special_rule_sentences(text, r"\bcastl(?:e|ing)\b"):
        if _is_negative_special_sentence(sentence):
            continue
        if "kingside" in sentence:
            castling.add("kingside")
        if "queenside" in sentence:
            castling.add("queenside")

    for sentence in _special_rule_sentences(text, r"\ben\s+passant\b"):
        if _is_negative_special_sentence(sentence):
            continue
        en_passant.update(move for move in _UCI_RE.findall(sentence) if len(move) == 4)

    for sentence in _special_rule_sentences(
        text,
        r"\bpromot(?:e|es|ed|ing|ion|ions)\b",
    ):
        if _is_negative_special_sentence(sentence):
            continue
        promotions.update(move for move in _UCI_RE.findall(sentence) if len(move) == 5)

    return {
        "castling": castling,
        "en_passant": en_passant,
        "promotions": promotions,
    }


def special_rules_accuracy(prediction: str, gold: str) -> float:
    """Score special-rule availability by castling sides and UCI special moves."""
    pred = _special_rules_signature(prediction)
    expected = _special_rules_signature(gold)
    return 1.0 if pred == expected else 0.0


def eval_bucket_accuracy(prediction: str, gold: str) -> float:
    """1.0 if prediction matches the gold side and evaluation bucket."""
    gold_key = _eval_bucket_key(gold)
    pred_key = _eval_bucket_key(prediction)
    return 1.0 if gold_key is not None and pred_key == gold_key else 0.0


def _eval_bucket_key(text: str) -> tuple[str, str] | None:
    """Parse side and bucket from a position-evaluation answer."""
    lower = text.strip().lower()
    if not lower:
        return None
    if "equal" in lower:
        return ("equal", "equal")
    if "white" in lower and "black" not in lower:
        side = "white"
    elif "black" in lower and "white" not in lower:
        side = "black"
    else:
        return None
    if "forced mate" in lower:
        return (side, "forced mate")
    for bucket in ("slight edge", "clear advantage", "winning", "decisive"):
        if bucket in lower:
            return (side, bucket)
    if "slight" in lower:
        return (side, "slight edge")
    return None


def normalize_prediction(prediction: str) -> str:
    """Strip chat-template wrappers and hidden thinking blocks."""
    text = prediction or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<|im_end|>", " ")
    text = text.replace("<|endoftext|>", " ")
    text = text.replace("<|assistant|>", " ")
    text = text.replace("<|user|>", " ")
    text = re.sub(r"<\|im_start\|>\s*assistant", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def format_compliance(prediction: str) -> float:
    """1.0 if valid ``<think>...</think><move>UCI</move>`` format."""
    return 1.0 if validate_think_move_format(prediction) else 0.0


def legal_move_rate(
    prediction: str,
    fen: str,
    chess960: bool = False,
) -> float | None:
    """1.0 if the tagged move is legal, None if no tag is present."""
    text = normalize_prediction(prediction)
    has_move_tag = _ANY_MOVE_TAG_RE.search(text) is not None
    uci = _extract_uci_from_move_tag(text)
    if uci is None:
        return 0.0 if has_move_tag else None
    return 1.0 if is_legal_move(fen, uci, chess960=chess960) else 0.0


def example_is_chess960(example: BenchmarkExample) -> bool:
    """Return True when benchmark metadata marks an example as Chess960."""
    metadata = example.metadata or {}
    return bool(metadata.get("is_chess960") or metadata.get("chess960_id") is not None)


def move_extraction_match(prediction: str, gold_move: str) -> float:
    """1.0 if the extracted UCI move matches the gold move."""
    uci = extract_move(prediction)
    if uci is None:
        return 0.0
    return 1.0 if uci.strip().lower() == gold_move.strip().lower() else 0.0


def _is_negated(text: str, move_start: int, move: str) -> bool:
    """Return True when nearby language rejects a mentioned candidate move."""
    global _NEGATION_RE
    if _NEGATION_RE is None:
        _NEGATION_RE = re.compile(
            r"\b(?:not|don'?t|never|avoid|reject|worse|inferior|"
            r"rather\s+than|instead\s+of)\b"
        )

    window_before = text[max(0, move_start - 30):move_start]
    if _NEGATION_RE.search(window_before):
        return True

    suffix = text[move_start + len(move):move_start + len(move) + 25]
    if re.match(r"\s+is\s+(?:not|worse|inferior|bad|wrong|weaker)", suffix):
        return True
    if re.match(r"[?!]\s*no\b", suffix):
        return True
    return False


def move_choice_match(
    prediction: str,
    gold_move: str,
    candidates: tuple[str, str] | None = None,
) -> float:
    """1.0 if the model chose the gold move in a binary-choice task."""
    gold = gold_move.strip().lower()
    pred = normalize_prediction(prediction).strip().lower()

    if candidates:
        ca, cb = candidates[0].strip().lower(), candidates[1].strip().lower()
        pos_a = [m.start() for m in re.finditer(re.escape(ca), pred)]
        pos_b = [m.start() for m in re.finditer(re.escape(cb), pred)]

        if not pos_a and not pos_b:
            return 0.0
        if pos_a and not pos_b:
            return 1.0 if ca == gold else 0.0
        if pos_b and not pos_a:
            return 1.0 if cb == gold else 0.0

        first_a, first_b = pos_a[0], pos_b[0]
        earlier = min(first_a, first_b)
        later = max(first_a, first_b)
        earlier_move = ca if first_a <= first_b else cb
        later_move = cb if first_a <= first_b else ca
        between = pred[earlier + len(earlier_move):later]

        is_preamble = (later - earlier) < 25 and re.search(r"\b(?:and|or)\b", between)
        if is_preamble:
            skip = later + len(later_move)
            next_a = next((p for p in pos_a if p >= skip), -1)
            next_b = next((p for p in pos_b if p >= skip), -1)
            if next_a >= 0 and (next_b < 0 or next_a <= next_b):
                return 1.0 if ca == gold else 0.0
            if next_b >= 0:
                return 1.0 if cb == gold else 0.0
            return 0.0

        first_move = earlier_move
        chosen = later_move if _is_negated(pred, earlier, first_move) else first_move
        return 1.0 if chosen == gold else 0.0

    tokens = _UCI_RE.findall(pred)
    if not tokens:
        return 0.0
    return 1.0 if tokens[0] == gold else 0.0


def extract_move(prediction: str) -> str | None:
    """Extract UCI from a ``<move>`` tag, bare UCI, or prose UCI."""
    return _extract_move_from_answer_text(normalize_prediction(prediction))


_extract_move = extract_move


def pass_at_k(predictions: list[str], gold_move: str) -> float:
    """1.0 if any prediction's extracted move matches gold_move."""
    gold = gold_move.strip().lower()
    for pred in predictions:
        uci = extract_move(pred)
        if uci and uci == gold:
            return 1.0
    return 0.0


def centipawn_loss(gold_cp: float, predicted_cp: float) -> float:
    """Centipawn loss when both scores use the same perspective."""
    return max(0.0, gold_cp - predicted_cp)


def threat_f1(prediction: str, gold: str) -> float:
    """F1 score for threat detection based on parsed threat sets."""
    pred_set = _parse_threat_set(prediction)
    gold_set = _parse_threat_set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _parse_threat_set(text: str) -> set[str]:
    parsed = text.strip().lower()
    if "no immediate threats" in parsed or "no " in parsed and "threat" in parsed:
        return set()
    for marker in ("threatens:", "pieces under attack:"):
        if marker in parsed:
            parsed = parsed.split(marker, 1)[1]
    return {item.strip().rstrip(".") for item in parsed.split(",") if item.strip()}


def continuation_rank(prediction: str, gold: str) -> float:
    """Reciprocal rank of the first predicted move in the gold continuation."""
    gold_moves = re.findall(r"([a-h][1-8][a-h][1-8][qrbn]?)\s*\(\d+%\)", gold)
    if not gold_moves:
        return 0.0
    pred_moves = _UCI_RE.findall(prediction.lower())
    if not pred_moves:
        return 0.0
    for pred_move in pred_moves:
        for rank, gold_move in enumerate(gold_moves, 1):
            if pred_move == gold_move:
                return 1.0 / rank
    return 0.0


def score_prediction(
    example: BenchmarkExample,
    prediction: str,
) -> dict[str, float | None]:
    """Score one prediction against one benchmark example."""
    raw_prediction = prediction
    prediction = normalize_prediction(prediction)
    metric = example.metric_type
    gold = example.gold_answer
    scores: dict[str, float | None] = {}
    if example.task_type in ("check_detection", "check_detection_960"):
        metric = "check_state"
    elif example.task_type == "side_piece_inventory":
        metric = "side_piece_inventory"
    elif example.task_type == "legality_check":
        metric = "legality_check"
    elif example.task_type == "special_rules":
        metric = "special_rules"
    elif example.task_type == "legal_moves_by_piece":
        metric = "legal_moves_by_piece"

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
    elif metric == "material_count":
        scores["primary"] = material_count_accuracy(prediction, gold)
        scores["material_piece_accuracy"] = material_piece_accuracy(prediction, gold)
    elif metric == "side_piece_inventory":
        scores["primary"] = side_piece_inventory_accuracy(prediction, gold)
    elif metric == "text_exact_match":
        scores["primary"] = text_exact_match(prediction, gold)
    elif metric == "legal_moves_by_piece":
        scores["primary"] = text_exact_match(prediction, gold)
        scores.update(legal_moves_by_piece_diagnostics(prediction, gold))
    elif metric == "board_exact_match":
        scores["primary"] = board_exact_match(prediction, gold)
    elif metric == "fen_exact_match":
        is_960 = example_is_chess960(example)
        scores["primary"] = fen_exact_match(
            prediction,
            gold,
            chess960=is_960,
        )
        scores["fen_first4"] = fen_first4_match(
            prediction,
            gold,
            chess960=is_960,
        )
    elif metric == "jaccard":
        scores["primary"] = jaccard_similarity(prediction, gold)
    elif metric == "uci_set_jaccard":
        scores["primary"] = uci_set_jaccard(prediction, gold)
    elif metric == "legality_check":
        scores["primary"] = legality_binary_accuracy(prediction, gold)
        reason_score = legality_reason_accuracy(
            prediction,
            gold,
            gold_reason_label=example.metadata.get("legality_reason_label"),
        )
        if reason_score is not None:
            scores["legality_reason_accuracy"] = reason_score
    elif metric == "check_state":
        scores["primary"] = check_state_accuracy(prediction, gold)
    elif metric == "special_rules":
        scores["primary"] = special_rules_accuracy(prediction, gold)
    elif metric == "eval_bucket":
        scores["primary"] = eval_bucket_accuracy(prediction, gold)
    elif metric == "threat_f1":
        scores["primary"] = threat_f1(prediction, gold)
    elif metric == "continuation_rank":
        scores["primary"] = continuation_rank(prediction, gold)
    else:
        scores["primary"] = exact_match(prediction, gold)

    if example.task_type in ("best_move", "puzzle_solve"):
        scores["format_compliance"] = format_compliance(raw_prediction)
        scores["legal_move"] = legal_move_rate(
            raw_prediction,
            example.fen,
            chess960=example_is_chess960(example),
        )
    return scores


def score_split(
    examples: list[BenchmarkExample],
    predictions: dict[str, str],
    acpl_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Aggregate benchmark metrics for one split."""
    task_scores: dict[str, list[float]] = defaultdict(list)
    task_secondary: dict[str, list[float]] = defaultdict(list)
    task_acpl: dict[str, list[float]] = defaultdict(list)

    for example in examples:
        pred = predictions.get(example.example_id, "")
        scores = score_prediction(example, pred)
        task_scores[example.task_type].append(float(scores.get("primary", 0.0)))

        for key in scores:
            if key == "primary":
                continue
            if scores[key] is not None:
                task_secondary[f"{example.task_type}_{key}"].append(float(scores[key]))

        if acpl_scores and example.example_id in acpl_scores:
            task_acpl[example.task_type].append(acpl_scores[example.example_id])

    result: dict[str, float] = {}
    for task_type, values in task_scores.items():
        if values:
            result[task_type] = sum(values) / len(values)
    for key, values in task_secondary.items():
        if values:
            result[key] = sum(values) / len(values)
    for task_type, values in task_acpl.items():
        if values:
            result[f"{task_type}_acpl"] = sum(values) / len(values)

    all_acpl = [value for values in task_acpl.values() for value in values]
    if all_acpl:
        result["acpl"] = sum(all_acpl) / len(all_acpl)

    all_primary = [value for values in task_scores.values() for value in values]
    if all_primary:
        result["overall"] = sum(all_primary) / len(all_primary)
    return result


def validate_oracle(examples: list[BenchmarkExample]) -> list[str]:
    """Return examples whose gold answer does not self-score as perfect."""
    failures: list[str] = []
    for example in examples:
        primary = score_prediction(example, example.gold_answer).get("primary", 0.0)
        if primary != 1.0:
            failures.append(
                f"{example.example_id} ({example.task_type}, "
                f"metric={example.metric_type}): gold self-score={primary}"
            )
    return failures


def save_benchmark(examples: list[BenchmarkExample], path: str | Path) -> None:
    """Save benchmark examples to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def load_benchmark(
    path: str | Path,
    *,
    max_examples: int | None = None,
    task_types: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
) -> list[BenchmarkExample]:
    """Load benchmark examples from JSONL, optionally stopping after a cap."""
    if max_examples is not None and max_examples <= 0:
        return []
    allowed_task_types = set(task_types) if task_types is not None else None
    examples: list[BenchmarkExample] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                example = BenchmarkExample.from_dict(json.loads(line))
                if (
                    allowed_task_types is not None
                    and example.task_type not in allowed_task_types
                ):
                    continue
                examples.append(example)
                if max_examples is not None and len(examples) >= max_examples:
                    break
    return examples


def _task_is_chess960(task_type: str, raw: dict) -> bool:
    return raw_is_chess960(raw) or task_type.endswith("_960")


def _board_from_fen(fen: str, *, chess960: bool = False) -> chess.Board:
    return chess.Board(fen, chess960=chess960)


def _cp_to_bucket(cp: int) -> str:
    abs_cp = abs(cp)
    for low, high, label in DEFAULT_EVAL_BUCKETS:
        if low <= abs_cp < high:
            if cp > 0:
                return f"White has a {label}"
            if cp < 0:
                return f"Black has a {label}"
            return label
    return "decisive"


def _legal_castling_sides(board: chess.Board) -> list[str]:
    sides: set[str] = set()
    for move in board.legal_moves:
        if not board.is_castling(move):
            continue
        if board.is_kingside_castling(move):
            sides.add("kingside")
        elif board.is_queenside_castling(move):
            sides.add("queenside")
    return [side for side in ("kingside", "queenside") if side in sides]


def _derive_check_detection(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    if board.is_checkmate():
        return "Checkmate."
    if board.is_stalemate():
        return "Stalemate."
    if board.is_check():
        return "Check."
    return "Normal position -- no check, checkmate, or stalemate."


def _derive_captures(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    captures = sorted(m.uci() for m in board.legal_moves if board.is_capture(m))
    if not captures:
        return "No captures available."
    return " ".join(captures)


def _derive_material_count(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
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


def _material_summary(board: chess.Board) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        inventory: dict[str, list[str]] = {name: [] for name in _PIECE_NAMES.values()}
        counts: dict[str, int] = {name: 0 for name in _PIECE_NAMES.values()}
        values: dict[str, int] = {name: 0 for name in _PIECE_NAMES.values()}
        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.color != color:
                continue
            name = _PIECE_NAMES[piece.piece_type]
            value = _PIECE_VALUES[piece.piece_type]
            inventory[name].append(chess.square_name(square))
            counts[name] += 1
            values[name] += value
            total += value
        summary[color_name] = {
            "inventory": inventory,
            "counts": counts,
            "values": values,
            "total": total,
        }
    return summary


def _material_inventory_vector(summary: dict[str, object]) -> str:
    inventory = summary["inventory"]
    assert isinstance(inventory, dict)
    parts = []
    for piece_type in _PIECE_ORDER:
        name = _PIECE_NAMES[piece_type]
        squares = inventory.get(name, [])
        square_text = ",".join(squares) if squares else "none"
        parts.append(f"{name}={square_text}")
    return "; ".join(parts)


def _material_count_vector_text(summary: dict[str, object]) -> str:
    counts = summary["counts"]
    assert isinstance(counts, dict)
    return "; ".join(
        f"{_PIECE_NAMES[piece_type]}={counts.get(_PIECE_NAMES[piece_type], 0)}"
        for piece_type in _PIECE_ORDER
    )


def _material_value_vector_text(summary: dict[str, object]) -> str:
    values = summary["values"]
    assert isinstance(values, dict)
    parts = [
        f"{_PIECE_NAMES[piece_type]}={values.get(_PIECE_NAMES[piece_type], 0)}"
        for piece_type in _PIECE_ORDER
    ]
    parts.append(f"total={int(summary['total'])}")
    return "; ".join(parts)


def _material_balance_sentence(white_total: int, black_total: int) -> str:
    balance = white_total - black_total
    if balance > 0:
        return f"White is up {balance} point(s) of material."
    if balance < 0:
        return f"Black is up {abs(balance)} point(s) of material."
    return "Material is equal."


def _derive_material_decomposition(
    task_type: str,
    fen: str,
    chess960: bool = False,
) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    summary = _material_summary(board)
    white = summary["white"]
    black = summary["black"]
    if task_type == "material_inventory":
        return (
            f"White inventory: {_material_inventory_vector(white)}.\n"
            f"Black inventory: {_material_inventory_vector(black)}."
        )
    if task_type == "material_piece_counts":
        return (
            f"White counts: {_material_count_vector_text(white)}.\n"
            f"Black counts: {_material_count_vector_text(black)}."
        )
    if task_type == "material_value_totals":
        return (
            f"White values: {_material_value_vector_text(white)}.\n"
            f"Black values: {_material_value_vector_text(black)}."
        )
    white_total = int(white["total"])
    black_total = int(black["total"])
    return "\n".join(
        [
            (
                "Inventory: "
                f"white {_material_inventory_vector(white)} | "
                f"black {_material_inventory_vector(black)}"
            ),
            (
                "Counts: "
                f"white {_material_count_vector_text(white)} | "
                f"black {_material_count_vector_text(black)}"
            ),
            f"Values: white total={white_total}; black total={black_total}",
            f"Balance: {_material_balance_sentence(white_total, black_total)}",
        ]
    )


def _derive_special_rules(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    parts: list[str] = []

    castling = _legal_castling_sides(board)
    if castling:
        parts.append(f"Castling available: {', '.join(castling)}.")

    if board.ep_square is not None:
        ep_moves = [
            move.uci()
            for move in board.legal_moves
            if move.to_square == board.ep_square and board.is_en_passant(move)
        ]
        if ep_moves:
            parts.append(f"En passant possible: {' '.join(ep_moves)}.")

    promotion_moves: list[str] = []
    for move in board.legal_moves:
        if move.promotion:
            promotion_moves.append(move.uci())
    if promotion_moves:
        parts.append(
            f"Promotion possible: {' '.join(sorted(promotion_moves))}."
        )

    return " ".join(parts) if parts else "No special moves available."


def _derive_hanging_pieces(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    hanging: list[str] = []

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        enemy_color = not piece.color
        friendly_color = piece.color
        if board.is_attacked_by(enemy_color, sq) and not board.is_attacked_by(friendly_color, sq):
            color_name = "white" if piece.color == chess.WHITE else "black"
            hanging.append(
                f"{color_name} {_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
            )

    if hanging:
        return f"Hanging pieces: {', '.join(hanging)}."
    return "No hanging pieces — all attacked pieces are defended."


def _derive_threats(fen: str, chess960: bool = False) -> tuple[str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    color = board.turn
    color_name = "white" if color == chess.WHITE else "black"
    opp_color = not color

    threatened: list[str] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == opp_color and board.is_attacked_by(color, sq):
            threatened.append(f"{_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}")

    if threatened:
        return f"{color_name.capitalize()} threatens: {', '.join(threatened)}.", color_name
    return f"{color_name.capitalize()} has no immediate threats.", color_name


def _derive_state_tracking(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    moves_played: list[str] = []
    for _ in range(rng.randint(1, _STATE_TRACKING_MAX_PLIES)):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        moves_played.append(move.uci())
        board.push(move)
    if not moves_played:
        return "", fen
    return " ".join(moves_played), board.fen()


def _piece_phrase(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    color_name = "white" if piece.color == chess.WHITE else "black"
    return f"{color_name} {_PIECE_NAMES[piece.piece_type]}"


def _side_name(color: bool) -> str:
    return "white" if color == chess.WHITE else "black"


def _side_piece_squares(board: chess.Board) -> list[int]:
    return [
        square
        for square in chess.SQUARES
        if (piece := board.piece_at(square)) is not None and piece.color == board.turn
    ]


def _derive_side_piece_inventory(
    fen: str,
    chess960: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    board = _board_from_fen(fen, chess960=chess960)
    inventory = [
        {
            "square": chess.square_name(square),
            "piece": _piece_phrase(board.piece_at(square)),
        }
        for square in _side_piece_squares(board)
    ]
    pieces = "; ".join(
        f"{item['square']} {item['piece']}"
        for item in inventory
    )
    if not pieces:
        pieces = "none"
    return f"Side to move: {_side_name(board.turn)}.\nPieces: {pieces}.", inventory


def _derive_piece_legal_moves(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    squares = _side_piece_squares(board)
    if not squares:
        return "", "", ""
    square = rng.choice(squares)
    square_name = chess.square_name(square)
    piece = board.piece_at(square)
    moves = sorted(move.uci() for move in board.legal_moves if move.from_square == square)
    answer = " ".join(moves) if moves else "No legal moves."
    piece_name = _PIECE_NAMES.get(piece.piece_type, "piece") if piece else "piece"
    return square_name, piece_name, answer


def _move_text(moves: list[str], *, empty: str = "none") -> str:
    return " ".join(moves) if moves else empty


def _pseudo_legal_moves_from_square(board: chess.Board, square: int) -> list[str]:
    return sorted(
        move.uci()
        for move in board.pseudo_legal_moves
        if move.from_square == square
    )


def _legal_moves_from_square(board: chess.Board, square: int) -> list[str]:
    return sorted(
        move.uci()
        for move in board.legal_moves
        if move.from_square == square
    )


def _rejected_move_text(board: chess.Board, moves: list[str]) -> str:
    if not moves:
        return "none"
    return "; ".join(
        f"{move_uci} {classify_move_legality(board, move_uci).reason_label}"
        for move_uci in moves
    )


def _select_decomposition_square(
    board: chess.Board,
    *,
    prefer_rejected: bool = False,
) -> int | None:
    candidates: list[tuple[int, list[str], list[str]]] = []
    for square in _side_piece_squares(board):
        pseudo = _pseudo_legal_moves_from_square(board, square)
        if not pseudo:
            continue
        legal = _legal_moves_from_square(board, square)
        candidates.append((square, pseudo, legal))
    if prefer_rejected:
        rejected = [
            item
            for item in candidates
            if sorted(set(item[1]) - set(item[2]))
        ]
        if rejected:
            return rejected[0][0]
    if candidates:
        return candidates[0][0]
    return None


def _derive_piece_pseudo_legal_moves(
    fen: str,
    chess960: bool = False,
) -> tuple[str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    square = _select_decomposition_square(board)
    if square is None:
        return "", ""
    square_name = chess.square_name(square)
    moves = _pseudo_legal_moves_from_square(board, square)
    return square_name, f"Pseudo-legal moves from {square_name}: {_move_text(moves)}"


def _derive_piece_legal_filter(
    fen: str,
    chess960: bool = False,
) -> tuple[str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    square = _select_decomposition_square(board, prefer_rejected=True)
    if square is None:
        return "", ""
    square_name = chess.square_name(square)
    pseudo = _pseudo_legal_moves_from_square(board, square)
    legal = _legal_moves_from_square(board, square)
    rejected = sorted(set(pseudo) - set(legal))
    return square_name, "\n".join(
        [
            f"Pseudo-legal from {square_name}: {_move_text(pseudo)}.",
            f"Legal: {_move_text(legal)}.",
            f"Rejected: {_rejected_move_text(board, rejected)}.",
        ]
    )


def _select_king_safety_move(board: chess.Board) -> str | None:
    rejected: list[str] = []
    legal: list[str] = []
    for square in _side_piece_squares(board):
        for move_uci in _pseudo_legal_moves_from_square(board, square):
            classification = classify_move_legality(board, move_uci)
            if classification.is_legal:
                legal.append(move_uci)
            else:
                rejected.append(move_uci)
    if rejected:
        return sorted(rejected)[0]
    if legal:
        return sorted(legal)[0]
    return None


def _derive_king_safety_filter(
    fen: str,
    chess960: bool = False,
) -> tuple[str, str, str, bool]:
    board = _board_from_fen(fen, chess960=chess960)
    move_uci = _select_king_safety_move(board)
    if move_uci is None:
        return "", "", "", False
    classification = classify_move_legality(board, move_uci)
    answer = "\n".join(
        [
            f"Move: {move_uci}.",
            "Pseudo-legal: yes.",
            f"King safe after move: {'yes' if classification.is_legal else 'no'}.",
            (
                "Final: legal; legal."
                if classification.is_legal
                else f"Final: illegal; {classification.reason_label}."
            ),
        ]
    )
    return move_uci, classification.reason_label, answer, classification.is_legal


def _derive_legal_moves_by_piece(
    fen: str,
    chess960: bool = False,
) -> tuple[str, dict[str, list[str]], str]:
    board = _board_from_fen(fen, chess960=chess960)
    grouped = {
        chess.square_name(square): []
        for square in _side_piece_squares(board)
    }
    for move in sorted(board.legal_moves, key=lambda item: item.uci()):
        grouped.setdefault(chess.square_name(move.from_square), []).append(move.uci())

    pieces = []
    move_lines = []
    for square in _side_piece_squares(board):
        square_name = chess.square_name(square)
        piece = board.piece_at(square)
        if piece is None:
            continue
        phrase = _piece_phrase(piece)
        pieces.append(f"{square_name} {phrase}")
        moves = grouped.get(square_name, [])
        move_lines.append(f"{square_name} {phrase}: {_move_text(moves, empty='no legal moves')}")
    legal_moves = sorted(move.uci() for move in board.legal_moves)
    answer = (
        f"Side to move: {_side_name(board.turn)}.\n"
        f"Pieces: {'; '.join(pieces) if pieces else 'none'}.\n"
        "Moves by piece:\n"
        + "\n".join(move_lines)
        + f"\nAll legal moves: {_move_text(legal_moves)}"
    )
    return answer, grouped, _move_text(legal_moves)


def _piece_fen_char(piece: chess.Piece | None) -> str:
    if piece is None:
        return "1"
    return piece.symbol()


def _fen_rank_row(board: chess.Board, rank: int) -> str:
    if rank < 1 or rank > 8:
        raise ValueError(f"rank must be in 1..8, got {rank!r}")
    return board.board_fen().split("/")[8 - rank]


def _derive_square_lookup(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    empty = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
    if occupied and (not empty or rng.random() < 0.5):
        square = rng.choice(occupied)
    else:
        square = rng.choice(empty or list(chess.SQUARES))
    square_name = chess.square_name(square)
    return square_name, f"{square_name}={_piece_phrase(board.piece_at(square))}"


def _derive_rank_lookup(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    rank = rng.randint(1, 8)
    row = _fen_rank_row(board, rank)
    return str(rank), row, f"rank {rank}: {row}"


def _square_coordinates_answer(square_name: str) -> str:
    square = chess.parse_square(square_name)
    file_name = chess.FILE_NAMES[chess.square_file(square)]
    rank_name = str(chess.square_rank(square) + 1)
    fen_row_from_top = 8 - chess.square_rank(square)
    file_index = chess.square_file(square) + 1
    return (
        f"{square_name}: file={file_name}; rank={rank_name}; "
        f"fen_row_from_top={fen_row_from_top}; file_index={file_index}"
    )


def _derive_square_coordinates(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    square, _lookup = _derive_square_lookup(fen, rng, chess960=chess960)
    return square, _square_coordinates_answer(square)


def _expand_fen_rank_row(row: str) -> list[str]:
    cells: list[str] = []
    for char in row:
        if char.isdigit():
            cells.extend(["1"] * int(char))
        else:
            cells.append(char)
    if len(cells) != 8:
        raise ValueError(f"FEN rank row must expand to 8 cells, got {row!r}")
    return cells


def _compress_fen_rank_cells(cells: list[str]) -> str:
    if len(cells) != 8:
        raise ValueError(f"FEN rank cells must contain 8 cells, got {len(cells)}")
    parts: list[str] = []
    empty_count = 0
    for cell in cells:
        if cell == "1":
            empty_count += 1
            continue
        if empty_count:
            parts.append(str(empty_count))
            empty_count = 0
        parts.append(cell)
    if empty_count:
        parts.append(str(empty_count))
    return "".join(parts)


def _format_fen_rank_expansion(rank: int, row: str) -> str:
    cells = _expand_fen_rank_row(row)
    assignments = [
        f"{file_name}={cell}"
        for file_name, cell in zip(chess.FILE_NAMES, cells, strict=True)
    ]
    return f"rank {rank}: {'; '.join(assignments)}"


def _derive_fen_rank_expansion(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str, str]:
    rank, row, _answer = _derive_rank_lookup(fen, rng, chess960=chess960)
    return rank, row, _format_fen_rank_expansion(int(rank), row)


def _apply_single_rank_cell_edit(row: str, file_name: str, after_fen: str) -> str:
    cells = _expand_fen_rank_row(row)
    cells[chess.FILE_NAMES.index(file_name)] = after_fen
    return _compress_fen_rank_cells(cells)


def _changed_square_edits(
    before: chess.Board,
    after: chess.Board,
    preferred_order: list[int] | None = None,
) -> list[dict[str, str]]:
    changed = []
    for square in chess.SQUARES:
        before_piece = before.piece_at(square)
        after_piece = after.piece_at(square)
        if before_piece == after_piece:
            continue
        changed.append({
            "square": chess.square_name(square),
            "before": _piece_phrase(before_piece),
            "after": _piece_phrase(after_piece),
            "before_fen": _piece_fen_char(before_piece),
            "after_fen": _piece_fen_char(after_piece),
        })
    if preferred_order:
        order = {square: index for index, square in enumerate(preferred_order)}
        changed.sort(
            key=lambda item: (
                order.get(chess.parse_square(item["square"]), len(order)),
                item["square"],
            )
        )
    return changed


def _format_move_square_edits(
    changed_squares: list[dict[str, str]],
) -> str:
    lookups = [
        f"{item['square']}={item['before']}"
        for item in changed_squares
    ]
    square_edits = [
        f"{item['square']} {item['before']}->{item['after']}"
        for item in changed_squares
    ]
    rank_edits = []
    for item in changed_squares:
        square = chess.parse_square(item["square"])
        file_name = chess.FILE_NAMES[chess.square_file(square)]
        rank_name = str(chess.square_rank(square) + 1)
        rank_edits.append(
            f"rank {rank_name} {file_name} {item['before_fen']}->{item['after_fen']}"
        )
    return "\n".join(
        [
            f"Lookup: {'; '.join(lookups)}.",
            f"Squares: {'; '.join(square_edits)}.",
            f"Ranks: {'; '.join(rank_edits)}.",
        ]
    )


def _rank_row_rewrites(
    before: chess.Board,
    after: chess.Board,
    changed_squares: list[dict[str, str]],
) -> list[dict[str, str]]:
    before_rows = before.board_fen().split("/")
    after_rows = after.board_fen().split("/")
    affected_ranks: list[int] = []
    for item in changed_squares:
        rank = chess.square_rank(chess.parse_square(item["square"]))
        if rank not in affected_ranks:
            affected_ranks.append(rank)

    rewrites = []
    for rank in affected_ranks:
        row_index = 7 - rank
        rewrites.append({
            "rank": str(rank + 1),
            "before": before_rows[row_index],
            "after": after_rows[row_index],
        })
    return rewrites


def _format_fen_row_application(
    rank_rewrites: list[dict[str, str]],
    result_fen: str,
) -> str:
    rewrites = [
        f"rank {item['rank']} {item['before']}->{item['after']}"
        for item in rank_rewrites
    ]
    return "\n".join(
        [
            f"Rows: {'; '.join(rewrites)}.",
            f"Result FEN: {result_fen}",
        ]
    )


def _derive_move_square_edits(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    before = _board_from_fen(fen, chess960=chess960)
    legal_moves = list(before.legal_moves)
    if not legal_moves:
        return "", ""
    move = rng.choice(legal_moves)
    after = before.copy(stack=False)
    after.push(move)
    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    return move.uci(), _format_move_square_edits(changed_squares)


def _derive_fen_rank_cell_edit(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> dict[str, str]:
    before = _board_from_fen(fen, chess960=chess960)
    legal_moves = list(before.legal_moves)
    if not legal_moves:
        return {}
    move = rng.choice(legal_moves)
    after = before.copy(stack=False)
    after.push(move)
    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    if not changed_squares:
        return {}
    changed = rng.choice(changed_squares)
    square = chess.parse_square(changed["square"])
    file_name = chess.FILE_NAMES[chess.square_file(square)]
    rank = str(chess.square_rank(square) + 1)
    before_row = _fen_rank_row(before, int(rank))
    after_row = _apply_single_rank_cell_edit(
        before_row,
        file_name,
        changed["after_fen"],
    )
    return {
        "move": move.uci(),
        "square": changed["square"],
        "rank": rank,
        "file": file_name,
        "before_row": before_row,
        "after_row": after_row,
        "before_fen": changed["before_fen"],
        "after_fen": changed["after_fen"],
        "answer": f"rank {rank}: {before_row} -> {after_row}",
    }


def _derive_fen_board_edit(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> dict[str, object]:
    before = _board_from_fen(fen, chess960=chess960)
    legal_moves = list(before.legal_moves)
    if not legal_moves:
        return {}
    move = rng.choice(legal_moves)
    after = before.copy(stack=False)
    after.push(move)
    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    if not changed_squares:
        return {}
    edit_text = "; ".join(
        f"{item['square']} {item['before_fen']}->{item['after_fen']}"
        for item in changed_squares
    )
    board_fen_after = after.board_fen()
    return {
        "move": move.uci(),
        "board_fen_before": before.board_fen(),
        "board_fen_after": board_fen_after,
        "edit_text": edit_text,
        "changed_squares": changed_squares,
        "answer": f"Result board FEN: {board_fen_after}",
    }


def _captured_piece_for_move(
    board: chess.Board,
    move: chess.Move,
) -> tuple[chess.Piece | None, str]:
    if not board.is_capture(move):
        return None, ""
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        square = move.to_square + offset
    else:
        square = move.to_square
    return board.piece_at(square), chess.square_name(square)


def _move_trace_description(board: chess.Board, move: chess.Move) -> str:
    piece = board.piece_at(move.from_square)
    if board.is_castling(move):
        side = (
            "kingside"
            if chess.square_file(move.to_square) > chess.square_file(move.from_square)
            else "queenside"
        )
        sentence = f"Move 1: {_piece_phrase(piece)} {move.uci()} ({side} castling)."
    else:
        sentence = f"Move 1: {_piece_phrase(piece)} {move.uci()}."

    captured_piece, captured_square = _captured_piece_for_move(board, move)
    if captured_piece is not None:
        sentence += f" Captures {_piece_phrase(captured_piece)} on {captured_square}."
    if move.promotion:
        promoted_piece = chess.Piece(move.promotion, piece.color if piece else board.turn)
        sentence += f" Promotes to {_piece_phrase(promoted_piece)}."
    return sentence


def _derive_fen_assembly(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str]:
    before = _board_from_fen(fen, chess960=chess960)
    legal_moves = list(before.legal_moves)
    if not legal_moves:
        return "", ""
    move = rng.choice(legal_moves)
    after = before.copy(stack=False)
    after.push(move)
    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    answer = "\n".join(
        [
            _move_trace_description(before, move),
            _format_move_square_edits(changed_squares),
            f"Result FEN: {after.fen()}",
        ]
    )
    return move.uci(), answer


def _derive_fen_row_application(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str, list[dict[str, str]]]:
    before = _board_from_fen(fen, chess960=chess960)
    legal_moves = list(before.legal_moves)
    if not legal_moves:
        return "", "", []
    move = rng.choice(legal_moves)
    after = before.copy(stack=False)
    after.push(move)
    changed_squares = _changed_square_edits(
        before,
        after,
        preferred_order=[move.from_square, move.to_square],
    )
    rank_rewrites = _rank_row_rewrites(before, after, changed_squares)
    return move.uci(), _format_fen_row_application(rank_rewrites, after.fen()), rank_rewrites


def _derive_legality_check(
    fen: str,
    rng: Random,
    chess960: bool = False,
) -> tuple[str, str, str]:
    board = _board_from_fen(fen, chess960=chess960)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "", "", ""

    if rng.random() < 0.5:
        move = rng.choice(legal_moves)
        classification = classify_move_legality(board, move.uci())
        return (
            move.uci(),
            classification.reason_label,
            format_legality_answer(classification.is_legal, classification.reason_label),
        )

    illegal = random_illegal_move_with_reason(board, rng)
    if illegal is not None:
        move_uci, reason_label = illegal
        return move_uci, reason_label, format_legality_answer(False, reason_label)

    move = rng.choice(legal_moves)
    classification = classify_move_legality(board, move.uci())
    return (
        move.uci(),
        classification.reason_label,
        format_legality_answer(classification.is_legal, classification.reason_label),
    )


def _derive_opening_continuation(raw: dict) -> str:
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
    board = _board_from_fen(fen, chess960=chess960)
    castling = _legal_castling_sides(board)
    if castling:
        return f"Castling available: {', '.join(castling)}."
    return "No castling available."


def _material_signature(board: chess.Board) -> str:
    piece_chars = {
        chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
        chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P",
    }
    order = "KQRBNP"
    white: list[str] = []
    black: list[str] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        ch = piece_chars.get(piece.piece_type, "")
        if piece.color == chess.WHITE:
            white.append(ch)
        else:
            black.append(ch)
    white.sort(key=lambda c: order.index(c) if c in order else 99)
    black.sort(key=lambda c: order.index(c) if c in order else 99)
    return "".join(white) + "".join(black)


def _answer_material_balance(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    white_mat = 0
    black_mat = 0
    white_pieces: dict[str, int] = {}
    black_pieces: dict[str, int] = {}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = _PIECE_VALUES.get(piece.piece_type, 0)
        name = _PIECE_NAMES[piece.piece_type]
        if piece.color == chess.WHITE:
            white_mat += val
            white_pieces[name] = white_pieces.get(name, 0) + 1
        else:
            black_mat += val
            black_pieces[name] = black_pieces.get(name, 0) + 1

    diff = white_mat - black_mat
    w_str = ", ".join(f"{v} {k}" for k, v in sorted(white_pieces.items()))
    b_str = ", ".join(f"{v} {k}" for k, v in sorted(black_pieces.items()))

    if diff > 0:
        balance = f"White is ahead by {diff} point(s)."
    elif diff < 0:
        balance = f"Black is ahead by {abs(diff)} point(s)."
    else:
        balance = "Material is equal."

    return f"White: {w_str} ({white_mat} pts). Black: {b_str} ({black_mat} pts). {balance}"


def _answer_position_eval(cp: int | None, mate: int | None) -> str:
    if mate is not None:
        if mate > 0:
            return "White has a decisive advantage (forced mate)."
        return "Black has a decisive advantage (forced mate)."
    if cp is not None:
        return _cp_to_bucket(cp) + "."
    return ""


def _analyze_pawn_structure(board: chess.Board) -> dict:
    result = {
        "white": {"doubled": [], "isolated": [], "passed": []},
        "black": {"doubled": [], "isolated": [], "passed": []},
    }

    for color in (chess.WHITE, chess.BLACK):
        color_name = "white" if color == chess.WHITE else "black"
        opp_color = not color
        pawns_by_file: dict[int, list[int]] = {}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                file_index = chess.square_file(sq)
                pawns_by_file.setdefault(file_index, []).append(sq)

        file_letters = "abcdefgh"
        for file_index, squares in pawns_by_file.items():
            if len(squares) > 1:
                result[color_name]["doubled"].append(file_letters[file_index])

        for file_index, squares in pawns_by_file.items():
            adjacent_files = [adj for adj in (file_index - 1, file_index + 1) if 0 <= adj <= 7]
            if not any(adj in pawns_by_file for adj in adjacent_files):
                for sq in squares:
                    result[color_name]["isolated"].append(chess.square_name(sq))

        for file_index, squares in pawns_by_file.items():
            adjacent_files = [adj for adj in (file_index - 1, file_index, file_index + 1) if 0 <= adj <= 7]
            for sq in squares:
                rank = chess.square_rank(sq)
                ahead_ranks = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
                is_passed = True
                for adj in adjacent_files:
                    for ahead_rank in ahead_ranks:
                        check_sq = chess.square(adj, ahead_rank)
                        piece = board.piece_at(check_sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color == opp_color:
                            is_passed = False
                            break
                    if not is_passed:
                        break
                if is_passed:
                    result[color_name]["passed"].append(chess.square_name(sq))

    return result


def _answer_pawn_structure(fen: str, chess960: bool = False) -> str:
    board = _board_from_fen(fen, chess960=chess960)
    analysis = _analyze_pawn_structure(board)
    parts: list[str] = []
    for color_name in ("white", "black"):
        info = analysis[color_name]
        if info["doubled"]:
            files = ", ".join(info["doubled"])
            parts.append(f"{color_name.capitalize()} has doubled pawns on file(s) {files}.")
        if info["isolated"]:
            squares = ", ".join(info["isolated"])
            parts.append(f"{color_name.capitalize()} has isolated pawn(s) on {squares}.")
        if info["passed"]:
            squares = ", ".join(info["passed"])
            parts.append(f"{color_name.capitalize()} has passed pawn(s) on {squares}.")
    return " ".join(parts) if parts else "No notable pawn structure features."


def _answer_endgame_classification(
    fen: str,
    material: str | None = None,
    chess960: bool = False,
) -> str:
    if material:
        signature = material
    else:
        signature = _material_signature(_board_from_fen(fen, chess960=chess960))
    return f"This is a {signature} endgame."


def _answer_endgame_wdl(wdl: int) -> str:
    return _WDL_LABELS.get(wdl, f"WDL value: {wdl}")


def _legal_gold_move(fen: str, move_uci: str | None, *, chess960: bool = False) -> str:
    move = (move_uci or "").strip().lower()
    if not move:
        return ""
    return move if is_legal_move(fen, move, chess960=chess960) else ""


def _legal_binary_choice_gold(raw: dict, *, chess960: bool = False) -> str:
    fen = raw.get("fen", "")
    move_a = (raw.get("move_a") or "").strip().lower()
    move_b = (raw.get("move_b") or "").strip().lower()
    better_move = (raw.get("better_move") or "").strip().lower()
    if not move_a or not move_b or better_move not in {move_a, move_b}:
        return ""
    if not is_legal_move(fen, move_a, chess960=chess960):
        return ""
    if not is_legal_move(fen, move_b, chess960=chess960):
        return ""
    return better_move


def _is_valid_empty_gold(task_type: str, raw: dict) -> bool:
    """Return True when an empty gold answer is a real oracle answer."""
    if task_type not in ("legal_moves", "legal_moves_960"):
        return False
    fen = raw.get("fen", "")
    chess960 = _task_is_chess960(task_type, raw)
    try:
        board = _board_from_fen(fen, chess960=chess960)
    except (ValueError, TypeError):
        return False
    return board.is_valid() and not any(board.legal_moves)


def derive_gold_answer(
    task_type: str,
    raw: dict,
    rng: Random,
) -> str:
    """Derive the ground-truth answer for a frozen benchmark row."""
    try:
        return _derive_gold_answer_inner(task_type, raw, rng)
    except (ValueError, TypeError, IndexError) as exc:
        logger.debug("derive_gold_answer(%r) failed: %s", task_type, exc)
        return ""


def _derive_gold_answer_inner(
    task_type: str,
    raw: dict,
    rng: Random,
) -> str:
    fen = raw.get("fen", "")
    is_960 = _task_is_chess960(task_type, raw)

    if task_type == "board_print":
        return render_ascii_board(_board_from_fen(fen, chess960=is_960))
    if task_type == "board_to_fen":
        return _board_from_fen(fen, chess960=is_960).fen()
    if task_type == "piece_id":
        board = _board_from_fen(fen, chess960=is_960)
        occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
        sq = rng.choice(occupied)
        piece = board.piece_at(sq)
        if piece is None:
            return ""
        raw["_piece_id_square"] = chess.square_name(sq)
        color_name = "white" if piece.color == chess.WHITE else "black"
        return f"{color_name} {chess.piece_name(piece.piece_type)}"
    if task_type == "material_count":
        return _derive_material_count(fen, chess960=is_960)
    if task_type in {
        "material_inventory",
        "material_piece_counts",
        "material_value_totals",
        "material_balance_trace",
    }:
        return _derive_material_decomposition(task_type, fen, chess960=is_960)
    if task_type == "square_lookup":
        square, answer = _derive_square_lookup(fen, rng, chess960=is_960)
        raw["_square_lookup_square"] = square
        return answer
    if task_type == "rank_lookup":
        rank, row, answer = _derive_rank_lookup(fen, rng, chess960=is_960)
        raw["_rank_lookup_rank"] = rank
        raw["_rank_lookup_row"] = row
        return answer
    if task_type == "square_coordinates":
        square, answer = _derive_square_coordinates(fen, rng, chess960=is_960)
        raw["_square_coordinates_square"] = square
        return answer
    if task_type == "fen_rank_expansion":
        rank, row, answer = _derive_fen_rank_expansion(fen, rng, chess960=is_960)
        raw["_fen_rank_expansion_rank"] = rank
        raw["_fen_rank_expansion_row"] = row
        return answer
    if task_type == "fen_rank_cell_edit":
        derived = _derive_fen_rank_cell_edit(fen, rng, chess960=is_960)
        if not derived:
            return ""
        for key, value in derived.items():
            if key != "answer":
                raw[f"_fen_rank_cell_edit_{key}"] = value
        return str(derived["answer"])
    if task_type == "fen_board_edit":
        derived = _derive_fen_board_edit(fen, rng, chess960=is_960)
        if not derived:
            return ""
        for key, value in derived.items():
            if key != "answer":
                raw[f"_fen_board_edit_{key}"] = value
        return str(derived["answer"])
    if task_type == "move_square_edits":
        move, answer = _derive_move_square_edits(fen, rng, chess960=is_960)
        raw["_move_square_edits_move"] = move
        return answer
    if task_type == "fen_assembly":
        move, answer = _derive_fen_assembly(fen, rng, chess960=is_960)
        raw["_fen_assembly_move"] = move
        return answer
    if task_type == "fen_row_application":
        move, answer, rank_rewrites = _derive_fen_row_application(fen, rng, chess960=is_960)
        raw["_fen_row_application_move"] = move
        raw["_fen_row_application_rank_rewrites"] = rank_rewrites
        return answer
    if task_type == "state_tracking":
        moves_str, result_fen = _derive_state_tracking(fen, rng, chess960=is_960)
        raw["_state_tracking_moves"] = moves_str
        raw["_state_tracking_result"] = result_fen
        return f"Result FEN: {result_fen}"
    if task_type in ("legal_moves", "legal_moves_960"):
        board = _board_from_fen(fen, chess960=is_960)
        legal_moves = sorted(m.uci() for m in board.legal_moves)
        side = _side_name(board.turn)
        if not legal_moves:
            return f"Side to move: {side}.\nLegal moves: No legal moves available."
        return f"Side to move: {side}.\nLegal moves: {' '.join(legal_moves)}"
    if task_type == "side_piece_inventory":
        answer, inventory = _derive_side_piece_inventory(fen, chess960=is_960)
        raw["_side_piece_inventory_side"] = _side_name(
            _board_from_fen(fen, chess960=is_960).turn
        )
        raw["_side_piece_inventory"] = inventory
        return answer
    if task_type == "piece_legal_moves":
        square, piece_name, answer = _derive_piece_legal_moves(fen, rng, chess960=is_960)
        raw["_piece_legal_moves_square"] = square
        raw["_piece_legal_moves_piece"] = piece_name
        return answer
    if task_type == "piece_pseudo_legal_moves":
        square, answer = _derive_piece_pseudo_legal_moves(fen, chess960=is_960)
        raw["_piece_pseudo_legal_moves_square"] = square
        return answer
    if task_type == "piece_legal_filter":
        square, answer = _derive_piece_legal_filter(fen, chess960=is_960)
        raw["_piece_legal_filter_square"] = square
        return answer
    if task_type == "king_safety_filter":
        move_uci, reason_label, answer, is_legal = _derive_king_safety_filter(
            fen,
            chess960=is_960,
        )
        raw["_king_safety_filter_move"] = move_uci
        raw["_king_safety_filter_reason_label"] = reason_label
        raw["_king_safety_filter_is_legal"] = is_legal
        return answer
    if task_type == "legal_moves_by_piece":
        answer, grouped, final_moves = _derive_legal_moves_by_piece(fen, chess960=is_960)
        raw["_legal_moves_by_piece_grouped"] = grouped
        raw["_legal_moves_by_piece_moves"] = final_moves
        return answer
    if task_type in ("check_detection", "check_detection_960"):
        return _derive_check_detection(fen, chess960=is_960)
    if task_type in ("captures", "capture_id"):
        return _derive_captures(fen, chess960=is_960)
    if task_type == "special_rules":
        return _derive_special_rules(fen, chess960=is_960)
    if task_type == "legality_check":
        move_uci, reason_label, answer = _derive_legality_check(fen, rng, chess960=is_960)
        raw["_legality_check_move"] = move_uci
        raw["_legality_check_reason_label"] = reason_label
        return answer
    if task_type == "hanging_pieces":
        return _derive_hanging_pieces(fen, chess960=is_960)
    if task_type == "threats":
        answer, color_name = _derive_threats(fen, chess960=is_960)
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
        return _answer_material_balance(fen, chess960=is_960)
    if task_type == "eval_bucket":
        return _answer_position_eval(raw.get("cp"), raw.get("mate"))
    if task_type == "pawn_structure":
        return _answer_pawn_structure(fen, chess960=is_960)
    if task_type == "opening_name":
        name = raw.get("name", "")
        eco = raw.get("eco", "")
        if not name or not eco:
            return ""
        _board_from_fen(fen, chess960=is_960)
        return f"{name} (ECO: {eco})"
    if task_type == "opening_continuation":
        return _derive_opening_continuation(raw)
    if task_type == "endgame_classification":
        return _answer_endgame_classification(fen, raw.get("material"), chess960=is_960)
    if task_type == "endgame_wdl":
        wdl = raw.get("wdl")
        return _answer_endgame_wdl(wdl) if wdl is not None else ""
    if task_type == "endgame_best_move":
        return _legal_gold_move(fen, raw.get("best_move"), chess960=is_960)
    if task_type == "best_move":
        return _legal_gold_move(fen, raw.get("best_move"), chess960=is_960)
    if task_type == "puzzle_solve":
        return _legal_gold_move(fen, raw.get("solution_first_move"), chess960=is_960)
    if task_type == "castling_rules_960":
        return _derive_castling_rules(fen, chess960=True)
    if task_type == "binary_choice":
        return _legal_binary_choice_gold(raw, chess960=is_960)
    return ""


def _render_prompt(task_type: str, raw: dict) -> str:
    """Render the deterministic prompt for a frozen benchmark task."""
    template = CANONICAL_PROMPTS[task_type]
    ctx = dict(raw)
    fen = ctx.get("fen", "")
    is_960 = _task_is_chess960(task_type, raw)

    if task_type in CANONICAL_START_FEN_PROMPT_TASK_TYPES and fen:
        ctx["fen"] = _canonical_start_fen_for_prompt(str(fen), chess960=is_960)
        fen = ctx["fen"]
    if fen:
        ctx = build_template_context({**ctx, "is_chess960": is_960})
    if task_type == "board_to_fen" and fen:
        ctx.setdefault("board", "")
    if task_type == "piece_id":
        ctx.setdefault("square", raw.get("_piece_id_square", "e1"))
    if task_type == "square_lookup":
        ctx.setdefault("square", raw.get("_square_lookup_square", "e1"))
    if task_type == "rank_lookup":
        ctx.setdefault("rank", raw.get("_rank_lookup_rank", "1"))
    if task_type == "square_coordinates":
        ctx.setdefault("square", raw.get("_square_coordinates_square", "e1"))
    if task_type == "fen_rank_expansion":
        ctx.setdefault("rank", raw.get("_fen_rank_expansion_rank", "1"))
        ctx.setdefault("fen_rank_row", raw.get("_fen_rank_expansion_row", "8"))
    if task_type == "fen_rank_cell_edit":
        ctx.setdefault("rank", raw.get("_fen_rank_cell_edit_rank", "1"))
        ctx.setdefault("file", raw.get("_fen_rank_cell_edit_file", "a"))
        ctx.setdefault("before_row", raw.get("_fen_rank_cell_edit_before_row", "8"))
        ctx.setdefault("before_fen", raw.get("_fen_rank_cell_edit_before_fen", "1"))
        ctx.setdefault("after_fen", raw.get("_fen_rank_cell_edit_after_fen", "1"))
    if task_type == "fen_board_edit":
        ctx.setdefault("board_fen_before", raw.get("_fen_board_edit_board_fen_before", "8/8/8/8/8/8/8/8"))
        ctx.setdefault("edit_text", raw.get("_fen_board_edit_edit_text", ""))
    if task_type == "move_square_edits":
        ctx.setdefault("move", raw.get("_move_square_edits_move", ""))
    if task_type == "fen_assembly":
        ctx.setdefault("move", raw.get("_fen_assembly_move", ""))
    if task_type == "fen_row_application":
        ctx.setdefault("move", raw.get("_fen_row_application_move", ""))
    if task_type == "state_tracking":
        ctx.setdefault("moves", raw.get("_state_tracking_moves", ""))
    if task_type == "piece_legal_moves":
        ctx.setdefault("source_square", raw.get("_piece_legal_moves_square", "e1"))
    if task_type == "piece_pseudo_legal_moves":
        ctx.setdefault("source_square", raw.get("_piece_pseudo_legal_moves_square", "e1"))
    if task_type == "piece_legal_filter":
        ctx.setdefault("source_square", raw.get("_piece_legal_filter_square", "e1"))
    if task_type == "king_safety_filter":
        ctx.setdefault("move", raw.get("_king_safety_filter_move", ""))
    if task_type == "legality_check":
        ctx.setdefault("move", raw.get("_legality_check_move", ""))
    if task_type == "threats":
        ctx.setdefault("color", raw.get("_threats_color", "white"))
    if task_type == "binary_choice":
        ctx.setdefault("move_a", raw.get("move_a", ""))
        ctx.setdefault("move_b", raw.get("move_b", ""))

    try:
        prompt = template.format(**ctx)
    except KeyError:
        return template
    if task_type in FULL_FEN_STATE_PROMPT_TASK_TYPES and fen:
        prompt = _append_full_fen_state(prompt, ctx)
    prompt = append_answer_contract(task_type, prompt)
    return prompt


def freeze_split(
    split_name: str,
    raw_examples: list[dict],
    seed: int,
) -> list[BenchmarkExample]:
    """Create frozen benchmark examples from raw eval split rows."""
    rng = Random(seed)
    task_types = SPLIT_TASK_TYPES.get(split_name, [])
    if not task_types:
        logger.warning("No task types defined for split %r", split_name)
        return []

    examples: list[BenchmarkExample] = []
    skipped = 0

    for index, raw in enumerate(raw_examples):
        if split_name == "planning":
            task_type = "puzzle_solve" if raw.get("puzzle_id") else "best_move"
            candidates = [task_type]
        else:
            primary = task_types[index % len(task_types)]
            candidates = [primary] + [task for task in task_types if task != primary]

        gold = ""
        accepted = False
        task_type = candidates[0]
        for candidate in candidates:
            gold = derive_gold_answer(candidate, raw, rng)
            if gold or _is_valid_empty_gold(candidate, raw):
                task_type = candidate
                accepted = True
                break

        if not accepted:
            skipped += 1
            continue

        metadata = {
            key: value
            for key, value in raw.items()
            if key in _META_KEYS or key.startswith("_")
        }
        if task_type in DIAGNOSTIC_TASK_TYPES:
            metadata["diagnostic"] = True
            metadata["hard_gate"] = False
            if task_type == "square_lookup":
                metadata["square"] = raw.get("_square_lookup_square", "")
            elif task_type == "rank_lookup":
                metadata["rank"] = raw.get("_rank_lookup_rank", "")
                metadata["fen_rank_row"] = raw.get("_rank_lookup_row", "")
            elif task_type == "square_coordinates":
                metadata["square"] = raw.get("_square_coordinates_square", "")
            elif task_type == "fen_rank_expansion":
                metadata["rank"] = raw.get("_fen_rank_expansion_rank", "")
                metadata["fen_rank_row"] = raw.get("_fen_rank_expansion_row", "")
            elif task_type == "fen_rank_cell_edit":
                metadata["move"] = raw.get("_fen_rank_cell_edit_move", "")
                metadata["square"] = raw.get("_fen_rank_cell_edit_square", "")
                metadata["rank"] = raw.get("_fen_rank_cell_edit_rank", "")
                metadata["file"] = raw.get("_fen_rank_cell_edit_file", "")
                metadata["before_row"] = raw.get("_fen_rank_cell_edit_before_row", "")
                metadata["after_row"] = raw.get("_fen_rank_cell_edit_after_row", "")
                metadata["before_fen"] = raw.get("_fen_rank_cell_edit_before_fen", "")
                metadata["after_fen"] = raw.get("_fen_rank_cell_edit_after_fen", "")
            elif task_type == "fen_board_edit":
                metadata["move"] = raw.get("_fen_board_edit_move", "")
                metadata["board_fen_before"] = raw.get("_fen_board_edit_board_fen_before", "")
                metadata["board_fen_after"] = raw.get("_fen_board_edit_board_fen_after", "")
                metadata["edit_text"] = raw.get("_fen_board_edit_edit_text", "")
                metadata["changed_squares"] = raw.get("_fen_board_edit_changed_squares", [])
            elif task_type == "move_square_edits":
                metadata["move"] = raw.get("_move_square_edits_move", "")
            elif task_type == "fen_assembly":
                metadata["move"] = raw.get("_fen_assembly_move", "")
            elif task_type == "fen_row_application":
                metadata["move"] = raw.get("_fen_row_application_move", "")
                metadata["rank_rewrites"] = raw.get("_fen_row_application_rank_rewrites", [])
            elif task_type == "side_piece_inventory":
                metadata["side_to_move"] = raw.get("_side_piece_inventory_side", "")
                metadata["side_piece_inventory"] = raw.get("_side_piece_inventory", [])
            elif task_type == "piece_legal_moves":
                metadata["source_square"] = raw.get("_piece_legal_moves_square", "")
                metadata["piece"] = raw.get("_piece_legal_moves_piece", "")
            elif task_type == "piece_pseudo_legal_moves":
                metadata["source_square"] = raw.get("_piece_pseudo_legal_moves_square", "")
            elif task_type == "piece_legal_filter":
                metadata["source_square"] = raw.get("_piece_legal_filter_square", "")
            elif task_type == "king_safety_filter":
                metadata["move"] = raw.get("_king_safety_filter_move", "")
                metadata["legality_reason_label"] = raw.get(
                    "_king_safety_filter_reason_label",
                    "",
                )
                metadata["expected_is_legal"] = raw.get("_king_safety_filter_is_legal", False)
            elif task_type == "legal_moves_by_piece":
                metadata["legal_moves_by_piece"] = raw.get("_legal_moves_by_piece_grouped", {})
                metadata["legal_moves"] = raw.get("_legal_moves_by_piece_moves", "")
        if task_type == "legality_check":
            metadata["move"] = raw.get("_legality_check_move", "")
            metadata["legality_reason_label"] = raw.get("_legality_check_reason_label", "")
        raw_metadata = raw.get("metadata")
        if isinstance(raw_metadata, dict):
            for key, value in raw_metadata.items():
                metadata.setdefault(key, value)
        examples.append(BenchmarkExample(
            example_id=f"{split_name}_{len(examples):05d}",
            split=split_name,
            task_type=task_type,
            fen=raw.get("fen", ""),
            prompt=_render_prompt(task_type, raw),
            gold_answer=gold,
            metric_type=TASK_METRIC_TYPE.get(task_type, "exact_match"),
            metadata=metadata,
        ))

    if skipped:
        logger.warning(
            "Split %r: %d / %d examples skipped (no viable task type). "
            "Frozen %d examples vs %d raw.",
            split_name, skipped, len(raw_examples),
            len(examples), len(raw_examples),
        )

    actual_counts: dict[str, int] = {}
    for example in examples:
        actual_counts[example.task_type] = actual_counts.get(example.task_type, 0) + 1
    for task_type in task_types:
        if actual_counts.get(task_type, 0) == 0:
            logger.warning(
                "Split %r: task type %r has 0 examples (all fell back to other types). "
                "Metric coverage for this task is missing.",
                split_name, task_type,
            )

    return examples


def freeze_and_save(
    splits: dict[str, list[dict]],
    output_dir: str | Path,
    seed: int,
    version: str = "chess-sft-eval-v1",
    clean: bool = True,
    strict_coverage: bool = True,
) -> dict:
    """Freeze eval splits into canonical benchmark JSONL files plus a manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if clean:
        for old_jsonl in out.glob("*.jsonl"):
            old_jsonl.unlink()

    manifest_path = out / "manifest.json"
    if not clean and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["timestamp"] = datetime.now(timezone.utc).isoformat()
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
        save_benchmark(frozen, out / f"{split_name}.jsonl")
        manifest["splits"][split_name] = len(frozen)
        total += len(frozen)
        all_frozen.extend(frozen)
        logger.info("Froze benchmark split %r: %d examples", split_name, len(frozen))

    if all_frozen:
        failures = validate_oracle(all_frozen)
        if failures:
            logger.error(
                "Oracle validation FAILED: %d / %d examples do not self-score 1.0",
                len(failures), len(all_frozen),
            )
            for failure in failures[:10]:
                logger.error("  %s", failure)
            raise ValueError(
                f"Oracle validation failed: {len(failures)} examples have gold answers "
                f"that don't self-score 1.0"
            )
        logger.info("Oracle validation passed: %d examples", len(all_frozen))

    coverage_gaps: list[str] = []
    for split_name in splits:
        planned = SPLIT_TASK_TYPES.get(split_name, [])
        actual = {example.task_type for example in all_frozen if example.split == split_name}
        for task_type in planned:
            if task_type not in actual:
                coverage_gaps.append(f"{split_name}/{task_type}")
    if coverage_gaps:
        msg = (
            f"Task coverage gaps: {', '.join(coverage_gaps)}. "
            f"Ensure source data covers all planned task types."
        )
        if strict_coverage:
            logger.error(
                "Task coverage gaps: %d task type(s) have 0 frozen examples: %s",
                len(coverage_gaps), ", ".join(coverage_gaps),
            )
            raise ValueError(msg)
        logger.warning(
            "Task coverage gaps (non-fatal): %d task type(s) have 0 frozen examples: %s",
            len(coverage_gaps), ", ".join(coverage_gaps),
        )

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Frozen benchmark: %d total examples, manifest at %s", total, manifest_path)
    return manifest
