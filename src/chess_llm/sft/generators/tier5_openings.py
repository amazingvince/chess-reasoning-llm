"""Tier 5: Openings tasks (~30K examples, multi-variant).

5.1 Opening identification (name the opening from position/moves)
5.2 Opening continuation (suggest next moves using Polyglot weights)
5.3 Opening principles (plans, character of the position)

Each generator produces 4-5 variants per opening, yielding ~10K candidates
per task from ~3,090 train openings.
"""

from __future__ import annotations

from typing import Iterator

import chess

from chess_llm.sft.context import board_from_raw, raw_is_chess960
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.templates import select_template


# Opening character taxonomy for principle descriptions
_OPENING_CHARACTER: dict[str, str] = {
    "A": "Flank opening — flexible, positional, transpositional play.",
    "B": "Semi-open game — asymmetric structures, positional imbalances.",
    "C": "Open game — tactical, piece activity, early castling.",
    "D": "Closed game — strategic, pawn chains, slower development.",
    "E": "Indian defense — hypermodern, fianchettoes, delayed center control.",
}

_PIECE_NAMES = {
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}

# Home squares for minor pieces (used to check development)
_WHITE_MINOR_HOME = {
    chess.B1, chess.G1,  # knights
    chess.C1, chess.F1,  # bishops
}
_BLACK_MINOR_HOME = {
    chess.B8, chess.G8,  # knights
    chess.C8, chess.F8,  # bishops
}


def _copy_chess960_metadata(source: dict, metadata: dict) -> None:
    if source.get("chess960_id") is not None:
        metadata["chess960_id"] = source.get("chess960_id")
    source_metadata = source.get("metadata")
    if isinstance(source_metadata, dict):
        if source_metadata.get("chess960_id") is not None:
            metadata["chess960_id"] = source_metadata.get("chess960_id")
        if source_metadata.get("is_chess960") is not None:
            metadata["is_chess960"] = bool(source_metadata.get("is_chess960"))
    if source.get("is_chess960") is not None:
        metadata["is_chess960"] = bool(source.get("is_chess960"))


# ── Board analysis helpers ─────────────────────────────────────────


def _count_developed_pieces(
    board: chess.Board, color: chess.Color
) -> tuple[int, list[str]]:
    """Count minor pieces off their home rank and describe them."""
    home_sqs = _WHITE_MINOR_HOME if color == chess.WHITE else _BLACK_MINOR_HOME
    home_rank = 0 if color == chess.WHITE else 7
    count = 0
    descriptions: list[str] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if (
            piece
            and piece.color == color
            and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
        ):
            if sq not in home_sqs and chess.square_rank(sq) != home_rank:
                name = _PIECE_NAMES[piece.piece_type]
                descriptions.append(f"{name} on {chess.square_name(sq)}")
                count += 1
    return count, descriptions


def _castling_status(board: chess.Board, color: chess.Color) -> str:
    """Describe castling status for *color*."""
    # Check if king has moved from home square (castled or lost rights)
    king_sq = board.king(color)
    if color == chess.WHITE:
        has_ks = board.has_kingside_castling_rights(chess.WHITE)
        has_qs = board.has_queenside_castling_rights(chess.WHITE)
        if king_sq == chess.G1:
            return "castled kingside"
        if king_sq == chess.C1:
            return "castled queenside"
    else:
        has_ks = board.has_kingside_castling_rights(chess.BLACK)
        has_qs = board.has_queenside_castling_rights(chess.BLACK)
        if king_sq == chess.G8:
            return "castled kingside"
        if king_sq == chess.C8:
            return "castled queenside"

    if has_ks and has_qs:
        return "can castle both sides"
    if has_ks:
        return "can castle kingside"
    if has_qs:
        return "can castle queenside"
    return "lost castling rights"


def _analyze_center(board: chess.Board) -> dict:
    """Analyze center occupation and attacks for both sides."""
    center = [chess.E4, chess.D4, chess.E5, chess.D5]
    result: dict = {"white_attacks": 0, "black_attacks": 0,
                    "white_occupies": 0, "black_occupies": 0}
    for sq in center:
        if board.is_attacked_by(chess.WHITE, sq):
            result["white_attacks"] += 1
        if board.is_attacked_by(chess.BLACK, sq):
            result["black_attacks"] += 1
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                result["white_occupies"] += 1
            else:
                result["black_occupies"] += 1
    return result


def _pawn_structure_summary(board: chess.Board) -> str:
    """Describe doubled, isolated, passed pawns and pawn chains."""
    parts: list[str] = []

    for color, color_name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
        opp = not color
        pawn_files: dict[int, list[int]] = {}  # file -> list of ranks
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                f = chess.square_file(sq)
                pawn_files.setdefault(f, []).append(chess.square_rank(sq))

        doubled: list[str] = []
        isolated: list[str] = []
        passed: list[str] = []

        for f, ranks in sorted(pawn_files.items()):
            file_letter = "abcdefgh"[f]

            # Doubled: multiple pawns on same file
            if len(ranks) > 1:
                doubled.append(f"{file_letter}-file")

            # Isolated: no friendly pawns on adjacent files
            has_neighbor = (f - 1 in pawn_files) or (f + 1 in pawn_files)
            if not has_neighbor:
                isolated.append(f"{file_letter}-file")

            # Passed: no enemy pawns on same or adjacent files ahead
            for rank in ranks:
                is_passed = True
                for adj_f in [f - 1, f, f + 1]:
                    if adj_f < 0 or adj_f > 7:
                        continue
                    for sq2 in chess.SQUARES:
                        p2 = board.piece_at(sq2)
                        if (
                            p2
                            and p2.piece_type == chess.PAWN
                            and p2.color == opp
                            and chess.square_file(sq2) == adj_f
                        ):
                            r2 = chess.square_rank(sq2)
                            if color == chess.WHITE and r2 > rank:
                                is_passed = False
                            elif color == chess.BLACK and r2 < rank:
                                is_passed = False
                if is_passed:
                    sq_name = chess.square_name(chess.square(f, rank))
                    passed.append(sq_name)

        color_parts: list[str] = []
        if doubled:
            color_parts.append(f"doubled pawns on {', '.join(doubled)}")
        if isolated:
            color_parts.append(f"isolated pawns on {', '.join(isolated)}")
        if passed:
            color_parts.append(f"passed pawns on {', '.join(passed)}")
        if not color_parts:
            color_parts.append("no notable pawn weaknesses")

        parts.append(f"{color_name}: {'; '.join(color_parts)}.")

    return " ".join(parts)


# ── 5.1 Opening Identification ─────────────────────────────────────


class OpeningIdentification(TaskGenerator):
    """Task 5.1: Name the opening from position/moves (5 variants)."""

    def task_id(self) -> str:
        return "5.1_opening_identification"

    def tier(self) -> int:
        return 5

    # -- answer builders (static) --

    @staticmethod
    def _ans_name_eco(name: str, eco: str, **_kw: object) -> str | None:
        return f"{name} (ECO: {eco})" if eco else name

    @staticmethod
    def _ans_name_only(name: str, **_kw: object) -> str | None:
        return name

    @staticmethod
    def _ans_eco_focus(name: str, eco: str, **_kw: object) -> str | None:
        if not eco:
            return None
        return f"ECO {eco}: {name}"

    @staticmethod
    def _ans_contextual(
        name: str, eco: str, eco_volume: str, **_kw: object
    ) -> str | None:
        if not eco_volume:
            return None
        character = _OPENING_CHARACTER.get(eco_volume, "")
        if not character:
            return None
        return f"{name} (ECO: {eco}). {character}"

    @staticmethod
    def _ans_from_moves(
        name: str, eco: str, moves_str: str, **_kw: object
    ) -> str | None:
        if not moves_str:
            return None
        return f"{name} (ECO: {eco})" if eco else name

    _VARIANTS: list[tuple[str, str, staticmethod]] = [
        ("name_eco", "5.1_opening_identification", _ans_name_eco),
        ("name_only", "5.1_id_name_only", _ans_name_only),
        ("eco_focus", "5.1_id_eco_focus", _ans_eco_focus),
        ("contextual", "5.1_id_contextual", _ans_contextual),
        ("from_moves", "5.1_id_from_moves", _ans_from_moves),
    ]

    def generate(self) -> Iterator[dict]:
        openings = self.config.get("openings", [])
        target = self.target_volume()
        count = 0
        cycle = 0

        while count < target:
            emitted_this_cycle = 0
            for opening in openings:
                if count >= target:
                    return
                fen = opening["fen"]
                if self.is_blocked(opening):
                    continue
                is_960 = raw_is_chess960(opening)

                name = opening.get("name", "Unknown")
                eco = opening.get("eco", "")
                eco_volume = opening.get("eco_volume", "")
                if not eco_volume and eco:
                    eco_volume = eco[0]
                uci_moves = opening.get("uci_moves", [])
                moves_str = " ".join(uci_moves) if uci_moves else ""

                for variant_tag, tpl_key, builder in self._VARIANTS:
                    if count >= target:
                        return
                    answer = builder(
                        name=name, eco=eco, eco_volume=eco_volume,
                        moves_str=moves_str,
                    )
                    if answer is None:
                        continue

                    metadata = {
                        "source": "lichess_openings",
                        "eco": eco,
                        "variant": variant_tag,
                        "cycle": cycle,
                    }
                    _copy_chess960_metadata(opening, metadata)
                    raw = {
                        "fen": fen,
                        "moves": moves_str,
                        "name": name,
                        "eco": eco,
                        "is_chess960": is_960,
                        "metadata": metadata,
                    }
                    tpl = select_template(tpl_key, self.rng)
                    user_text = self.render_template(raw, tpl)
                    yield self.format_example(
                        raw, template_text=user_text, assistant_content=answer,
                    )
                    count += 1
                    emitted_this_cycle += 1
            if emitted_this_cycle == 0:
                return
            cycle += 1


# ── 5.2 Opening Continuation ──────────────────────────────────────


class OpeningContinuation(TaskGenerator):
    """Task 5.2: Suggest next moves using Polyglot weights (4 variants)."""

    def task_id(self) -> str:
        return "5.2_opening_continuation"

    def tier(self) -> int:
        return 5

    # -- answer builders (static) --

    @staticmethod
    def _fmt_moves(moves: list[tuple[str, int]], limit: int = 5) -> list[str]:
        """Format book moves as 'uci (pct%)' strings."""
        total = sum(w for _, w in moves)
        parts: list[str] = []
        for uci, weight in moves[:limit]:
            pct = weight / total * 100 if total > 0 else 0
            parts.append(f"{uci} ({pct:.0f}%)")
        return parts

    @staticmethod
    def _ans_top_n(
        moves: list[tuple[str, int]], **_kw: object,
    ) -> str | None:
        parts = OpeningContinuation._fmt_moves(moves)
        return "Top continuations: " + ", ".join(parts) + "."

    @staticmethod
    def _ans_best_single(
        moves: list[tuple[str, int]], **_kw: object,
    ) -> str | None:
        parts = OpeningContinuation._fmt_moves(moves, limit=1)
        return f"The most popular move is {parts[0]}."

    @staticmethod
    def _ans_with_context(
        moves: list[tuple[str, int]], name: str, **_kw: object,
    ) -> str | None:
        parts = OpeningContinuation._fmt_moves(moves)
        return f"In the {name}, the main continuations are: " + ", ".join(parts) + "."

    @staticmethod
    def _ans_alternatives(
        moves: list[tuple[str, int]], **_kw: object,
    ) -> str | None:
        if len(moves) < 2:
            return None
        top_uci = moves[0][0]
        # Use full total for percentage calculation, not just the subset
        total = sum(w for _, w in moves)
        parts: list[str] = []
        for uci, weight in moves[1:5]:
            pct = weight / total * 100 if total > 0 else 0
            parts.append(f"{uci} ({pct:.0f}%)")
        return f"Besides {top_uci}, the alternatives are: " + ", ".join(parts) + "."

    _VARIANTS: list[tuple[str, str, staticmethod]] = [
        ("top_n", "5.2_opening_continuation", _ans_top_n),
        ("best_single", "5.2_cont_best_single", _ans_best_single),
        ("with_context", "5.2_cont_with_context", _ans_with_context),
        ("alternatives", "5.2_cont_alternatives", _ans_alternatives),
    ]

    def generate(self) -> Iterator[dict]:
        openings = self.config.get("openings", [])
        book_moves = self.config.get("book_moves", {})
        target = self.target_volume()
        count = 0
        cycle = 0

        while count < target:
            emitted_this_cycle = 0
            for opening in openings:
                if count >= target:
                    return
                fen = opening["fen"]
                if self.is_blocked(opening):
                    continue
                is_960 = raw_is_chess960(opening)

                name = opening.get("name", "Unknown")
                moves = book_moves.get(fen, [])
                if not moves:
                    continue

                for variant_tag, tpl_key, builder in self._VARIANTS:
                    if count >= target:
                        return
                    answer = builder(moves=moves, name=name)
                    if answer is None:
                        continue

                    metadata = {
                        "source": "polyglot_books",
                        "variant": variant_tag,
                        "cycle": cycle,
                    }
                    _copy_chess960_metadata(opening, metadata)
                    raw = {
                        "fen": fen,
                        "name": name,
                        "is_chess960": is_960,
                        "metadata": metadata,
                    }
                    tpl = select_template(tpl_key, self.rng)
                    user_text = self.render_template(raw, tpl)
                    yield self.format_example(
                        raw, template_text=user_text, assistant_content=answer,
                    )
                    count += 1
                    emitted_this_cycle += 1
            if emitted_this_cycle == 0:
                return
            cycle += 1


# ── 5.3 Opening Principles ────────────────────────────────────────


class OpeningPrinciples(TaskGenerator):
    """Task 5.3: Plans, character of the position (5 variants)."""

    def task_id(self) -> str:
        return "5.3_opening_principles"

    def tier(self) -> int:
        return 5

    # -- answer builders --

    @staticmethod
    def _ans_general(
        name: str, character: str, board: chess.Board, **_kw: object,
    ) -> str | None:
        desc_parts = [f"This is the {name}."]
        if character:
            desc_parts.append(character)

        center = _analyze_center(board)
        if center["white_attacks"] > center["black_attacks"]:
            desc_parts.append("White has more central control.")
        elif center["black_attacks"] > center["white_attacks"]:
            desc_parts.append("Black has more central control.")

        return " ".join(desc_parts)

    @staticmethod
    def _ans_white_plans(
        name: str, board: chess.Board, **_kw: object,
    ) -> str | None:
        parts: list[str] = []
        w_count, w_desc = _count_developed_pieces(board, chess.WHITE)
        if w_desc:
            parts.append(f"White has developed {w_count} piece(s): {', '.join(w_desc)}.")
        else:
            parts.append("White has not yet developed any minor pieces.")

        center = _analyze_center(board)
        if center["white_occupies"]:
            parts.append(f"White occupies {center['white_occupies']} central square(s) with pawns.")
        parts.append(f"White {_castling_status(board, chess.WHITE)}.")
        return " ".join(parts)

    @staticmethod
    def _ans_black_plans(
        name: str, board: chess.Board, **_kw: object,
    ) -> str | None:
        parts: list[str] = []
        b_count, b_desc = _count_developed_pieces(board, chess.BLACK)
        if b_desc:
            parts.append(f"Black has developed {b_count} piece(s): {', '.join(b_desc)}.")
        else:
            parts.append("Black has not yet developed any minor pieces.")

        center = _analyze_center(board)
        if center["black_occupies"]:
            parts.append(f"Black occupies {center['black_occupies']} central square(s) with pawns.")
        parts.append(f"Black {_castling_status(board, chess.BLACK)}.")
        return " ".join(parts)

    @staticmethod
    def _ans_development(
        name: str, board: chess.Board, **_kw: object,
    ) -> str | None:
        parts: list[str] = []
        for color, color_name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            cnt, desc = _count_developed_pieces(board, color)
            status = _castling_status(board, color)
            if desc:
                parts.append(f"{color_name}: {cnt} piece(s) developed ({', '.join(desc)}); {status}.")
            else:
                parts.append(f"{color_name}: no minor pieces developed; {status}.")
        return " ".join(parts)

    @staticmethod
    def _ans_pawn_structure(
        board: chess.Board, **_kw: object,
    ) -> str | None:
        return _pawn_structure_summary(board)

    _VARIANTS: list[tuple[str, str, staticmethod]] = [
        ("general", "5.3_opening_principles", _ans_general),
        ("white_plans", "5.3_white_plans", _ans_white_plans),
        ("black_plans", "5.3_black_plans", _ans_black_plans),
        ("development", "5.3_development", _ans_development),
        ("pawn_structure", "5.3_pawn_structure", _ans_pawn_structure),
    ]

    def generate(self) -> Iterator[dict]:
        openings = self.config.get("openings", [])
        target = self.target_volume()
        count = 0
        cycle = 0

        while count < target:
            emitted_this_cycle = 0
            for opening in openings:
                if count >= target:
                    return
                fen = opening["fen"]
                if self.is_blocked(opening):
                    continue
                is_960 = raw_is_chess960(opening)

                name = opening.get("name", "Unknown")
                eco = opening.get("eco", "")
                eco_volume = opening.get("eco_volume", "")
                if not eco_volume and eco:
                    eco_volume = eco[0]

                character = _OPENING_CHARACTER.get(eco_volume, "")
                board = board_from_raw(opening)
                if board is None:
                    continue

                for variant_tag, tpl_key, builder in self._VARIANTS:
                    if count >= target:
                        return
                    answer = builder(
                        name=name, character=character, board=board,
                    )
                    if answer is None:
                        continue

                    metadata = {
                        "source": "lichess_openings",
                        "eco": eco,
                        "variant": variant_tag,
                        "cycle": cycle,
                    }
                    _copy_chess960_metadata(opening, metadata)
                    raw = {
                        "fen": fen,
                        "name": name,
                        "is_chess960": is_960,
                        "metadata": metadata,
                    }
                    tpl = select_template(tpl_key, self.rng)
                    user_text = self.render_template(raw, tpl)
                    yield self.format_example(
                        raw, template_text=user_text, assistant_content=answer,
                    )
                    count += 1
                    emitted_this_cycle += 1
            if emitted_this_cycle == 0:
                return
            cycle += 1
