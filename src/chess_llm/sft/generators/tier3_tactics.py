"""Tier 3: Tactics tasks (~260K examples).

3.1 Available captures
3.2 Threats
3.3 Attacked/defended square analysis
3.4 Tactical patterns (from Lichess puzzles by theme)
3.5 Hanging pieces (undefended pieces under attack)
"""

from __future__ import annotations

import re
from typing import Iterator

import chess

from chess_llm.core.hanging import select_hanging_claim_case
from chess_llm.sft.context import board_from_raw
from chess_llm.sft.generators.base import TaskGenerator
from chess_llm.sft.templates import select_template

_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}

_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# Lichess puzzle themes that describe an actual tactical motif.  Everything
# else (difficulty, length, source tags like masterVsMaster) is noise.
_TACTICAL_THEME_WHITELIST = (
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "doubleCheck",
    "sacrifice",
    "deflection",
    "attraction",
    "clearance",
    "interference",
    "zugzwang",
    "backRankMate",
    "smotheredMate",
    "hangingPiece",
    "trappedPiece",
    "exposedKing",
    "xRayAttack",
    "zwischenzug",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
)

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z0-9])")


def _humanize_theme(theme: str) -> str:
    """Turn a camelCase theme into words, e.g. 'mateIn2' -> 'mate in 2'."""
    return _CAMEL_SPLIT_RE.sub(" ", theme).lower()


def _tactical_theme_phrases(themes: list[str]) -> list[str]:
    """Filter raw puzzle themes to humanized tactical motifs."""
    return [
        _humanize_theme(theme)
        for theme in themes
        if theme in _TACTICAL_THEME_WHITELIST
    ]


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _piece_description(piece: chess.Piece | None) -> str:
    if piece is None:
        return "empty"
    return f"{_color_name(piece.color)} {_PIECE_NAMES[piece.piece_type]}"


def _format_piece_list(board: chess.Board, squares: list[int]) -> str:
    if not squares:
        return "none"
    return ", ".join(
        f"{_PIECE_NAMES[board.piece_at(square).piece_type]} on {chess.square_name(square)}"
        for square in squares
        if board.piece_at(square) is not None
    )


class AvailableCaptures(TaskGenerator):
    """Task 3.1: List all capture moves."""

    def task_id(self) -> str:
        return "3.1_available_captures"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            captures = sorted(
                m.uci() for m in board.legal_moves if board.is_capture(m)
            )
            if not captures:
                answer = "No captures available."
            else:
                answer = " ".join(captures)

            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class Threats(TaskGenerator):
    """Task 3.2: Identify threats to pieces."""

    def task_id(self) -> str:
        return "3.2_threats"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            color = board.turn
            color_name = "white" if color == chess.WHITE else "black"
            opp_color = not color

            raw["color"] = color_name
            tpl = select_template(self.task_id(), self.rng)

            # Determine if the template asks "what is {color} threatening?"
            # (answer = opponent pieces attacked) or "which {color} pieces
            # are under attack?" (answer = same-color pieces being attacked)
            tpl_lower = tpl.lower()
            if "under attack" in tpl_lower:
                # "Which {color} pieces are under attack?" ->
                # list pieces of {color} that the opponent attacks
                under_attack = []
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece and piece.color == color:
                        if board.is_attacked_by(opp_color, sq):
                            under_attack.append(
                                f"{_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
                            )
                if under_attack:
                    answer = f"{color_name.capitalize()} pieces under attack: {', '.join(under_attack)}."
                else:
                    answer = f"No {color_name} pieces are under attack."
            else:
                # "What is {color} threatening?" / generic ->
                # list opponent pieces attacked by {color}
                threatened = []
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

            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class AttackedDefended(TaskGenerator):
    """Task 3.3: Attacked/defended square analysis."""

    def task_id(self) -> str:
        return "3.3_attacked_defended"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0
        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            # Pick a square probe from this position.
            sq = self.rng.choice(chess.SQUARES)
            square_name = chess.square_name(sq)
            piece = board.piece_at(sq)

            white_attackers = sorted(board.attackers(chess.WHITE, sq))
            black_attackers = sorted(board.attackers(chess.BLACK, sq))
            defenders = sorted(board.attackers(piece.color, sq)) if piece else []
            defender_text = _format_piece_list(board, defenders) if piece else "not applicable"
            occupant = _piece_description(piece)

            answer = "\n".join(
                [
                    f"Square: {square_name}",
                    f"Occupant: {occupant}",
                    f"White attackers ({len(white_attackers)}): {_format_piece_list(board, white_attackers)}",
                    f"Black attackers ({len(black_attackers)}): {_format_piece_list(board, black_attackers)}",
                    f"Defenders ({len(defenders)}): {defender_text}",
                ]
            )
            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "source": "attacked_defended",
                    "query_square": square_name,
                    "occupant": occupant,
                    "white_attacker_count": len(white_attackers),
                    "black_attacker_count": len(black_attackers),
                    "defender_count": len(defenders),
                    "white_attackers": [
                        chess.square_name(square) for square in white_attackers
                    ],
                    "black_attackers": [
                        chess.square_name(square) for square in black_attackers
                    ],
                    "defenders": [
                        chess.square_name(square) for square in defenders
                    ],
                }
            )
            raw.update(
                {
                    "square": square_name,
                    "query_square": square_name,
                    "piece": occupant,
                    "metadata": metadata,
                }
            )
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class TacticalPatterns(TaskGenerator):
    """Task 3.4: Fork, pin, skewer, etc. from Lichess puzzles by theme."""

    def task_id(self) -> str:
        return "3.4_tactical_patterns"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        puzzles = self.config.get("puzzles", [])
        target = self.target_volume()
        count = 0

        for puzzle in puzzles:
            if count >= target:
                return
            raw = self.source_row(puzzle)
            if self.is_blocked(raw):
                continue

            themes = puzzle.get("themes", [])
            solution = puzzle.get("solution_first_move", "")
            if not solution:
                continue

            theme_phrases = _tactical_theme_phrases(themes)
            theme_str = ", ".join(theme_phrases) if theme_phrases else "tactical"
            answer = f"The tactic is {theme_str}. Best move: {solution}"

            metadata = dict(raw.get("metadata", {}))
            metadata.update({
                "themes": themes,
                "source": "lichess_puzzles",
                "rating": puzzle.get("rating", 0),
            })
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class HangingPieces(TaskGenerator):
    """Task 3.5: Undefended pieces under attack."""

    def task_id(self) -> str:
        return "3.5_hanging_pieces"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            hanging = []
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
                answer = f"Hanging pieces: {', '.join(hanging)}."
            else:
                answer = "No hanging pieces — all attacked pieces are defended."

            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class HangingPieceStatus(TaskGenerator):
    """Task 3.6: Decide whether one concrete piece is attacked, defended, and hanging."""

    def task_id(self) -> str:
        return "3.6_hanging_piece_status"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0
        bucket_order = ("hanging", "attacked_defended", "safe")

        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            candidates: dict[str, list[tuple[int, chess.Piece, bool, bool]]] = {
                "hanging": [],
                "attacked_defended": [],
                "safe": [],
            }
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None or piece.piece_type == chess.KING:
                    continue
                attacked = board.is_attacked_by(not piece.color, sq)
                defended = board.is_attacked_by(piece.color, sq)
                if attacked and not defended:
                    bucket = "hanging"
                elif attacked and defended:
                    bucket = "attacked_defended"
                else:
                    bucket = "safe"
                candidates[bucket].append((sq, piece, attacked, defended))

            desired_bucket = bucket_order[count % len(bucket_order)]
            selected = candidates.get(desired_bucket) or next(
                (values for bucket in bucket_order if (values := candidates[bucket])),
                [],
            )
            if not selected:
                continue

            sq, piece, attacked, defended = selected[0]
            square_name = chess.square_name(sq)
            color_name = "white" if piece.color == chess.WHITE else "black"
            piece_name = _PIECE_NAMES[piece.piece_type]
            piece_description = f"{color_name} {piece_name} on {square_name}"
            hanging = attacked and not defended
            status = (
                "hanging"
                if hanging
                else "attacked_defended"
                if attacked and defended
                else "safe"
            )
            answer = "\n".join(
                [
                    f"Piece: {piece_description}",
                    f"Attacked: {'yes' if attacked else 'no'}",
                    f"Defended: {'yes' if defended else 'no'}",
                    f"Hanging: {'yes' if hanging else 'no'}",
                ]
            )

            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "query_square": square_name,
                    "query_piece": piece_name,
                    "query_color": color_name,
                    "hanging_status": status,
                    "attacked": attacked,
                    "defended": defended,
                    "hanging": hanging,
                }
            )
            raw.update(
                {
                    "metadata": metadata,
                    "query_square": square_name,
                    "square": square_name,
                    "piece": piece_name,
                    "piece_description": piece_description,
                }
            )
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class HangingPieceFilter(TaskGenerator):
    """Task 3.7: Audit attacked pieces before listing true hanging pieces."""

    def task_id(self) -> str:
        return "3.7_hanging_piece_filter"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue

            audit_lines: list[str] = []
            hanging: list[str] = []
            attacked_defended_count = 0
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None or piece.piece_type == chess.KING:
                    continue
                attacked = board.is_attacked_by(not piece.color, sq)
                if not attacked:
                    continue
                defended = board.is_attacked_by(piece.color, sq)
                is_hanging = not defended
                color_name = "white" if piece.color == chess.WHITE else "black"
                description = (
                    f"{color_name} {_PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)}"
                )
                audit_lines.append(
                    f"Attacked {description}: Defended: {'yes' if defended else 'no'}; "
                    f"Hanging: {'yes' if is_hanging else 'no'}"
                )
                if is_hanging:
                    hanging.append(description)
                else:
                    attacked_defended_count += 1

            if not audit_lines:
                audit_lines.append("Attacked pieces: none")
            if hanging:
                final_line = f"Hanging pieces: {', '.join(hanging)}."
            else:
                final_line = "No hanging pieces — all attacked pieces are defended."
            answer = "\n".join([*audit_lines, final_line])

            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "attacked_piece_count": len(audit_lines)
                    if audit_lines != ["Attacked pieces: none"]
                    else 0,
                    "hanging_count": len(hanging),
                    "attacked_defended_count": attacked_defended_count,
                }
            )
            raw["metadata"] = metadata
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(raw, template_text=user_text, assistant_content=answer)
            count += 1


class HangingPieceClaimVerification(TaskGenerator):
    """Task 3.8: Verify whether one hanging-piece claim is correct."""

    def task_id(self) -> str:
        return "3.8_hanging_piece_claim_verification"

    def tier(self) -> int:
        return 3

    def generate(self) -> Iterator[dict]:
        pool = self.config.get("fen_pool", [])
        target = self.target_volume()
        count = 0

        for entry in pool:
            if count >= target:
                return
            raw = self.source_row(entry)
            if self.is_blocked(raw):
                continue
            board = board_from_raw(raw)
            if board is None:
                continue
            case = select_hanging_claim_case(board, count)
            if case is None:
                continue

            metadata = dict(raw.get("metadata", {}))
            metadata.update(
                {
                    "source": "hanging_piece_claim_verification",
                    "verification_claim": case.verification_claim,
                    "corruption_kind": case.corruption_kind,
                    "query_square": case.query_square,
                    "query_piece": case.query_piece,
                    "query_color": case.query_color,
                    "attacked": case.attacked,
                    "defended": case.defended,
                    "hanging": case.hanging,
                    "claimed_hanging": case.claimed_hanging,
                    "verification_verdict": case.verdict,
                    "hanging_status": case.hanging_status,
                    "correction": case.correction,
                }
            )
            raw.update(
                {
                    "metadata": metadata,
                    "verification_claim": case.verification_claim,
                    "corruption_kind": case.corruption_kind,
                    "query_square": case.query_square,
                    "query_piece": case.query_piece,
                    "query_color": case.query_color,
                }
            )
            tpl = select_template(self.task_id(), self.rng)
            user_text = self.render_template(raw, tpl)
            yield self.format_example(
                raw,
                template_text=user_text,
                assistant_content=case.answer,
            )
            count += 1
