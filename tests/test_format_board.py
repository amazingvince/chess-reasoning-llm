import chess

from chess_llm.formats import render_ascii_board
from chess_llm.sft import build_template_context


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_render_ascii_board_matches_training_diagram():
    board = chess.Board(STARTING_FEN)

    assert render_ascii_board(board) == "\n".join(
        [
            "8 r n b q k b n r",
            "7 p p p p p p p p",
            "6 . . . . . . . .",
            "5 . . . . . . . .",
            "4 . . . . . . . .",
            "3 . . . . . . . .",
            "2 P P P P P P P P",
            "1 R N B Q K B N R",
            "  a b c d e f g h",
        ]
    )


def test_build_template_context_adds_common_board_state_fields():
    context = build_template_context({"fen": AFTER_E4_FEN})

    assert context["board"].startswith("8 r n b q k b n r")
    assert context["side_to_move"] == "black"
    assert context["castling_rights"] == "KQkq"
    assert context["en_passant_square"] == "e3"


def test_build_template_context_preserves_explicit_fields_and_bad_fen():
    explicit = build_template_context({"fen": STARTING_FEN, "board": "custom"})
    invalid = build_template_context({"fen": "not a fen", "board": "custom"})

    assert explicit["board"] == "custom"
    assert invalid == {"fen": "not a fen", "board": "custom"}
