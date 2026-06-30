import json

from chess_llm.formats.prompts import SYSTEM_PROMPT
from chess_llm.sft import (
    SftExample,
    build_sft_messages,
    build_sft_row,
    write_legacy_sft_jsonl,
)
from chess_llm.sft.context import board_from_raw


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_build_sft_messages_uses_standard_chat_roles():
    messages = build_sft_messages(
        user_prompt="FEN: ...\nChoose a move.",
        assistant_content="<move>e2e4</move>",
    )

    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "FEN: ...\nChoose a move."},
        {"role": "assistant", "content": "<move>e2e4</move>"},
    ]


def test_build_sft_row_preserves_legacy_shape_and_copies_metadata():
    metadata = {"source": "unit"}
    row = build_sft_row(
        task="7.1_best_move_selection",
        tier=7,
        fen=STARTING_FEN,
        user_prompt="FEN: ...\nChoose a move.",
        assistant_content="<think>center</think>\n<move>e2e4</move>",
        is_chess960=True,
        metadata=metadata,
    )
    metadata["source"] = "mutated"

    assert row["task"] == "7.1_best_move_selection"
    assert row["tier"] == 7
    assert row["fen"] == STARTING_FEN
    assert row["is_chess960"] is True
    assert row["metadata"] == {"source": "unit"}
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]


def test_sft_example_round_trips_to_legacy_dict():
    example = SftExample(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content="board",
        metadata={"source": "unit"},
    )

    row = example.to_dict()

    assert row == build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content="board",
        metadata={"source": "unit"},
    )


def test_write_legacy_sft_jsonl_writes_dict_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = build_sft_row(
        task="1.1_fen_to_board",
        tier=1,
        fen=STARTING_FEN,
        user_prompt="Render this board.",
        assistant_content="board",
    )

    write_legacy_sft_jsonl(path, [row])

    assert json.loads(path.read_text(encoding="utf-8")) == row


def test_board_from_raw_honors_chess960_flag_for_castling_fen():
    raw = {
        "fen": "bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1",
        "is_chess960": True,
    }

    board = board_from_raw(raw)

    assert board is not None
    assert board.is_valid()
    assert "d1c1" in {move.uci() for move in board.legal_moves}


def test_board_from_raw_rejects_parseable_but_invalid_fen():
    board = board_from_raw({"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})

    assert board is None
