import json
import logging
import sys
from pathlib import Path

import chess

from chess_llm.core.board import canonical_fen_key, variant_fen_key


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _game_fens(*ucis: str) -> list[str]:
    board = chess.Board()
    fens = [board.fen()]
    for uci in ucis:
        board.push_uci(uci)
        fens.append(board.fen())
    return fens


def _clear_legacy_modules() -> None:
    for name in list(sys.modules):
        if (
            name.startswith("pool")
            or name.startswith("validation")
            or name == "config"
            or name.startswith("config.")
        ):
            sys.modules.pop(name, None)


def test_package_eval_split_imports_without_legacy_modules(tmp_path):
    _clear_legacy_modules()

    from chess_llm.sft import eval_split

    sources = {
        "perception": [{"fen": f"pos_{i}"} for i in range(6)],
        "rules": [{"fen": f"pos_{i}"} for i in range(6)],
    }

    splits = eval_split.generate_all_eval_splits(
        sources,
        seed=7,
        split_sizes={"perception": 3, "rules": 3},
    )

    assert len(splits["perception"]) == 3
    assert len(splits["rules"]) == 3
    assert {row["fen"] for row in splits["perception"]}.isdisjoint(
        row["fen"] for row in splits["rules"]
    )
    assert "pool.eval_split" not in sys.modules
    assert "validation.decontamination" not in sys.modules
    assert "config" not in sys.modules


def test_package_eval_split_uses_canonical_keys_across_splits():
    from chess_llm.sft import eval_split

    fen_a = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    fen_b = "8/8/8/8/8/8/4K3/4k3 w - - 17 42"
    sources = {
        "perception": [{"fen": fen_a}],
        "rules": [{"fen": fen_b}],
    }

    splits = eval_split.generate_all_eval_splits(
        sources,
        seed=7,
        split_sizes={"perception": 1, "rules": 1},
    )

    assert splits["perception"] == [{"fen": fen_a}]
    assert splits["rules"] == []


def test_package_eval_split_dedups_canonical_keys_within_split():
    from chess_llm.sft import eval_split

    fen_with_impossible_ep = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq d3 0 1"
    )
    fen_without_ep = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    )
    sources = {
        "rules": [
            {"fen": fen_with_impossible_ep},
            {"fen": fen_without_ep},
        ],
    }

    splits = eval_split.generate_all_eval_splits(
        sources,
        seed=7,
        split_sizes={"rules": 10},
    )

    assert len(splits["rules"]) == 1


def test_package_eval_split_round_trips_chess960_blocklist(tmp_path):
    from chess_llm.sft import eval_split

    board = chess.Board.from_chess960_pos(0)
    board.chess960 = True
    fen = board.fen()
    split_dir = tmp_path / "splits"
    splits = {"chess960": [{"fen": fen, "is_chess960": True, "chess960_id": 0}]}

    expected_key = variant_fen_key(fen, chess960=True)
    eval_split.save_eval_splits(splits, split_dir)
    loaded = eval_split.load_blocklist(split_dir / "blocklist.txt")

    assert expected_key in eval_split.build_blocklist(splits)
    assert expected_key in loaded
    assert expected_key.removeprefix("960:").split()[2] == "KQkq"


def test_package_eval_split_accepts_metadata_only_chess960_marker():
    from chess_llm.sft import eval_split

    board = chess.Board.from_chess960_pos(0)
    board.chess960 = True
    fen = board.fen()

    blocklist = eval_split.build_blocklist(
        {"chess960": [{"fen": fen, "metadata": {"chess960_id": 0}}]}
    )

    assert variant_fen_key(fen, chess960=True) in blocklist


def test_package_eval_split_keeps_standard_and_chess960_fens_separate():
    from chess_llm.sft import eval_split

    sources = {
        "standard": [{"fen": STARTING_FEN, "is_chess960": False}],
        "chess960": [{"fen": STARTING_FEN, "metadata": {"chess960_id": 518}}],
    }

    splits = eval_split.generate_all_eval_splits(
        sources,
        seed=7,
        split_sizes={"standard": 1, "chess960": 1},
    )

    assert splits["standard"] == [{"fen": STARTING_FEN, "is_chess960": False}]
    assert splits["chess960"] == [
        {"fen": STARTING_FEN, "metadata": {"chess960_id": 518}}
    ]


def test_build_blocklist_blocks_every_pool_position_from_sampled_games():
    from chess_llm.sft import eval_split

    game_fens = _game_fens("e2e4", "e7e5")
    other_fen = "k7/8/8/8/8/8/8/K7 w - - 0 1"
    lone_fen = "7k/8/8/8/8/8/8/K7 w - - 0 1"
    fen_pool = [
        {"fen": game_fens[0], "game_id": "aaaa0000bbbb1111"},
        {"fen": game_fens[1], "game_id": "aaaa0000bbbb1111"},
        {"fen": game_fens[2], "game_id": "aaaa0000bbbb1111"},
        {"fen": other_fen, "game_id": "cccc2222dddd3333"},
        {"fen": lone_fen},
    ]
    splits = {"perception": [fen_pool[1]]}

    blocklist = eval_split.build_blocklist(splits, fen_pool)

    for fen in game_fens:
        assert variant_fen_key(fen) in blocklist
    assert variant_fen_key(other_fen) not in blocklist
    assert variant_fen_key(lone_fen) not in blocklist


def test_build_blocklist_without_pool_blocks_exact_positions_only():
    from chess_llm.sft import eval_split

    game_fens = _game_fens("e2e4")
    splits = {"perception": [{"fen": game_fens[0], "game_id": "aaaa0000bbbb1111"}]}

    blocklist = eval_split.build_blocklist(splits)

    assert blocklist == frozenset({variant_fen_key(game_fens[0])})


def test_save_eval_splits_persists_game_neighbors_and_extra_keys(tmp_path):
    from chess_llm.sft import eval_split

    game_fens = _game_fens("e2e4")
    fen_pool = [
        {"fen": game_fens[0], "game_id": "aaaa0000bbbb1111"},
        {"fen": game_fens[1], "game_id": "aaaa0000bbbb1111"},
    ]
    splits = {"perception": [fen_pool[0]]}

    eval_split.save_eval_splits(
        splits,
        tmp_path,
        fen_pool=fen_pool,
        extra_blocklist_keys={"std:legacy-key"},
    )
    loaded = eval_split.load_blocklist(tmp_path / "blocklist.txt")

    assert variant_fen_key(game_fens[0]) in loaded
    assert variant_fen_key(game_fens[1]) in loaded
    assert "std:legacy-key" in loaded


def test_rules_eval_split_stratifies_check_and_terminal_positions():
    from chess_llm.sft import eval_split

    board = chess.Board()
    quiet = []
    for move in board.legal_moves:
        board.push(move)
        quiet.append({"fen": board.fen()})
        board.pop()
    check_fen = "rnbqkbnr/ppp1pppp/8/1B1p4/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2"
    checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    stalemate_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    sources = {
        "rules": quiet
        + [{"fen": check_fen}, {"fen": checkmate_fen}, {"fen": stalemate_fen}]
    }

    splits = eval_split.generate_all_eval_splits(
        sources,
        seed=7,
        split_sizes={"rules": 10},
    )

    fens = [row["fen"] for row in splits["rules"]]
    assert len(fens) == 10
    assert check_fen in fens
    assert checkmate_fen in fens
    assert stalemate_fen in fens


def test_rules_eval_split_warns_loudly_when_pool_lacks_terminal_positions(caplog):
    from chess_llm.sft import eval_split

    board = chess.Board()
    quiet = []
    for move in board.legal_moves:
        board.push(move)
        quiet.append({"fen": board.fen()})
        board.pop()

    with caplog.at_level(logging.WARNING, logger="chess_llm.sft.eval_split"):
        splits = eval_split.generate_all_eval_splits(
            sources={"rules": quiet},
            seed=7,
            split_sizes={"rules": 10},
        )

    assert len(splits["rules"]) == 10
    assert "RULES EVAL SPLIT IS UNDER-STRATIFIED" in caplog.text


def test_partition_eco_codes_keeps_missing_eco_rows_in_train():
    from chess_llm.sft.eval_split import partition_eco_codes

    rows = [
        {"fen": "with_a", "eco": "A00"},
        {"fen": "with_b", "eco": "B00"},
        {"fen": "blank", "eco": ""},
        {"fen": "missing"},
    ]

    eval_rows, train_rows = partition_eco_codes(rows, holdout_fraction=0.5, seed=1)

    assigned = eval_rows + train_rows
    train_fens = {row["fen"] for row in train_rows}
    assert sorted(row["fen"] for row in assigned) == sorted(row["fen"] for row in rows)
    assert {"blank", "missing"} <= train_fens


def test_partition_eco_codes_zero_holdout_assigns_all_rows_to_train():
    from chess_llm.sft.eval_split import partition_eco_codes

    rows = [
        {"fen": "a", "eco": "A00"},
        {"fen": "b", "eco": "B00"},
        {"fen": "c", "eco": "C00"},
    ]

    eval_rows, train_rows = partition_eco_codes(rows, holdout_fraction=0.0, seed=1)

    assert eval_rows == []
    assert sorted(row["fen"] for row in train_rows) == ["a", "b", "c"]


def test_package_decontamination_uses_canonical_fen_and_row_chess960(tmp_path):
    from chess_llm.sft.decontamination import audit_output_files

    standard_eval = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    standard_train_variant = "8/8/8/8/8/8/4K3/4k3 w - - 17 42"

    chess960_board = chess.Board.from_chess960_pos(0)
    chess960_board.chess960 = True
    chess960_fen = chess960_board.fen()

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    train_path = output_dir / "tier.jsonl"
    rows = [
        {"fen": standard_train_variant},
        {"fen": chess960_fen, "metadata": {"chess960_id": 0}},
        {"fen": "not in blocklist"},
    ]
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    blocklist = frozenset(
        {
            variant_fen_key(standard_eval),
            variant_fen_key(chess960_fen, chess960=True),
        }
    )

    report = audit_output_files(output_dir, blocklist)

    assert report == {str(train_path): [standard_train_variant, chess960_fen]}


def test_package_decontamination_does_not_cross_block_chess_variants(tmp_path):
    from chess_llm.sft.decontamination import audit_output_files

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    train_path = output_dir / "tier.jsonl"
    rows = [
        {"fen": STARTING_FEN, "is_chess960": False},
        {"fen": STARTING_FEN, "metadata": {"chess960_id": 518}},
    ]
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    blocklist = frozenset({variant_fen_key(STARTING_FEN, chess960=True)})

    assert audit_output_files(output_dir, blocklist) == {
        str(train_path): [STARTING_FEN]
    }


def test_check_no_contamination_blocks_castleless_positions_across_variants():
    from chess_llm.sft.decontamination import check_no_contamination

    castleless_fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    blocklist = frozenset({variant_fen_key(castleless_fen, chess960=True)})

    assert check_no_contamination(castleless_fen, blocklist, chess960=False) is False
    assert check_no_contamination(castleless_fen, blocklist, chess960=True) is False

    # Positions with castling rights stay variant-scoped.
    castling_blocklist = frozenset({variant_fen_key(STARTING_FEN, chess960=True)})
    assert check_no_contamination(STARTING_FEN, castling_blocklist, chess960=False) is True


def test_row_level_contamination_catches_metadata_and_answer_fens(tmp_path):
    from chess_llm.sft.decontamination import (
        audit_output_files,
        check_row_no_contamination,
    )

    blocked = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    clean = "k7/8/8/8/8/8/8/K7 w - - 0 1"
    blocklist = frozenset({variant_fen_key(blocked)})

    metadata_row = {"fen": clean, "metadata": {"result_fen": blocked}}
    answer_row = {
        "fen": clean,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {clean}"},
            {"role": "assistant", "content": f"Result FEN: {blocked}"},
        ],
    }
    clean_row = {"fen": clean, "metadata": {"result_fen": clean}}

    assert check_row_no_contamination(metadata_row, blocklist) is False
    assert check_row_no_contamination(answer_row, blocklist) is False
    assert check_row_no_contamination(clean_row, blocklist) is True

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    train_path = output_dir / "tier.jsonl"
    train_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (metadata_row, answer_row, clean_row)
        ),
        encoding="utf-8",
    )

    assert audit_output_files(output_dir, blocklist) == {
        str(train_path): [blocked, blocked]
    }


def test_legacy_decontamination_wrapper_delegates_to_package():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.sft import decontamination as package_decontamination
    from validation import decontamination as legacy_decontamination

    assert (
        legacy_decontamination.audit_output_files
        is package_decontamination.audit_output_files
    )
    assert (
        legacy_decontamination.check_no_contamination
        is package_decontamination.check_no_contamination
    )


def test_legacy_eval_split_wrapper_preserves_monkeypatchable_sizes(monkeypatch):
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.sft import eval_split as package_eval_split
    from pool import eval_split as legacy_eval_split

    monkeypatch.setattr(legacy_eval_split, "EVAL_SPLIT_SIZES", {"perception": 2})
    sources = {"perception": [{"fen": f"pos_{i}"} for i in range(10)]}

    splits = legacy_eval_split.generate_all_eval_splits(sources, seed=42)

    assert len(splits["perception"]) == 2
    assert legacy_eval_split.build_blocklist is package_eval_split.build_blocklist
    assert legacy_eval_split.partition_eco_codes is package_eval_split.partition_eco_codes
