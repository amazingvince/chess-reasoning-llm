import json
import sys
from pathlib import Path

import chess

from chess_llm.core.board import canonical_fen_key, variant_fen_key


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


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
