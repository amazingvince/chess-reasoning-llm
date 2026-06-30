import sys
from random import Random
from pathlib import Path

from chess_llm.core.board import variant_fen_key


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KRK_FEN = "8/8/8/4k3/8/8/8/4K2R w - - 0 1"
PROMOTION_FEN = "8/P7/8/8/8/8/8/4K2k w - - 0 1"


def _clear_legacy_modules() -> None:
    for name in list(sys.modules):
        if name.startswith("pool") or name == "config" or name.startswith("config."):
            sys.modules.pop(name, None)


def test_package_fen_pool_imports_without_legacy_modules():
    _clear_legacy_modules()

    from chess_llm.sft.fen_pool import FENPool

    pool = FENPool()
    pool.add(STARTING_FEN, source="unit")

    assert pool.get_tags(STARTING_FEN) == {"source": "unit"}
    assert "pool.fen_pool" not in sys.modules
    assert "config" not in sys.modules


def test_package_fen_pool_dedups_by_canonical_fen_key():
    from chess_llm.sft.fen_pool import FENPool

    fen_a = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    fen_b = "8/8/8/8/8/8/4K3/4k3 w - - 17 42"
    pool = FENPool()

    pool.add(fen_a, source="first")
    pool.add(fen_b, source="second")

    assert len(pool) == 1
    assert pool.all_fens() == [fen_a]
    assert fen_b in pool
    assert pool.get_tags(fen_b) == {"source": "first"}
    assert pool.identity_key(fen_b) == variant_fen_key(fen_a)


def test_package_fen_pool_merges_chess960_tags_from_canonical_duplicate():
    from chess_llm.sft.fen_pool import FENPool

    fen = "bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1"
    pool = FENPool()

    pool.add(fen, source="first")
    pool.add(fen, source="second", metadata={"chess960_id": 3}, is_chess960=True)

    assert len(pool) == 1
    assert pool.all_fens() == [fen]
    assert pool.get_tags(fen) == {
        "source": "first",
        "metadata": {"chess960_id": 3},
        "is_chess960": True,
    }


def test_package_fen_pool_keeps_standard_and_chess960_identities_distinct():
    from chess_llm.sft.fen_pool import FENPool

    pool = FENPool()

    pool.add(STARTING_FEN, source="standard", is_chess960=False)
    pool.add(STARTING_FEN, source="chess960", metadata={"chess960_id": 518})

    assert len(pool) == 2
    assert pool.identity_key(STARTING_FEN) == variant_fen_key(STARTING_FEN)
    assert pool.identity_key(
        STARTING_FEN,
        metadata={"chess960_id": 518},
    ) == variant_fen_key(STARTING_FEN, chess960=True)
    assert pool.sample(10, filters={"is_chess960": True}) == [
            {
                "fen": STARTING_FEN,
                "source": "chess960",
                "metadata": {"chess960_id": 518},
                "is_chess960": True,
            }
        ]


def test_package_fen_pool_exports_metadata_only_chess960_start_position():
    from chess_llm.sft.fen_pool import FENPool

    pool = FENPool()

    pool.add(STARTING_FEN, source="chess960", metadata={"chess960_id": 518})

    assert pool.all_rows() == [
        {
            "fen": STARTING_FEN,
            "source": "chess960",
            "is_chess960": True,
            "metadata": {"chess960_id": 518},
        }
    ]


def test_package_fen_pool_all_rows_preserves_duplicate_standard_and_chess960_fen():
    from chess_llm.sft.fen_pool import FENPool

    pool = FENPool()

    pool.add(STARTING_FEN, source="standard", is_chess960=False)
    pool.add(STARTING_FEN, source="chess960", metadata={"chess960_id": 518})

    assert pool.all_rows() == [
        {
            "fen": STARTING_FEN,
            "source": "standard",
            "is_chess960": False,
        },
        {
            "fen": STARTING_FEN,
            "source": "chess960",
            "is_chess960": True,
            "metadata": {"chess960_id": 518},
        },
    ]


def test_package_fen_pool_load_dedups_canonical_variants(tmp_path):
    from chess_llm.sft.fen_pool import FENPool

    path = tmp_path / "pool.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1", "source": "first"}',
                '{"fen": "8/8/8/8/8/8/4K3/4k3 w - - 17 42", "label": "second"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pool = FENPool()
    pool.load(path)

    assert len(pool) == 1
    assert pool.get_tags("8/8/8/8/8/8/4K3/4k3 w - - 17 42") == {
        "source": "first",
        "label": "second",
    }


def test_package_fen_pool_reconstruction_preserves_merged_chess960_tags():
    from chess_llm.sft.fen_pool import FENPool

    fen = "bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1"
    pool = FENPool()

    pool.add(fen, source="standard")
    pool.add(fen, metadata={"chess960_id": 3})
    rows = [{"fen": fen, **pool.get_tags(fen)} for fen in pool.all_fens()]

    assert rows == [
        {
            "fen": fen,
            "source": "standard",
            "is_chess960": True,
            "metadata": {"chess960_id": 3},
        }
    ]


def test_package_fen_pool_save_load_roundtrip(tmp_path):
    from chess_llm.sft.fen_pool import FENPool

    path = tmp_path / "pool.jsonl"
    pool = FENPool()
    pool.add(STARTING_FEN, source="lichess")
    pool.add(KRK_FEN, source="syzygy")

    pool.save(path)
    loaded = FENPool()
    loaded.load(path)

    assert loaded.all_fens() == [STARTING_FEN, KRK_FEN]
    assert loaded.get_tags(STARTING_FEN) == {"source": "lichess"}


def test_package_fen_pool_sampling_stays_reproducible():
    from chess_llm.sft.fen_pool import FENPool

    pool = FENPool()
    pool.add(STARTING_FEN)
    pool.add(KRK_FEN)
    pool.add(PROMOTION_FEN)

    first = pool.sample(2, rng=Random(42))
    second = pool.sample(2, rng=Random(42))

    assert [row["fen"] for row in first] == [row["fen"] for row in second]


def test_legacy_fen_pool_wrapper_delegates_to_package():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.sft.fen_pool import FENPool as PackageFENPool
    from pool.fen_pool import FENPool as LegacyFENPool

    assert LegacyFENPool is PackageFENPool
