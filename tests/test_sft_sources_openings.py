import chess
import sys
from pathlib import Path


def test_parse_opening_row_replays_uci_and_preserves_source_fields():
    from chess_llm.sft.sources.lichess_openings import parse_opening_row

    row = {
        "eco-volume": "B",
        "eco": "B20",
        "name": "Sicilian Defense",
        "pgn": "1. e4 c5",
        "uci": "e2e4 c7c5",
        "epd": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
    }

    parsed = parse_opening_row(row)

    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("c7c5")
    assert parsed == {
        "eco_volume": "B",
        "eco": "B20",
        "name": "Sicilian Defense",
        "pgn": "1. e4 c5",
        "uci_moves": ["e2e4", "c7c5"],
        "epd": row["epd"],
        "fen": board.fen(),
    }


def test_parse_opening_row_accepts_eco_volume_alias_and_rejects_bad_uci():
    from chess_llm.sft.sources.lichess_openings import parse_opening_row

    good = {
        "eco_volume": "A",
        "eco": "A00",
        "name": "Amar Opening",
        "pgn": "1. Nh3",
        "uci": "g1h3",
        "epd": "",
    }
    bad = {**good, "uci": "g1h3 g1f3"}

    assert parse_opening_row(good)["eco_volume"] == "A"
    assert parse_opening_row(bad) is None


def test_parse_opening_row_marks_mismatched_epd_but_keeps_uci_authoritative():
    from chess_llm.sft.sources.lichess_openings import parse_opening_row

    row = {
        "eco": "A00",
        "name": "Mismatched",
        "pgn": "1. Nh3",
        "uci": "g1h3",
        "epd": "8/8/8/8/8/8/8/8 w - -",
    }

    parsed = parse_opening_row(row)

    assert parsed["fen"] == chess.Board("rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1").fen()
    assert parsed["epd_mismatch"] is True


def test_parse_opening_row_rejects_missing_or_blank_uci():
    from chess_llm.sft.sources.lichess_openings import parse_opening_row

    assert parse_opening_row({"eco": "A00"}) is None
    assert parse_opening_row({"eco": "A00", "uci": " "}) is None


def test_load_openings_uses_injected_dataset_loader_and_counts_valid_rows():
    from chess_llm.sft.sources.lichess_openings import load_openings

    rows = [
        {"eco": "bad", "name": "Bad", "uci": "g1f3 g1h3"},
        {"eco": "A00", "name": "Amar", "uci": "g1h3"},
        {"eco": "A00", "name": "Anderssen", "uci": "a2a3"},
    ]
    calls = []

    def fake_loader(name: str, *, split: str):
        calls.append((name, split))
        return rows

    loaded = list(load_openings(max_openings=2, dataset_loader=fake_loader))

    assert calls == [("Lichess/chess-openings", "train")]
    assert [row["name"] for row in loaded] == ["Amar", "Anderssen"]


def test_legacy_openings_wrapper_delegates_to_package():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.sft.sources import lichess_openings as package_openings
    from sources import lichess_openings as legacy_openings

    assert legacy_openings.load_openings is package_openings.load_openings
    assert legacy_openings.parse_opening_row is package_openings.parse_opening_row


def test_legacy_openings_wrapper_does_not_eagerly_import_datasets(monkeypatch):
    import importlib

    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))
    sys.modules.pop("sources.lichess_openings", None)
    sys.modules.pop("datasets", None)

    module = importlib.import_module("sources.lichess_openings")

    assert module.load_openings
    assert "datasets" not in sys.modules
