import json
from pathlib import Path

from chess_llm.sft.sources.self_play import POSITION_RECORD_KEYS, load_self_play_positions


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _position(fen: str = STARTING_FEN, **overrides) -> dict:
    row = {
        "fen": fen,
        "move_played_uci": "e2e4",
        "game_phase": "opening",
        "material_balance": 0,
        "ply": 0,
        "game_id": "abc123def456abcd",
        "mover": "model",
        "model_id": "test-model",
        "run_id": "run-1",
    }
    row.update(overrides)
    return row


def _write_run(root: Path, run_id: str, rows: list) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "positions.jsonl"
    path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_missing_dir_and_none_return_empty_list(tmp_path: Path):
    assert load_self_play_positions(None) == []
    assert load_self_play_positions(tmp_path / "does-not-exist") == []


def test_dir_without_position_files_returns_empty_list(tmp_path: Path):
    (tmp_path / "run-empty").mkdir()
    assert load_self_play_positions(tmp_path) == []


def test_positions_from_multiple_run_dirs_are_merged(tmp_path: Path):
    _write_run(tmp_path, "run-a", [_position(run_id="run-a")])
    _write_run(
        tmp_path,
        "run-b",
        [_position(run_id="run-b", ply=1), _position(run_id="run-b", ply=2)],
    )

    rows = load_self_play_positions(tmp_path)

    assert len(rows) == 3
    assert sorted({row["run_id"] for row in rows}) == ["run-a", "run-b"]


def test_malformed_lines_and_missing_contract_keys_are_skipped(tmp_path: Path, caplog):
    incomplete = _position()
    incomplete.pop("mover")
    _write_run(
        tmp_path,
        "run-a",
        [
            "{not valid json",
            json.dumps(incomplete),
            '"not a dict"',
            json.dumps(_position()),
        ],
    )

    with caplog.at_level("WARNING"):
        rows = load_self_play_positions(tmp_path)

    assert len(rows) == 1
    assert all(key in rows[0] for key in POSITION_RECORD_KEYS)
    assert "malformed self-play position" in caplog.text
    assert "missing contract keys" in caplog.text
