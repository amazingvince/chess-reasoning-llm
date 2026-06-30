import json
import os
from pathlib import Path

import pytest


def _example(tier: int = 1, fen: str = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"):
    return {
        "task": f"{tier}.x_test",
        "tier": tier,
        "fen": fen,
        "is_chess960": False,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"FEN: {fen}"},
            {"role": "assistant", "content": "a"},
        ],
        "metadata": {"heterogeneous": {"payload": tier}},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_package_load_tier_data_sanitizes_pipeline_columns(tmp_path):
    from chess_llm.training.data.loader import load_tier_data

    _write_jsonl(tmp_path / "tier1" / "rows.jsonl", [_example(tier=1)])

    ds = load_tier_data(1, tmp_path)

    assert len(ds) == 1
    assert set(ds.column_names) == {
        "messages",
        "task",
        "tier",
        "fen",
        "is_chess960",
        "chess960_id",
    }
    assert ds[0]["messages"][0]["role"] == "system"
    assert ds[0]["is_chess960"] is False
    assert ds[0]["chess960_id"] is None


def test_load_sanitized_jsonl_uses_local_dataset_json_loader(monkeypatch, tmp_path):
    from chess_llm.training.data import loader

    path = tmp_path / "sanitized.jsonl"
    path.write_text(json.dumps(_example()) + "\n", encoding="utf-8")
    calls = []
    sentinel = object()

    def fake_from_json(data_files):
        calls.append(data_files)
        return sentinel

    monkeypatch.setattr(loader.Dataset, "from_json", fake_from_json)

    result = loader._load_sanitized_jsonl(path)

    assert result is sentinel
    assert calls == [str(path)]


def test_package_load_tier_data_rejects_missing_training_columns(tmp_path):
    from chess_llm.training.data.loader import load_tier_data

    _write_jsonl(
        tmp_path / "tier1" / "bad.jsonl",
        [{"task": "1.x", "tier": 1, "fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1"}],
    )

    with pytest.raises(ValueError, match="missing required keys: messages"):
        load_tier_data(1, tmp_path)


def test_sanitized_cache_path_changes_when_same_size_file_content_changes(tmp_path):
    from chess_llm.training.data.loader import _sanitized_jsonl_path

    src = tmp_path / "tier1" / "rows.jsonl"
    src.parent.mkdir()
    src.write_text(json.dumps(_example(fen="8/8/8/8/8/8/4K3/4k3 w - - 0 1")), encoding="utf-8")
    first_stat = src.stat()
    first_path = _sanitized_jsonl_path(src, tmp_path / "cache")

    src.write_text(json.dumps(_example(fen="8/8/8/8/8/8/4K3/4k3 b - - 0 1")), encoding="utf-8")
    os.utime(src, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
    assert src.stat().st_size == first_stat.st_size

    second_path = _sanitized_jsonl_path(src, tmp_path / "cache")

    assert second_path != first_path
