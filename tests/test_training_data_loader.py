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


def test_package_load_tier_data_handles_late_chess960_ids(tmp_path):
    from chess_llm.training.data.loader import load_tier_data

    rows = [_example(tier=2) for _ in range(8)]
    chess960_row = _example(tier=2)
    chess960_row["is_chess960"] = True
    chess960_row["chess960_id"] = 518
    rows.append(chess960_row)
    _write_jsonl(tmp_path / "tier2" / "rows.jsonl", rows)

    ds = load_tier_data(2, tmp_path)

    assert ds[0]["chess960_id"] is None
    assert ds[-1]["is_chess960"] is True
    assert ds[-1]["chess960_id"] == 518


def test_write_sanitized_jsonl_applies_task_filter_and_move_only_target(tmp_path):
    from chess_llm.training.data.loader import (
        TrainingDataTransformConfig,
        _write_sanitized_jsonl,
    )

    keep = _example(tier=7)
    keep["task"] = "7.1_best_move_selection"
    keep["messages"][-1]["content"] = "<think>noisy trace</think><move>a2a3</move>"
    keep["metadata"]["target_move"] = "e2e4"
    drop = _example(tier=7)
    drop["task"] = "7.8_candidate_ratings"
    drop["metadata"]["target_move"] = "d2d4"
    src = tmp_path / "tier7" / "rows.jsonl"
    dst = tmp_path / "cache" / "rows.stripped.jsonl"
    _write_jsonl(src, [keep, drop])

    _write_sanitized_jsonl(
        src,
        dst,
        transform_config=TrainingDataTransformConfig(
            task_include=frozenset({"7.1_best_move_selection"}),
            move_only_tasks=frozenset({"7.1_best_move_selection"}),
        ),
    )

    rows = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["task"] == "7.1_best_move_selection"
    assert rows[0]["messages"][-1]["content"] == (
        "<think>Choose the final legal move.</think>\n<move>e2e4</move>"
    )
    assert "metadata" not in rows[0]


def test_sanitized_cache_path_includes_non_default_transform_config(tmp_path):
    from chess_llm.training.data.loader import (
        TrainingDataTransformConfig,
        _sanitized_jsonl_path,
    )

    src = tmp_path / "tier7" / "rows.jsonl"
    src.parent.mkdir()
    src.write_text(json.dumps(_example(tier=7)), encoding="utf-8")

    default_path = _sanitized_jsonl_path(src, tmp_path / "cache")
    filtered_path = _sanitized_jsonl_path(
        src,
        tmp_path / "cache",
        transform_config=TrainingDataTransformConfig(
            task_include=frozenset({"7.1_best_move_selection"}),
            move_only_tasks=frozenset({"7.1_best_move_selection"}),
        ),
    )

    assert filtered_path != default_path
    assert ".xf" in filtered_path.name


def test_load_sanitized_jsonl_uses_local_dataset_json_loader(monkeypatch, tmp_path):
    from chess_llm.training.data import loader

    path = tmp_path / "sanitized.jsonl"
    path.write_text(json.dumps(_example()) + "\n", encoding="utf-8")
    calls = []
    sentinel = object()

    def fake_from_json(data_files, **kwargs):
        calls.append((data_files, kwargs))
        return sentinel

    monkeypatch.setattr(loader.Dataset, "from_json", fake_from_json)

    result = loader._load_sanitized_jsonl(path)

    assert result is sentinel
    assert calls == [(str(path), {"features": loader._SANITIZED_FEATURES})]


def test_write_sanitized_jsonl_uses_unique_temp_path(monkeypatch, tmp_path):
    from chess_llm.training.data import loader

    src = tmp_path / "tier1" / "rows.jsonl"
    dst = tmp_path / "cache" / "rows.stripped.jsonl"
    _write_jsonl(src, [_example(tier=1)])
    replace_sources: list[str] = []
    original_replace = Path.replace

    def capture_replace(self, target):
        replace_sources.append(self.name)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", capture_replace)

    loader._write_sanitized_jsonl(src, dst)

    assert dst.exists()
    assert replace_sources
    assert f"{dst.name}.tmp." in replace_sources[0]


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


def test_sanitized_cache_path_includes_sanitizer_version(monkeypatch, tmp_path):
    from chess_llm.training.data import loader

    src = tmp_path / "tier1" / "rows.jsonl"
    src.parent.mkdir()
    src.write_text(json.dumps(_example()), encoding="utf-8")

    assert loader._SANITIZER_VERSION >= 2
    first_path = loader._sanitized_jsonl_path(src, tmp_path / "cache")
    assert f".v{loader._SANITIZER_VERSION}." in first_path.name

    monkeypatch.setattr(loader, "_SANITIZER_VERSION", loader._SANITIZER_VERSION + 1)
    bumped_path = loader._sanitized_jsonl_path(src, tmp_path / "cache")

    assert bumped_path != first_path
