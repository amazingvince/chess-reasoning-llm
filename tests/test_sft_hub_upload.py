from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )


def test_package_hub_upload_imports_without_legacy_script_or_hf_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    sys.modules.pop("scripts.push_to_hub", None)
    sys.modules.pop("sft.make_data.scripts.push_to_hub", None)

    module = importlib.import_module("chess_llm.sft.hub_upload")

    assert module.upload_training
    assert module.upload_eval
    assert "scripts.push_to_hub" not in sys.modules
    assert "sft.make_data.scripts.push_to_hub" not in sys.modules


def test_collect_training_file_stats_counts_rows_and_tiers(tmp_path: Path):
    from chess_llm.sft.hub_upload import collect_training_file_stats

    _write_jsonl(
        tmp_path / "tier3" / "3.4_tactical_patterns.jsonl",
        [{"task": "3.4_tactical_patterns"}, {"task": "3.4_tactical_patterns"}],
    )

    stats = collect_training_file_stats(tmp_path)

    assert len(stats) == 1
    assert stats[0].task_id == "3.4_tactical_patterns"
    assert stats[0].tier == 3
    assert stats[0].rows == 2
    assert stats[0].relative_path.as_posix() == "tier3/3.4_tactical_patterns.jsonl"


def test_collect_training_file_stats_ignores_hidden_cache_jsonl(tmp_path: Path):
    from chess_llm.sft.hub_upload import collect_training_file_stats

    _write_jsonl(tmp_path / "tier1" / "1.1_fen_to_board.jsonl", [{"task": "1.1"}])
    _write_jsonl(
        tmp_path / ".training_cache" / "tier1" / "1.1_fen_to_board.stripped.jsonl",
        [{"task": "cache"}],
    )

    stats = collect_training_file_stats(tmp_path)

    assert [stat.relative_path.as_posix() for stat in stats] == [
        "tier1/1.1_fen_to_board.jsonl"
    ]


def test_stage_training_upload_copies_tiers_and_writes_dataset_card(tmp_path: Path):
    from chess_llm.sft.hub_upload import (
        collect_training_file_stats,
        stage_training_upload,
    )

    tier_root = tmp_path / "output"
    staging_dir = tmp_path / "staging"
    _write_jsonl(tier_root / "tier1" / "1.1_fen_to_board.jsonl", [{"task": "1.1"}])
    stats = collect_training_file_stats(tier_root)

    stage_training_upload(tier_root, staging_dir, stats, org="ExampleOrg")

    assert (staging_dir / "tier1" / "1.1_fen_to_board.jsonl").is_file()
    readme = (staging_dir / "README.md").read_text(encoding="utf-8")
    assert "Chess SFT Training Data" in readme
    assert "ExampleOrg/chess-sft-eval" in readme
    assert 'path: "tier1/*.jsonl"' in readme


def test_upload_eval_uses_injected_hf_api_and_skips_blocklist(tmp_path: Path):
    from chess_llm.sft.hub_upload import upload_eval

    eval_dir = tmp_path / "eval_splits"
    bench_dir = tmp_path / "benchmark"
    _write_jsonl(eval_dir / "rules.jsonl", [{"fen": "fen"}])
    (eval_dir / "blocklist.txt").write_text("fen\n", encoding="utf-8")
    _write_jsonl(bench_dir / "rules.jsonl", [{"example_id": "rules_0"}])
    (bench_dir / "manifest.json").write_text('{"version": 1}\n', encoding="utf-8")

    class FakeApi:
        def __init__(self) -> None:
            self.created: list[tuple[str, str, bool]] = []
            self.uploads: list[dict] = []

        def create_repo(self, repo_id, repo_type, exist_ok):
            self.created.append((repo_id, repo_type, exist_ok))

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            assert (folder / "eval_splits" / "rules.jsonl").is_file()
            assert not (folder / "eval_splits" / "blocklist.txt").exists()
            assert (folder / "benchmark" / "manifest.json").is_file()
            assert "Upload chess SFT eval" in kwargs["commit_message"]
            self.uploads.append(kwargs)

    api = FakeApi()

    upload_eval(
        "ExampleOrg",
        eval_dir=eval_dir,
        benchmark_dir=bench_dir,
        api_factory=lambda: api,
    )

    assert api.created == [("ExampleOrg/chess-sft-eval", "dataset", True)]
    assert api.uploads[0]["repo_id"] == "ExampleOrg/chess-sft-eval"
    assert api.uploads[0]["repo_type"] == "dataset"


def test_collect_eval_file_stats_raises_when_benchmark_dir_missing(tmp_path: Path):
    import pytest

    from chess_llm.sft.hub_upload import collect_eval_file_stats

    eval_dir = tmp_path / "eval_splits"
    _write_jsonl(eval_dir / "rules.jsonl", [{"fen": "fen"}])

    with pytest.raises(FileNotFoundError, match="benchmark dir not found"):
        collect_eval_file_stats(eval_dir, tmp_path / "missing-benchmark")


def test_collect_eval_file_stats_raises_when_benchmark_dir_empty(tmp_path: Path):
    import pytest

    from chess_llm.sft.hub_upload import collect_eval_file_stats

    eval_dir = tmp_path / "eval_splits"
    bench_dir = tmp_path / "benchmark"
    _write_jsonl(eval_dir / "rules.jsonl", [{"fen": "fen"}])
    bench_dir.mkdir()

    with pytest.raises(ValueError, match="no benchmark JSONL files"):
        collect_eval_file_stats(eval_dir, bench_dir)


def test_upload_eval_fails_instead_of_publishing_zero_benchmark_files(tmp_path: Path):
    from chess_llm.sft.hub_upload import main

    eval_dir = tmp_path / "eval_splits"
    bench_dir = tmp_path / "benchmark"
    _write_jsonl(eval_dir / "rules.jsonl", [{"fen": "fen"}])
    bench_dir.mkdir()

    exit_code = main(
        [
            "--eval",
            "--dry-run",
            "--eval-splits-dir",
            str(eval_dir),
            "--benchmark-dir",
            str(bench_dir),
        ]
    )

    assert exit_code == 1
