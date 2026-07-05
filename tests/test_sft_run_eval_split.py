from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_freeze_existing_eval_splits_loads_jsonl_and_uses_non_strict_coverage(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import run_eval_split

    split_dir = tmp_path / "splits"
    benchmark_dir = tmp_path / "benchmark"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": "fen-a"}])
    (split_dir / "blocklist.txt").write_text("std:fen-a\n", encoding="utf-8")
    (split_dir / "manifest.json").write_text(
        json.dumps({"split_sizes": {"rules": 1}}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_freeze(splits, output_dir, *, seed, strict_coverage=True):
        captured["splits"] = splits
        captured["output_dir"] = output_dir
        captured["seed"] = seed
        captured["strict_coverage"] = strict_coverage
        return {"splits": {"rules": 1}}

    monkeypatch.setattr(run_eval_split, "freeze_and_save", fake_freeze)

    assert (
        run_eval_split.freeze_existing_eval_splits(
            split_dir,
            benchmark_dir,
            split_names=("rules", "missing"),
            seed=123,
        )
        == 1
    )
    assert captured == {
        "splits": {"rules": [{"fen": "fen-a"}]},
        "output_dir": benchmark_dir,
        "seed": 123,
        "strict_coverage": False,
    }


def test_freeze_existing_eval_splits_skips_stale_and_uncovered_splits(
    monkeypatch,
    tmp_path: Path,
    caplog,
):
    import logging

    from chess_llm.sft import run_eval_split

    split_dir = tmp_path / "splits"
    benchmark_dir = tmp_path / "benchmark"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": "fen-a"}])
    _write_jsonl(split_dir / "perception.jsonl", [{"fen": "fen-uncovered"}])
    _write_jsonl(split_dir / "tactics.jsonl", [{"fen": "fen-a"}])
    (split_dir / "blocklist.txt").write_text("std:fen-a\n", encoding="utf-8")
    (split_dir / "manifest.json").write_text(
        json.dumps({"split_sizes": {"rules": 1, "perception": 1}}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_freeze(splits, _output_dir, *, seed, strict_coverage=True):
        captured["splits"] = splits
        return {"splits": {name: len(rows) for name, rows in splits.items()}}

    monkeypatch.setattr(run_eval_split, "freeze_and_save", fake_freeze)

    with caplog.at_level(logging.WARNING, logger="chess_llm.sft.run_eval_split"):
        count = run_eval_split.freeze_existing_eval_splits(
            split_dir,
            benchmark_dir,
            seed=123,
        )

    assert count == 1
    assert captured["splits"] == {"rules": [{"fen": "fen-a"}]}
    assert "Skipping stale eval split 'tactics'" in caplog.text
    assert "Skipping eval split 'perception'" in caplog.text


def test_freeze_existing_eval_splits_skips_everything_without_blocklist(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import run_eval_split

    split_dir = tmp_path / "splits"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": "fen-a"}])
    (split_dir / "manifest.json").write_text(
        json.dumps({"split_sizes": {"rules": 1}}),
        encoding="utf-8",
    )

    def fail_freeze(*_args, **_kwargs):
        raise AssertionError("uncovered splits must not be frozen")

    monkeypatch.setattr(run_eval_split, "freeze_and_save", fail_freeze)

    assert (
        run_eval_split.freeze_existing_eval_splits(
            split_dir,
            tmp_path / "benchmark",
            seed=123,
        )
        == 0
    )


def test_run_eval_split_generates_splits_and_relaxes_volume_freeze(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import run_eval_split

    strict_values: list[bool | None] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_eval_split, "EVAL_SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(run_eval_split, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(run_eval_split, "MIN_DEPTH_EVAL_BENCHMARK", 40)
    monkeypatch.setattr(run_eval_split, "MASTER_SEED", 99)
    monkeypatch.setattr(
        run_eval_split,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": "fen-a", "is_chess960": False}],
            "openings": [],
            "position_evals": [{"fen": "fen-b", "depth": 40}],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        },
    )

    def fake_generate(sources, seed, split_sizes=None):
        captured["sources"] = sources
        captured["seed"] = seed
        captured["split_sizes"] = split_sizes
        return {"perception": list(sources["perception"])[: split_sizes["perception"]]}

    monkeypatch.setattr(run_eval_split, "generate_all_eval_splits", fake_generate)
    monkeypatch.setattr(
        run_eval_split,
        "save_eval_splits",
        lambda splits, path, **_kwargs: captured.update(saved=(splits, path)),
    )
    monkeypatch.setattr(
        run_eval_split,
        "build_blocklist",
        lambda splits, *_args, **_kwargs: {
            row["fen"]
            for rows in splits.values()
            for row in rows
        },
    )

    def fake_freeze(_splits, _output_dir, *, seed, strict_coverage=True):
        strict_values.append(strict_coverage)
        return {"splits": {}}

    monkeypatch.setattr(run_eval_split, "freeze_and_save", fake_freeze)

    result = run_eval_split.generate_eval_splits(volume_override=10)

    assert result.total_eval_examples == 0
    assert result.blocklist_size == 0
    assert captured["seed"] == 99
    split_sizes = captured["split_sizes"]
    assert all(size <= 10 for size in split_sizes.values())
    assert split_sizes["perception"] == 0
    assert split_sizes["rules"] == 0
    assert strict_values == [False]


def test_run_eval_split_caps_split_targets_for_volume_smoke_run(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import run_eval_split

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        run_eval_split,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": "fen-a", "is_chess960": False}],
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
            "book_moves": {},
        },
    )

    def fake_generate(sources, *, seed, split_sizes=None):
        captured["sources"] = sources
        captured["seed"] = seed
        captured["split_sizes"] = split_sizes
        return {split_name: [] for split_name in split_sizes}

    monkeypatch.setattr(run_eval_split, "generate_all_eval_splits", fake_generate)
    monkeypatch.setattr(run_eval_split, "save_eval_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_split, "freeze_and_save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_split, "build_blocklist", lambda *_args, **_kwargs: frozenset())

    result = run_eval_split.generate_eval_splits(
        volume_override=20,
        eval_splits_dir=tmp_path / "splits",
        benchmark_dir=tmp_path / "benchmark",
        seed=123,
    )

    assert result.blocklist_size == 0
    assert captured["seed"] == 123
    split_sizes = captured["split_sizes"]
    assert all(size <= 20 for size in split_sizes.values())
    assert split_sizes["perception"] == 0
    assert split_sizes["rules"] == 0


def test_run_eval_split_main_backfills_existing_splits_without_loading_sources(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import run_eval_split

    split_dir = tmp_path / "splits"
    benchmark_dir = tmp_path / "benchmark"
    split_dir.mkdir()
    (split_dir / "blocklist.txt").write_text("fen\n", encoding="utf-8")
    monkeypatch.setattr(run_eval_split, "EVAL_SPLITS_DIR", split_dir)
    monkeypatch.setattr(run_eval_split, "BENCHMARK_DIR", benchmark_dir)
    monkeypatch.setattr(
        run_eval_split,
        "load_sources",
        lambda volume_override=None: (_ for _ in ()).throw(
            AssertionError("existing blocklist should skip source loading")
        ),
    )
    monkeypatch.setattr(run_eval_split, "freeze_existing_eval_splits", lambda *_args, **_kwargs: 0)

    assert run_eval_split.main([]) == 0


def test_run_eval_split_passes_seed_to_opening_holdout(monkeypatch, tmp_path: Path):
    from chess_llm.sft import run_eval_split

    captured: dict[str, object] = {}
    openings = [
        {"fen": "a", "eco": "A00"},
        {"fen": "b", "eco": "B00"},
        {"fen": "c", "eco": "C00"},
        {"fen": "d", "eco": "D00"},
    ]

    monkeypatch.setattr(
        run_eval_split,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [],
            "openings": list(openings),
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
            "book_moves": {},
        },
    )
    monkeypatch.setattr(
        run_eval_split,
        "generate_all_eval_splits",
        lambda sources, seed, split_sizes=None: captured.update(
            sources=sources,
            seed=seed,
            split_sizes=split_sizes,
        )
        or {},
    )
    monkeypatch.setattr(run_eval_split, "save_eval_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_split, "freeze_and_save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_split, "build_blocklist", lambda *_args, **_kwargs: frozenset())

    run_eval_split.generate_eval_splits(
        volume_override=1,
        eval_splits_dir=tmp_path / "splits",
        benchmark_dir=tmp_path / "benchmark",
        seed=1,
    )
    eval_seed_1 = [row["fen"] for row in captured["sources"]["openings"]]

    run_eval_split.generate_eval_splits(
        volume_override=1,
        eval_splits_dir=tmp_path / "splits2",
        benchmark_dir=tmp_path / "benchmark2",
        seed=2,
    )
    eval_seed_2 = [row["fen"] for row in captured["sources"]["openings"]]

    assert eval_seed_1 != eval_seed_2


def test_generate_eval_splits_writes_manifest_the_pipeline_reuses(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline, run_eval_split

    def make_config() -> dict:
        return {
            "fen_pool": [
                {"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "is_chess960": False},
                {"fen": "7k/8/8/8/8/8/8/K7 w - - 0 1", "is_chess960": False},
                {"fen": "7k/8/8/8/8/8/8/1K6 w - - 0 1", "is_chess960": False},
                {"fen": "k7/8/8/8/8/8/8/1K6 w - - 0 1", "is_chess960": False},
            ],
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
            "book_moves": {},
        }

    splits_dir = tmp_path / "splits"
    benchmark_dir = tmp_path / "benchmark"
    monkeypatch.setattr(
        run_eval_split,
        "load_sources",
        lambda volume_override=None: make_config(),
    )
    monkeypatch.setattr(run_eval_split, "freeze_and_save", lambda *_args, **_kwargs: None)

    run_eval_split.generate_eval_splits(
        volume_override=2,
        eval_splits_dir=splits_dir,
        benchmark_dir=benchmark_dir,
    )

    assert (splits_dir / "manifest.json").exists()
    assert (splits_dir / "blocklist.txt").exists()

    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", splits_dir)
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", benchmark_dir)
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "freeze_and_save", lambda *_args, **_kwargs: None)

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("pipeline must reuse the CLI-generated eval splits")

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fail_generate)

    from chess_llm.sft.eval_split import load_blocklist

    blocklist = pipeline.run_eval_splits(make_config(), volume_override=2)

    assert blocklist == load_blocklist(splits_dir / "blocklist.txt")
    assert len(blocklist) == 2


def test_run_eval_split_module_help_runs_as_python_m():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "chess_llm.sft.run_eval_split", "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate eval splits" in result.stdout
    assert "--eval-splits-dir" in result.stdout
