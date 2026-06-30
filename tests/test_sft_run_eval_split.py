from __future__ import annotations

import importlib
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


def test_package_run_eval_split_imports_without_legacy_modules():
    for name in list(sys.modules):
        if (
            name.startswith("scripts")
            or name.startswith("pool")
            or name.startswith("validation")
            or name == "config"
            or name.startswith("config.")
        ):
            sys.modules.pop(name, None)

    module = importlib.import_module("chess_llm.sft.run_eval_split")

    assert module.main
    assert module.freeze_existing_eval_splits
    assert "scripts.run_eval_split" not in sys.modules
    assert "pool.eval_split" not in sys.modules
    assert "validation.benchmark" not in sys.modules
    assert "config" not in sys.modules


def test_freeze_existing_eval_splits_loads_jsonl_and_uses_non_strict_coverage(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import run_eval_split

    split_dir = tmp_path / "splits"
    benchmark_dir = tmp_path / "benchmark"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": "fen-a"}])
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
    monkeypatch.setattr(run_eval_split, "save_eval_splits", lambda splits, path: captured.update(saved=(splits, path)))
    monkeypatch.setattr(
        run_eval_split,
        "build_blocklist",
        lambda splits: {
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
    monkeypatch.setattr(run_eval_split, "build_blocklist", lambda _splits: frozenset())

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
    monkeypatch.setattr(run_eval_split, "build_blocklist", lambda _splits: frozenset())

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


def test_legacy_run_eval_split_import_aliases_package_module(monkeypatch):
    package_module = importlib.import_module("chess_llm.sft.run_eval_split")
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    sys.modules.pop("scripts.run_eval_split", None)

    legacy_module = importlib.import_module("scripts.run_eval_split")

    assert legacy_module is package_module
