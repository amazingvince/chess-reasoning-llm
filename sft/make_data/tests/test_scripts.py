"""Tests for make_data command-line helper modules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import chess


def test_validate_outputs_fails_when_required_task_files_are_missing(tmp_path: Path):
    from scripts import validate_outputs

    assert validate_outputs.main(["--output-dir", str(tmp_path)]) == 1


def test_extract_polyglot_books_legacy_script_aliases_package_module():
    package_module = importlib.import_module("chess_llm.sft.extract_polyglot_books")
    legacy_module = importlib.import_module("scripts.extract_polyglot_books")

    assert legacy_module is package_module


def test_download_tablebases_legacy_script_aliases_package_module():
    package_module = importlib.import_module("chess_llm.sft.download_tablebases")
    legacy_module = importlib.import_module("scripts.download_tablebases")

    assert legacy_module is package_module


def test_syzygy_preflight_fails_when_probe_raises(monkeypatch, tmp_path: Path):
    import chess.syzygy
    from scripts import preflight_check

    (tmp_path / "dummy.rtbw").write_text("not a real tablebase")
    (tmp_path / "dummy.rtbz").write_text("not a real tablebase")
    monkeypatch.setattr(preflight_check, "SYZYGY_PATH", str(tmp_path))

    def raise_probe_error(_path):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(chess.syzygy, "open_tablebase", raise_probe_error)

    ok, detail = preflight_check.check_syzygy()

    assert ok is False
    assert "probe failed" in detail


def test_run_pipeline_eval_splits_filter_chess960_from_standard_sources(
    monkeypatch,
    tmp_path: Path,
):
    from scripts import run_pipeline

    standard = {"fen": "standard", "source": "unit"}
    chess960 = {"fen": "chess960", "metadata": {"chess960_id": 3}}
    config = {
        "fen_pool": [standard, chess960],
        "openings": [],
        "position_evals": [],
        "puzzles": [],
        "best_move_evals": [],
        "endgame_positions": [],
        "mate_rows": [],
    }
    captured: dict[str, dict] = {}

    def fake_generate(sources):
        captured["sources"] = sources
        return {
            "perception": sources["perception"],
            "rules": sources["rules"],
            "chess960": sources["chess960"],
        }

    monkeypatch.setattr(run_pipeline, "EVAL_SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(run_pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(run_pipeline, "generate_all_eval_splits", fake_generate)
    monkeypatch.setattr(run_pipeline, "save_eval_splits", lambda _splits, _path: None)
    monkeypatch.setattr(
        run_pipeline,
        "freeze_and_save",
        lambda *_args, **_kwargs: {"splits": {}},
    )
    monkeypatch.setattr(run_pipeline, "build_blocklist", lambda _splits: frozenset())

    run_pipeline.run_eval_splits(config)

    assert captured["sources"]["perception"] == [standard]
    assert captured["sources"]["rules"] == [standard]
    assert captured["sources"]["chess960"] == [chess960]


def test_run_eval_split_volume_relaxes_freeze_coverage(monkeypatch, tmp_path: Path):
    from scripts import run_eval_split
    from scripts import run_pipeline

    strict_values: list[bool | None] = []

    monkeypatch.setattr(sys, "argv", ["run_eval_split.py", "--volume", "10"])
    monkeypatch.setattr(run_eval_split, "EVAL_SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(run_eval_split, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(run_pipeline, "load_sources", lambda volume_override=None: {
        "fen_pool": [],
        "openings": [],
        "position_evals": [],
    })
    monkeypatch.setattr(
        run_eval_split,
        "generate_all_eval_splits",
        lambda _sources, seed: {"perception": []},
    )
    monkeypatch.setattr(run_eval_split, "save_eval_splits", lambda _splits, _path: None)
    monkeypatch.setattr(run_eval_split, "build_blocklist", lambda _splits: frozenset())

    def fake_freeze(_splits, _output_dir, *, seed, strict_coverage=True):
        strict_values.append(strict_coverage)
        return {"splits": {}}

    monkeypatch.setattr(run_eval_split, "freeze_and_save", fake_freeze)

    assert run_eval_split.main() == 0
    assert strict_values == [False]


def test_run_benchmark_acpl_converts_white_cp_metadata_for_black_to_move():
    """Legacy metadata cp is White-centric; ACPL compares side-to-move scores."""
    from scripts import run_benchmark
    from validation.benchmark import BenchmarkExample

    class FakeEngine:
        def analyse(self, _board, _limit):
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE),
            }

    example = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="e7e5",
        metric_type="move_extraction",
        metadata={"cp": -100},
    )

    scores = run_benchmark.compute_acpl(
        FakeEngine(),
        [example],
        {"planning_00000": "<move>e7e5</move>"},
        depth=1,
    )

    assert scores["planning_00000"] == 200.0


def test_run_benchmark_acpl_accepts_chess960_castling_prediction():
    from scripts import run_benchmark
    from validation.benchmark import BenchmarkExample

    class FakeEngine:
        def analyse(self, _board, _limit):
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE),
            }

    example = BenchmarkExample(
        example_id="chess960_00000",
        split="chess960",
        task_type="best_move",
        fen="bqrkrnnb/pppppppp/8/8/8/8/PPPPPPPP/BQRKRNNB w KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="d1c1",
        metric_type="move_extraction",
        metadata={"cp": 100, "is_chess960": True},
    )

    scores = run_benchmark.compute_acpl(
        FakeEngine(),
        [example],
        {"chess960_00000": "<move>d1c1</move>"},
        depth=1,
    )

    assert scores["chess960_00000"] == 0.0


def test_run_benchmark_example_is_chess960_accepts_chess960_id():
    from scripts import run_benchmark
    from validation.benchmark import BenchmarkExample

    example = BenchmarkExample(
        example_id="chess960_00001",
        split="chess960",
        task_type="best_move",
        fen="bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="f1g1",
        metric_type="move_extraction",
        metadata={"chess960_id": 3},
    )

    assert run_benchmark._example_is_chess960(example) is True
