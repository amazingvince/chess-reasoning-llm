from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from random import Random
from pathlib import Path

import chess
import pytest

from chess_llm.core.board import variant_fen_key
from chess_llm.sft.source_preparation import build_eval_split_sources


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _valid_example(task: str, tier: int, idx: int = 0) -> dict:
    return {
        "task": task,
        "tier": tier,
        "fen": STARTING_FEN,
        "is_chess960": False,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"FEN: {STARTING_FEN}"},
            {"role": "assistant", "content": f"answer {idx}"},
        ],
        "metadata": {},
    }


def test_package_pipeline_imports_without_legacy_script_module():
    sys.modules.pop("scripts.run_pipeline", None)
    sys.modules.pop("sft.make_data.scripts.run_pipeline", None)

    module = importlib.import_module("chess_llm.sft.pipeline")

    assert module.run_tier
    assert module.load_sources
    assert module.main
    assert "scripts.run_pipeline" not in sys.modules
    assert "sft.make_data.scripts.run_pipeline" not in sys.modules


def test_package_run_tier_regenerates_incomplete_existing_output(monkeypatch, tmp_path: Path):
    from chess_llm.sft.output import PipelineStats
    from chess_llm.sft import pipeline

    class CompleteGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.1_fake_complete"

        def target_volume(self) -> int:
            return 2

        def generate(self):
            yield _valid_example(self.task_id(), 9, 0)
            yield _valid_example(self.task_id(), 9, 1)

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.1_fake_complete.jsonl"
    output_path.write_text(json.dumps(_valid_example("9.1_fake_complete", 9, 0)) + "\n")

    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, [CompleteGenerator])

    pipeline.run_tier(9, {}, frozenset(), PipelineStats())

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2


def test_pipeline_stats_for_volume_override_reports_effective_targets():
    from chess_llm.sft import pipeline

    stats = pipeline._pipeline_stats_for_volume_override(50)
    stats.record("1.6_square_lookup")

    report = stats.report()

    assert "1.6_square_lookup" in report
    assert "       50 " in report


def test_package_run_tier_regenerates_existing_output_contaminated_by_blocklist(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.core.board import variant_fen_key
    from chess_llm.sft.output import PipelineStats
    from chess_llm.sft import pipeline

    clean_fen = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"

    class CleanGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.3_fake_clean"

        def target_volume(self) -> int:
            return 1

        def generate(self):
            yield _valid_example(self.task_id(), 9, 0) | {"fen": clean_fen}

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.3_fake_clean.jsonl"
    output_path.write_text(
        json.dumps(_valid_example("9.3_fake_clean", 9, 0)) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, [CleanGenerator])

    pipeline.run_tier(
        9,
        {},
        frozenset({variant_fen_key(STARTING_FEN)}),
        PipelineStats(),
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [_valid_example("9.3_fake_clean", 9, 0) | {"fen": clean_fen}]


def test_package_run_tier_discards_partial_temp_output_on_generator_error(monkeypatch, tmp_path: Path):
    from chess_llm.sft.output import PipelineStats
    from chess_llm.sft import pipeline

    class CrashingGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def task_id(self) -> str:
            return "9.2_fake_crash"

        def target_volume(self) -> int:
            return 2

        def generate(self):
            yield _valid_example(self.task_id(), 9, 99)
            raise RuntimeError("boom")

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.2_fake_crash.jsonl"
    old_example = _valid_example("9.2_fake_crash", 9, 0)
    output_path.write_text(json.dumps(old_example) + "\n")

    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, [CrashingGenerator])

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_tier(9, {}, frozenset(), PipelineStats())

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [old_example]
    assert not (tier_dir / "9.2_fake_crash.jsonl.tmp").exists()


def test_package_pipeline_fen_pool_export_preserves_chess960_identity():
    from chess_llm.sft.fen_pool import FENPool

    pool = FENPool()
    pool.add(STARTING_FEN, source="standard", is_chess960=False)
    pool.add(STARTING_FEN, source="chess960", metadata={"chess960_id": 518})

    fen_pool_entries = pool.all_rows()
    prepared = build_eval_split_sources(
        {
            "fen_pool": fen_pool_entries,
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        }
    )

    assert prepared.standard_fen_pool == [
        {
            "fen": STARTING_FEN,
            "source": "standard",
            "is_chess960": False,
        }
    ]
    assert prepared.chess960_fen_pool == [
        {
            "fen": STARTING_FEN,
            "source": "chess960",
            "is_chess960": True,
            "metadata": {"chess960_id": 518},
        }
    ]


def test_legacy_run_pipeline_import_aliases_package_module(monkeypatch):
    package_module = importlib.import_module("chess_llm.sft.pipeline")
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    sys.modules.pop("scripts.run_pipeline", None)

    legacy_module = importlib.import_module("scripts.run_pipeline")

    assert legacy_module is package_module


def test_pipeline_import_does_not_set_hf_home(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    sys.modules.pop("chess_llm.sft.pipeline", None)

    importlib.import_module("chess_llm.sft.pipeline")

    assert os.environ.get("HF_HOME") is None


def test_pipeline_no_args_prints_help_without_creating_data_dirs_or_loading_sources(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("CHESS_SFT_OUTPUT", str(tmp_path / "data-root"))
    sys.modules.pop("chess_llm.sft.pipeline", None)
    pipeline = importlib.import_module("chess_llm.sft.pipeline")

    def fail_load_sources(*_args, **_kwargs):
        raise AssertionError("no-arg pipeline invocation should not load sources")

    monkeypatch.setattr(pipeline, "load_sources", fail_load_sources)

    assert pipeline.main([]) == 0

    output = capsys.readouterr().out
    assert "Chess SFT data generation pipeline" in output
    assert not (tmp_path / "data-root").exists()


def test_pipeline_cli_flushes_and_uses_hard_exit(monkeypatch):
    from chess_llm.sft import pipeline

    calls = []

    class HardExit(Exception):
        pass

    monkeypatch.setattr(pipeline, "main", lambda argv=None: 7)
    monkeypatch.setattr(
        pipeline.sys.stdout,
        "flush",
        lambda: calls.append("stdout"),
    )
    monkeypatch.setattr(
        pipeline.sys.stderr,
        "flush",
        lambda: calls.append("stderr"),
    )

    def fake_exit(code):
        calls.append(f"exit:{code}")
        raise HardExit

    monkeypatch.setattr(pipeline.os, "_exit", fake_exit)

    with pytest.raises(HardExit):
        pipeline.cli(["--fake"])

    assert calls == ["stdout", "stderr", "exit:7"]


def test_package_pipeline_module_help_runs_as_python_m():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "chess_llm.sft.pipeline", "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Chess SFT data generation pipeline" in result.stdout
    assert "--validate-only" in result.stdout


def test_pipeline_help_lists_source_readiness_controls():
    from chess_llm.sft import pipeline

    help_text = pipeline.build_arg_parser().format_help()

    assert "--allow-source-gaps" in help_text
    assert "--source-readiness-report" in help_text
    assert "--tier" in help_text


def test_pipeline_fails_before_eval_when_required_sources_missing(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "POOL_DIR", tmp_path / "fen_pool")
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(
        pipeline,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "position_evals": [],
        },
    )

    def fail_run_eval_splits(*_args, **_kwargs):
        raise AssertionError("source readiness should fail before eval splits")

    monkeypatch.setattr(pipeline, "run_eval_splits", fail_run_eval_splits)

    assert pipeline.main(["--tier", "4"]) == 1


def test_pipeline_allows_source_gaps_when_requested(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    called = {"eval": False}

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "POOL_DIR", tmp_path / "fen_pool")
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(
        pipeline,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "position_evals": [],
        },
    )

    def fake_run_eval_splits(*_args, **_kwargs):
        called["eval"] = True
        return frozenset()

    monkeypatch.setattr(pipeline, "run_eval_splits", fake_run_eval_splits)

    assert pipeline.main(["--eval-only", "--allow-source-gaps"]) == 0
    assert called["eval"] is True


def test_phase_a_tier_selection_does_not_require_openings_for_strict_eval():
    from chess_llm.sft.source_readiness import build_source_readiness_report

    report = build_source_readiness_report(
        {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "game_positions": [{"fen": STARTING_FEN, "move_played_uci": "e2e4"}],
            "openings": [],
        },
        tiers=(1, 2),
        strict_eval_splits=True,
    )

    assert report.ok is True
    assert "openings" not in report.required_sources


def test_pipeline_volume_runs_report_source_gaps_without_blocking(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    called = {"eval": False}

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "POOL_DIR", tmp_path / "fen_pool")
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(
        pipeline,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "position_evals": [],
        },
    )

    def fake_run_eval_splits(*_args, **_kwargs):
        called["eval"] = True
        return frozenset()

    monkeypatch.setattr(pipeline, "run_eval_splits", fake_run_eval_splits)

    assert pipeline.main(["--eval-only", "--volume", "1"]) == 0
    assert called["eval"] is True


def test_pipeline_eval_only_respects_tier_scope(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    captured: dict[str, object] = {}

    class ReadyReport:
        ok = True
        counts = {"fen_pool": 1}

        def format_lines(self):
            return []

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "POOL_DIR", tmp_path / "fen_pool")
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(
        pipeline,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "position_evals": [],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "build_source_readiness_report",
        lambda *_args, **_kwargs: ReadyReport(),
    )

    def fake_run_eval_splits(_config, *, volume_override=None, tiers=None):
        captured["volume_override"] = volume_override
        captured["tiers"] = tiers
        return frozenset()

    monkeypatch.setattr(pipeline, "run_eval_splits", fake_run_eval_splits)

    assert pipeline.main(["--eval-only", "--tier", "1", "2", "--volume", "200"]) == 0
    assert captured == {"volume_override": 200, "tiers": [1, 2]}


def test_load_sources_restricts_lichess_game_files_for_volume_runs(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    captured: dict[str, object] = {}

    def fake_stream_games(*, max_games=None, data_files=None):
        captured["max_games"] = max_games
        captured["data_files"] = data_files
        return [
            {
                "WhiteElo": "1600",
                "BlackElo": "1700",
                "Result": "1-0",
                "movetext": "1. e4 e5",
            }
        ]

    monkeypatch.setattr(pipeline, "stream_games", fake_stream_games)
    monkeypatch.setattr(pipeline, "load_puzzles", lambda max_puzzles=None: [])
    monkeypatch.setattr(pipeline, "load_openings", lambda max_openings=None: [])
    monkeypatch.setattr(
        pipeline,
        "stream_evals",
        lambda min_depth=None, max_rows=None: [],
    )
    monkeypatch.setattr(pipeline, "sample_chess960_positions", lambda **_kwargs: [])
    monkeypatch.setattr(pipeline, "load_mate", lambda max_rows=None: [])
    monkeypatch.setattr(pipeline, "POLYGLOT_DIR", str(tmp_path / "missing-books"))

    config = pipeline.load_sources(volume_override=2)

    assert captured == {
        "max_games": 20,
        "data_files": list(pipeline.VOLUME_LICHESS_GAME_DATA_FILES),
    }
    assert config["game_positions"]


def test_lichess_game_file_restriction_is_disabled_for_full_runs():
    from chess_llm.sft import pipeline

    assert pipeline.lichess_game_data_files_for_volume(None) is None
    assert pipeline.lichess_game_data_files_for_volume(20) == list(
        pipeline.VOLUME_LICHESS_GAME_DATA_FILES
    )


def test_pipeline_validate_only_passes_tier_scope(monkeypatch):
    from chess_llm.sft import pipeline

    captured: dict[str, object] = {}

    def fake_validate_existing_outputs(*, volume_override=None, tiers=None):
        captured["volume_override"] = volume_override
        captured["tiers"] = tiers
        return 0

    monkeypatch.setattr(pipeline, "validate_existing_outputs", fake_validate_existing_outputs)

    assert pipeline.main(["--validate-only", "--tier", "1", "2", "--volume", "100"]) == 0
    assert captured == {"volume_override": 100, "tiers": [1, 2]}


def test_run_eval_splits_caps_split_targets_for_volume_smoke_run(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")

    def fake_generate(sources, *, split_sizes=None):
        captured["sources"] = sources
        captured["split_sizes"] = split_sizes
        return {split_name: [] for split_name in split_sizes}

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fake_generate)
    monkeypatch.setattr(pipeline, "save_eval_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "freeze_and_save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "build_blocklist", lambda _splits: frozenset())

    blocklist = pipeline.run_eval_splits(
        {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        },
        volume_override=20,
    )

    assert blocklist == frozenset()
    split_sizes = captured["split_sizes"]
    assert all(size <= 20 for size in split_sizes.values())
    assert split_sizes["perception"] == 0
    assert split_sizes["rules"] == 0


def test_run_eval_splits_scopes_strict_coverage_to_selected_tiers(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")

    def fake_generate(sources, *, split_sizes=None):
        captured["sources"] = sources
        captured["split_sizes"] = split_sizes
        return {
            split_name: [{"fen": STARTING_FEN, "is_chess960": False}]
            for split_name in split_sizes
        }

    def fake_freeze(splits, _output_dir, **kwargs):
        captured["frozen_splits"] = sorted(splits)
        captured["strict_coverage"] = kwargs.get("strict_coverage")

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fake_generate)
    monkeypatch.setattr(pipeline, "save_eval_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "freeze_and_save", fake_freeze)
    monkeypatch.setattr(pipeline, "build_blocklist", lambda _splits: frozenset())

    pipeline.run_eval_splits(
        {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        },
        tiers=[1, 2],
    )

    assert captured["frozen_splits"] == ["perception", "rules"]
    assert captured["strict_coverage"] is True
    assert sorted(captured["split_sizes"]) == ["perception", "rules"]


def test_run_eval_splits_rebuilds_existing_blocklist_without_manifest(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    pipeline.EVAL_SPLITS_DIR.mkdir()
    (pipeline.EVAL_SPLITS_DIR / "blocklist.txt").write_text("old-fen\n", encoding="utf-8")

    def fake_generate(_sources, *, split_sizes=None):
        captured["split_sizes"] = split_sizes
        return {"perception": [{"fen": STARTING_FEN, "is_chess960": False}]}

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fake_generate)
    monkeypatch.setattr(pipeline, "save_eval_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "freeze_and_save", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "build_blocklist", lambda _splits: frozenset({"new-fen"}))

    blocklist = pipeline.run_eval_splits(
        {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        },
        volume_override=20,
    )

    assert blocklist == frozenset({"new-fen"})
    assert captured["split_sizes"] is not None


def test_run_eval_splits_reuses_existing_blocklist_when_manifest_matches(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    pipeline.EVAL_SPLITS_DIR.mkdir()
    pipeline.BENCHMARK_DIR.mkdir()
    (pipeline.EVAL_SPLITS_DIR / "blocklist.txt").write_text(
        variant_fen_key(STARTING_FEN) + "\n",
        encoding="utf-8",
    )
    (pipeline.BENCHMARK_DIR / "manifest.json").write_text("{}\n", encoding="utf-8")

    config = {
        "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
        "openings": [],
        "position_evals": [],
        "puzzles": [],
        "best_move_evals": [],
        "endgame_positions": [],
        "mate_rows": [],
    }
    prepared_sources = pipeline.build_eval_split_sources(
        config,
        min_depth_eval_benchmark=pipeline.MIN_DEPTH_EVAL_BENCHMARK,
        seed=pipeline.MASTER_SEED,
    )
    split_sizes = pipeline.reserve_training_rows_for_volume(
        pipeline.effective_eval_split_sizes(
            pipeline.EVAL_SPLIT_SIZES,
            volume_override=20,
        ),
        prepared_sources,
        volume_override=20,
    )
    manifest = pipeline.build_eval_split_manifest(
        prepared_sources,
        split_sizes=split_sizes,
        volume_override=20,
    )
    pipeline.write_eval_split_manifest(manifest, pipeline.EVAL_SPLITS_DIR)

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("matching eval split manifest should reuse blocklist")

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fail_generate)

    assert pipeline.run_eval_splits(config, volume_override=20) == frozenset(
        {variant_fen_key(STARTING_FEN)}
    )


def test_volume_smoke_chess960_source_target_covers_train_and_eval():
    from chess_llm.sft import pipeline

    target = pipeline.chess960_source_target_count(
        n_standard=1000,
        volume_override=100,
    )

    assert target >= 350


def test_volume_smoke_eval_split_does_not_block_all_tier1_fens(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline
    from chess_llm.sft.generators.tier1_perception import FENToBoard

    fen_pool = []
    for chess960_id in range(20):
        board = chess.Board.from_chess960_pos(chess960_id)
        board.chess960 = True
        fen_pool.append(
            {
                "fen": board.fen(),
                "metadata": {"chess960_id": chess960_id},
            }
        )

    captured: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(
        pipeline,
        "save_eval_splits",
        lambda splits, _path: captured.update(splits=splits),
    )
    monkeypatch.setattr(pipeline, "freeze_and_save", lambda *_args, **_kwargs: None)

    blocklist = pipeline.run_eval_splits(
        {
            "fen_pool": fen_pool,
            "openings": [],
            "position_evals": [],
            "puzzles": [],
            "best_move_evals": [],
            "endgame_positions": [],
            "mate_rows": [],
        },
        volume_override=20,
    )

    assert captured["splits"]["chess960"] == []
    generator = FENToBoard(
        config={"fen_pool": fen_pool, "volume_override": 20},
        blocklist=blocklist,
        rng=Random(1),
    )

    assert len(list(generator.generate())) == 20


def test_pipeline_writes_source_readiness_report(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    report_path = tmp_path / "readiness.json"

    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "data")
    monkeypatch.setattr(pipeline, "POOL_DIR", tmp_path / "data" / "fen_pool")
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "data" / "eval_splits")
    monkeypatch.setattr(pipeline, "ANNOTATIONS_DIR", tmp_path / "data" / "annotations")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "data" / "output")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "data" / "benchmark")
    monkeypatch.setattr(
        pipeline,
        "load_sources",
        lambda volume_override=None: {
            "fen_pool": [{"fen": STARTING_FEN, "is_chess960": False}],
            "position_evals": [],
        },
    )
    monkeypatch.setattr(pipeline, "run_eval_splits", lambda *_args, **_kwargs: frozenset())

    assert (
        pipeline.main(
            [
                "--eval-only",
                "--allow-source-gaps",
                "--source-readiness-report",
                str(report_path),
            ]
        )
        == 0
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "sft_source_readiness"
    assert payload["counts"]["fen_pool"] == 1
    assert payload["ok"] is False
