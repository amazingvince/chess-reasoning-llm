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


def test_package_run_tier_extends_valid_output_without_reusing_examples(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline
    from chess_llm.sft.output import PipelineStats

    def example(idx: int) -> dict:
        row = _valid_example("9.4_fake_append", 9, idx)
        row["metadata"] = {"example_identity": f"fake-{idx}"}
        return row

    class ExtendingGenerator:
        def __init__(self, config=None, *args, **kwargs):
            self.config = config or {}

        def task_id(self) -> str:
            return "9.4_fake_append"

        def target_volume(self) -> int:
            return int(self.config.get("volume_override", 0))

        def generate(self):
            for idx in range(self.target_volume()):
                yield example(idx)

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.4_fake_append.jsonl"
    output_path.write_text(json.dumps(example(0)) + "\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, [ExtendingGenerator])

    pipeline.run_tier(9, {}, frozenset(), PipelineStats(), volume_override=3)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["example_identity"] for row in rows] == [
        "fake-0",
        "fake-1",
        "fake-2",
    ]
    assert [row["messages"][2]["content"] for row in rows] == [
        "answer 0",
        "answer 1",
        "answer 2",
    ]
    manifest = json.loads(
        (tier_dir / "9.4_fake_append.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["previous_count"] == 1
    assert manifest["target_count"] == 3
    assert manifest["appended_count"] == 2
    assert manifest["skipped_duplicate_count"] >= 1
    assert manifest["generation_mode"] == "extended_existing"
    assert manifest["freshness_policy"] == "append_unseen_examples"
    assert manifest["extension_candidate_count"] >= 3
    assert manifest["extension_candidate_budget"] >= 4


def test_package_run_tier_extension_searches_past_duplicate_prefix(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline
    from chess_llm.sft.output import PipelineStats

    def example(identity: str, idx: int) -> dict:
        row = _valid_example("9.5_fake_sparse_append", 9, idx)
        row["metadata"] = {"example_identity": identity}
        return row

    class SparseFreshGenerator:
        def __init__(self, config=None, *args, **kwargs):
            self.config = config or {}

        def task_id(self) -> str:
            return "9.5_fake_sparse_append"

        def target_volume(self) -> int:
            return int(self.config.get("volume_override", 0))

        def generate(self):
            for idx in range(self.target_volume()):
                if idx < 5:
                    yield example("fake-0", idx)
                else:
                    yield example(f"fake-{idx - 4}", idx)

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.5_fake_sparse_append.jsonl"
    output_path.write_text(json.dumps(example("fake-0", 0)) + "\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, [SparseFreshGenerator])

    pipeline.run_tier(9, {}, frozenset(), PipelineStats(), volume_override=3)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["example_identity"] for row in rows] == [
        "fake-0",
        "fake-1",
        "fake-2",
    ]
    manifest = json.loads(
        (tier_dir / "9.5_fake_sparse_append.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["generation_mode"] == "extended_existing"
    assert manifest["extension_candidate_budget"] > 4
    assert manifest["extension_candidate_count"] == 7
    assert manifest["skipped_duplicate_count"] == 5


def test_package_run_tier_extension_underfill_reports_duplicate_supply_problem(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline
    from chess_llm.sft.output import PipelineStats

    def duplicate_example(idx: int) -> dict:
        row = _valid_example("9.6_fake_exhausted_append", 9, idx)
        row["metadata"] = {"example_identity": "fake-0"}
        return row

    class DuplicateOnlyGenerator:
        def __init__(self, config=None, *args, **kwargs):
            self.config = config or {}

        def task_id(self) -> str:
            return "9.6_fake_exhausted_append"

        def target_volume(self) -> int:
            return int(self.config.get("volume_override", 0))

        def generate(self):
            for idx in range(self.target_volume()):
                yield duplicate_example(idx)

    tier_dir = tmp_path / "tier9"
    tier_dir.mkdir()
    output_path = tier_dir / "9.6_fake_exhausted_append.jsonl"
    output_path.write_text(json.dumps(duplicate_example(0)) + "\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, [DuplicateOnlyGenerator])

    with pytest.raises(RuntimeError, match="underfilled while extending.*duplicate"):
        pipeline.run_tier(9, {}, frozenset(), PipelineStats(), volume_override=2)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [duplicate_example(0)]
    assert not (tier_dir / "9.6_fake_exhausted_append.jsonl.tmp").exists()


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


def test_eval_split_sources_include_multipv_planning_tasks():
    candidate_rows = []
    for index, fen in enumerate(
        [
            STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
        ]
    ):
        candidate_rows.append(
            {
                "fen": fen,
                "candidate_ratings": [
                    {"uci": "e2e4", "cp": 42 + index},
                    {"uci": "d2d4", "cp": 15},
                    {"uci": "g1f3", "cp": 5},
                    {"uci": "c2c4", "cp": -20},
                    {"uci": "b1c3", "cp": -80},
                ],
            }
        )

    prepared = build_eval_split_sources(
        {
            "fen_pool": [],
            "openings": [],
            "position_evals": [],
            "puzzles": [
                {
                    "fen": STARTING_FEN,
                    "puzzle_id": "duplicate-source",
                    "solution_first_move": "e2e4",
                }
            ],
            "best_move_evals": [{"fen": STARTING_FEN, "best_move": "e2e4"}],
            "candidate_rating_evals": candidate_rows,
            "endgame_positions": [],
            "mate_rows": [],
        }
    )

    planning_rows = prepared.sources["planning"]
    assert [row["task_type"] for row in planning_rows[:3]] == [
        "candidate_ratings",
        "best_line_trace",
        "step_verification",
    ]
    assert planning_rows[1]["best_line_trace"] is True
    verifier = planning_rows[2]
    assert "1. Candidate" in verifier["verification_trace"]
    assert verifier["expected_answer"].startswith("Verdict:")
    assert verifier["metadata"]["source_task"] == "7.8_candidate_ratings"


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
    assert "--eval-split-volume" in help_text
    assert "--refresh-eval-splits" in help_text
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

    def fake_run_eval_splits(
        _config,
        *,
        volume_override=None,
        tiers=None,
        reuse_eval_splits=False,
    ):
        captured["volume_override"] = volume_override
        captured["tiers"] = tiers
        captured["reuse_eval_splits"] = reuse_eval_splits
        return frozenset()

    monkeypatch.setattr(pipeline, "run_eval_splits", fake_run_eval_splits)

    assert (
        pipeline.main(
            [
                "--eval-only",
                "--tier",
                "1",
                "2",
                "--volume",
                "200",
                "--reuse-eval-splits",
            ]
        )
        == 0
    )
    assert captured == {
        "volume_override": 200,
        "tiers": [1, 2],
        "reuse_eval_splits": True,
    }


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
    monkeypatch.setattr(pipeline, "build_blocklist", lambda *_args, **_kwargs: frozenset())

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
    monkeypatch.setattr(pipeline, "build_blocklist", lambda *_args, **_kwargs: frozenset())

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
    monkeypatch.setattr(pipeline, "build_blocklist", lambda *_args, **_kwargs: frozenset({"new-fen"}))

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

    assert blocklist == frozenset({"new-fen", "std:old-fen"})
    assert captured["split_sizes"] is not None


def test_run_eval_splits_refuses_manifest_change_when_outputs_exist(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    pipeline.EVAL_SPLITS_DIR.mkdir()
    (pipeline.EVAL_SPLITS_DIR / "blocklist.txt").write_text("old-fen\n", encoding="utf-8")
    output_dir = pipeline.TIER_OUTPUT_DIR / "tier1"
    output_dir.mkdir(parents=True)
    (output_dir / "1.1_fen_to_board.jsonl").write_text("{}\n", encoding="utf-8")

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("eval split generation should not run")

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fail_generate)

    with pytest.raises(RuntimeError, match="eval split manifest changed"):
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
            volume_override=20,
            tiers=[1],
        )


def test_run_eval_splits_can_reuse_existing_blocklist_when_manifest_changes(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", tmp_path / "eval_splits")
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    pipeline.EVAL_SPLITS_DIR.mkdir()
    (pipeline.EVAL_SPLITS_DIR / "blocklist.txt").write_text("old-fen\n", encoding="utf-8")
    output_dir = pipeline.TIER_OUTPUT_DIR / "tier3"
    output_dir.mkdir(parents=True)
    (output_dir / "3.1_available_captures.jsonl").write_text("{}\n", encoding="utf-8")

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("eval split generation should not run")

    monkeypatch.setattr(pipeline, "generate_all_eval_splits", fail_generate)

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
        volume_override=50,
        tiers=[3],
        reuse_eval_splits=True,
    )

    assert blocklist == frozenset({"std:old-fen"})


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
        lambda splits, _path, **_kwargs: captured.update(splits=splits),
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


def _empty_source_fakes(monkeypatch, pipeline, tmp_path: Path) -> None:
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
    monkeypatch.setattr(pipeline, "SYZYGY_PATH", str(tmp_path / "missing-syzygy"))


def test_load_sources_threads_game_id_into_fen_pool_entries(
    monkeypatch,
    tmp_path: Path,
):
    import hashlib

    from chess_llm.sft import pipeline

    movetext = "1. e4 e5"
    game_id = hashlib.sha256(movetext.encode("utf-8")).hexdigest()[:16]
    after_e4_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    def fake_extract_game_positions(_game):
        yield {
            "fen": STARTING_FEN,
            "move_played_uci": "e2e4",
            "game_phase": "opening",
            "material_balance": 0,
            "ply": 0,
            "game_id": game_id,
        }
        # Rows without game_id must be tolerated (exact-position fallback).
        yield {
            "fen": after_e4_fen,
            "move_played_uci": "e7e5",
            "game_phase": "opening",
            "material_balance": 0,
            "ply": 1,
        }

    monkeypatch.setattr(
        pipeline,
        "stream_games",
        lambda **_kwargs: [{"moves": movetext}],
    )
    monkeypatch.setattr(
        pipeline,
        "extract_game_positions",
        fake_extract_game_positions,
    )
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)

    config = pipeline.load_sources(volume_override=2)

    by_fen = {entry["fen"]: entry for entry in config["fen_pool"]}
    assert by_fen[STARTING_FEN]["game_id"] == game_id
    assert "game_id" not in by_fen[after_e4_fen]


def _write_self_play_run(root: Path, rows: list[dict]) -> None:
    run_dir = root / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "positions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _self_play_position(fen: str, move_uci: str, ply: int) -> dict:
    return {
        "fen": fen,
        "move_played_uci": move_uci,
        "game_phase": "opening",
        "material_balance": 0,
        "ply": ply,
        "game_id": "abc123def456abcd",
        "mover": "model",
        "model_id": "test-model",
        "run_id": "run-1",
    }


def test_load_sources_samples_self_play_positions_with_ratio_cap(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    self_play_dir = tmp_path / "self_play"
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    after_e4_e5 = (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    )
    _write_self_play_run(
        self_play_dir,
        [
            _self_play_position(STARTING_FEN, "e2e4", 0),
            _self_play_position(after_e4, "e7e5", 1),
            _self_play_position(after_e4_e5, "g1f3", 2),
        ],
    )

    def fake_extract_game_positions(_game):
        for index in range(4):
            yield {
                "fen": f"base-fen-{index}",
                "move_played_uci": "e2e4",
                "game_phase": "opening",
                "material_balance": 0,
                "ply": index,
            }

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [{"moves": "1. e4"}])
    monkeypatch.setattr(pipeline, "extract_game_positions", fake_extract_game_positions)
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(pipeline, "SELF_PLAY_DIR", self_play_dir)
    monkeypatch.setattr(pipeline, "SELF_PLAY_MAX_POSITIONS", 100)
    monkeypatch.setattr(pipeline, "SELF_PLAY_RATIO", 0.5)

    config = pipeline.load_sources(volume_override=5)

    self_play_entries = [
        entry
        for entry in config["fen_pool"]
        if entry.get("source") == "self_play"
    ]
    # cap = min(100, int(0.5 * 4 base FENs)) = 2 of the 3 harvested rows.
    assert len(self_play_entries) == 2
    for entry in self_play_entries:
        assert entry["is_chess960"] is False
        assert entry["game_phase"] == "opening"
        assert entry["game_id"] == "abc123def456abcd"
    harvested_game_positions = [
        pos for pos in config["game_positions"] if pos.get("mover") == "model"
    ]
    assert len(harvested_game_positions) == 2


def test_load_sources_respects_self_play_max_positions(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    self_play_dir = tmp_path / "self_play"
    _write_self_play_run(
        self_play_dir,
        [_self_play_position(STARTING_FEN, "e2e4", 0)],
    )

    def fake_extract_game_positions(_game):
        for index in range(4):
            yield {
                "fen": f"base-fen-{index}",
                "move_played_uci": "e2e4",
                "game_phase": "opening",
                "material_balance": 0,
                "ply": index,
            }

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [{"moves": "1. e4"}])
    monkeypatch.setattr(pipeline, "extract_game_positions", fake_extract_game_positions)
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(pipeline, "SELF_PLAY_DIR", self_play_dir)
    monkeypatch.setattr(pipeline, "SELF_PLAY_MAX_POSITIONS", 0)
    monkeypatch.setattr(pipeline, "SELF_PLAY_RATIO", 1.0)

    config = pipeline.load_sources(volume_override=5)

    assert not [
        entry
        for entry in config["fen_pool"]
        if entry.get("source") == "self_play"
    ]


def test_load_sources_ignores_absent_self_play_dir(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [])
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(pipeline, "SELF_PLAY_DIR", tmp_path / "missing-self-play")

    config = pipeline.load_sources(volume_override=2)

    assert not [
        entry
        for entry in config["fen_pool"]
        if entry.get("source") == "self_play"
    ]


def test_load_sources_normalizes_polyglot_weights_per_book(
    monkeypatch,
    tmp_path: Path,
):
    from contextlib import contextmanager

    from chess_llm.sft import pipeline
    from chess_llm.sft.sources import polyglot_books

    books_dir = tmp_path / "books"
    books_dir.mkdir()
    (books_dir / "small.bin").write_bytes(b"")
    (books_dir / "large.bin").write_bytes(b"")
    per_book_moves = {
        "small.bin": [("e2e4", 3), ("d2d4", 1)],
        "large.bin": [("d2d4", 60_000), ("e2e4", 40_000)],
    }

    class FakeReader:
        def __init__(self, name: str) -> None:
            self.name = name

    @contextmanager
    def fake_load_book(path: str):
        yield FakeReader(Path(path).name)

    monkeypatch.setattr(polyglot_books, "load_book", fake_load_book)
    monkeypatch.setattr(
        polyglot_books,
        "get_weighted_moves",
        lambda reader, _board: list(per_book_moves[reader.name]),
    )
    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [])
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(
        pipeline,
        "load_openings",
        lambda max_openings=None: [
            {"fen": STARTING_FEN, "eco": "B00", "name": "Test Opening"}
        ],
    )
    monkeypatch.setattr(pipeline, "POLYGLOT_DIR", str(books_dir))

    config = pipeline.load_sources(volume_override=1)

    merged = dict(config["book_moves"][STARTING_FEN])
    # Raw summing would rank d2d4 first (60_001 vs 40_003); per-book
    # normalization ranks by aggregate relative popularity instead.
    assert merged["e2e4"] == pytest.approx(0.75 + 0.4)
    assert merged["d2d4"] == pytest.approx(0.25 + 0.6)
    assert config["book_moves"][STARTING_FEN][0][0] == "e2e4"


def test_load_sources_derives_book_moves_from_opening_prefixes_without_polyglot(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    prefix = ["e2e4", "e7e5", "g1f3", "b8c6"]
    board = chess.Board()
    for uci in prefix:
        board.push(chess.Move.from_uci(uci))
    prefix_fen = board.fen()

    spanish = list(prefix)
    board.push(chess.Move.from_uci("f1b5"))
    spanish_fen = board.fen()

    italian_board = chess.Board()
    for uci in prefix + ["f1c4"]:
        italian_board.push(chess.Move.from_uci(uci))
    italian_fen = italian_board.fen()

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [])
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(
        pipeline,
        "load_openings",
        lambda max_openings=None: [
            {
                "fen": prefix_fen,
                "eco": "C44",
                "name": "King Pawn Game",
                "uci_moves": prefix,
            },
            {
                "fen": spanish_fen,
                "eco": "C60",
                "name": "Ruy Lopez",
                "uci_moves": spanish + ["f1b5"],
            },
            {
                "fen": italian_fen,
                "eco": "C50",
                "name": "Italian Game",
                "uci_moves": prefix + ["f1c4"],
            },
        ],
    )

    config = pipeline.load_sources(volume_override=1)

    derived = dict(config["book_moves"][prefix_fen])
    assert derived == {"f1b5": 1.0, "f1c4": 1.0}


def test_load_sources_caps_syzygy_sampling_for_large_volume_overrides(
    monkeypatch,
    tmp_path: Path,
):
    from contextlib import contextmanager

    from chess_llm.sft import pipeline
    from chess_llm.sft.sources import syzygy_probing

    syzygy_dir = tmp_path / "syzygy"
    syzygy_dir.mkdir()
    calls: list[tuple[str, int]] = []

    @contextmanager
    def fake_open_tablebase(_path: str):
        yield object()

    def fake_sample_endgame_positions(_tb, material: str, n: int, _rng):
        calls.append((material, n))
        yield {
            "fen": "8/8/8/8/8/8/6K1/6kQ w - - 0 1",
            "wdl": 2,
            "dtz": 1,
            "material": material,
        }

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [])
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(pipeline, "SYZYGY_PATH", str(syzygy_dir))
    monkeypatch.setattr(syzygy_probing, "MATERIAL_CONFIGS", ["KQK", "KRK"])
    monkeypatch.setattr(syzygy_probing, "open_tablebase", fake_open_tablebase)
    monkeypatch.setattr(
        syzygy_probing,
        "sample_endgame_positions",
        fake_sample_endgame_positions,
    )
    monkeypatch.setattr(syzygy_probing, "best_dtz_move", lambda _tb, _board: "h1h8")

    config = pipeline.load_sources(volume_override=25_000)

    assert calls == [("KQK", 5_000), ("KRK", 5_000)]
    assert len(config["endgame_positions"]) == 2


def test_load_sources_shuffles_depth_ordered_evals(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    rows = [
        {"fen": f"fen-{index}", "depth": 60 - index, "best_move": "e2e4"}
        for index in range(30)
    ]

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [])
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(
        pipeline,
        "stream_evals",
        lambda min_depth=None, max_rows=None: list(rows),
    )

    config = pipeline.load_sources(volume_override=5)

    depth_ordered = [row["fen"] for row in rows]
    position_order = [row["fen"] for row in config["position_evals"]]
    best_move_order = [row["fen"] for row in config["best_move_evals"]]
    assert sorted(position_order) == sorted(depth_ordered)
    assert position_order != depth_ordered
    assert sorted(best_move_order) == sorted(depth_ordered)
    assert best_move_order != depth_ordered


def test_load_sources_reads_candidate_rating_annotation_file(monkeypatch, tmp_path: Path):
    from chess_llm.sft import pipeline

    candidate_row = {
        "fen": STARTING_FEN,
        "candidate_ratings": [
            {"uci": "e2e4", "cp": 42},
            {"uci": "d2d4", "cp": 15},
            {"uci": "g1f3", "cp": 5},
            {"uci": "c2c4", "cp": -20},
            {"uci": "b1c3", "cp": -80},
        ],
        "multipv_depth": 8,
    }
    candidate_path = tmp_path / "candidate_ratings.jsonl"
    candidate_path.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "stream_games", lambda **_kwargs: [])
    _empty_source_fakes(monkeypatch, pipeline, tmp_path)
    monkeypatch.setattr(pipeline, "CANDIDATE_RATINGS_PATH", candidate_path)

    config = pipeline.load_sources(volume_override=1)

    assert config["candidate_rating_evals"] == [candidate_row]


def test_task_rng_is_stable_and_order_independent(monkeypatch, tmp_path: Path):
    from random import Random

    from chess_llm.sft import pipeline
    from chess_llm.sft.output import PipelineStats

    assert (
        pipeline._task_rng("1.1_fen_to_board").random()
        == Random(f"{pipeline.MASTER_SEED}:1.1_fen_to_board").random()
    )
    assert (
        pipeline._task_rng("1.1_fen_to_board").random()
        != pipeline._task_rng("1.2_board_to_fen").random()
    )

    def make_generator(task: str):
        class FakeGenerator:
            def __init__(self, config=None, blocklist=frozenset(), rng=None):
                self.config = config or {}
                self.rng = rng or Random()

            def task_id(self) -> str:
                return task

            def target_volume(self) -> int:
                return 2

            def generate(self):
                for idx in range(2):
                    row = _valid_example(task, 9, idx)
                    row["messages"][2]["content"] = f"answer {self.rng.random()}"
                    row["metadata"] = {"example_identity": f"{task}-{idx}"}
                    yield row

        return FakeGenerator

    generators = [make_generator("9.6_fake_a"), make_generator("9.7_fake_b")]

    first_root = tmp_path / "first"
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", first_root)
    monkeypatch.setitem(pipeline.TIER_GENERATORS, 9, generators)
    pipeline.run_tier(9, {}, frozenset(), PipelineStats())
    first_b = (first_root / "tier9" / "9.7_fake_b.jsonl").read_text(encoding="utf-8")

    # Resume: task A already complete on disk, so only B regenerates. B's
    # output must not depend on whether A actually ran in this invocation.
    second_root = tmp_path / "second"
    (second_root / "tier9").mkdir(parents=True)
    (second_root / "tier9" / "9.6_fake_a.jsonl").write_text(
        (first_root / "tier9" / "9.6_fake_a.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", second_root)
    pipeline.run_tier(9, {}, frozenset(), PipelineStats())
    second_b = (second_root / "tier9" / "9.7_fake_b.jsonl").read_text(encoding="utf-8")

    assert first_b == second_b


def test_run_eval_splits_merges_blocklist_and_keeps_other_benchmark_splits(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline
    from chess_llm.sft.eval_split import load_blocklist

    eval_dir = tmp_path / "eval_splits"
    bench_dir = tmp_path / "benchmark"
    monkeypatch.setattr(pipeline, "EVAL_SPLITS_DIR", eval_dir)
    monkeypatch.setattr(pipeline, "BENCHMARK_DIR", bench_dir)
    monkeypatch.setattr(pipeline, "TIER_OUTPUT_DIR", tmp_path / "output")
    eval_dir.mkdir(parents=True)
    bench_dir.mkdir(parents=True)

    old_key = variant_fen_key("8/8/8/8/8/8/4K3/4k3 w - - 0 1")
    (eval_dir / "blocklist.txt").write_text(old_key + "\n", encoding="utf-8")
    (bench_dir / "tactics.jsonl").write_text("{}\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_freeze(splits, _output_dir, **kwargs):
        captured["clean"] = kwargs.get("clean")

    monkeypatch.setattr(pipeline, "freeze_and_save", fake_freeze)

    fen_pool = [
        {"fen": "k7/8/8/8/8/8/8/K7 w - - 0 1", "is_chess960": False},
        {"fen": "7k/8/8/8/8/8/8/K7 w - - 0 1", "is_chess960": False},
        {"fen": "7k/8/8/8/8/8/8/1K6 w - - 0 1", "is_chess960": False},
        {"fen": "k7/8/8/8/8/8/8/1K6 w - - 0 1", "is_chess960": False},
    ]
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
        volume_override=2,
        tiers=[1],
    )

    saved = load_blocklist(eval_dir / "blocklist.txt")
    assert old_key in saved
    assert old_key in blocklist
    assert len(saved) > 1
    assert captured["clean"] is False
    assert (bench_dir / "tactics.jsonl").exists()


def test_source_fingerprints_cache_skips_reparsing_unchanged_sources(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import pipeline

    rows = [{"fen": STARTING_FEN, "is_chess960": False}]
    cache_path = tmp_path / "source_fingerprints.cache.json"

    first = pipeline._source_fingerprints({"perception": rows}, cache_path=cache_path)
    assert cache_path.exists()

    def fail_fingerprint(_rows):
        raise AssertionError("unchanged sources must reuse the cached fingerprint")

    monkeypatch.setattr(pipeline, "_source_fingerprint", fail_fingerprint)
    second = pipeline._source_fingerprints({"perception": rows}, cache_path=cache_path)
    assert second == first

    changed = rows + [{"fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1", "is_chess960": False}]
    with pytest.raises(AssertionError):
        pipeline._source_fingerprints({"perception": changed}, cache_path=cache_path)


def test_batch_annotator_cache_lookup_respects_requested_depth(tmp_path: Path):
    from chess_llm.sft.annotation import BatchAnnotator

    annotator = BatchAnnotator(
        stockfish=None,
        cache_path=str(tmp_path / "annotations.sqlite"),
    )
    try:
        annotator.preload_from_evals(
            iter(
                [
                    {
                        "fen": STARTING_FEN,
                        "cp": 30,
                        "mate": None,
                        "best_move": "e2e4",
                        "pv_line": "e2e4 e7e5",
                        "depth": 12,
                    }
                ]
            )
        )

        assert annotator.annotate(STARTING_FEN, depth=10)["best_move"] == "e2e4"
        assert annotator.annotate(STARTING_FEN, depth=12)["best_move"] == "e2e4"
        assert annotator.annotate(STARTING_FEN)["cp"] == 30
        # A deeper request must not be satisfied by the shallow cached eval.
        assert annotator.annotate(STARTING_FEN, depth=30) == {
            "cp": None,
            "mate": None,
            "best_move": None,
            "pv_line": "",
        }
    finally:
        annotator.close()


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
