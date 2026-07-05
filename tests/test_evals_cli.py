import json
import sys
from pathlib import Path

import chess

from chess_llm.evals.benchmark import BenchmarkExample, load_benchmark, save_benchmark


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _clear_legacy_modules() -> None:
    for name in list(sys.modules):
        if (
            name.startswith("validation")
            or name.startswith("scripts")
            or name == "config"
            or name.startswith("config.")
        ):
            sys.modules.pop(name, None)


def test_package_eval_cli_modules_import_without_legacy_modules():
    _clear_legacy_modules()

    from chess_llm.evals import freeze_benchmark, run_benchmark, run_eval_harness

    assert freeze_benchmark is not None
    assert run_benchmark is not None
    assert run_eval_harness is not None
    assert "validation.benchmark" not in sys.modules
    assert "validation.eval_harness" not in sys.modules
    assert "scripts.run_benchmark" not in sys.modules
    assert "config" not in sys.modules


def test_freeze_benchmark_cli_writes_manifest_and_split(tmp_path):
    from chess_llm.evals import freeze_benchmark

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": STARTING_FEN}])

    exit_code = freeze_benchmark.main(
        [
            "--split",
            "rules",
            "--split-dir",
            str(split_dir),
            "--output-dir",
            str(output_dir),
            "--version",
            "unit-v1",
            "--seed",
            "42",
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = load_benchmark(output_dir / "rules.jsonl")

    assert exit_code == 0
    assert manifest["version"] == "unit-v1"
    assert manifest["splits"] == {"rules": 1}
    assert rows[0].task_type == "legal_moves"


def test_freeze_benchmark_cli_full_freeze_rejects_coverage_gaps(tmp_path):
    from chess_llm.evals import freeze_benchmark

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": STARTING_FEN}])

    exit_code = freeze_benchmark.main(
        [
            "--split-dir",
            str(split_dir),
            "--output-dir",
            str(output_dir),
            "--version",
            "unit-v1",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 1
    assert not (output_dir / "manifest.json").exists()


def test_freeze_benchmark_cli_can_allow_full_freeze_coverage_gaps(tmp_path):
    from chess_llm.evals import freeze_benchmark

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    output_dir = tmp_path / "benchmark"
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": STARTING_FEN}])

    exit_code = freeze_benchmark.main(
        [
            "--split-dir",
            str(split_dir),
            "--output-dir",
            str(output_dir),
            "--version",
            "unit-v1",
            "--seed",
            "42",
            "--allow-coverage-gaps",
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest["splits"] == {"rules": 1}


def test_run_eval_harness_cli_returns_failure_for_invalid_split(tmp_path):
    from chess_llm.evals import run_eval_harness

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    _write_jsonl(split_dir / "rules.jsonl", [{"fen": "8/8/8/8/8/8/8/8 w - - 0 1"}])

    exit_code = run_eval_harness.main(
        [
            "--split",
            "rules",
            "--split-dir",
            str(split_dir),
        ]
    )

    assert exit_code == 1


def test_run_benchmark_package_acpl_accepts_chess960_id():
    from chess_llm.evals import run_benchmark

    class FakeEngine:
        def analyse(self, _board, _limit):
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE),
            }

    example = BenchmarkExample(
        example_id="chess960_00000",
        split="chess960",
        task_type="best_move",
        fen="bqnnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="f1g1",
        metric_type="move_extraction",
        metadata={"cp": 100, "chess960_id": 3},
    )

    scores = run_benchmark.compute_acpl(
        FakeEngine(),
        [example],
        {"chess960_00000": "<move>f1g1</move>"},
        depth=1,
    )

    assert run_benchmark.example_is_chess960(example) is True
    assert scores["chess960_00000"] == 0.0


def test_run_benchmark_acpl_invalid_move_uses_shared_clamp():
    from chess_llm.evals import run_benchmark
    from chess_llm.evals.benchmark import ACPL_INVALID_MOVE_PENALTY

    class FakeEngine:
        def analyse(self, _board, _limit):
            raise AssertionError("invalid moves should not call Stockfish")

    example = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={"cp": 23},
    )

    scores = run_benchmark.compute_acpl(
        FakeEngine(),
        [example],
        {"planning_00000": "no move tag"},
        depth=1,
    )

    assert scores["planning_00000"] == ACPL_INVALID_MOVE_PENALTY


def test_run_benchmark_acpl_reuses_sqlite_score_cache_between_runs(tmp_path):
    from chess_llm.evals import run_benchmark

    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def analyse(self, board, _limit):
            self.calls += 1
            cp = 100 if not board.move_stack else 40
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
            }

    class FailingEngine:
        def analyse(self, _board, _limit):
            raise AssertionError("cached ACPL should not call the engine")

    example = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={"cp": 100},
    )
    cache_path = tmp_path / "eval_cache.sqlite"
    first_engine = FakeEngine()

    first = run_benchmark.compute_acpl(
        first_engine,
        [example],
        {example.example_id: "<move>e2e4</move>"},
        depth=1,
        cache_path=cache_path,
    )
    second = run_benchmark.compute_acpl(
        FailingEngine(),
        [example],
        {example.example_id: "<move>e2e4</move>"},
        depth=1,
        cache_path=cache_path,
    )

    assert first == {example.example_id: 60.0}
    assert second == first
    assert first_engine.calls == 1


def test_run_benchmark_cli_writes_prediction_analysis_report(tmp_path):
    from chess_llm.evals import run_benchmark

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    save_benchmark(
        [
            BenchmarkExample(
                example_id="perception_00000",
                split="perception",
                task_type="board_to_fen",
                fen=STARTING_FEN,
                prompt="Here is the current board:\n...\nWrite the FEN.",
                gold_answer=STARTING_FEN,
                metric_type="fen_exact_match",
                metadata={},
            )
        ],
        benchmark_dir / "perception.jsonl",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "example_id": "perception_00000",
                "prediction": "Lookup: d1=white queen. Result FEN: 8/8/8/8/8/8/8/K6k w - - 0 1",
            }
        ],
    )

    exit_code = run_benchmark.main(
        [
            "--benchmark-dir",
            str(benchmark_dir),
            "--predictions",
            str(predictions_path),
        ]
    )

    analysis = json.loads(
        predictions_path.with_suffix(".analysis.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert analysis["tasks"]["board_to_fen"]["format_bleed_count"] == 1
    assert analysis["tasks"]["board_to_fen"]["failure_count"] == 1


def test_run_benchmark_cli_reports_puzzle_rating_and_theme_strata(tmp_path):
    from chess_llm.evals import run_benchmark

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    save_benchmark(
        [
            BenchmarkExample(
                example_id="planning_00000",
                split="planning",
                task_type="puzzle_solve",
                fen=STARTING_FEN,
                prompt="FEN: ...\nSolve this puzzle.",
                gold_answer="e2e4",
                metric_type="move_extraction",
                metadata={"rating": 1420, "themes": ["fork", "pin"]},
            ),
            BenchmarkExample(
                example_id="planning_00001",
                split="planning",
                task_type="puzzle_solve",
                fen=STARTING_FEN,
                prompt="FEN: ...\nSolve this puzzle.",
                gold_answer="d2d4",
                metric_type="move_extraction",
                metadata={"rating": 1810, "themes": ["fork", "backRankMate"]},
            ),
        ],
        benchmark_dir / "planning.jsonl",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {"example_id": "planning_00000", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00001", "prediction": "<move>e2e4</move>"},
        ],
    )

    exit_code = run_benchmark.main(
        [
            "--benchmark-dir",
            str(benchmark_dir),
            "--predictions",
            str(predictions_path),
        ]
    )

    analysis = json.loads(
        predictions_path.with_suffix(".analysis.json").read_text(encoding="utf-8")
    )
    strata = analysis["puzzle_strata"]
    assert exit_code == 0
    assert strata["rating_buckets"]["1200-1599"]["primary_accuracy"] == 1.0
    assert strata["rating_buckets"]["1600-1999"]["primary_accuracy"] == 0.0
    assert strata["themes"]["fork"]["row_count"] == 2
    assert strata["themes"]["fork"]["primary_accuracy"] == 0.5
    assert strata["themes"]["pin"]["primary_accuracy"] == 1.0
    assert strata["themes"]["backRankMate"]["primary_accuracy"] == 0.0


def test_load_predictions_returns_normalized_and_raw_maps(tmp_path):
    from chess_llm.evals.run_benchmark import load_predictions

    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "example_id": "a",
                "prediction": "e2e4",
                "raw_prediction": "<think>center</think><move>e2e4</move>",
            },
            {"example_id": "b", "prediction": "d2d4"},
        ],
    )

    predictions, raw_predictions = load_predictions(path)

    assert predictions == {"a": "e2e4", "b": "d2d4"}
    assert raw_predictions == {
        "a": "<think>center</think><move>e2e4</move>",
        "b": "d2d4",
    }


def test_run_benchmark_cli_scores_format_compliance_on_raw_prediction(tmp_path, capsys):
    from chess_llm.evals import run_benchmark

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    save_benchmark(
        [
            BenchmarkExample(
                example_id="planning_00000",
                split="planning",
                task_type="best_move",
                fen=STARTING_FEN,
                prompt="FEN: ...\nWhat is the best move?",
                gold_answer="e2e4",
                metric_type="move_extraction",
                metadata={},
            )
        ],
        benchmark_dir / "planning.jsonl",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "example_id": "planning_00000",
                "prediction": "e2e4",
                "raw_prediction": "<think>center</think>\n<move>e2e4</move>",
            }
        ],
    )

    exit_code = run_benchmark.main(
        [
            "--benchmark-dir",
            str(benchmark_dir),
            "--predictions",
            str(predictions_path),
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    format_line = next(
        line for line in out.splitlines() if line.strip().startswith("format_compliance")
    )
    assert "100.0%" in format_line
    best_move_line = next(
        line for line in out.splitlines() if line.strip().startswith("best_move ")
    )
    assert "100.0%" in best_move_line


def test_run_benchmark_cli_reports_uncovered_and_partially_covered_splits(tmp_path, capsys):
    from chess_llm.evals import run_benchmark

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    save_benchmark(
        [
            BenchmarkExample(
                example_id=f"perception_{index:05d}",
                split="perception",
                task_type="board_to_fen",
                fen=STARTING_FEN,
                prompt="Write the FEN.",
                gold_answer=STARTING_FEN,
                metric_type="fen_exact_match",
                metadata={},
            )
            for index in range(2)
        ],
        benchmark_dir / "perception.jsonl",
    )
    save_benchmark(
        [
            BenchmarkExample(
                example_id="planning_00000",
                split="planning",
                task_type="best_move",
                fen=STARTING_FEN,
                prompt="FEN: ...\nWhat is the best move?",
                gold_answer="e2e4",
                metric_type="move_extraction",
                metadata={},
            )
        ],
        benchmark_dir / "planning.jsonl",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [{"example_id": "perception_00000", "prediction": STARTING_FEN}],
    )

    exit_code = run_benchmark.main(
        [
            "--benchmark-dir",
            str(benchmark_dir),
            "--predictions",
            str(predictions_path),
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert (
        "[WARN] split 'planning': 0/1 examples have predictions; "
        "split is uncovered and excluded from scoring and ACPL"
    ) in out
    assert "Planning (1 examples): UNCOVERED (0 predictions)" in out
    assert "[WARN] split 'perception': MISSING predictions for 1/2 examples" in out
    assert "coverage" in out
    assert "1/2" in out
    assert "best_move" not in out


def test_prediction_analysis_result_fen_family_is_not_state_tracking_bleed():
    from chess_llm.evals.prediction_analysis import (
        is_prediction_format_bleed,
        prediction_format_family,
    )

    family = prediction_format_family(f"Result FEN: {STARTING_FEN}")

    assert family == "result_fen"
    assert is_prediction_format_bleed("state_tracking", family) is False
    assert prediction_format_family(STARTING_FEN) == "fen"
    assert prediction_format_family("b2b4/8/8/8/7k/8/8/K7 w - - 0 1") == "fen"
