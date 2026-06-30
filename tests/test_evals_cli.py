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


def test_legacy_eval_scripts_delegate_to_package():
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    sys.path.insert(0, str(make_data_root))

    from chess_llm.evals import freeze_benchmark, run_benchmark, run_eval_harness
    from scripts import freeze_benchmark as legacy_freeze
    from scripts import run_benchmark as legacy_benchmark
    from scripts import run_eval_harness as legacy_harness

    assert legacy_freeze.main is freeze_benchmark.main
    assert legacy_harness.main is run_eval_harness.main
    assert legacy_benchmark.main is run_benchmark.main
    assert legacy_benchmark.compute_acpl is run_benchmark.compute_acpl
