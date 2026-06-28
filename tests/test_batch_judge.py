import json
from pathlib import Path

import chess
import chess.engine
import pytest

from chess_llm.artifacts.jsonl import read_jsonl
from chess_llm.artifacts.schemas import JudgmentArtifact, PromptArtifact, RolloutArtifact
from chess_llm.autodata.failure_buckets import LEGAL_UNSCORED
from chess_llm.evals.batch_judge import judge_prediction_file, main


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _benchmark_row(example_id: str = "planning_00000") -> dict:
    return {
        "example_id": example_id,
        "split": "planning",
        "task_type": "best_move",
        "fen": STARTING_FEN,
        "prompt": f"FEN: {STARTING_FEN}\nWhat is the best move?",
        "gold_answer": "e2e4",
        "metric_type": "move_extraction",
        "metadata": {"source": "unit"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_benchmark(tmp_path: Path) -> Path:
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(benchmark_dir / "planning.jsonl", [_benchmark_row()])
    return benchmark_dir


class FakeStockfish:
    def __init__(self) -> None:
        self.configured: list[dict] = []
        self.quit_called = False

    def configure(self, options: dict) -> None:
        self.configured.append(dict(options))

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        if len(board.move_stack) == 0:
            return {
                "score": chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE),
                "pv": [chess.Move.from_uci("e2e4")],
            }
        return {"score": chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE)}

    def quit(self) -> None:
        self.quit_called = True


class FakeOpenedStockfish(FakeStockfish):
    @property
    def id(self) -> dict[str, str]:
        if self.quit_called:
            raise RuntimeError("engine already closed")
        return {"name": "Fake Stockfish"}


def test_batch_judge_writes_prompt_rollout_judgment_and_manifest(tmp_path):
    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [{"example_id": "planning_00000", "prediction": "<move>e2e4</move>"}],
    )
    output_dir = tmp_path / "artifacts"

    result = judge_prediction_file(
        benchmark_dir,
        predictions,
        output_dir,
        model_id="model-a",
    )

    assert result.prompt_count == 1
    assert result.rollout_count == 1
    assert result.judgment_count == 1
    assert (output_dir / "prompts.jsonl").exists()
    assert (output_dir / "rollouts.jsonl").exists()
    assert (output_dir / "judgments.jsonl").exists()
    assert (output_dir / "manifest.json").exists()

    prompts = list(read_jsonl(output_dir / "prompts.jsonl", PromptArtifact))
    rollouts = list(read_jsonl(output_dir / "rollouts.jsonl", RolloutArtifact))
    judgments = list(read_jsonl(output_dir / "judgments.jsonl", JudgmentArtifact))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert [prompt.prompt_id for prompt in prompts] == ["planning_00000"]
    assert rollouts[0].prompt_id == "planning_00000"
    assert rollouts[0].model_id == "model-a"
    assert rollouts[0].parsed_answer.move_uci == "e2e4"
    assert judgments[0].rollout_id == rollouts[0].rollout_id
    assert judgments[0].legal is True
    assert judgments[0].failure_bucket == LEGAL_UNSCORED
    assert manifest["prompt_count"] == 1
    assert manifest["rollout_count"] == 1
    assert manifest["judgment_count"] == 1
    assert manifest["model_id"] == "model-a"
    assert manifest["splits"] == ["planning"]


def test_batch_judge_supports_multiple_predictions_per_example(tmp_path):
    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {"example_id": "planning_00000", "prediction": "<move>e2e4</move>"},
            {"example_id": "planning_00000", "prediction": "<move>d2d4</move>"},
        ],
    )

    judge_prediction_file(benchmark_dir, predictions, tmp_path / "artifacts", "model-a")

    rollouts = list(read_jsonl(tmp_path / "artifacts" / "rollouts.jsonl", RolloutArtifact))
    judgments = list(
        read_jsonl(tmp_path / "artifacts" / "judgments.jsonl", JudgmentArtifact)
    )

    assert len(rollouts) == 2
    assert len(judgments) == 2
    assert rollouts[0].rollout_id != rollouts[1].rollout_id
    assert [rollout.parsed_answer.move_uci for rollout in rollouts] == ["e2e4", "d2d4"]


def test_batch_judge_prefers_raw_prediction_for_rollout_output(tmp_path):
    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "example_id": "planning_00000",
                "prediction": "e2e4",
                "raw_prediction": "<think>center</think>\n<move>e2e4</move>",
            }
        ],
    )

    judge_prediction_file(benchmark_dir, predictions, tmp_path / "artifacts", "model-a")

    rollouts = list(read_jsonl(tmp_path / "artifacts" / "rollouts.jsonl", RolloutArtifact))
    assert rollouts[0].raw_output == "<think>center</think>\n<move>e2e4</move>"
    assert rollouts[0].parsed_answer.format_type == "move_tag"


def test_batch_judge_raises_for_unknown_example_id(tmp_path):
    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [{"example_id": "missing_00000", "prediction": "e2e4"}])

    with pytest.raises(ValueError, match="missing_00000"):
        judge_prediction_file(benchmark_dir, predictions, tmp_path / "artifacts", "model-a")


def test_batch_judge_cli_writes_expected_files(tmp_path):
    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "artifacts"
    _write_jsonl(
        predictions,
        [{"example_id": "planning_00000", "prediction": "<move>e2e4</move>"}],
    )

    exit_code = main(
        [
            "--benchmark-dir",
            str(benchmark_dir),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
            "--model-id",
            "model-a",
            "--split",
            "planning",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "prompts.jsonl").exists()
    assert (output_dir / "rollouts.jsonl").exists()
    assert (output_dir / "judgments.jsonl").exists()
    assert (output_dir / "manifest.json").exists()


def test_batch_judge_can_score_predictions_with_stockfish_engine(tmp_path):
    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "artifacts"
    _write_jsonl(
        predictions,
        [{"example_id": "planning_00000", "prediction": "<move>d2d4</move>"}],
    )
    engine = FakeStockfish()

    result = judge_prediction_file(
        benchmark_dir,
        predictions,
        output_dir,
        model_id="model-a",
        analysis_engine=engine,
        stockfish_depth=16,
        stockfish_threads=4,
        stockfish_hash_mb=512,
        syzygy_path=tmp_path / "syzygy",
    )

    judgments = list(read_jsonl(result.judgments_path, JudgmentArtifact))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert judgments[0].legal is True
    assert judgments[0].failure_bucket is None
    assert judgments[0].teacher_move_uci == "e2e4"
    assert judgments[0].regret_cp == 25.0
    assert judgments[0].metadata["judge"] == "stockfish"
    assert judgments[0].metadata["stockfish_depth"] == 16
    assert manifest["judge"]["mode"] == "stockfish"
    assert manifest["judge"]["stockfish_depth"] == 16
    assert manifest["judge"]["stockfish_threads"] == 4
    assert manifest["judge"]["stockfish_hash_mb"] == 512
    assert manifest["judge"]["syzygy_path"] == str(tmp_path / "syzygy")
    assert engine.quit_called is False


def test_batch_judge_records_engine_name_before_closing_opened_stockfish(
    monkeypatch,
    tmp_path,
):
    from chess_llm.evals import batch_judge

    benchmark_dir = _write_benchmark(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "artifacts"
    _write_jsonl(
        predictions,
        [{"example_id": "planning_00000", "prediction": "<move>d2d4</move>"}],
    )
    engine = FakeOpenedStockfish()

    monkeypatch.setattr(batch_judge, "open_stockfish", lambda _config: engine)

    result = judge_prediction_file(
        benchmark_dir,
        predictions,
        output_dir,
        model_id="model-a",
        stockfish_path=tmp_path / "stockfish.exe",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["judge"]["stockfish_engine_name"] == "Fake Stockfish"
    assert engine.quit_called is True
