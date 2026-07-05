from __future__ import annotations

import json
import logging
import types
from argparse import Namespace
from pathlib import Path

import pytest

from chess_llm.evals.benchmark import BenchmarkExample


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _eval_args(tmp_path: Path, **overrides) -> Namespace:
    benchmark_dir = tmp_path / "benchmark"
    output_path = tmp_path / "predictions.jsonl"
    defaults = {
        "model": "model-id",
        "benchmark_dir": benchmark_dir,
        "output": output_path,
        "inference_backend": "transformers",
        "attn_implementation": "auto",
        "vllm_gpu_memory_utilization": 0.85,
        "vllm_max_model_len": None,
        "pass_k": 1,
        "temperature": None,
        "max_new_tokens": 16,
        "batch_size": 1,
        "max_examples_per_split": None,
        "splits": None,
        "full_benchmark": False,
        "stockfish_path": "stockfish",
        "acpl_depth": 1,
        "no_acpl": True,
        "full_acpl_report": False,
        "baseline": None,
        "phase": None,
        "report_only": True,
        "soft_gate": True,
        "wandb_project": "chess-sft",
        "wandb_run_name": None,
        "wandb_group": None,
        "wandb_job_type": "benchmark-eval",
        "no_wandb": True,
        "allow_wandb_offline": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _benchmark_example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )


def _puzzle_example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="planning_00001",
        split="planning",
        task_type="puzzle_solve",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )


def _split_example(split: str) -> BenchmarkExample:
    return BenchmarkExample(
        example_id=f"{split}_00000",
        split=split,
        task_type="board_print" if split == "perception" else "legal_moves",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        prompt="FEN: ...",
        gold_answer="answer",
        metric_type="exact_match",
        metadata={},
    )


def test_vllm_prediction_generator_reuses_engine_across_generate_calls(monkeypatch):
    from chess_llm.training import evaluate

    class FakeTokenizer:
        name_or_path = "fake-tokenizer"

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCandidate:
        def __init__(self, text):
            self.text = text

    class FakeOutput:
        def __init__(self, texts):
            self.outputs = [FakeCandidate(text) for text in texts]

    class FakeLLM:
        init_calls: list[dict] = []

        def __init__(self, **kwargs):
            self.init_calls.append(kwargs)
            self.generate_calls: list[tuple[list[str], dict]] = []

        def generate(self, prompts, sampling_params):
            self.generate_calls.append((list(prompts), sampling_params.kwargs))
            return [
                FakeOutput([f"{prompt} sample {idx}" for idx in range(sampling_params.kwargs["n"])])
                for prompt in prompts
            ]

    monkeypatch.setattr(evaluate, "format_prompt", lambda ex, _tokenizer: ex.prompt)

    generator = evaluate.VllmPredictionGenerator(
        "model-id",
        FakeTokenizer(),
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
    )

    first = generator.generate(
        [_benchmark_example()],
        num_samples=1,
        temperature=0.0,
        max_new_tokens=8,
        seed=11,
    )
    second = generator.generate(
        [_puzzle_example()],
        num_samples=2,
        temperature=0.7,
        max_new_tokens=16,
        seed=12,
    )

    assert len(FakeLLM.init_calls) == 1
    assert FakeLLM.init_calls[0]["model"] == "model-id"
    assert FakeLLM.init_calls[0]["tokenizer"] == "fake-tokenizer"
    assert FakeLLM.init_calls[0]["gpu_memory_utilization"] == 0.7
    assert FakeLLM.init_calls[0]["max_model_len"] == 4096
    assert first == {"planning_00000": ["FEN: ... sample 0"]}
    assert second == {
        "planning_00001": ["FEN: ... sample 0", "FEN: ... sample 1"]
    }
    assert generator.llm.generate_calls == [
        (["FEN: ..."], {"n": 1, "temperature": 0.0, "max_tokens": 8, "seed": 11}),
        (
            ["FEN: ..."],
            {
                "n": 2,
                "temperature": 0.7,
                "max_tokens": 16,
                "seed": 12,
                "top_p": 0.95,
            },
        ),
    ]


def test_qwen_eval_prompt_uses_template_thinking_flag_without_text_directive():
    from chess_llm.training import evaluate

    captured = {}

    class FakeQwenTokenizer:
        name_or_path = "Qwen/Qwen3.5-0.8B"

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "formatted prompt"

    prompt = evaluate.format_prompt(_split_example("rules"), FakeQwenTokenizer())

    assert prompt == "formatted prompt"
    assert captured["kwargs"]["enable_thinking"] is False
    assert captured["messages"][1]["content"] == _split_example("rules").prompt
    assert "/no_think" not in captured["messages"][1]["content"]


def test_trace_protocol_eval_prompt_enables_thinking_prefill():
    from chess_llm.training import evaluate

    captured = {}

    class FakeQwenTokenizer:
        name_or_path = "Qwen/Qwen3.5-0.8B"

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "formatted prompt<think>\n"

    prompt, prefill = evaluate.format_prompt_with_assistant_prefill(
        _benchmark_example(),
        FakeQwenTokenizer(),
    )

    assert prompt == "formatted prompt<think>\n"
    assert prefill == "<think>\n"
    assert captured["kwargs"]["enable_thinking"] is True
    assert captured["messages"][1]["content"] == _benchmark_example().prompt


def test_eval_prompt_disables_thinking_for_local_checkpoint_tokenizer():
    from chess_llm.training import evaluate

    captured = {}

    class FakeLocalCheckpointTokenizer:
        # A local checkpoint directory: no "qwen" anywhere in the path.
        name_or_path = "/home/amazi/chess_sft_checkpoints/phase_a/best"

        def apply_chat_template(self, messages, **kwargs):
            captured["kwargs"] = kwargs
            return "formatted prompt"

    prompt = evaluate.format_prompt(
        _split_example("rules"), FakeLocalCheckpointTokenizer()
    )

    assert prompt == "formatted prompt"
    assert captured["kwargs"]["enable_thinking"] is False


def test_stitch_assistant_prefill_preserves_explicit_think():
    from chess_llm.training import evaluate

    assert (
        evaluate._stitch_assistant_prefill("<think>\n", "reason</think><move>e2e4</move>")
        == "<think>\nreason</think><move>e2e4</move>"
    )
    assert (
        evaluate._stitch_assistant_prefill("<think>\n", "<think>x</think><move>e2e4</move>")
        == "<think>x</think><move>e2e4</move>"
    )


def test_eval_prompt_falls_back_when_template_rejects_thinking_flag():
    from chess_llm.training import evaluate

    calls = []

    class FakeStrictTokenizer:
        name_or_path = "/checkpoints/phase_a/best"

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ):
            calls.append(
                {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
            )
            return "formatted prompt"

    prompt = evaluate.format_prompt(_benchmark_example(), FakeStrictTokenizer())

    assert prompt == "formatted prompt"
    assert calls == [{"tokenize": False, "add_generation_prompt": True}]


def test_vllm_eval_reuses_one_generator_for_pass_k(monkeypatch, tmp_path):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_SUCCESS_EXIT_CODE

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")
    instances = []

    class FakeVllmPredictionGenerator:
        def __init__(
            self,
            model_path,
            tokenizer,
            *,
            gpu_memory_utilization,
            max_model_len,
        ):
            self.model_path = model_path
            self.tokenizer = tokenizer
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_model_len = max_model_len
            self.calls = []
            instances.append(self)

        def generate(
            self,
            examples,
            *,
            num_samples=1,
            temperature=0.0,
            max_new_tokens=512,
            seed=42,
        ):
            self.calls.append(
                {
                    "example_ids": [example.example_id for example in examples],
                    "num_samples": num_samples,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                }
            )
            return {
                example.example_id: [
                    "<think>play a legal opening move</think><move>e2e4</move>"
                    for _ in range(num_samples)
                ]
                for example in examples
            }

    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: _eval_args(
            tmp_path,
            benchmark_dir=benchmark_dir,
            inference_backend="vllm",
            pass_k=3,
            temperature=0.9,
            max_new_tokens=32,
            vllm_max_model_len=4096,
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [_puzzle_example()],
    )
    monkeypatch.setattr(evaluate, "load_prompt_tokenizer", lambda _model: "tokenizer")
    monkeypatch.setattr(evaluate, "VllmPredictionGenerator", FakeVllmPredictionGenerator)

    assert evaluate._main_impl() == EVAL_SUCCESS_EXIT_CODE

    assert len(instances) == 1
    assert instances[0].model_path == "model-id"
    assert instances[0].tokenizer == "tokenizer"
    assert instances[0].gpu_memory_utilization == 0.85
    assert instances[0].max_model_len == 4096
    assert instances[0].calls == [
        {
            "example_ids": ["planning_00001"],
            "num_samples": 1,
            "temperature": 0.0,
            "max_new_tokens": 32,
            "seed": 42,
        },
        {
            "example_ids": ["planning_00001"],
            "num_samples": 2,
            "temperature": 0.9,
            "max_new_tokens": 32,
            "seed": 42,
        },
    ]


def test_transformers_pass_k_samples_are_generated_in_one_batched_call(monkeypatch):
    from chess_llm.training import evaluate

    class FakeTorch:
        @staticmethod
        def no_grad():
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return Context()

        @staticmethod
        def manual_seed(_seed):
            return None

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def manual_seed_all(_seed):
                return None

    class FakeTensor:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)

        def __getitem__(self, key):
            if isinstance(key, slice):
                return FakeTensor(self.values[key])
            return self.values[key]

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            self.prompts = prompts
            self.kwargs = kwargs
            return FakeInputs(
                {
                    "input_ids": [
                        FakeTensor([10, 11]),
                        FakeTensor([20, 21]),
                    ]
                }
            )

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(token) for token in token_ids.values)

    class FakeModel:
        device = "cuda"

        def __init__(self):
            self.generate_calls = []

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return [
                FakeTensor([10, 11, 100]),
                FakeTensor([10, 11, 101]),
                FakeTensor([10, 11, 102]),
                FakeTensor([20, 21, 200]),
                FakeTensor([20, 21, 201]),
                FakeTensor([20, 21, 202]),
            ]

    monkeypatch.setattr(evaluate, "_get_torch", lambda: FakeTorch)
    monkeypatch.setattr(evaluate, "format_prompt", lambda ex, _tokenizer: ex.prompt)

    model = FakeModel()
    predictions = evaluate.generate_predictions_transformers(
        model,
        FakeTokenizer(),
        [_benchmark_example(), _puzzle_example()],
        num_samples=3,
        temperature=0.7,
        max_new_tokens=8,
        batch_size=6,
        seed=123,
    )

    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["num_return_sequences"] == 3
    assert predictions == {
        "planning_00000": ["100", "101", "102"],
        "planning_00001": ["200", "201", "202"],
    }


def test_transformers_pass_k_with_greedy_temperature_raises_value_error():
    from chess_llm.training import evaluate

    class FakeModel:
        def generate(self, **_kwargs):
            raise AssertionError("generation must not run for greedy pass@k")

    with pytest.raises(ValueError, match="temperature > 0"):
        evaluate.generate_predictions_transformers(
            FakeModel(),
            object(),
            [_puzzle_example()],
            num_samples=3,
            temperature=0.0,
        )


def test_vllm_pass_k_with_greedy_temperature_raises_value_error(monkeypatch):
    from chess_llm.training import evaluate

    class FakeTokenizer:
        name_or_path = "fake-tokenizer"

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **_kwargs):
            self.generate_calls = []

        def generate(self, prompts, sampling_params):
            self.generate_calls.append((list(prompts), sampling_params.kwargs))
            raise AssertionError("generation must not run for greedy pass@k")

    monkeypatch.setattr(evaluate, "format_prompt", lambda ex, _tokenizer: ex.prompt)

    generator = evaluate.VllmPredictionGenerator(
        "model-id",
        FakeTokenizer(),
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
    )

    with pytest.raises(ValueError, match="temperature > 0"):
        generator.generate(
            [_puzzle_example()],
            num_samples=2,
            temperature=0.0,
        )
    assert generator.llm.generate_calls == []


def test_run_evaluation_rejects_pass_k_with_zero_temperature_before_model_load(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")

    def fail_model_load(*_args, **_kwargs):
        raise AssertionError("model must not load for invalid pass@k config")

    monkeypatch.setattr(evaluate, "load_model_and_tokenizer", fail_model_load)

    with pytest.raises(ValueError, match="temperature > 0"):
        evaluate.run_evaluation(
            evaluate.EvaluationConfig(
                model="model-id",
                benchmark_dir=benchmark_dir,
                output=tmp_path / "predictions.jsonl",
                pass_k=3,
                temperature=0.0,
                no_acpl=True,
                no_wandb=True,
                report_only=True,
                soft_gate=True,
            )
        )


def test_merge_predictions_for_save_does_not_mutate_greedy_predictions():
    from chess_llm.training import evaluate

    predictions = {"puzzle": ["greedy-normalized"]}
    raw_predictions = {"puzzle": ["greedy-raw"]}
    sampled_predictions = {"puzzle": ["sampled-normalized"]}
    raw_sampled_predictions = {"puzzle": ["sampled-raw"]}

    merged, raw_merged = evaluate._merge_predictions_for_save(
        predictions,
        raw_predictions,
        sampled_predictions,
        raw_sampled_predictions,
    )

    assert merged == {"puzzle": ["greedy-normalized", "sampled-normalized"]}
    assert raw_merged == {"puzzle": ["greedy-raw", "sampled-raw"]}
    assert predictions == {"puzzle": ["greedy-normalized"]}
    assert raw_predictions == {"puzzle": ["greedy-raw"]}


def test_save_predictions_can_include_benchmark_context(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"

    evaluate.save_predictions(
        {"planning_00000": ["e2e4"]},
        output_path,
        examples_by_id={"planning_00000": _benchmark_example()},
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert row["example_id"] == "planning_00000"
    assert row["sample_index"] == 0
    assert row["split"] == "planning"
    assert row["task_type"] == "best_move"
    assert row["metric_type"] == "move_extraction"
    assert row["fen"] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert row["gold_answer"] == "e2e4"


def test_save_predictions_includes_fen_diagnostics(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    example = BenchmarkExample(
        example_id="perception_00000",
        split="perception",
        task_type="state_tracking",
        fen="7k/8/8/8/8/8/4P3/4K3 w - - 0 1",
        prompt="Starting FEN: 7k/8/8/8/8/8/4P3/4K3 w - - 0 1\nAfter e2e4?",
        gold_answer="7k/8/8/8/4P3/8/8/4K3 b - - 0 1",
        metric_type="fen_exact_match",
        metadata={},
    )

    evaluate.save_predictions(
        {"perception_00000": ["Result FEN: 7k/8/8/8/4P3/8/8/4K3 b - - 9 9"]},
        output_path,
        examples_by_id={"perception_00000": example},
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert row["prompt"] == example.prompt
    assert row["score"]["primary"] == 0.0
    assert row["diagnostics"]["fen"]["prediction_fen"] == "7k/8/8/8/4P3/8/8/4K3 b - - 9 9"
    assert row["diagnostics"]["fen"]["gold_fen"] == "7k/8/8/8/4P3/8/8/4K3 b - - 0 1"
    assert row["diagnostics"]["fen"]["first4_match"] is True
    assert row["diagnostics"]["fen"]["full_match"] is False


def test_save_predictions_includes_invalid_fen_candidate_diagnostics(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    example = BenchmarkExample(
        example_id="perception_00000",
        split="perception",
        task_type="fen_assembly",
        fen="7k/8/8/8/8/8/4P3/4K3 w - - 0 1",
        prompt="Starting FEN: ...\nMove: e2e4",
        gold_answer="Result FEN: 7k/8/8/8/4P3/8/8/4K3 b - - 0 1",
        metric_type="fen_exact_match",
        metadata={},
    )
    invalid_prediction = (
        "Lookup: e2=white pawn; e4=empty.\n"
        "Squares: e2 white pawn->empty; e4 empty->white pawn.\n"
        "Result FEN: 7k/8/8/8/4P3/8/8/4K2K b - - 0 1"
    )

    evaluate.save_predictions(
        {"perception_00000": [invalid_prediction]},
        output_path,
        examples_by_id={"perception_00000": example},
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))

    fen_diag = row["diagnostics"]["fen"]
    assert row["score"]["primary"] == 0.0
    assert fen_diag["prediction_fen"] is None
    assert fen_diag["prediction_fen_candidate"] == "7k/8/8/8/4P3/8/8/4K2K b - - 0 1"
    assert fen_diag["prediction_fen_syntax_valid"] is True
    assert fen_diag["prediction_fen_board_valid"] is False
    assert fen_diag["gold_fen_syntax_valid"] is True
    assert fen_diag["gold_fen_board_valid"] is True


def test_save_predictions_includes_uci_set_diagnostics(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    example = BenchmarkExample(
        example_id="rules_00000",
        split="rules",
        task_type="legal_moves",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        prompt="FEN: start\nList all legal moves.",
        gold_answer="d2d4 e2e4",
        metric_type="uci_set_jaccard",
        metadata={},
    )

    evaluate.save_predictions(
        {"rules_00000": ["e2e4 a1a2"]},
        output_path,
        examples_by_id={"rules_00000": example},
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert row["score"]["primary"] == 1 / 3
    assert row["diagnostics"]["uci_set"] == {
        "prediction_count": 2,
        "gold_count": 2,
        "true_positive_count": 1,
        "extra_count": 1,
        "missing_count": 1,
        "precision": 0.5,
        "recall": 0.5,
        "illegal_extra_count": 1,
    }


def test_save_predictions_includes_legality_reason_diagnostics(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    example = BenchmarkExample(
        example_id="rules_00000",
        split="rules",
        task_type="legality_check",
        fen=STARTING_FEN,
        prompt="FEN: start\nIs a3a4 legal?",
        gold_answer="No, illegal. Reason: empty source.",
        metric_type="legality_check",
        metadata={
            "move": "a3a4",
            "legality_reason_label": "empty_source",
        },
    )

    evaluate.save_predictions(
        {"rules_00000": ["No, illegal. Reason: wrong side piece."]},
        output_path,
        examples_by_id={"rules_00000": example},
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert row["score"]["primary"] == 1.0
    assert row["score"]["legality_reason_accuracy"] == 0.0
    assert row["diagnostics"]["legality_check"] == {
        "prediction_is_legal": False,
        "gold_is_legal": False,
        "binary_match": True,
        "prediction_reason_label": "wrong_side_piece",
        "gold_reason_label": "empty_source",
        "reason_match": False,
    }


def test_save_predictions_and_analysis_include_trace_metrics(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    prediction = (
        "<think>"
        "Candidates: e2e4 d2d4. "
        "Line: e2e4 e7e5 g1f3. "
        "Backtrack: d2d4 is slower. "
        "Best: e2e4."
        "</think>\n"
        "<move>e2e4</move>"
    )
    example = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )

    evaluate.save_predictions(
        {"planning_00000": [prediction]},
        output_path,
        examples_by_id={"planning_00000": example},
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))

    assert row["score"]["trace_referenced_move_accuracy"] == 1.0
    assert row["score"]["trace_step_accuracy"] == 1.0
    assert row["diagnostics"]["trace"]["trace_candidate_count"] == 2
    assert row["diagnostics"]["trace"]["trace_line_depth"] == 3

    report_path = evaluate.write_prediction_analysis_report(
        output_path,
        examples_by_id={"planning_00000": example},
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert (
        report["tasks"]["best_move"]["score_metrics"][
            "trace_referenced_move_accuracy"
        ]
        == 1.0
    )
    assert report["tasks"]["best_move"]["score_metrics"]["trace_line_depth"] == 3.0


def test_prediction_analysis_report_flags_move_edit_format_bleed(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    wrong_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    board_to_fen = BenchmarkExample(
        example_id="perception_board_to_fen",
        split="perception",
        task_type="board_to_fen",
        fen=STARTING_FEN,
        prompt="Here is the current board:\n...\nWrite the FEN for this position.",
        gold_answer=STARTING_FEN,
        metric_type="fen_exact_match",
        metadata={},
    )
    fen_assembly = BenchmarkExample(
        example_id="perception_fen_assembly",
        split="perception",
        task_type="fen_assembly",
        fen=STARTING_FEN,
        prompt="Starting FEN: ...\nMove: d1f3\nUse square lookups and rank edits.",
        gold_answer=wrong_fen,
        metric_type="fen_exact_match",
        metadata={},
    )
    trace_prediction = "\n".join(
        [
            "Source d1=white queen.",
            "Destination f3=empty.",
            "Rank 1 d Q->1.",
            f"Result FEN: {wrong_fen}",
        ]
    )

    evaluate.save_predictions(
        {
            "perception_board_to_fen": [trace_prediction],
            "perception_fen_assembly": [trace_prediction],
        },
        output_path,
        examples_by_id={
            "perception_board_to_fen": board_to_fen,
            "perception_fen_assembly": fen_assembly,
        },
    )

    report_path = evaluate.write_prediction_analysis_report(output_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "prediction_analysis.v1"
    assert report["predictions_path"] == str(output_path)
    assert report["row_count"] == 2
    assert report["tasks"]["board_to_fen"]["format_families"]["square_edit_trace"] == 1
    assert report["tasks"]["board_to_fen"]["format_bleed_count"] == 1
    assert report["tasks"]["fen_assembly"]["format_bleed_count"] == 0
    assert report["format_bleed"][0]["example_id"] == "perception_board_to_fen"
    assert report["tasks"]["board_to_fen"]["failure_examples"][0]["prediction_excerpt"].startswith(
        "Source d1=white queen."
    )


def test_prediction_analysis_report_classifies_legal_move_lists_as_uci(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    example = BenchmarkExample(
        example_id="rules_legal_moves",
        split="rules",
        task_type="legal_moves",
        fen=STARTING_FEN,
        prompt="FEN: start\nList all legal moves.",
        gold_answer="a2a3 a2a4",
        metric_type="uci_set_jaccard",
        metadata={},
    )

    evaluate.save_predictions(
        {"rules_legal_moves": ["Side to move: white. Legal moves: a2a3 a2a4"]},
        output_path,
        examples_by_id={"rules_legal_moves": example},
    )

    report_path = evaluate.write_prediction_analysis_report(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["tasks"]["legal_moves"]["format_families"] == {"uci_moves": 1}


def test_prediction_analysis_report_refreshes_legal_moves_by_piece_score_metrics(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    gold = (
        "Side to move: white.\n"
        "Pieces: a1 white rook.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1b1\n"
        "All legal moves: a1a2 a1b1"
    )
    prediction = (
        "Side to move: white.\n"
        "Pieces: a1 white rook.\n"
        "Moves by piece:\n"
        "a1 white rook: a1a2 a1a3\n"
        "All legal moves: a1a2 a1a3"
    )
    example = BenchmarkExample(
        example_id="rules_grouped",
        split="rules",
        task_type="legal_moves_by_piece",
        fen=STARTING_FEN,
        prompt="FEN: start\nGroup all legal moves by side-to-move piece.",
        gold_answer=gold,
        metric_type="text_exact_match",
        metadata={},
    )
    output_path.write_text(
        json.dumps(
            {
                "example_id": "rules_grouped",
                "sample_index": 0,
                "prediction": prediction,
                "score": {"primary": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = evaluate.write_prediction_analysis_report(
        output_path,
        examples_by_id={"rules_grouped": example},
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    task = report["tasks"]["legal_moves_by_piece"]
    assert round(task["score_metrics"]["all_moves_jaccard"], 3) == 0.333
    assert task["score_metrics"]["all_legal_line_present"] == 1.0
    assert task["score_metrics"]["illegal_extra_count"] == 1.0
    assert task["score_metrics"]["missing_move_count"] == 1.0
    assert round(task["failure_examples"][0]["score"]["all_moves_jaccard"], 3) == 0.333


def test_prediction_analysis_report_breaks_down_step_verification_metadata(tmp_path):
    from chess_llm.training import evaluate
    from chess_llm.evals.benchmark import BenchmarkExample

    output_path = tmp_path / "predictions.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "example_id": "verify_1",
                "sample_index": 0,
                "prediction": "\n".join(
                    [
                        "Verdict: broken",
                        "Faulty line: 1",
                        "Error type: wrong_bucket",
                        "Correction: Candidate e2e4 bucket should be equal.",
                    ]
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    example = BenchmarkExample(
        example_id="verify_1",
        split="planning",
        task_type="step_verification",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="\n".join(
            [
                "Verdict: broken",
                "Faulty line: 1",
                "Error type: wrong_bucket",
                "Correction: Candidate e2e4 bucket should be equal.",
            ]
        ),
        metric_type="step_verification",
        metadata={
            "source_task": "7.8_candidate_ratings",
            "error_type": "wrong_bucket",
            "corruption_kind": "candidate_rating_wrong_bucket",
        },
    )

    report_path = evaluate.write_prediction_analysis_report(
        output_path,
        examples_by_id={"verify_1": example},
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    breakdowns = report["tasks"]["step_verification"]["metadata_breakdowns"]

    assert breakdowns["source_task"] == {"7.8_candidate_ratings": 1}
    assert breakdowns["error_type"] == {"wrong_bucket": 1}
    assert breakdowns["corruption_kind"] == {"candidate_rating_wrong_bucket": 1}


def test_compute_wpd_scores_multiple_samples_per_prompt(monkeypatch, tmp_path):
    from chess_llm.training import evaluate

    calls: list[str] = []

    class FakeDiagnostics:
        def __init__(self, move: str) -> None:
            self.move = move

        def to_metric_dict(self) -> dict[str, object]:
            return {
                "predicted_move": self.move,
                "best_move": "e2e4",
                "best_expectation": 0.6,
                "predicted_expectation": 0.6 if self.move == "e2e4" else 0.3,
                "wpd": 0.0 if self.move == "e2e4" else 0.3,
                "reward": 1.0 if self.move == "e2e4" else 0.7,
                "reward_bucket": "best" if self.move == "e2e4" else "playable",
                "multipv_hit": True,
                "postmove_hit": False,
                "cache_hit": False,
            }

    def fake_score_move_wpd(_engine, _fen, uci, **_kwargs):
        calls.append(uci)
        return FakeDiagnostics(uci)

    monkeypatch.setattr(evaluate, "score_move_wpd", fake_score_move_wpd)
    example = _benchmark_example()

    scores = evaluate.compute_wpd(
        object(),
        [example],
        {
            example.example_id: [
                "<think>x</think><move>e2e4</move>",
                "<think>x</think><move>d2d4</move>",
                "no move",
            ]
        },
        depth=3,
        cache_path=tmp_path / "multipv.sqlite",
    )

    assert calls == ["e2e4", "d2d4"]
    assert isinstance(scores[example.example_id], list)
    assert [item["predicted_move"] for item in scores[example.example_id]] == [
        "e2e4",
        "d2d4",
        None,
    ]
    assert scores[example.example_id][2]["reward_bucket"] == "missing_move"


def test_compute_acpl_uses_shared_clamp_for_invalid_and_tail_losses(monkeypatch):
    from chess_llm.evals.benchmark import ACPL_INVALID_MOVE_PENALTY
    from chess_llm.training import evaluate

    invalid = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )
    tail = BenchmarkExample(
        example_id="planning_00001",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )

    monkeypatch.setattr(evaluate, "_evaluate_position", lambda *_args, **_kwargs: 10000)
    monkeypatch.setattr(
        evaluate,
        "_evaluate_predicted_move",
        lambda *_args, **_kwargs: -10000,
    )

    scores = evaluate.compute_acpl(
        object(),
        [invalid, tail],
        {
            invalid.example_id: "no move tag",
            tail.example_id: "<move>e2e4</move>",
        },
        depth=1,
    )

    assert scores[invalid.example_id] == ACPL_INVALID_MOVE_PENALTY
    assert scores[tail.example_id] == ACPL_INVALID_MOVE_PENALTY


def test_prediction_analysis_report_classifies_fen_row_rewrite_trace(tmp_path):
    from chess_llm.training import evaluate

    output_path = tmp_path / "predictions.jsonl"
    example = BenchmarkExample(
        example_id="perception_fen_row_application",
        split="perception",
        task_type="fen_row_application",
        fen=STARTING_FEN,
        prompt="Starting FEN: ...\nMove: e2e4\nRewrite the affected rows.",
        gold_answer=(
            "Rows: rank 2 PPPPPPPP->PPPP1PPP; rank 4 8->4P3.\n"
            "Result FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        ),
        metric_type="fen_exact_match",
        metadata={"move": "e2e4"},
    )

    evaluate.save_predictions(
        {
            "perception_fen_row_application": [
                "Rows: rank 2 PPPPPPPP->PPPP1PPP; rank 4 8->4P3.\n"
                "Result FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ]
        },
        output_path,
        examples_by_id={"perception_fen_row_application": example},
    )

    report_path = evaluate.write_prediction_analysis_report(output_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["tasks"]["fen_row_application"]["format_families"] == {
        "fen_row_rewrite_trace": 1
    }
    assert report["tasks"]["fen_row_application"]["format_bleed_count"] == 0


def test_run_evaluation_returns_structured_result_without_cli_parsing(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_SUCCESS_EXIT_CODE

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")
    output_path = tmp_path / "predictions.jsonl"

    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )
    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [_benchmark_example()],
    )
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {
            "planning_00000": [
                "<think>claim the center</think><move>e2e4</move>",
            ],
        },
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=output_path,
            max_new_tokens=32,
            batch_size=1,
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert result.return_code == EVAL_SUCCESS_EXIT_CODE
    assert result.predictions_path == output_path
    assert result.results_path == output_path.with_suffix(".results.json")
    assert result.eval_run_path == output_path.with_suffix(".eval_run.json")
    assert result.split_counts == {"planning": 1}
    assert result.split_results["planning"]["best_move"] == 1.0
    assert output_path.exists()
    assert result.results_path.exists()
    assert result.eval_run_path.exists()
    analysis_path = output_path.with_suffix(".analysis.json")
    assert analysis_path.exists()

    eval_run = json.loads(result.eval_run_path.read_text(encoding="utf-8"))
    assert eval_run["schema_version"] == "artifact.v1"
    assert eval_run["artifact_type"] == "evaluation_run"
    assert eval_run["run_id"].startswith("eval-")
    assert eval_run["created_at_utc"]
    assert eval_run["model_id"] == "model-id"
    assert eval_run["phase"] is None
    assert eval_run["benchmark_dir"] == str(benchmark_dir)
    assert eval_run["benchmark_manifest_path"] == str(benchmark_dir / "manifest.json")
    assert eval_run["benchmark_version"] == "unknown"
    assert eval_run["predictions_path"] == str(output_path)
    assert eval_run["results_path"] == str(result.results_path)
    assert eval_run["return_code"] == EVAL_SUCCESS_EXIT_CODE
    assert eval_run["split_counts"] == {"planning": 1}
    assert eval_run["has_acpl"] is False
    assert eval_run["n_failures"] == 0
    assert eval_run["inference"]["backend"] == "transformers"
    assert eval_run["inference"]["pass_k"] == 1
    assert eval_run["inference"]["primary_temperature"] == 0.0
    assert eval_run["inference"]["sample_temperature"] == 0.7
    assert eval_run["inference"]["max_new_tokens"] == 32
    assert eval_run["inference"]["batch_size"] == 1
    assert eval_run["scoring"]["no_acpl"] is True
    assert eval_run["scoring"]["acpl_depth"] == 20
    assert eval_run["gate"]["baseline_path"] is None
    assert eval_run["gate"]["report_only"] is True
    assert eval_run["gate"]["soft_gate"] is True
    assert eval_run["metadata"]["eval_run_path"] == str(result.eval_run_path)
    assert eval_run["metadata"]["prediction_analysis_path"] == str(analysis_path)


def test_planning_legal_move_rate_counts_tagless_predictions_as_zero(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")

    tagless_example = BenchmarkExample(
        example_id="planning_00002",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )

    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )
    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [_benchmark_example(), tagless_example],
    )
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {
            "planning_00000": [
                "<think>claim the center</think><move>e2e4</move>",
            ],
            "planning_00002": ["I do not see a good continuation here."],
        },
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    planning = result.split_results["planning"]
    # The tag-less prediction stays in the denominator as 0.0 instead of
    # silently inflating the Phase C gate metric.
    assert planning["legal_move_rate"] == 0.5
    assert planning["missing_move_tag_count"] == 1.0


def test_eval_output_budget_defaults_to_512_tokens(monkeypatch):
    from chess_llm.training import evaluate

    monkeypatch.setattr(
        "sys.argv",
        [
            "chess-llm-evaluate",
            "--model", "model-id",
            "--benchmark-dir", "benchmark",
            "--output", "predictions.jsonl",
        ],
    )

    args = evaluate.parse_args()

    assert args.max_new_tokens == 512
    config = evaluate.EvaluationConfig(
        model="model-id",
        benchmark_dir=Path("benchmark"),
        output=Path("predictions.jsonl"),
    )
    assert config.max_new_tokens == 512


def test_flash_attention_eval_loads_bfloat16_weights(monkeypatch):
    from chess_llm.training import evaluate

    captured: dict[str, object] = {}
    fake_model = types.SimpleNamespace(eval=lambda: None, to=lambda _device: fake_model)

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
        ),
    )

    monkeypatch.setattr(evaluate, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(evaluate, "load_prompt_tokenizer", lambda _model: "tokenizer")

    def fake_load_causal_lm_with_attention(
        _model_cls,
        _model_path,
        model_kwargs,
        *,
        requested_attn,
        logger,
    ):
        captured["model_kwargs"] = model_kwargs
        captured["requested_attn"] = requested_attn
        return fake_model, requested_attn

    monkeypatch.setattr(
        evaluate,
        "load_causal_lm_with_attention",
        fake_load_causal_lm_with_attention,
    )

    model, tokenizer = evaluate.load_model_and_tokenizer(
        "checkpoint",
        attn_implementation="flash_attention_3",
    )

    assert model is fake_model
    assert tokenizer == "tokenizer"
    assert captured["requested_attn"] == "flash_attention_3"
    assert captured["model_kwargs"]["torch_dtype"] is fake_torch.bfloat16


def test_phase_a_evaluation_loads_only_foundation_splits_by_default(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    for split_name in ["perception", "rules", "planning"]:
        (benchmark_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

    loaded_paths: list[str] = []
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )

    def fake_load_benchmark(path, **_kwargs):
        loaded_paths.append(Path(path).stem)
        return [_split_example(Path(path).stem)]

    monkeypatch.setattr(evaluate, "load_benchmark", fake_load_benchmark)
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {
            "perception_00000": ["answer"],
            "rules_00000": ["answer"],
        },
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            phase="a",
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert loaded_paths == ["perception", "rules"]
    assert result.split_counts == {"perception": 1, "rules": 1}
    eval_run = json.loads(result.eval_run_path.read_text(encoding="utf-8"))
    assert eval_run["inference"]["effective_splits"] == ["perception", "rules"]
    assert eval_run["inference"]["full_benchmark"] is False


def test_phase_a_evaluation_filters_non_foundation_task_types(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    for split_name in ["perception", "rules"]:
        (benchmark_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )

    def fake_load_benchmark(path, **_kwargs):
        split = Path(path).stem
        if split == "perception":
            return [
                _split_example("perception"),
                BenchmarkExample(
                    example_id="perception_square_lookup",
                    split="perception",
                    task_type="square_lookup",
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    prompt="FEN: ...",
                    gold_answer="e2=white pawn",
                    metric_type="text_exact_match",
                    metadata={"diagnostic": True, "hard_gate": False},
                ),
                BenchmarkExample(
                    example_id="perception_square_coordinates",
                    split="perception",
                    task_type="square_coordinates",
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    prompt="FEN: ...",
                    gold_answer=(
                        "e2: file=e; rank=2; fen_row_from_top=7; file_index=5"
                    ),
                    metric_type="text_exact_match",
                    metadata={"diagnostic": True, "hard_gate": False},
                ),
                BenchmarkExample(
                    example_id="perception_fen_rank_expansion",
                    split="perception",
                    task_type="fen_rank_expansion",
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    prompt="FEN: ...",
                    gold_answer=(
                        "rank 2: a=P; b=P; c=P; d=P; e=P; f=P; g=P; h=P"
                    ),
                    metric_type="text_exact_match",
                    metadata={"diagnostic": True, "hard_gate": False},
                ),
                BenchmarkExample(
                    example_id="perception_fen_rank_cell_edit",
                    split="perception",
                    task_type="fen_rank_cell_edit",
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    prompt="FEN: ...",
                    gold_answer="rank 2: PPPPPPPP -> PPPP1PPP",
                    metric_type="text_exact_match",
                    metadata={"diagnostic": True, "hard_gate": False},
                ),
                BenchmarkExample(
                    example_id="perception_fen_board_edit",
                    split="perception",
                    task_type="fen_board_edit",
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    prompt="FEN: ...",
                    gold_answer=(
                        "Result board FEN: "
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR"
                    ),
                    metric_type="text_exact_match",
                    metadata={"diagnostic": True, "hard_gate": False},
                ),
            ]
        return [
            BenchmarkExample(
                example_id="rules_legal",
                split="rules",
                task_type="legal_moves",
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                prompt="FEN: ...",
                gold_answer="e2e4",
                metric_type="jaccard",
                metadata={},
            ),
            BenchmarkExample(
                example_id="rules_captures",
                split="rules",
                task_type="captures",
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                prompt="FEN: ...",
                gold_answer="No captures available.",
                metric_type="jaccard",
                metadata={},
            ),
            BenchmarkExample(
                example_id="rules_piece_legal_moves",
                split="rules",
                task_type="piece_legal_moves",
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                prompt="FEN: ...",
                gold_answer="No legal moves.",
                metric_type="uci_set_jaccard",
                metadata={"diagnostic": True, "hard_gate": False},
            ),
        ]

    seen_example_ids: list[str] = []

    def fake_generate(_model, _tokenizer, examples, **_kwargs):
        seen_example_ids.extend(example.example_id for example in examples)
        return {example.example_id: ["answer"] for example in examples}

    monkeypatch.setattr(evaluate, "load_benchmark", fake_load_benchmark)
    monkeypatch.setattr(evaluate, "generate_predictions_transformers", fake_generate)

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            phase="a",
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert seen_example_ids == [
        "perception_00000",
        "perception_square_lookup",
        "perception_square_coordinates",
        "perception_fen_rank_expansion",
        "perception_fen_rank_cell_edit",
        "perception_fen_board_edit",
        "rules_legal",
        "rules_piece_legal_moves",
    ]
    assert result.split_counts == {"perception": 6, "rules": 2}
    predictions_text = result.predictions_path.read_text(encoding="utf-8")
    assert "perception_square_lookup" in predictions_text
    assert "perception_square_coordinates" in predictions_text
    assert "perception_fen_rank_expansion" in predictions_text
    assert "perception_fen_rank_cell_edit" in predictions_text
    assert "perception_fen_board_edit" in predictions_text
    assert "rules_legal" in predictions_text
    assert "rules_piece_legal_moves" in predictions_text
    assert "rules_captures" not in predictions_text


def test_phase_a_evaluation_can_run_full_benchmark_override(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    for split_name in ["perception", "rules", "planning"]:
        (benchmark_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

    loaded_paths: list[str] = []
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )

    def fake_load_benchmark(path, **_kwargs):
        loaded_paths.append(Path(path).stem)
        return [_split_example(Path(path).stem)]

    monkeypatch.setattr(evaluate, "load_benchmark", fake_load_benchmark)
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {
            "perception_00000": ["answer"],
            "rules_00000": ["answer"],
            "planning_00000": ["answer"],
        },
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            phase="a",
            full_benchmark=True,
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert loaded_paths == ["perception", "planning", "rules"]
    assert result.split_counts == {
        "perception": 1,
        "planning": 1,
        "rules": 1,
    }


def test_phase_c_default_evaluation_skips_auxiliary_splits(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    for split_name in [
        "perception",
        "rules",
        "tactics",
        "evaluation",
        "openings",
        "endgames",
        "planning",
        "chess960",
        "mate",
    ]:
        (benchmark_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

    loaded_paths: list[str] = []
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )

    def fake_load_benchmark(path, **_kwargs):
        split = Path(path).stem
        loaded_paths.append(split)
        return [_split_example(split)]

    monkeypatch.setattr(evaluate, "load_benchmark", fake_load_benchmark)
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {
            f"{split}_00000": ["answer"]
            for split in loaded_paths
        },
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            phase="c",
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert loaded_paths == [
        "endgames",
        "evaluation",
        "openings",
        "perception",
        "planning",
        "rules",
        "tactics",
    ]
    assert "chess960" not in result.split_counts
    assert "mate" not in result.split_counts


def test_evaluation_passes_split_cap_into_benchmark_loader(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    for split_name in ["perception", "rules"]:
        (benchmark_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

    seen_limits: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )

    def fake_load_benchmark(path, *, max_examples=None, **_kwargs):
        seen_limits.append((Path(path).stem, max_examples))
        return [_split_example(Path(path).stem)]

    monkeypatch.setattr(evaluate, "load_benchmark", fake_load_benchmark)
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {
            "perception_00000": ["answer"],
            "rules_00000": ["answer"],
        },
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            phase="a",
            max_examples_per_split=1,
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert result.split_counts == {"perception": 1, "rules": 1}
    assert seen_limits == [("perception", 1), ("rules", 1)]


def test_phase_task_filter_is_applied_while_loading_capped_splits(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    for split_name in ["rules"]:
        (benchmark_dir / f"{split_name}.jsonl").write_text("{}\n", encoding="utf-8")

    captured_task_types: list[frozenset[str] | None] = []
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: ("model", "tokenizer"),
    )

    def fake_load_benchmark(path, *, max_examples=None, task_types=None):
        captured_task_types.append(None if task_types is None else frozenset(task_types))
        assert max_examples == 1
        assert task_types is not None
        assert "legal_moves" in task_types
        assert "captures" not in task_types
        return [_split_example(Path(path).stem)]

    monkeypatch.setattr(evaluate, "load_benchmark", fake_load_benchmark)
    monkeypatch.setattr(
        evaluate,
        "generate_predictions_transformers",
        lambda *_args, **_kwargs: {"rules_00000": ["answer"]},
    )

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=benchmark_dir,
            output=tmp_path / "predictions.jsonl",
            phase="a",
            max_examples_per_split=1,
            splits=("rules",),
            no_acpl=True,
            no_wandb=True,
            report_only=True,
            soft_gate=True,
        )
    )

    assert result.split_counts == {"rules": 1}
    assert captured_task_types


def test_run_evaluation_returns_structured_infra_failure_without_model_load(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    def fail_model_load(*_args, **_kwargs):
        raise AssertionError("model should not load for missing benchmark")

    monkeypatch.setattr(evaluate, "load_model_and_tokenizer", fail_model_load)

    result = evaluate.run_evaluation(
        evaluate.EvaluationConfig(
            model="model-id",
            benchmark_dir=tmp_path / "missing",
            output=tmp_path / "predictions.jsonl",
            no_acpl=True,
            no_wandb=True,
        )
    )

    assert result.return_code == EVAL_INFRA_FAILURE_EXIT_CODE
    assert result.split_results == {}
    assert result.split_counts == {}
    assert result.predictions_path == tmp_path / "predictions.jsonl"
    assert result.results_path == (tmp_path / "predictions.results.json")
    assert result.eval_run_path == (tmp_path / "predictions.eval_run.json")
    assert result.eval_run_path.exists()

    eval_run = json.loads(result.eval_run_path.read_text(encoding="utf-8"))
    assert eval_run["return_code"] == EVAL_INFRA_FAILURE_EXIT_CODE
    assert eval_run["split_counts"] == {}
    assert eval_run["metadata"]["error"] == "benchmark_dir_missing"


def test_eval_only_strict_success_writes_phase_sentinels(monkeypatch, tmp_path):
    from chess_llm.training import train
    from chess_llm.training.eval_exit_codes import EVAL_SUCCESS_EXIT_CODE
    from chess_llm.training.phases import PHASE_PASSED_SENTINEL, PHASE_READY_SENTINEL

    output_root = tmp_path / "checkpoints"
    args = Namespace(
        phase="a",
        data_root=tmp_path / "data",
        output_root=output_root,
        benchmark_dir=tmp_path / "benchmark",
        dry_run=False,
        smoke_run=False,
        eval_only=True,
        skip_eval=False,
        require_phase_gate=True,
        max_train_examples=None,
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        wandb_project="chess-sft",
        no_wandb=True,
        allow_wandb_offline=False,
        run_name=None,
        wandb_group=None,
        disable_liger_kernel=False,
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=1,
        eval_max_new_tokens=16,
        eval_acpl_depth=1,
        no_acpl=True,
        full_acpl_report=False,
        stockfish_path="stockfish",
    )
    calls = []
    (output_root / "phase_a" / "best").mkdir(parents=True)

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(
        train,
        "_run_eval",
        lambda *call_args, **call_kwargs: calls.append((call_args, call_kwargs))
        or EVAL_SUCCESS_EXIT_CODE,
    )

    assert train.main() == EVAL_SUCCESS_EXIT_CODE
    assert calls
    assert (output_root / "phase_a" / PHASE_READY_SENTINEL).exists()
    assert (output_root / "phase_a" / PHASE_PASSED_SENTINEL).exists()


def test_train_preflights_missing_post_training_benchmark_before_dataset_load(
    monkeypatch,
    tmp_path,
):
    from chess_llm.training import train
    from chess_llm.training.data import mixer
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    args = Namespace(
        phase="a",
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        benchmark_dir=tmp_path / "missing-benchmark",
        base_model=None,
        dry_run=False,
        smoke_run=False,
        eval_only=False,
        skip_eval=False,
        require_phase_gate=False,
        max_train_examples=None,
        task_upsample=[],
        max_eval_examples=None,
        max_benchmark_examples_per_split=None,
        full_benchmark_eval=False,
        max_steps=None,
        trainer_eval_steps=None,
        trainer_save_steps=None,
        skip_trainer_eval=False,
        wandb_project="chess-sft",
        no_wandb=True,
        allow_wandb_offline=False,
        run_name=None,
        wandb_group=None,
        disable_liger_kernel=False,
        inference_backend="transformers",
        attn_implementation="auto",
        eval_batch_size=1,
        eval_max_new_tokens=16,
        eval_acpl_depth=1,
        no_acpl=True,
        full_acpl_report=False,
        stockfish_path="stockfish",
    )

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(
        mixer,
        "build_phase_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dataset should not load when post-training eval inputs are missing")
        ),
    )

    assert train.main() == EVAL_INFRA_FAILURE_EXIT_CODE


def test_train_preflight_allows_existing_phase_selected_benchmark_split(tmp_path):
    from chess_llm.training import train

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "perception.jsonl").write_text("{}\n", encoding="utf-8")

    assert train._post_training_eval_preflight_error(
        benchmark_dir,
        phase="a",
        full_benchmark=False,
        inference_backend="transformers",
    ) is None


def test_train_preflight_rejects_empty_selected_benchmark_split(tmp_path):
    from chess_llm.training import train

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "perception.jsonl").write_text("", encoding="utf-8")

    assert "empty" in train._post_training_eval_preflight_error(
        benchmark_dir,
        phase="a",
        full_benchmark=False,
        inference_backend="transformers",
    )


def test_main_uses_evaluation_config_and_returns_result_code(monkeypatch, tmp_path):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_METRIC_FAILURE_EXIT_CODE

    captured = {}
    monkeypatch.setattr(evaluate, "parse_args", lambda: _eval_args(tmp_path, phase="c"))

    def fake_run_evaluation(config):
        captured["config"] = config
        return evaluate.EvaluationResult(
            return_code=EVAL_METRIC_FAILURE_EXIT_CODE,
            version="test",
            split_results={},
            split_counts={},
            has_acpl=False,
            n_failures=2,
            predictions_path=config.output,
            results_path=config.output.with_suffix(".results.json"),
            eval_run_path=config.output.with_suffix(".eval_run.json"),
        )

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)

    assert evaluate.main() == EVAL_METRIC_FAILURE_EXIT_CODE
    assert isinstance(captured["config"], evaluate.EvaluationConfig)
    assert captured["config"].phase == "c"
    assert captured["config"].output == tmp_path / "predictions.jsonl"


def test_evaluate_returns_infra_failure_for_missing_benchmark(monkeypatch, tmp_path):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    monkeypatch.setattr(evaluate, "parse_args", lambda: _eval_args(tmp_path))

    assert evaluate.main() == EVAL_INFRA_FAILURE_EXIT_CODE


def test_evaluate_returns_infra_failure_for_empty_benchmark(monkeypatch, tmp_path):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    monkeypatch.setattr(evaluate, "parse_args", lambda: _eval_args(tmp_path))
    monkeypatch.setattr(
        evaluate,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model should not load for empty benchmark")
        ),
    )

    assert evaluate.main() == EVAL_INFRA_FAILURE_EXIT_CODE
    eval_run = tmp_path / "predictions.eval_run.json"
    assert eval_run.exists()
    payload = json.loads(eval_run.read_text(encoding="utf-8"))
    assert payload["metadata"]["error"] == "benchmark_empty"


def test_evaluate_converts_runtime_error_to_infra_failure(monkeypatch, tmp_path):
    from chess_llm.training import evaluate
    from chess_llm.training.eval_exit_codes import EVAL_INFRA_FAILURE_EXIT_CODE

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "planning.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "parse_args", lambda: _eval_args(tmp_path))
    monkeypatch.setattr(
        evaluate,
        "load_benchmark",
        lambda _path, **_kwargs: [_benchmark_example()],
    )

    def fail_model_load(*_args, **_kwargs):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(evaluate, "load_model_and_tokenizer", fail_model_load)

    assert evaluate.main() == EVAL_INFRA_FAILURE_EXIT_CODE


def test_soft_gate_only_softens_metric_failures():
    from chess_llm.training.train import _post_training_eval_failure_return_code
    from chess_llm.training.eval_exit_codes import (
        EVAL_INFRA_FAILURE_EXIT_CODE,
        EVAL_METRIC_FAILURE_EXIT_CODE,
    )

    assert (
        _post_training_eval_failure_return_code(
            EVAL_METRIC_FAILURE_EXIT_CODE,
            require_phase_gate=False,
        )
        is None
    )
    assert _post_training_eval_failure_return_code(
        EVAL_METRIC_FAILURE_EXIT_CODE,
        require_phase_gate=True,
    ) == EVAL_METRIC_FAILURE_EXIT_CODE
    assert _post_training_eval_failure_return_code(
        EVAL_INFRA_FAILURE_EXIT_CODE,
        require_phase_gate=False,
    ) == EVAL_INFRA_FAILURE_EXIT_CODE


def test_soft_gate_eval_failures_are_summarized_in_training_log(
    caplog,
    tmp_path,
):
    from chess_llm.training.train import _warn_if_soft_gate_failures

    pred_path = tmp_path / "eval_predictions.jsonl"
    eval_run_path = pred_path.with_suffix(".eval_run.json")
    eval_run_path.write_text(
        json.dumps(
            {
                "return_code": 0,
                "n_failures": 3,
                "results_path": str(pred_path.with_suffix(".results.json")),
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        _warn_if_soft_gate_failures(pred_path, require_phase_gate=False)

    assert "soft-gated benchmark eval recorded 3 failure" in caplog.text
    assert str(eval_run_path) in caplog.text


def test_training_resource_release_returns_dropped_refs_and_clears_cuda():
    from chess_llm.training.train import (
        _clear_cuda_cache,
        _release_training_resources,
    )

    events: list[str] = []

    class FakeAccelerator:
        def free_memory(self):
            events.append("free_memory")

    class FakeTrainer:
        def __init__(self):
            self.accelerator = FakeAccelerator()
            self.model = object()

    class FakeCuda:
        @staticmethod
        def is_available():
            events.append("cuda_available")
            return True

        @staticmethod
        def empty_cache():
            events.append("empty_cache")

        @staticmethod
        def ipc_collect():
            events.append("ipc_collect")

    class FakeTorch:
        cuda = FakeCuda()

    trainer = FakeTrainer()
    model = object()

    trainer, model = _release_training_resources(trainer, model)
    _clear_cuda_cache(torch_module=FakeTorch, collect=lambda: events.append("gc"))

    assert trainer is None
    assert model is None
    assert events == ["free_memory", "gc", "cuda_available", "empty_cache", "ipc_collect"]
