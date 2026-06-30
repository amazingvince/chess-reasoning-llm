from __future__ import annotations

from chess_llm.training import evaluate, train


def test_training_and_eval_defaults_do_not_embed_local_machine_paths():
    defaults = [
        str(train.DEFAULT_DATA_ROOT),
        str(train.DEFAULT_OUTPUT_ROOT),
        str(train.DEFAULT_BENCHMARK_DIR),
        train.DEFAULT_STOCKFISH_PATH,
        evaluate.DEFAULT_STOCKFISH_PATH,
    ]

    for value in defaults:
        normalized = value.replace("\\", "/")
        assert not normalized.startswith("E:/")
        assert "C:/Users/amazi/" not in normalized
