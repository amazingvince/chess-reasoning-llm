# Batch Judge Artifact Export

The batch judge is the first file-based bridge from the existing frozen
benchmark workflow into the new artifact system. It does not run inference. It
assumes predictions have already been written by another evaluator or script.

## Inputs

Benchmark input is a directory containing frozen benchmark `*.jsonl` files. Each
row should match the existing `BenchmarkExample.to_dict()` shape:

```json
{
  "example_id": "planning_00000",
  "split": "planning",
  "task_type": "best_move",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "prompt": "FEN: ...\nWhat is the best move?",
  "gold_answer": "e2e4",
  "metric_type": "move_extraction",
  "metadata": {"source": "unit"}
}
```

Prediction input is one JSONL file with one or more rows per benchmark example:

```json
{"example_id":"planning_00000","prediction":"<move>e2e4</move>"}
{"example_id":"planning_00000","prediction":"e2e4","raw_prediction":"<think>...</think>\n<move>e2e4</move>"}
```

When `raw_prediction` exists, it becomes the rollout `raw_output`. Otherwise the
`prediction` field is used. Unknown `example_id` values fail fast with
`ValueError` so bad joins are not silently exported.

## Command

Run these commands after installing the root package with
`python -m pip install -e .`, or set `PYTHONPATH=src` for local development.

Legality-only export:

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B
```

Add repeatable `--split` flags to restrict the loaded benchmark rows:

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B \
  --split planning \
  --split rules
```

Engine-backed export:

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B \
  --stockfish-path C:/path/to/stockfish.exe \
  --stockfish-depth 20 \
  --stockfish-threads 4 \
  --stockfish-hash-mb 1024 \
  --syzygy-path E:/syzygy
```

`--syzygy-path` is passed to Stockfish's `SyzygyPath` option. Direct
python-chess tablebase probing remains a separate future layer.

## Outputs

The output directory contains:

- `prompts.jsonl`: one `PromptArtifact` per referenced benchmark example.
- `rollouts.jsonl`: one `RolloutArtifact` per prediction row.
- `judgments.jsonl`: one `JudgmentArtifact` per rollout.
- `manifest.json`: counts, source paths, model ID, split list, and output paths.

Multiple prediction rows for one `example_id` intentionally produce multiple
rollouts and judgments. That preserves pass@k or sampling runs for later
failure analysis.

## Scope Boundary

This module always uses the bootstrap parser and legality judge. It can
classify `parse_failure`, `illegal_move`, `missing_fen`, and `legal_unscored`.

When Stockfish is configured, legal rollouts are upgraded with:

- `regret_cp`: non-negative centipawn loss versus the engine position score.
- `teacher_move_uci`: first move from Stockfish's principal variation when
  available.
- metadata fields for depth, best eval, model-move eval, and scoring status.

This still does not run live model inference, synthesize missing predictions,
create preference pairs, or create SDPO feedback examples.
