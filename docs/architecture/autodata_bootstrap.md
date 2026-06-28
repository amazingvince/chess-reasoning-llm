# Autodata Bootstrap

## What Exists Now

The first Autodata layer is a library substrate for model-output auditing. It
does not run inference. It assumes another caller already has a prompt and raw
model text.

Available pieces:

- `chess_llm.core.board`: FEN identity, validation, legal moves, and legality
  checks backed by `python-chess`.
- `chess_llm.autodata.rollouts.build_rollout`: creates a `RolloutArtifact`
  and stores both raw output and parsed answer.
- `chess_llm.autodata.judge.judge_rollout`: creates a `JudgmentArtifact` for
  missing FEN, parse failure, illegal move, or legal unscored move.
- `chess_llm.autodata.stockfish_judge.judge_rollout_with_stockfish`: upgrades
  legal rollouts with Stockfish teacher moves and regret scores.
- `chess_llm.autodata.failure_buckets`: shared constants for the bootstrap
  failure taxonomy.
- `chess_llm.external.stockfish`: opens and configures Stockfish UCI engines.
- `chess_llm.evals.benchmark_artifacts`: converts frozen benchmark rows into
  prompt artifacts.
- `chess_llm.evals.batch_judge`: reads benchmark JSONL plus prediction JSONL
  and writes prompt, rollout, judgment, and manifest artifacts.
- `chess_llm.autodata.sft_refresh`: converts judged single-move failures into
  legacy-compatible Tier 7 SFT refresh rows.

## Intentionally Excluded

This layer does not include:

- live model inference
- direct python-chess Syzygy WDL/DTZ probing
- puzzle solution validation
- preference-pair generation
- SDPO feedback-target construction

Those pieces depend on this layer but should be added as separate slices so the
artifact contracts remain easy to test.

## Minimal Flow

```python
from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact
from chess_llm.autodata import build_rollout, judge_rollout

prompt = PromptArtifact(
    prompt_id="prompt-001",
    messages=[ChatMessage(role="user", content="Choose a legal move.")],
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    task_type="best_move",
)

rollout = build_rollout(prompt, "Qwen/Qwen3-4B", "<move>e2e4</move>")
judgment = judge_rollout(prompt, rollout)
```

The judgment will use `legal_unscored` when the parsed move is legal and no
engine is configured. Stockfish-backed judgment upgrades legal moves with
regret and teacher-move fields.

## Benchmark Prediction Export

Existing frozen benchmark examples and prediction files can now be converted
into artifacts without rerunning inference:

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B
```

The prediction file keeps the current benchmark shape:

```json
{"example_id":"planning_00000","prediction":"<move>e2e4</move>"}
```

If `raw_prediction` is present, it becomes the rollout `raw_output`; otherwise
`prediction` is used. Multiple rows with the same `example_id` produce multiple
rollouts and judgments, which keeps pass@k style outputs inspectable.

Add Stockfish options to score legal predictions:

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B \
  --stockfish-path C:/path/to/stockfish.exe \
  --stockfish-depth 20 \
  --syzygy-path E:/syzygy
```

## Next Layers

The natural next slices are:

1. Direct Syzygy WDL/DTZ probing for tablebase-covered positions.
2. Preference-pair and feedback-distillation builders from judged failures.
