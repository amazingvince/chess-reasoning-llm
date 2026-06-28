# Chess LLM

This repository is becoming a package-oriented chess model improvement
system. The near-term goal is to bootstrap a model with supervised chess
competence, then build the loop that evaluates its failures, generates
targeted data, and trains the next iteration.

The stable lifecycle is:

```text
data -> train -> evaluate -> rollout -> judge -> targeted data -> train again
```

## Current Repo Shape

- `sft/make_data/` contains the existing SFT data-generation pipeline,
  frozen benchmark construction, validation, and decontamination tooling.
- `sft/training/` contains the existing phase-aware SFT training and
  benchmark evaluation harness.
- `src/chess_llm/` is the new package home for shared interfaces that should
  outlive any one training method: artifact schemas, answer parsing, future
  Autodata components, preference data, SDPO feedback examples, and inference.
- `plan/` contains the older design documents. They remain useful historical
  context, but new architecture decisions should land under `docs/`.

## First Package Layer

The first package layer intentionally does not move the working SFT code. It
adds shared contracts that later systems can reuse:

- `chess_llm.artifacts.schemas` defines versioned JSONL artifacts.
- `chess_llm.artifacts.jsonl` reads and writes ordered artifact files.
- `chess_llm.formats.answers` parses flexible model outputs into normalized
  move fields without forcing JSON as the model's text format.
- `chess_llm.core.board` provides small python-chess helpers for FEN identity,
  FEN validation, legal move listing, and legality checks.
- `chess_llm.autodata.rollouts` and `chess_llm.autodata.judge` provide the
  first prompt -> rollout -> judgment substrate for the self-guided loop.
- `chess_llm.autodata.stockfish_judge` and `chess_llm.external.stockfish`
  provide the optional engine-backed scoring layer for legal rollouts.
- `chess_llm.autodata.sft_refresh` converts judged failures into legacy
  Tier 7 SFT refresh rows.

The current `<think>...</think><move>...</move>` training format remains valid.
Future rollout and training artifacts store both raw model text and parsed
fields so evaluation, preference training, and feedback distillation can share
the same data.

## Rollout + Judgment Bootstrap

The first runnable Autodata layer is library-first. It does not call a model.
It takes a prompt artifact and raw model output, parses the move, then judges
whether the move is parseable and legal in the prompt FEN.

```python
from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact
from chess_llm.autodata import build_rollout, judge_rollout

prompt = PromptArtifact(
    prompt_id="prompt-001",
    messages=[ChatMessage(role="user", content="FEN: ...")],
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    task_type="best_move",
)

rollout = build_rollout(prompt, "Qwen/Qwen3-4B", "<move>e2e4</move>")
judgment = judge_rollout(prompt, rollout)
```

The initial failure buckets are `parse_failure`, `illegal_move`,
`missing_fen`, and `legal_unscored`. When Stockfish is configured, legal moves
can be upgraded with `regret_cp` and `teacher_move_uci`.

## Batch Judge Artifact Export

The next bridge converts frozen benchmark examples plus existing prediction
JSONL rows into artifact files. It writes prompt, rollout, judgment, and
manifest files under an output directory. By default it uses the legality-only
bootstrap judge.

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B
```

Use repeatable `--split` flags to restrict the loaded benchmark splits:

```bash
python -m chess_llm.evals.batch_judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B \
  --split planning \
  --split rules
```

Pass Stockfish options when you want engine-backed regret and teacher moves:

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

## Autodata SFT Refresh

Judged artifacts can now be turned into targeted SFT rows without changing the
trainer. The builder reads the prompt, rollout, and judgment files produced by
batch judge and writes legacy-compatible Tier 7 JSONL files:

```bash
python -m chess_llm.autodata.sft_refresh \
  --prompts path/to/artifacts/prompts.jsonl \
  --rollouts path/to/artifacts/rollouts.jsonl \
  --judgments path/to/artifacts/judgments.jsonl \
  --output-dir path/to/autodata_refresh
```

The output directory contains `tier7/7.4_autodata_format_repair.jsonl`,
`tier7/7.5_autodata_move_correction.jsonl`, and `manifest.json`. Rows are
created only for single-move benchmark tasks with a legal target move.

## Useful Commands

Install the root package in editable mode before using `python -m chess_llm...`
commands from a normal shell:

```bash
python -m pip install -e .
```

Run the root package tests:

```bash
python -m pytest tests -q
```

Run the existing SFT data tests:

```bash
cd sft/make_data
python -m pytest tests -q
```

Run the existing training harness tests:

```bash
cd sft/training
python -m pytest tests -q
```
