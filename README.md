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
- `sft/training/` now contains compatibility wrappers and legacy docs/tests
  for the phase-aware SFT training and benchmark evaluation harness.
- `src/chess_llm/` is the new package home for shared interfaces that should
  outlive any one training method: artifact schemas, answer parsing, future
  Autodata components, preference data, SDPO feedback examples, and inference.
- `plan/` contains the older design documents. They remain useful historical
  context, but new architecture decisions should land under `docs/`.

## Install

Install the root package in editable mode before using package CLIs:

```bash
python -m pip install -e ".[dev]"
```

Optional GPU/training extras are separated from the lightweight package:

```bash
python -m pip install -e ".[data]"
python -m pip install -e ".[eval]"
python -m pip install -e ".[train]"
```

For one WSL or Linux GPU environment that will both generate data and train,
install all three extras together:

```bash
python -m pip install -e ".[data,train,eval]"
```

Stockfish remains an external binary. Pass `--stockfish-path` to tools that
need engine-backed scoring, or set `STOCKFISH_PATH` for legacy training eval.

## Current Package Layer

The current package layer intentionally does not move all working SFT code at
once. It adds shared contracts and the first reusable builders:

- `chess_llm.artifacts.schemas` defines versioned JSONL artifacts.
- `chess_llm.artifacts.jsonl` reads and writes ordered artifact files.
- `chess_llm.formats.answers` parses flexible model outputs into normalized
  move fields and owns the strict `<think>...</think><move>...</move>`
  protocol helpers used by SFT validation and benchmark scoring.
- `chess_llm.formats.board` renders deterministic ASCII boards for benchmark
  prompts and SFT rows.
- `chess_llm.formats.prompts` stores the shared SFT system prompt.
- `chess_llm.core.board` provides small python-chess helpers for FEN identity,
  variant-aware FEN identity, FEN validation, legal move listing, and legality
  checks. Use `canonical_fen_key()` for unprefixed board normalization and
  `variant_fen_key()` for dedupe, blocklists, and train/eval split identity.
- `chess_llm.sft` builds legacy-compatible SFT chat rows, JSONL files, common
  template context fields, package-owned data settings, and generated-row
  validation. The legacy `sft/make_data/config/settings.py` module now wraps
  this package layer.
- `chess_llm.sft.sources` owns reusable source-row parsers, validators, and the
  first package-owned source loaders as data sources move out of the legacy
  `sft/make_data/sources/` tree. Lichess game streaming, Lichess puzzle
  loading, Lichess position-eval streaming/deduplication, Chess960 generation,
  and Lichess opening loading are package-owned; their legacy source files now
  forward into `chess_llm.sft.sources`. Lichess eval rows carry an explicit
  `eval_perspective="white"` contract. MATE ZIP archive loading is
  package-owned and lazy-imports Hugging Face Hub; parsed rows preserve
  strategy/tactic annotations for Tier 7 traces.
- `chess_llm.sft.context` owns shared row-to-board inference, including
  Chess960 detection from top-level flags and benchmark metadata such as
  `chess960_id`.
- `chess_llm.sft.eval_split` and `chess_llm.sft.decontamination` own
  deterministic held-out split sampling, ECO-code holdout partitioning,
  variant-aware FEN blocklists, and generated-output contamination audits. The
  legacy `sft/make_data/pool/eval_split.py` and
  `sft/make_data/validation/decontamination.py` modules now wrap these package
  helpers.
- `chess_llm.sft.fen_pool` and `chess_llm.sft.source_preparation` own
  canonical FEN-pool deduplication and the source-map assembly used before
  freezing eval splits. The legacy `sft/make_data/pool/fen_pool.py` module now
  wraps the package pool.
- `chess_llm.evals.benchmark` owns frozen benchmark schema, gold-answer
  derivation, prompt rendering, freeze/write manifests, oracle validation, and
  scoring. The legacy `sft/make_data/validation/benchmark.py` module now wraps
  this package layer for compatibility.
- `chess_llm.evals.eval_harness` owns eval-split oracle self-checks for raw
  held-out rows. The legacy `sft/make_data/validation/eval_harness.py` module
  now wraps this package layer.
- `chess_llm.evals.freeze_benchmark`,
  `chess_llm.evals.run_eval_harness`, and
  `chess_llm.evals.run_benchmark` own the benchmark/eval command-line tools.
  The legacy `sft/make_data/scripts/` entry points now forward into these
  package CLIs.
- `chess_llm.sft.templates` and `chess_llm.sft.generators` own the prompt
  templates, task-generator base class, reasoning traces, tier modules, and
  tier registry. The legacy `sft/make_data/config/templates.py` and
  `sft/make_data/generators/` modules now alias the package modules.
- `chess_llm.sft.annotation`,
  `chess_llm.sft.sources.polyglot_books`, and
  `chess_llm.sft.sources.syzygy_probing` own the Stockfish annotation cache,
  Polyglot opening-book helpers, and Syzygy WDL/DTZ probing helpers. The legacy
  `pool/annotator.py`, `sources/polyglot_books.py`, and
  `sources/syzygy_probing.py` files now alias the package modules.
- `chess_llm.sft.pipeline` owns the main SFT data-generation orchestration and
  `chess-llm-make-data` CLI. It now loads package-owned source helpers
  directly.
- `chess_llm.sft.run_eval_split` owns the standalone eval-split refresh CLI
  for cases where you want to generate or backfill held-out splits without
  running tier generation. The legacy `scripts/run_eval_split.py` entry point
  aliases the package module.
- `chess_llm.sft.source_readiness` owns the tier-aware source availability
  contract for data generation. `chess-llm-make-data` reports loaded source
  counts before eval split generation, fails early for missing required source
  families on full runs, and can emit a JSON readiness manifest for run
  tracking.
- `chess_llm.sft.hub_upload` owns Hugging Face dataset-card generation,
  upload staging, and the `chess-llm-upload-data` CLI. The legacy
  `scripts/push_to_hub.py` entry point aliases the package module.
- `chess_llm.sft.preflight` owns environment/data preflight checks and the
  `chess-llm-preflight` CLI. The legacy `scripts/preflight_check.py` entry
  point aliases the package module.
- `chess_llm.sft.extract_polyglot_books` owns local Polyglot archive extraction
  and the `chess-llm-extract-polyglot-books` CLI. The legacy
  `scripts/extract_polyglot_books.py` entry point aliases the package module.
- `chess_llm.sft.download_tablebases` owns Syzygy tablebase index discovery,
  safe download staging, and the `chess-llm-download-tablebases` CLI. The
  legacy `scripts/download_tablebases.py` entry point aliases the package
  module.
- `chess_llm.sft.validate_outputs` owns post-hoc generated-output validation,
  completeness auditing, and decontamination checks for tier JSONL files. The
  legacy `scripts/validate_outputs.py` entry point aliases the package module.
- `chess_llm.training.data` owns tier JSONL loading and phase-aware train/eval
  mixing. The legacy `sft/training/data/` modules now wrap this package layer,
  and sanitized JSONL files are loaded through HuggingFace's JSON loader so the
  trainer can use Arrow-backed datasets instead of materializing every row in a
  Python list.
- `chess_llm.training.train`, `chess_llm.training.evaluate`, and
  `chess_llm.training.run_curriculum` own the training, benchmark evaluation,
  and A -> B -> C curriculum CLIs. The legacy `sft/training/train.py`,
  `sft/training/evaluate.py`, and `sft/training/run_curriculum.py` files are
  thin forwarders during migration.
- `chess_llm.training.model_loading` owns attention backend selection for
  training and transformers eval loads. `auto` tries CUDA FlashAttention 3,
  FlashAttention 2, CUDA SDPA, eager, then the Transformers default.
- `chess_llm.training.training_args` owns TRL `SFTConfig` construction for the
  optional training extra, including the measured Qwen3.5 defaults for TF32,
  Liger, trainer eval, and FlashAttention-only packing.
- `chess_llm.training.phase_gate` owns the lightweight phase-gate criteria used
  by training/evaluation scripts; the legacy `sft/training/phase_gate.py` now
  wraps this package layer.
- `chess_llm.autodata.rollouts` and `chess_llm.autodata.judge` provide the
  first prompt -> rollout -> judgment substrate for the self-guided loop.
- `chess_llm.autodata.stockfish_judge` and `chess_llm.external.stockfish`
  provide the optional engine-backed scoring layer for legal rollouts and the
  package-owned Stockfish process wrapper used by source annotation.
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
chess-llm-batch-judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B
```

Use repeatable `--split` flags to restrict the loaded benchmark splits:

```bash
chess-llm-batch-judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B \
  --split planning \
  --split rules
```

Pass Stockfish options when you want engine-backed regret and teacher moves:

```bash
chess-llm-batch-judge \
  --benchmark-dir path/to/frozen_benchmark \
  --predictions path/to/predictions.jsonl \
  --output-dir path/to/artifacts \
  --model-id Qwen/Qwen3-4B \
  --stockfish-path C:/path/to/stockfish.exe \
  --stockfish-depth 20 \
  --stockfish-threads 4 \
  --stockfish-hash-mb 1024 \
  --syzygy-path path/to/syzygy
```

## Autodata SFT Refresh

Judged artifacts can now be turned into targeted SFT rows without changing the
trainer. The builder reads the prompt, rollout, and judgment files produced by
batch judge and writes legacy-compatible Tier 7 JSONL files:

```bash
chess-llm-sft-refresh \
  --prompts path/to/artifacts/prompts.jsonl \
  --rollouts path/to/artifacts/rollouts.jsonl \
  --judgments path/to/artifacts/judgments.jsonl \
  --output-dir path/to/autodata_refresh
```

The output directory contains `tier7/7.4_autodata_format_repair.jsonl`,
`tier7/7.5_autodata_move_correction.jsonl`, and `manifest.json`. Rows are
created only for single-move benchmark tasks with a legal target move.

## Training Package CLIs

Prefer the package CLIs for new runs:

```bash
chess-llm-make-data --volume 100 --all
chess-llm-make-data --tier 1 2
chess-llm-make-data --eval-only
chess-llm-make-data --all --source-readiness-report readiness.json
chess-llm-make-data --tier 4 --allow-source-gaps
chess-llm-preflight
chess-llm-extract-polyglot-books --polyglot-dir path/to/polyglot_opening_books
chess-llm-download-tablebases --pieces 3,4,5 --dry-run
chess-llm-validate-outputs \
  --output-dir path/to/output \
  --blocklist path/to/eval_splits/blocklist.txt
chess-llm-upload-data --all --dry-run
chess-llm-run-eval-split --volume 100
chess-llm-run-eval-harness --split-dir path/to/eval_splits
chess-llm-freeze-benchmark \
  --split-dir path/to/eval_splits \
  --output-dir path/to/benchmark
chess-llm-run-benchmark \
  --benchmark-dir path/to/benchmark \
  --predictions path/to/predictions.jsonl
chess-llm-train --phase a --dry-run
chess-llm-train --phase a --smoke-run --wandb-project chess-sft --run-name phase-a-smoke
chess-llm-evaluate \
  --model path/to/checkpoint \
  --benchmark-dir path/to/benchmark \
  --output path/to/predictions.jsonl \
  --wandb-project chess-sft \
  --wandb-run-name phase-a-eval
chess-llm-run-curriculum --start-phase a --end-phase c --wandb-project chess-sft --run-prefix full-curriculum
```

`chess-llm-evaluate` writes three coordinated outputs: prediction JSONL,
`*.results.json` metrics, and `*.eval_run.json` metadata using the
`EvaluationRunArtifact` schema. The metadata sidecar records model, benchmark,
backend, generation settings, split counts, exit status, and artifact paths for
batch judging and later self-improvement loops.

Full benchmark freezes fail on missing planned task coverage by default. Use
`--split` for partial refreshes, or `--allow-coverage-gaps` for exploratory
full-directory freezes.

`chess-llm-make-data` writes a source-readiness summary before generation. Full
runs fail early when a selected tier or strict eval split needs an empty source
family, such as missing `position_evals` for Tier 4. `--volume` runs are
permissive for quick smoke tests, and `--allow-source-gaps` keeps exploratory
runs moving while still logging and optionally writing the manifest.

The old `python sft/make_data/scripts/run_pipeline.py ...` and
`python sft/make_data/scripts/run_eval_split.py ...` styles still work, but
they now forward into package modules.

The old `python sft/training/train.py ...` style still works, but it now
forwards into `chess_llm.training`.

## Useful Commands

The `python -m` form is still supported for package CLIs:

```bash
python -m chess_llm.evals.batch_judge --help
python -m chess_llm.evals.freeze_benchmark --help
python -m chess_llm.evals.run_eval_harness --help
python -m chess_llm.evals.run_benchmark --help
python -m chess_llm.autodata.sft_refresh --help
python -m chess_llm.sft.pipeline --help
python -m chess_llm.sft.preflight
python -m chess_llm.sft.extract_polyglot_books --help
python -m chess_llm.sft.download_tablebases --help
python -m chess_llm.sft.validate_outputs --help
python -m chess_llm.sft.hub_upload --help
python -m chess_llm.sft.run_eval_split --help
python -m chess_llm.training.train --help
python -m chess_llm.training.evaluate --help
python -m chess_llm.training.run_curriculum --help
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

Run the package training tests:

```bash
python -m pytest tests -q
```
