# Chess SFT Data Generation

This package builds the supervised fine-tuning data used to teach a small
language model chess. The goal is not only to teach syntax like FEN and UCI,
but to build a curriculum that moves from board perception, to rules, to
tactics, to evaluation, to planning.

At full target volume the pipeline generates about 1.94M examples across
7 tiers and 32 generator tasks. It also freezes a held-out benchmark before training
data is generated, then uses a canonical FEN blocklist to prevent train/eval
position leakage.

## Mental Model

The pipeline has six layers:

1. Source loaders collect chess positions and labels from real games,
   puzzles, openings, engine evaluations, tablebases, and generated Chess960.
2. A FEN pool deduplicates positions while preserving metadata like source,
   game phase, and `is_chess960`.
3. Eval splits are sampled first, frozen, and converted into a blocklist.
4. Task generators turn source rows into chat-style SFT examples.
5. The JSONL writer validates each example before writing it.
6. Benchmark tooling creates deterministic prompts and scoring metrics for
   model evaluation.

The important design choice is that training data generation is downstream of
the eval split. A position can only enter training if it is not in the frozen
benchmark blocklist, using a canonical key based on the first four FEN fields
so move counters do not create false "new" positions.

## Directory Layout

```text
sft/make_data/
  config/
    settings.py          Paths, volumes, source IDs, depth filters, seeds
    templates.py         Compatibility wrapper for chess_llm.sft.templates
    system_prompt.py     Shared system prompt for SFT examples
  sources/
    lichess_games.py     Streams games and extracts per-ply positions
    lichess_puzzles.py   Loads puzzles and applies setup-move preprocessing
    lichess_openings.py  Loads ECO/name/opening move data
    lichess_evals.py     Streams engine evals with SQLite dedup
    polyglot_books.py    Compatibility wrapper for package Polyglot helpers
    syzygy_probing.py    Compatibility wrapper for package Syzygy helpers
    stockfish_engine.py  Thin python-chess Stockfish wrapper
    chess960.py          Generates Chess960 starts and random continuations
    mate_dataset.py      Loads MATE binary move-comparison data
  pool/
    fen_pool.py          In-memory FEN dedup and tagging
    eval_split.py        Eval split sampling, ECO holdout, blocklist
    annotator.py         Compatibility wrapper for package annotation cache
  generators/
    *.py                 Compatibility wrappers for chess_llm.sft.generators
  validation/
    validator.py         Per-example validation before write
    benchmark.py         Frozen benchmark schema, gold derivation, scoring
    eval_harness.py      Oracle checks for benchmark examples
    decontamination.py   Output audit against eval blocklist
  output/
    writer.py            Validating JSONL writer
    stats.py             Generation counts and Chess960 mix checks
  scripts/
    run_pipeline.py      Compatibility wrapper for chess_llm.sft.pipeline
    freeze_benchmark.py  Refreeze benchmark JSONL from eval splits
    run_benchmark.py     Score model predictions against benchmark
    extract_polyglot_books.py  Compatibility alias for package extractor
    download_tablebases.py     Compatibility alias for package downloader
    run_eval_split.py    Compatibility alias for chess_llm.sft.run_eval_split
    run_eval_harness.py  Validate eval split oracle consistency
    push_to_hub.py       Compatibility alias for chess_llm.sft.hub_upload
    validate_outputs.py  Compatibility alias for chess_llm.sft.validate_outputs
    preflight_check.py   Compatibility alias for chess_llm.sft.preflight
```

Large runtime outputs default to `E:/chess_sft_data`, controlled by
`CHESS_SFT_OUTPUT`. Hugging Face cache defaults to `E:/hf_cache`.

## Data Sources

The pipeline currently uses nine source families.

| Source | What It Provides | Used For |
| --- | --- | --- |
| Lichess games | Realistic game FENs, played move, ply, phase, material | General FEN pool, state tracking, rules, tactics, evaluation |
| Lichess puzzles | Puzzle FEN after setup move, solution line, themes, rating | Tactics and puzzle-solving supervision |
| Lichess openings | ECO code, name, PGN, UCI move sequence, replayed FEN | Opening ID and opening principle tasks |
| Lichess position evals | FEN, depth, best move, PV, cp/mate | Position evaluation, best move, move consequence |
| Polyglot books | Weighted book moves for opening positions | Opening continuation labels |
| Syzygy tablebases | WDL and DTZ for <=7-piece endings | Endgame WDL and DTZ-optimal moves |
| Stockfish | Live fallback eval, best move, PV | Annotation cache and future backfills |
| Chess960 generator | Chess960 start IDs plus random legal continuations | Rules/perception robustness and Chess960 benchmark |
| MATE dataset | Two legal candidate moves and the better move | Binary move-choice benchmark/data |

Source-specific safeguards are built in:

- Lichess puzzle rows are preprocessed by pushing the opponent's setup move,
  so the FEN shown to the model is the real puzzle position.
- Opening rows are replayed from UCI moves with python-chess before use.
- Position-eval PV lines are parsed through python-chess, including castling
  normalization, before best moves are stored.
- Eval rows are deduplicated in SQLite, keeping higher depth and then higher
  `knodes` when the same FEN appears multiple times.
- Chess960 positions are parsed with `chess960=True` anywhere castling/legal
  move semantics matter.

Package-owned source parsers and the first source loaders live under
`src/chess_llm/sft/sources/` and define the contracts the legacy data pipeline
consumes. Lichess game streaming now normalizes the HF
`WhiteElo`/`BlackElo`/`Result`/`movetext` schema into the game rows consumed by
the pipeline, and rejects malformed PGN rows instead of emitting a valid
prefix. Lichess puzzle loading now normalizes the HF
`PuzzleId`/`FEN`/`Moves`/`Rating`/`Themes` schema before setup-move
preprocessing. Lichess position-eval streaming and SQLite deduplication now
live in the package: rows are validated through python-chess, highest depth
then highest `knodes` wins per FEN, persisted DB reruns are emitted in
deterministic quality order, and old king-to-rook castling PVs are normalized
when read back. Position eval rows preserve Stockfish `cp` and `mate` values as
White-perspective scores and emit `eval_perspective="white"`. MATE ZIP archive
loading is also package-owned: Hugging Face Hub is imported lazily, bad archives
and malformed JSONL rows are skipped, parseable-but-invalid FENs are rejected,
and strategy/tactic annotations are preserved for Tier 7 traces. Lichess opening
rows are replayed from UCI in package code and treat replayed UCI as
authoritative; if the source EPD disagrees, the row is marked with
`epd_mismatch`. Chess960 generation is also package-owned and reports the actual
number of random legal moves pushed, not just the requested rollout length.
Stockfish process configuration and the live annotation wrapper now live in
`chess_llm.external.stockfish`; the legacy `sources/stockfish_engine.py` module
only supplies the old default path behavior. Stockfish scores are stored from
White's perspective, matching the Lichess eval-row contract.

## Pipeline Flow

### 1. Load Sources

`scripts/run_pipeline.py` calls `load_sources()`. It streams or loads the
source datasets, applies memory caps, and builds a shared config dict consumed
by generators:

```python
config["fen_pool"]
config["game_positions"]
config["puzzles"]
config["openings"]
config["position_evals"]
config["best_move_evals"]
config["consequence_evals"]
config["book_moves"]
config["endgame_positions"]
config["mate_rows"]
```

The FEN pool is shuffled after deduplication so individual generators see a
varied mixture of positions.

After sources load, `chess_llm.sft.source_readiness` counts each source family
and reports which selected tiers or strict eval splits depend on empty inputs.
Full-volume runs fail early on missing required families instead of failing
later as generator underfills. `--volume` and `--allow-source-gaps` keep smoke
and exploratory runs permissive while still logging the missing inputs.

### 2. Freeze Eval Splits First

Before training examples are generated, `run_eval_splits()` creates held-out
splits:

| Split | Target Size | Main Source |
| --- | ---: | --- |
| perception | 2,000 | FEN pool |
| rules | 2,000 | FEN pool |
| tactics | 2,000 | puzzles |
| evaluation | 1,500 | depth >= 40 evals |
| openings | 500 | held-out ECO codes |
| endgames | 1,500 | Syzygy positions |
| planning | 2,000 | puzzles plus depth >= 30 evals |
| chess960 | 500 | Chess960-only positions |
| mate | 1,000 | MATE rows |

Openings get an extra leakage guard: rows are held out by ECO code, not by
individual row, so closely related opening positions stay on the same side of
the train/eval boundary.

The frozen blocklist stores canonical FEN keys:

```text
piece-placement side-to-move castling-rights en-passant-square
```

That means positions that only differ by halfmove or fullmove counter are
treated as the same position for decontamination.

Eval split sampling, ECO-code holdout partitioning, canonical FEN blocklist
save/load, and output decontamination audits now live in
`chess_llm.sft.eval_split` and `chess_llm.sft.decontamination`. The legacy
`pool/eval_split.py` and `validation/decontamination.py` modules are
compatibility wrappers.

FEN-pool deduplication now lives in `chess_llm.sft.fen_pool`. It deduplicates
by canonical board identity rather than literal FEN text, so positions that
only differ by move counters or impossible en-passant fields collapse to one
pool entry. Later duplicate rows can still contribute missing metadata such as
Chess960 markers.

The eval-source map used by both `run_pipeline.py` and `run_eval_split.py` now
comes from `chess_llm.sft.source_preparation`, which keeps Chess960 positions
out of standard perception/rules splits and enriches eval-opening rows on
copies rather than mutating the loaded opening rows.

Opening rows with blank or missing ECO codes are assigned to the training side
of the opening partition instead of being dropped. `holdout_fraction=0.0`
means no ECO holdout.

### 3. Freeze the Benchmark

Raw eval splits are converted into a deterministic benchmark under
`E:/chess_sft_data/benchmark`. Benchmark examples use a fixed schema:

```json
{
  "example_id": "rules_00042",
  "split": "rules",
  "task_type": "legal_moves",
  "fen": "...",
  "prompt": "FEN: ...\nList all legal moves.",
  "gold_answer": "a2a3 a2a4 ...",
  "metric_type": "jaccard",
  "metadata": {}
}
```

Unlike training data, benchmark prompts are canonical and deterministic. Gold
answers are derived from source data or from python-chess/Syzygy/engine-backed
oracles, then self-scored to verify that the expected answer receives a perfect
primary metric.

The canonical benchmark schema, gold derivation, prompt rendering, freeze/write
manifest logic, oracle validation, and scoring now live in
`chess_llm.evals.benchmark`. The benchmark command-line tools live in
`chess_llm.evals.freeze_benchmark`, `chess_llm.evals.run_eval_harness`, and
`chess_llm.evals.run_benchmark`. This legacy folder keeps compatibility imports
and script wrappers, but benchmark behavior should be changed in the package
modules first.

Raw eval-split oracle checks now live in `chess_llm.evals.eval_harness`.
`validation/eval_harness.py` is a compatibility wrapper used by the existing
scripts. New checks should be added to the package module first so the training
and future Autodata loops can reuse the same oracles without importing legacy
generator modules.

### 4. Generate Training Examples

Each task is implemented as a `TaskGenerator`. The base class handles:

- target volume lookup and small-run overrides
- Chess960 target ratios by tier
- eval blocklist checks
- prompt rendering from templates
- common derived prompt fields like ASCII board, side to move, castling rights,
  and en passant square
- wrapping raw labels into the shared chat schema

Every generated example is written only if `validation.validator.validate_example`
accepts it.

## Output Schema

Training rows are JSONL objects with a three-message chat format:

```json
{
  "task": "2.1_legal_move_gen",
  "tier": 2,
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "is_chess960": false,
  "messages": [
    {"role": "system", "content": "You are a chess reasoning engine..."},
    {"role": "user", "content": "FEN: ...\nList all legal moves."},
    {"role": "assistant", "content": "a7a5 a7a6 b7b5 b7b6 ..."}
  ],
  "metadata": {
    "source": "lichess_games"
  }
}
```

Tier 7 move-selection tasks use a stricter answer format:

```xml
<think>short chess reasoning trace</think>
<move>e2e4</move>
```

That format lets benchmark scoring separately measure answer extraction,
format compliance, and move legality.

## Curriculum Tiers

The tiers are ordered from board literacy to actual chess decision-making.

### Tier 1: Perception and State

Target volume: about 920K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `1.1_fen_to_board` | 80K | Render FEN as ASCII board |
| `1.2_board_to_fen` | 80K | Recover FEN from ASCII board |
| `1.3_piece_identification` | 100K | Identify a square or list pieces |
| `1.4_piece_counting` | 80K | Count pieces/material |
| `1.6_square_lookup` | 100K | Read a single square's contents from FEN |
| `1.7_rank_lookup` | 80K | Read one compressed rank row from FEN |
| `1.8_move_square_edits` | 100K | Trace square lookups and rank edits for one move |
| `1.9_fen_assembly` | 100K | Apply one legal move and assemble the resulting full FEN |
| `1.10_fen_row_application` | 100K | Rewrite affected compressed rank rows and return Result FEN |
| `1.5_state_tracking` | 100K | Apply one legal move and return resulting FEN |

This tier teaches the model how to read board state, perform explicit
square/rank lookup mechanics, and then update board state through moves.
By default `1.4_piece_counting` uses benchmark-aligned full material-count
targets; partial color/piece-count prompts remain available with
`piece_counting_include_partial=True`.

### Tier 2: Rules

Target volume: about 370K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `2.1_legal_move_gen` | 100K | Exact legal move set |
| `2.2_piece_specific_moves` | 80K | Legal moves from one square |
| `2.3_move_legality_check` | 80K | Yes/no legality for a candidate move |
| `2.4_check_detection` | 60K | Check, checkmate, stalemate, or normal |
| `2.5_special_rules` | 50K | Castling, en passant, and promotion availability |

This tier is where python-chess is used most directly as a rules oracle. It
also carries the heaviest Chess960 mix, because castling and legal move
generation are where Chess960 errors are most likely.

### Tier 3: Tactics

Target volume: about 260K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `3.1_available_captures` | 60K | All legal captures |
| `3.2_threats` | 50K | Threatened pieces |
| `3.3_attacked_defended` | 60K | Attack/defense status of a square |
| `3.4_tactical_patterns` | 50K | Puzzle-backed tactic and best move |
| `3.5_hanging_pieces` | 40K | Attacked and undefended pieces |

This tier connects rule knowledge to tactical features.

### Tier 4: Evaluation

Target volume: about 150K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `4.1_material_balance` | 50K | Standard material count/difference |
| `4.2_position_evaluation` | 60K | Engine cp/mate converted into buckets |
| `4.3_pawn_structure` | 40K | Passed, isolated, doubled pawn features |

This tier teaches static assessment: material, structure, and broad engine
evaluation categories such as equal, slight edge, winning, or decisive.

### Tier 5: Openings

Target volume: about 30K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `5.1_opening_identification` | 10K | ECO/name from position or move sequence |
| `5.2_opening_continuation` | 10K | Top book moves and alternatives |
| `5.3_opening_principles` | 10K | Opening plans, development, pawn structure |

This tier mixes factual opening labels with book-move continuations and
short strategic descriptions.

### Tier 6: Endgames

Target volume: about 140K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `6.1_endgame_classification` | 30K | Material signature such as KRK or KPK |
| `6.2_endgame_wdl` | 40K | Syzygy win/draw/loss |
| `6.3_endgame_best_move` | 40K | DTZ-optimal move |
| `6.4_endgame_principles` | 30K | Opposition, rook technique, pawn-race ideas |

This tier is tablebase-backed where possible, so the labels are not just
heuristic.

### Tier 7: Planning

Target volume: about 170K examples.

| Task | Target | Supervision |
| --- | ---: | --- |
| `7.1_best_move_selection` | 80K | Engine best move with reasoning trace |
| `7.2_puzzle_solving` | 50K | First solution move from Lichess puzzle |
| `7.3_move_consequence` | 40K | PV/consequence analysis after a candidate move |

This tier is the first explicit bridge from chess knowledge to chess
reasoning. Best-move and puzzle tasks require the `<think>`/`<move>` format.

## Prompt Templates

Training prompts are randomized through `chess_llm.sft.templates`; the legacy
`config/templates.py` path aliases that package module. Each task has multiple
phrasings, and many include richer context than raw FEN:

- ASCII board rendering
- side to move
- castling rights
- en passant square
- selected square, piece, color, or candidate move
- opening ECO/name and move sequence
- puzzle side and themes

This is meant to reduce overfitting to one instruction phrasing while keeping
the output labels deterministic.

## Chess960 Support

Chess960 is not a separate afterthought. It is mixed into the training data by
tier:

| Tier | Target Chess960 Ratio |
| --- | ---: |
| 1 | 20% |
| 2 | 20% |
| 3 | 15% |
| 4 | 10% |
| 5 | 5% |
| 6 | 5% |
| 7 | 10% |

The code uses package row-to-board helpers from `chess_llm.sft.context` for
Chess960-sensitive validation and answer derivation, especially legal moves and
castling. Under python-chess, Chess960 castling legality and UCI handling depend
on constructing the board with `chess960=True`, so generators should use
`board_from_raw()` instead of calling `chess.Board(fen)` directly when a source
row may carry `is_chess960` or `metadata.chess960_id`. The pipeline stats report
compares actual ratio against target after generation.

## Validation and Quality Gates

Validation happens at write time and can also be run after the fact.

Current checks include:

1. FEN parses under standard chess or Chess960 as appropriate.
2. Legal move generation exactly matches python-chess legal moves.
3. Tagged moves in Tier 7 are legal in the position.
4. Move-legality answers agree with the actual legal move set.
5. State-tracking result FEN matches replaying the listed moves.
6. Prompt and answer text contain no unfilled `{placeholder}` fields.
7. Tier 7 answers follow strict `<think>...</think><move>...</move>` format.
8. Output files can be audited against the eval blocklist for contamination.

Run validation only:

```bash
chess-llm-validate-outputs \
  --output-dir E:/chess_sft_data/output \
  --blocklist E:/chess_sft_data/eval_splits/blocklist.txt
```

Validation fails when no eval blocklist is available because contamination
cannot be checked. Use `--skip-decontamination` only for intentional local
smoke/debug validation.

## Benchmark and Scoring

The benchmark layer is separate from the training JSONL. It creates fixed
prompts and gold answers for held-out splits. Scoring supports:

- exact match for deterministic text labels
- Jaccard overlap for legal-move and capture sets
- evaluation-bucket accuracy for cp/mate labels
- continuation-rank scoring for book moves
- move extraction from `<move>...</move>`
- move-choice accuracy for MATE-style binary comparisons
- optional Stockfish ACPL for move-prediction tasks
- format compliance and legal-move secondary metrics

Run benchmark scoring against model predictions:

```bash
chess-llm-run-benchmark --predictions predictions.jsonl --benchmark-dir E:/chess_sft_data/benchmark
```

Full benchmark freezes are strict by default and fail when planned task types
have no frozen examples. Use `chess-llm-freeze-benchmark --split ...` for
partial refreshes, or pass `--allow-coverage-gaps` for exploratory full freezes.

The legacy `python scripts/run_benchmark.py ...` command still works as a
compatibility wrapper.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Small smoke run:

```bash
chess-llm-make-data --volume 100 --all
```

`--volume` is intended for smoke and bring-up runs. For those bounded runs the
Lichess game source uses a fixed small parquet shard instead of discovering the
full multi-terabyte game dataset. Full generation without `--volume` keeps the
unrestricted source path.

Environment/data preflight:

```bash
chess-llm-preflight
```

Extract local Polyglot book archives:

```bash
chess-llm-extract-polyglot-books --polyglot-dir polyglot_opening_books
```

Preview or download Syzygy tablebase files:

```bash
chess-llm-download-tablebases --pieces 3,4,5 --dry-run
```

Preview or upload generated datasets to Hugging Face Hub:

```bash
chess-llm-upload-data --all --dry-run
chess-llm-upload-data --training --org Chess-Nut-Engine
```

Single tier:

```bash
chess-llm-make-data --tier 2
```

Eval split and benchmark only:

```bash
chess-llm-make-data --eval-only
```

Standalone eval split refresh:

```bash
chess-llm-run-eval-split --volume 100
```

Full generation:

```bash
chess-llm-make-data --all
```

Write a source-readiness manifest, or continue an exploratory run when a
selected source family is empty:

```bash
chess-llm-make-data --all --source-readiness-report readiness.json
chess-llm-make-data --tier 4 --allow-source-gaps
```

The legacy `python scripts/run_pipeline.py ...` command still works from this
directory, but it now aliases the package-owned `chess_llm.sft.pipeline`
module. Task generators, templates, Syzygy helpers, Polyglot helpers, and the
annotation cache are package-owned. The legacy `python scripts/push_to_hub.py`
command also works, but it now aliases `chess_llm.sft.hub_upload`.

## How This Supports SFT, SDPO, and Self-Distillation

The current data is primarily SFT data. It teaches the model:

- board representation and FEN/state manipulation
- legal move generation and rule edge cases
- tactical features and puzzle first moves
- material/positional/endgame evaluation
- opening knowledge and book continuations
- best-move selection with a strict answer format

That is the cold-start foundation for later on-policy work. Once a model can
read positions and produce legal moves, the same benchmark/eval infrastructure
can support SDPO-style loops:

1. sample moves or reasoning traces from the current student
2. verify legality, outcome, engine score, puzzle success, or tablebase result
3. feed that feedback to a teacher or feedback-conditioned student
4. distill the corrected trajectory back into the deployment-time prompt format

The pieces already built for that future loop are the most important data
plumbing pieces: canonical FEN identity, held-out benchmark splits, legality
oracles, engine/tablebase feedback, prompt templates, and strict move-answer
extraction. The next missing layer is a richer trajectory/preference schema
that stores student rollout, environment feedback, teacher correction, and
chosen/rejected responses side by side.
