# make_data — Chess SFT Data Generation Pipeline

Generates ~1.5M chess SFT training examples across 7 tiers (25 tasks) from 9 data sources. All examples use a consistent `messages` format with FEN positions and UCI move notation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Small test run (100 examples per task)
python scripts/run_pipeline.py --volume 100

# Full pipeline
python scripts/run_pipeline.py --all

# Single tier
python scripts/run_pipeline.py --tier 1

# Validate existing outputs
python scripts/run_pipeline.py --validate-only
```

## Directory Layout

```
make_data/
├── config/
│   ├── settings.py          # Paths, volumes, depths, mix ratios, seeds
│   ├── templates.py          # 5-10 prompt templates per task (25 tasks)
│   └── system_prompt.py      # Single system prompt constant
├── sources/                   # 9 data source loaders
│   ├── lichess_games.py       # Stream PGN, extract FEN per ply, Elo >= 2000
│   ├── lichess_puzzles.py     # ~4M puzzles with setup-move preprocessing
│   ├── lichess_openings.py    # 3,630 named openings with ECO codes
│   ├── lichess_evals.py       # 845M position evals, SQLite dedup, partitioning
│   ├── polyglot_books.py      # Polyglot .bin book reader, weighted moves
│   ├── syzygy_probing.py      # WDL/DTZ probing, endgame position sampling
│   ├── stockfish_engine.py    # UCI wrapper for live Stockfish evaluation
│   ├── chess960.py            # Generate all 960 starting positions + variants
│   └── mate_dataset.py        # MATE dataset binary choice (zip/JSONL from HF)
├── pool/
│   ├── fen_pool.py            # Unified FEN collection, dedup, tagging
│   ├── eval_split.py          # Generate eval splits FIRST, build FEN blocklist
│   └── annotator.py           # Stockfish annotation with SQLite cache
├── generators/
│   ├── base.py                # Abstract TaskGenerator base class
│   ├── tier1_perception.py    # Tasks 1.1-1.5 (~440K)
│   ├── tier2_rules.py         # Tasks 2.1-2.5 (~370K)
│   ├── tier3_tactics.py       # Tasks 3.1-3.5 (~260K)
│   ├── tier4_evaluation.py    # Tasks 4.1-4.3 (~150K)
│   ├── tier5_openings.py      # Tasks 5.1-5.3 (~9K)
│   ├── tier6_endgames.py      # Tasks 6.1-6.4 (~140K)
│   ├── tier7_planning.py      # Tasks 7.1-7.3 (~170K)
│   └── reasoning_traces.py    # Shared <think>/<move> trace generation
├── validation/
│   ├── validator.py           # 7 validation checks (FEN, moves, templates, etc.)
│   ├── benchmark.py           # Frozen benchmark: gold answers, metrics, oracle validation
│   ├── eval_harness.py        # Self-consistency checks on eval split data
│   └── decontamination.py     # Eval blocklist enforcement
├── output/
│   ├── writer.py              # JSONL writer with inline validation
│   └── stats.py               # Volume tracking, Chess960 mix verification
├── scripts/
│   ├── run_pipeline.py        # Main CLI: full pipeline or single tier
│   ├── run_eval_split.py      # Generate + freeze eval splits independently
│   ├── run_eval_harness.py    # Run eval harness checks on eval splits
│   ├── run_benchmark.py       # Score model predictions against frozen benchmark
│   ├── freeze_benchmark.py    # Refreeze benchmark from existing eval splits
│   ├── validate_outputs.py    # Post-hoc validation of generated JSONL
│   └── download_tablebases.py # Download Syzygy tablebases
├── tests/                     # 251 pytest tests covering all tiers + benchmark
└── polyglot_opening_books/    # Polyglot book archives (.bin, .zip, .7z)
```

### Runtime Data (not in repo)

All large/generated data lives on the E: drive:

```
E:/hf_cache/                   # HuggingFace datasets cache
E:/chess_sft_data/
├── pool/                      # FEN pools
├── eval_splits/               # Raw eval split data (9 splits + blocklist)
│   ├── blocklist.txt          # One FEN per line — training must exclude these
│   ├── perception.jsonl
│   ├── rules.jsonl
│   └── ...
├── benchmark/                 # Frozen benchmark (canonical prompts + gold answers)
│   ├── manifest.json          # Version, seed, split sizes
│   ├── perception.jsonl
│   └── ...
├── annotations/               # Stockfish annotation cache (SQLite)
└── output/                    # Final JSONL per tier
    ├── tier1/
    ├── tier2/
    └── ...
```

## Tasks

| Tier | Task | ID | Volume | Description |
|------|------|----|--------|-------------|
| 1 | FEN to Board | 1.1 | 80K | FEN -> ASCII board diagram |
| 1 | Board to FEN | 1.2 | 80K | ASCII board -> FEN |
| 1 | Piece Identification | 1.3 | 100K | What piece is on a square / where are pieces |
| 1 | Piece Counting | 1.4 | 80K | Material count and balance |
| 1 | State Tracking | 1.5 | 100K | Apply 1-8 moves, report resulting FEN |
| 2 | Legal Move Gen | 2.1 | 100K | List all legal moves (incl. Chess960) |
| 2 | Piece-Specific Moves | 2.2 | 80K | Legal moves for a specific piece/square |
| 2 | Move Legality Check | 2.3 | 80K | Is a given move legal? (50/50 split) |
| 2 | Check Detection | 2.4 | 60K | Check / checkmate / stalemate / normal |
| 2 | Special Rules | 2.5 | 50K | Castling, en passant, promotion |
| 3 | Available Captures | 3.1 | 60K | List all capture moves |
| 3 | Threats | 3.2 | 50K | Identify threats to opponent pieces |
| 3 | Attacked/Defended | 3.3 | 60K | Square attack/defense analysis |
| 3 | Tactical Patterns | 3.4 | 50K | Fork, pin, skewer from Lichess puzzles |
| 3 | Hanging Pieces | 3.5 | 40K | Undefended pieces under attack |
| 4 | Material Balance | 4.1 | 50K | Count material difference |
| 4 | Position Evaluation | 4.2 | 60K | Centipawn -> 5-bucket eval labels |
| 4 | Pawn Structure | 4.3 | 40K | Doubled, isolated, passed pawns |
| 5 | Opening ID | 5.1 | 3K | Name the opening from position/moves |
| 5 | Opening Continuation | 5.2 | 3K | Suggest next moves via Polyglot weights |
| 5 | Opening Principles | 5.3 | 3K | Plans and character of the position |
| 6 | Endgame Classification | 6.1 | 30K | Categorize by material (KRK, KPK, etc.) |
| 6 | Endgame WDL | 6.2 | 40K | Syzygy win/draw/loss evaluation |
| 6 | Endgame Best Move | 6.3 | 40K | DTZ-optimal move from tablebases |
| 6 | Endgame Principles | 6.4 | 30K | Opposition, Lucena, Philidor, zugzwang |
| 7 | Best Move Selection | 7.1 | 80K | `<think>`/`<move>` format, depth >= 30 |
| 7 | Puzzle Solving | 7.2 | 50K | Lichess puzzles rated 1000-2500 |
| 7 | Move Consequence | 7.3 | 40K | PV line analysis and consequence prediction |

## Output Format

Every example follows this structure:

```json
{
  "task": "2.1_legal_move_gen",
  "tier": 2,
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "is_chess960": false,
  "messages": [
    {"role": "system", "content": "You are a chess reasoning engine..."},
    {"role": "user", "content": "What are the legal moves in this position?\nFEN: rnbqkbnr/..."},
    {"role": "assistant", "content": "a7a5 a7a6 b7b5 b7b6 ..."}
  ],
  "metadata": {
    "source": "lichess_games",
    "game_phase": "opening",
    "stockfish_eval_cp": 35,
    "template_id": 3
  }
}
```

Tier 7 tasks use `<think>...</think>` and `<move>...</move>` tags in assistant responses.

## Data Sources

| # | Source | Dataset |
|---|--------|---------|
| 1 | Lichess Games | `Lichess/standard-chess-games` |
| 2 | Lichess Puzzles | `Lichess/chess-puzzles` |
| 3 | Lichess Openings | `Lichess/chess-openings` |
| 4 | Lichess Evals | `Lichess/chess-position-evaluations` (845M rows) |
| 5 | Polyglot Books | Local `.bin` files in `polygloy_opening_books/` |
| 6 | Syzygy Tablebases | Downloaded via `scripts/download_tablebases.py` |
| 7 | Stockfish | Local engine binary (17+ with NNUE) |
| 8 | Chess960 | Generated via `chess.Board.from_chess960_pos()` |
| 9 | MATE Dataset | `OutFlankShu/MATE_DATASET` |

## Key Design Decisions

- **Eval splits first** — 13K held-out examples are generated before any training data. A FEN blocklist prevents contamination.
- **SQLite dedup** — The 40GB Position Evals dataset is deduplicated via SQLite WAL-mode, keeping the highest-depth row per FEN.
- **Annotation cache** — SQLite DB maps FEN -> eval. Position Evals pre-populates it; live Stockfish only runs for uncached FENs.
- **Template-driven variety** — Each task has 5-10 prompt templates randomly selected per example.
- **Chess960 mix** — Tiers 1-2: 20%, Tier 3: 15%, Tiers 4-7: 5-10% Chess960 positions.
- **Resumption** — The pipeline skips tasks with existing non-empty output files.

## Eval Benchmark

| Split | Size | Sources |
|-------|------|---------|
| Perception | 2,000 | Random FEN + Chess960 |
| Rules | 2,000 | Random FEN + Chess960 |
| Tactics | 2,000 | Lichess puzzles by theme |
| Evaluation | 1,500 | Stockfish-evaluated positions |
| Openings | 500 | Held-out ECO codes |
| Endgames | 1,500 | Syzygy-backed positions |
| Planning | 2,000 | Puzzles + engine-evaluated positions |
| Chess960 | 500 | Chess960 positions only |
| MATE | 1,000 | MATE dataset held-out |
| **Total** | **13,000** | |

## Validation

Every generated example passes 7 checks:

1. FEN parseable by python-chess
2. Legal move lists match `board.legal_moves` exactly
3. All UCI moves in outputs are legal in the position
4. State tracking results match python-chess replay
5. No unfilled `{placeholder}` template variables
6. Tier 7 outputs have valid `<think>`/`<move>` tags
7. No training FEN appears in the eval blocklist
