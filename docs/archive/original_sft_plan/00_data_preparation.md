# Stage 0: Data Preparation

**Phase**: Pre-training infrastructure
**Inputs**: Raw datasets from HuggingFace, Stockfish binary, Syzygy tablebases
**Outputs**: Unified FEN pool, Stockfish annotations, eval splits, JSONL per task
**Estimated effort**: 1-2 weeks

---

## Overview

Before any SFT training begins, all data sources must be downloaded,
preprocessed, deduplicated, annotated with Stockfish evaluations, and
split into training and evaluation sets. This document covers everything
that happens before Phase A.

---

## Data Sources

### Source 1: Lichess Standard Chess Games

**URL**: `https://huggingface.co/datasets/Lichess/standard-chess-games`
**Content**: Millions of complete games in PGN format with Elo ratings.
**Usage**: Extract FEN positions at each ply, filter by Elo ≥ 2000 for both
players. Provides realistic position distributions and real move sequences
for state tracking tasks.

**Processing**:
- Parse PGN, replay with python-chess, extract FEN at each half-move
- Record the move played (in UCI) and Stockfish evaluation
- Tag game phase: opening (moves 1-15), middlegame (16-35), endgame (36+)
- Tag material balance at each position

### Source 2: Lichess Chess Puzzles

**URL**: `https://huggingface.co/datasets/Lichess/chess-puzzles`
**Content**: ~4M tactical puzzles mined from 300M analyzed games, re-analyzed
with Stockfish at 40 meganodes per position.
**Fields**: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity,
NbPlays, Themes, GameUrl, OpeningTags.
**Usage**: Primary source for tactical pattern training and puzzle-solving tasks.

**Critical preprocessing required** — see Puzzle Preprocessing below.

### Source 3: Lichess Chess Openings

**URL**: `https://huggingface.co/datasets/Lichess/chess-openings`
**Content**: 3,630 named openings with ECO codes, PGN, UCI move sequences,
EPD (FEN-like position strings), and board images.
**Fields**: eco-volume, eco, name, pgn, uci, epd.
**Usage**: Opening identification, continuation, and principle evaluation tasks.
All move sequences are already in UCI format — zero conversion needed.

### Source 4: Lichess Chess Position Evaluations

**URL**: `https://huggingface.co/datasets/Lichess/chess-position-evaluations`
**Content**: 342M unique chess positions with 845M total evaluation rows from
Stockfish at various depths and node counts. Produced by the Lichess analysis
board running Stockfish within user browsers. Updated monthly (last: Jan 2026).

**Fields**:
- `fen`: Position (pieces, active color, castling, en passant — no move counters)
- `line`: Principal variation in UCI format
- `depth`: Search depth reached (uint8, range 1-245)
- `knodes`: Kilo-nodes searched (int32)
- `cp`: Centipawn evaluation (int16, range -20000 to 20000; null if mate found)
- `mate`: Mate-in-N evaluation (int8, range -108 to 96; null if no forced mate)

**Key properties**:
- Multiple evaluations per position at different depths — same FEN appears
  with different depth/knodes, allowing depth-filtering for quality control
- Principal variations are already in UCI format
- Centipawn and mate evaluations are mutually exclusive (one is always null)
- Massive scale (845M rows) means aggressive filtering is fine

**Usage across tiers**:

| Tier | How This Dataset Helps |
|------|----------------------|
| T4: Evaluation | Ground-truth centipawn evals for position assessment calibration. Filter depth ≥ 20 for reliable evals. Use cp buckets to train eval language ("equal", "slight edge", "winning") |
| T4: Pawn Structure | Positions with known evals help correlate structural features with outcomes |
| T7: Best Move | The `line` field's first move is Stockfish's best move at given depth. Filter depth ≥ 30 for high-quality best-move labels. Replaces/supplements live Stockfish annotation |
| T7: Move Consequence | PV lines show expected continuations after the best move — direct training signal for consequence analysis |
| T3: Tactical Patterns | Positions where `mate` is not null are forcing/tactical. Mate-in-N positions (small N) are excellent tactical training data |
| Eval Benchmark | High-depth evaluations (depth ≥ 40) serve as near-perfect ground truth for evaluation benchmark |

**Preprocessing**:

```python
from datasets import load_dataset

def preprocess_position_evals(min_depth=20, max_rows=None):
    """Load and filter the position evaluations dataset."""
    dset = load_dataset("Lichess/chess-position-evaluations", split="train",
                        streaming=True)  # Stream due to 40GB size

    for i, row in enumerate(dset):
        if max_rows and i >= max_rows:
            break

        # Filter by depth for quality
        if row["depth"] < min_depth:
            continue

        fen = row["fen"]
        line = row["line"]
        depth = row["depth"]
        cp = row["cp"]
        mate = row["mate"]

        # Extract best move (first move in PV line)
        best_move = line.split()[0] if line else None

        # Classify evaluation
        if mate is not None:
            eval_type = "mate"
            eval_value = mate
        else:
            eval_type = "cp"
            eval_value = cp

        yield {
            "fen": fen,
            "best_move": best_move,
            "pv_line": line,
            "depth": depth,
            "knodes": row["knodes"],
            "eval_type": eval_type,
            "eval_value": eval_value,
            "cp": cp,
            "mate": mate,
        }
```

**Depth filtering guidelines**:
- Depth ≥ 15: Acceptable for bulk training data
- Depth ≥ 20: Good quality for most tasks
- Depth ≥ 30: High quality for best-move ground truth
- Depth ≥ 40: Near-perfect for evaluation benchmark

**Deduplication strategy**: When multiple evaluations exist for the same FEN,
keep the highest-depth evaluation. If tied on depth, keep the one with more
knodes searched.

```python
def deduplicate_by_depth(position_evals):
    """Keep only the highest-depth eval per FEN."""
    best = {}
    for row in position_evals:
        fen = row["fen"]
        if fen not in best or row["depth"] > best[fen]["depth"]:
            best[fen] = row
        elif (row["depth"] == best[fen]["depth"] and
              row["knodes"] > best[fen]["knodes"]):
            best[fen] = row
    return best
```

### Source 5: Polyglot Opening Book

**URL**: `https://python-chess.readthedocs.io/en/latest/polyglot.html`
**Content**: Binary opening book format containing weighted moves for positions.
**Usage**: Move frequency/weight data for opening positions. Supplements
Lichess openings with statistical depth.

```python
import chess.polyglot

with chess.polyglot.open_reader("opening_book.bin") as reader:
    board = chess.Board()
    entries = list(reader.find_all(board))
    for entry in entries:
        move = entry.move
        weight = entry.weight  # popularity/strength weight
```

### Source 6: Syzygy Endgame Tablebases

**Content**: Perfect WDL and DTZ for all positions with ≤ 7 pieces.
**Practical scope**: 3-5 piece tables (~1GB) for training data generation.
6-piece tables (~150GB) optional for expanded coverage.

```python
import chess.syzygy

tablebase = chess.syzygy.open_tablebase("/path/to/syzygy/")
board = chess.Board("8/8/4k3/8/3K4/8/4P3/8 w - - 0 1")

wdl = tablebase.probe_wdl(board)   # 2=win, 0=draw, -2=loss
dtz = tablebase.probe_dtz(board)   # distance to zeroing move
```

### Source 7: Stockfish (Engine)

**Version**: Stockfish 17+ with NNUE
**Configuration**: Depth 15-20 for training data generation. Depth 25+ for
evaluation benchmark ground truth.
**Throughput**: ~100-500 positions/second depending on depth and hardware.

### Source 8: Chess960 Positions

**Generation**: `chess.Board.from_chess960_pos(n)` for n in 0..959.
**Note**: Standard chess is Chess960 position #518.

### Source 9: MATE Dataset

**URL**: `https://huggingface.co/datasets/OutFlankShu/MATE_DATASET`
**Content**: ~1M chess positions with candidate moves annotated by chess
experts (including four-time world champion Yifan Hou).
**Subsets**: MATE-N (no explanation), MATE-S (strategy), MATE-T (tactic),
MATE-ST (strategy + tactic).
**Usage**: Reasoning trace generation (Tier 7) and binary choice evaluation.

---

## Puzzle Preprocessing

### The Lichess Puzzle Gotcha

The Lichess puzzles dataset has a critical structural detail: the `FEN`
field stores the position BEFORE the opponent makes their move, and the
`Moves` field begins with that opponent's move. The puzzle's actual challenge
starts after applying the first move.

**Raw data structure**:
```
FEN: r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 10
Moves: d1d4 c6e5 d4e5 d6e5
       ^^^^^                   <- Opponent's move (setup)
             ^^^^^^^^^^^^^^^^^  <- Puzzle solution moves
```

### Preprocessing Pipeline

```python
import chess

def preprocess_puzzle(fen, moves_str, themes):
    """Convert raw Lichess puzzle into training-ready format."""
    board = chess.Board(fen)
    moves = moves_str.split()

    # Step 1: Apply the first move (opponent's setup move)
    setup_move = chess.Move.from_uci(moves[0])
    board.push(setup_move)

    # Step 2: The position after setup is where the solver must act
    puzzle_fen = board.fen()
    side_to_move = "White" if board.turn == chess.WHITE else "Black"

    # Step 3: Solution moves are the remaining moves
    solution_moves = moves[1:]  # These alternate: solver, opponent, solver...

    # Step 4: The first solution move is the answer for single-move tasks
    first_solution = solution_moves[0]

    # Step 5: Validate the solution is legal
    assert chess.Move.from_uci(first_solution) in board.legal_moves

    # Step 6: Get legal moves list for prompt
    legal_moves = [m.uci() for m in board.legal_moves]

    return {
        "fen": puzzle_fen,
        "side": side_to_move,
        "legal_moves": legal_moves,
        "solution_first_move": first_solution,
        "solution_full": solution_moves,
        "themes": themes.split() if themes else [],
        "is_mate_in_1": len(solution_moves) == 1 and "mate" in themes,
    }
```

### Puzzle Quality Notes

- Player moves in the solution are intended to be "only moves" — the single
  best response. Exception: mate-in-1 cases where several mates may be valid.
- Puzzle ratings range from ~400 to ~3000. For our training:
  - Tiers 3-4 tasks: puzzles rated 800-1800 (pattern recognition)
  - Tier 7 tasks: puzzles rated 1000-2500 (reasoning challenges)
  - Eval benchmark: stratified sample across all ratings
- Theme tags (fork, pin, skewer, discoveredAttack, backRankMate, etc.) are
  used to ensure balanced tactical coverage in training data.

### Multi-Move Puzzle Handling

For puzzles requiring multiple moves, the training format shows only the
first move as the answer, but the reasoning trace can reference the full
line:

```
<think>
Looking for tactical patterns. The knight on e5 forks the king and queen
after the discovered check from the bishop. After e5g6 check, the king
must move, then I play g6e7 winning the queen.
</think>
<move>e5g6</move>
```

For deeper puzzles (3+ solution moves), also generate intermediate position
tasks: apply 2 moves of the solution, then ask for the 3rd.

---

## Position Evaluations Preprocessing

The Lichess position evaluations dataset requires special handling due to
its massive scale (845M rows, ~40GB).

### Streaming Strategy

```python
from datasets import load_dataset
import chess

def stream_high_quality_evals(min_depth=20):
    """Stream and filter position evaluations."""
    dset = load_dataset("Lichess/chess-position-evaluations",
                        split="train", streaming=True)

    seen_fens = set()
    for row in dset:
        if row["depth"] < min_depth:
            continue

        fen = row["fen"]

        # Validate FEN
        try:
            board = chess.Board(fen)
        except ValueError:
            continue

        # Validate best move from PV line
        line = row["line"]
        if not line:
            continue
        best_move_uci = line.split()[0]
        try:
            move = chess.Move.from_uci(best_move_uci)
            if move not in board.legal_moves:
                continue
        except ValueError:
            continue

        # Dedup: keep first (often highest depth for same FEN)
        if fen in seen_fens:
            continue
        seen_fens.add(fen)

        yield {
            "fen": fen,
            "best_move": best_move_uci,
            "pv_line": line,
            "depth": row["depth"],
            "cp": row["cp"],
            "mate": row["mate"],
        }
```

### Partitioning by Use Case

After streaming and filtering, partition positions for different tasks:

```python
def partition_evals(evals):
    """Partition filtered evals by training task suitability."""
    mate_positions = []      # For tactical training (T3, T7)
    high_eval = []           # |cp| > 200, for clear advantage (T4)
    balanced = []            # |cp| < 100, for nuanced eval (T4)
    endgame = []             # Few pieces, for endgame tasks (T6)
    best_move_pool = []      # High depth, for best move tasks (T7)

    for row in evals:
        board = chess.Board(row["fen"])
        piece_count = len(board.piece_map())

        if row["mate"] is not None:
            mate_positions.append(row)
        if row["cp"] is not None:
            if abs(row["cp"]) > 200:
                high_eval.append(row)
            elif abs(row["cp"]) < 100:
                balanced.append(row)
        if piece_count <= 10:
            endgame.append(row)
        if row["depth"] >= 30:
            best_move_pool.append(row)

    return {
        "mate": mate_positions,
        "high_eval": high_eval,
        "balanced": balanced,
        "endgame": endgame,
        "best_move": best_move_pool,
    }
```

---

## Chess960 Generation

### Why Chess960

Chess960 has 960 legal starting positions where back-rank pieces are
shuffled (bishops on opposite colors, king between rooks). This forces
the model to learn general rules rather than memorize standard patterns.

### Generation

```python
import chess
import random

def generate_chess960_position():
    """Generate a random Chess960 starting position."""
    pos_id = random.randint(0, 959)
    board = chess.Board.from_chess960_pos(pos_id)
    board.chess960 = True  # Enable Chess960 castling rules
    return board, pos_id
```

### Chess960 Castling

Chess960 castling is more complex than standard. King and rook may start on
various squares but ALWAYS end on the standard squares:

- Kingside ("g-castling"): King to g1, rook to f1
- Queenside ("c-castling"): King to c1, rook to d1

In UCI notation for Chess960, castling is encoded as king-captures-rook
(e.g., if king is on d1 and rook is on a1, queenside castling is `d1a1`).

### Mix Ratio

- **Tiers 1-2** (Perception, Rules): 20% Chess960
- **Tier 3** (Tactics): 15% Chess960
- **Tiers 4-7**: 5-10% Chess960

---

## Opening Theory Data

### Lichess Openings Database

The 3,630 named openings provide: eco-volume (A-E), eco code, name, pgn,
uci (already in target format), and epd (resulting position).

### Polyglot Book Enhancement

Polyglot books give weighted move distributions — something the named
openings list doesn't provide. This enables:

1. **Move popularity**: "e2e4 is the most popular (weight 382), followed
   by d2d4 (weight 295) and g1f3 (weight 98)."
2. **Continuation depth**: Books go 10-20 moves deep into main lines.
3. **Off-book detection**: No entry = left established theory.

### Opening Character Taxonomy

Openings are categorized for training insight generation:

- **Open games** (1.e4 e5): Tactical, piece activity, early castling
- **Semi-open** (1.e4, no e5): Asymmetric structures, positional imbalances
- **Closed games** (1.d4 d5): Strategic, pawn chains, slower development
- **Indian defenses** (1.d4 Nf6): Hypermodern, fianchettoes, delayed center
- **Flank openings** (1.c4, 1.Nf3): Flexible, positional, transpositional

---

## Endgame Theory Data

### Syzygy Integration

For each endgame type, generate positions by:

1. **Tablebase sampling**: Enumerate positions, sample across WDL categories
   and DTZ distances.
2. **Lichess game extraction**: Find positions from real games with ≤7 pieces.
3. **Stockfish analysis**: For 6+ piece positions outside tablebase coverage.

### Endgame Types to Cover

**Basic mates** (mandatory, high volume): KQ vs K, KR vs K, KBB vs K, KBN vs K

**Pawn endgames** (critical): KP vs K (key squares, opposition, rule of square),
KPP vs KP, complex pawn endings (triangulation, zugzwang, breakthrough)

**Rook endgames** (most common): KR+P vs KR (Lucena/Philidor), KR+PP vs KR+P

**Minor piece endgames**: KB+P vs K, KN+P vs K, KB vs KN with pawns

**Queen endgames**: KQ vs KP (pawn on 7th)

---

## MATE Dataset Preprocessing

Since the HuggingFace viewer shows a parsing error, raw data may need manual
inspection. Processing steps:

1. Load JSON/JSONL files directly
2. Extract FEN, candidate moves, annotations
3. Convert any SAN moves to UCI using python-chess
4. Validate all positions and moves
5. Split into training (reasoning traces) and eval (benchmark)

---

## Pipeline Architecture

```
┌─────────────────────────────┐
│    Source Data Downloaders    │
│  (Lichess, MATE, Openings,   │
│   Position Evaluations)      │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│       FEN Pool Builder       │
│  Extract + deduplicate FENs  │
│  Tag: phase, material, type  │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│     Stockfish Annotator      │
│  Batch eval, best move, PV   │
│  ~200 positions/sec @ d15    │
│  (Skip for Position Evals    │
│   dataset — already has SF)  │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│     Eval Split Generator     │
│  FIRST: reserve eval data    │
│  Block eval FENs from train  │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│      Task Generators         │
│  One module per task tier    │
│  Template selection + output │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│     Validator + Formatter    │
│  python-chess verification   │
│  Output: JSONL per task      │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│       Parquet Converter      │
│  SDPO/verl-compatible format │
└─────────────────────────────┘
```

**Note on Position Evaluations dataset**: Because this dataset already
contains Stockfish evaluations and PV lines, it can bypass the Stockfish
Annotator step entirely. This dramatically reduces compute for the ~342M
unique positions it covers. Use live Stockfish annotation only for positions
not found in this dataset (e.g., Chess960 positions, custom endgame
positions, game-extracted FENs not in the dataset).

---

## Validation Rules

Every generated example passes through validation:

1. **FEN validity**: `chess.Board(fen)` does not throw
2. **Legal move lists**: Match `[m.uci() for m in board.legal_moves]` exactly
3. **Move validity**: All UCI moves in outputs are legal in the position
4. **State tracking**: Resulting FEN matches python-chess replay
5. **Stockfish consistency**: Best moves verified independently
6. **Template completeness**: All template placeholders are filled
7. **No eval contamination**: Training FENs do not appear in eval set

---

## Output Format

Training data is output as JSONL, one file per task:

```json
{
  "task": "legal_move_gen",
  "tier": 2,
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "is_chess960": false,
  "messages": [
    {"role": "system", "content": "You are a chess reasoning engine..."},
    {"role": "user", "content": "What moves are available?\nPosition: rnbqkbnr/..."},
    {"role": "assistant", "content": "a7a6 a7a5 b7b6 b7b5..."}
  ],
  "metadata": {
    "source": "lichess_games",
    "game_phase": "opening",
    "stockfish_eval_cp": 35,
    "template_id": 3
  }
}
```

---

## Prompt Template System

Every task uses a pool of 5-10 prompt templates. During data generation,
one is randomly selected per example.

```python
TEMPLATES = {
    "board_print": [
        "Position (FEN): {fen}\nShow me the board.",
        "Display this position:\n{fen}",
        "Render the board for FEN: {fen}",
        "FEN: {fen}\nPrint the chess board.",
        "What does this position look like?\n{fen}",
        "Here is a FEN string: {fen}\nLay out the board in ASCII.",
        "Draw the board state for: {fen}",
    ],
    # ... templates for every task defined in phase specs
}
```

---

## Eval Split Generation

**Critical**: Eval splits are generated FIRST and their FEN positions are
stored in a blocklist. No training example may share a FEN with any eval
example. See `04_evaluation_benchmark.md` for full details.

```python
def generate_eval_split(task_name, source_data, n_eval, seed=42):
    """Generate a held-out eval split before training data."""
    rng = random.Random(seed)
    eval_indices = rng.sample(range(len(source_data)), n_eval)
    eval_data = [source_data[i] for i in eval_indices]
    train_data = [source_data[i] for i in range(len(source_data))
                  if i not in set(eval_indices)]
    return eval_data, train_data
```

---

## Checklist

- [ ] Download all datasets from HuggingFace
- [ ] Install and configure Stockfish 17+
- [ ] Download Syzygy 3-5 piece tablebases
- [ ] Obtain Polyglot opening book(s)
- [ ] Build FEN pool from Lichess games (Elo ≥ 2000)
- [ ] Preprocess Lichess puzzles (apply setup move)
- [ ] Stream and filter Position Evaluations dataset (depth ≥ 20)
- [ ] Partition Position Evaluations by task suitability
- [ ] Generate Chess960 positions
- [ ] Preprocess MATE dataset (SAN → UCI conversion)
- [ ] Generate opening insights from Lichess openings + Polyglot
- [ ] Sample endgame positions from Syzygy tablebases
- [ ] Generate eval splits BEFORE any training data (see `04_evaluation_benchmark.md`)
- [ ] Run Stockfish batch annotation on FENs not covered by Position Evals
- [ ] Validate all generated data with python-chess
- [ ] Output JSONL per task, ready for phase-specific training
