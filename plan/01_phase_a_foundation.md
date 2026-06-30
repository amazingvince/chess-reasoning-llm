# Phase A: Foundation — Perception + Rules (Tiers 1-2)

**Phase**: SFT Stage 1, Phase A
**Prerequisites**: Stage 0 complete (data preparation, eval splits generated)
**Data volume**: ~1.19M examples
**Training**: 2-3 epochs
**Focus**: FEN comprehension, legal move generation, board manipulation

---

## Overview

Phase A teaches the model to see the board. After this phase, the model
should be able to parse any FEN, produce an accurate board diagram,
identify pieces, track state through move sequences, and generate
complete legal move lists. This is pure chess literacy — no evaluation
or strategy yet.

---

## Tier 1: Perception — "What's on the board?"

**Purpose**: Teach the model to parse FEN, build internal board
representation, and convert between FEN and visual board layouts.

### Task 1.1: FEN → Board

Given a FEN, produce an ASCII board diagram.

**Templates**: 7 variants
```python
TEMPLATES_BOARD_PRINT = [
    "Position (FEN): {fen}\nShow me the board.",
    "Display this position:\n{fen}",
    "Render the board for FEN: {fen}",
    "FEN: {fen}\nPrint the chess board.",
    "What does this position look like?\n{fen}",
    "Here is a FEN string: {fen}\nLay out the board in ASCII.",
    "Draw the board state for: {fen}",
]
```

**Output format**:
```
  a b c d e f g h
8 r n b q k b n r
7 p p p p p p p p
6 . . . . . . . .
5 . . . . . . . .
4 . . . . P . . .
3 . . . . . . . .
2 P P P P . P P P
1 R N B Q K B N R

Side to move: Black
Castling: KQkq
En passant target: e3
```

**Volume**: ~80K (includes ~15K Chess960 positions)
**Source**: Random FEN from Lichess games + Chess960 generation

### Task 1.2: Board → FEN

Given an ASCII board and metadata, produce the FEN string.

**Templates**: 6 variants
**Input**: ASCII board + side to move + castling rights
**Output**: Complete FEN string
**Volume**: ~80K
**Source**: Same positions as Task 1.1, reversed
**Validation**: `chess.Board(output_fen)` must not throw

### Task 1.3: Piece Identification

Given a FEN, answer questions about piece placement.

**Templates**: 8 variants across subtask types
**Subtask types**:
- "What is on {square}?" → piece name or "empty"
- "Where is the {color} king?" → square
- "Which squares have {color} {piece_type}s?" → list
- "What {color} pieces are on rank {n}?" → list
- "List all pieces on the queenside." → list

**Volume**: ~100K (2-3 queries per position)
**Source**: Random FEN from Lichess games + Chess960

### Task 1.4: Piece Counting

Count pieces by type and side. Compute material balance.

**Templates**: 6 variants across subtask types
**Subtask types**:
- Default Phase A data: full count for both sides plus material balance
- Optional partial-count prompts: color totals, piece-type totals, and minor
  pieces via `piece_counting_include_partial=True`

**Output**: Compact counts and material balance only; detailed square inventories
belong in the side-piece inventory task.

**Volume**: ~80K
**Source**: Positions sampled across material-balance spectrum

### Task 1.5: State Tracking

Given a FEN and a short sequence of legal moves, predict the resulting FEN.

**Templates**: 6 variants
**Move sequences**: one ply by default
**Output**: `Result FEN: ...` only. The square/rank trace is taught separately
by Task 1.8 and Task 1.9.
**Volume**: ~100K
**Source**: Lichess game positions with FEN-pool fallback, replayed with python-chess
**Validation**: Every resulting FEN is verified with python-chess

### Task 1.6: Square Lookup

Given a FEN and a square, return the explicit square lookup.

**Output format**: `d1=white queen` or `f3=empty`
**Volume**: ~100K
**Purpose**: Teach direct FEN square addressing before full state tracking.

### Task 1.7: Rank Lookup

Given a FEN and a rank, return the compressed FEN row for that rank.

**Output format**: `rank 1: RNBQKBNR`
**Volume**: ~80K
**Purpose**: Teach rank-row indexing and compressed empty-square notation.

### Task 1.8: Move Square Edits

Given a FEN and one legal move, return only the square lookup/edit trace.

**Output format**:
```
Lookup: d1=white queen; f3=empty.
Squares: d1 white queen->empty; f3 empty->white queen.
Ranks: rank 1 d Q->1; rank 3 f 1->Q.
```

**Volume**: ~100K
**Purpose**: Teach the edit mechanics used inside state tracking without also
requiring the final full-FEN reconstruction.

### Task 1.10: FEN Row Application

Given a FEN and one legal move, return the affected compressed FEN rank-row
rewrite(s), then the final full FEN.

**Output format**:
```
Rows: rank 8 3qk3->4k3; rank 4 8->3q4.
Result FEN: 4k3/8/8/8/3q4/8/8/4K3 w - - 1 2
```

**Volume**: ~100K
**Purpose**: Bridge square edits to actual FEN assembly by teaching whole-row
replacement directly, without repeating the longer lookup trace.

---

## Tier 2: Rules & Mechanics — "What can I do?"

**Purpose**: Teach legal move generation, special rules, and position status.

### Task 2.1: Legal Move Generation

List all legal moves for the side to move in UCI notation.

**Templates**: 6 variants
**Output**: Space-separated UCI moves + total count
**Volume**: ~100K (includes ~20K Chess960)
**Validation**: Must exactly match `[m.uci() for m in board.legal_moves]`

### Task 2.2: Piece-Specific Legal Moves

List legal moves for a specific piece on a named square.

**Templates**: 6 variants
**Output**: UCI moves from that square + count
**Volume**: ~80K
**Source**: Sample 1-2 pieces per position, prefer pieces with 3+ legal moves

### Task 2.3: Move Legality Verification

Check whether a specific move is legal. Brief factual answer only.

**Templates**: 6 variants
**Output**: "Yes. [one-sentence reason]" or "No. [one-sentence reason]"
**Volume**: ~80K (50% legal, 50% illegal)

**Illegal move generation** — create plausible-looking but illegal moves:
- Wrong piece movement (knight moving like bishop)
- Path blocked (bishop through occupied square)
- Would leave king in check
- Castling when rights forfeited
- En passant when not available

### Task 2.4: Check and Checkmate Detection

Determine position status.

**Templates**: 6 variants
**Output categories**: Normal play, {color} in check (from {piece} on
{square}), Checkmate ({color} wins), Stalemate (draw)
**Volume**: ~60K (balanced: ~20K check, ~15K checkmate, ~5K stalemate, ~20K normal)
**Source**: Curate checkmates and stalemates from Lichess games + puzzles;
normal positions are abundant.

### Task 2.5: Special Rules

Castling (including Chess960), en passant, promotion.

**Templates**: 8 variants across rule types
**Subtask types**:
- "Can {color} castle? Which side(s)?" — verify rights, path, check
- "Is en passant available?" — verify en passant square
- "What promotions are possible?" — verify pawn on 7th rank

**Volume**: ~50K (heavy on castling, including Chess960 castling)
**Chess960 castling examples**: ~10K specifically for non-standard castling

---

## Chess960 in Phase A

Phase A has the highest Chess960 mix ratio (20%) because this is where
piece movement rules are learned. The model must internalize that:

- Bishops move diagonally from ANY starting square
- Knights make L-shapes regardless of position
- Castling generalizes: king always ends on g1/c1, rook on f1/d1
- No reliance on "knights start on b1/g1"

Include a dedicated Chess960 eval split:
- Legal move generation on 960 positions
- Castling rule verification on 960 positions
- Piece movement on non-standard starting squares

---

## Training Configuration

**Data**: ~1.19M examples (820K Tier 1 + 370K Tier 2)
**Epochs**: 2-3
**Mixing**: Shuffle all Tier 1 and Tier 2 examples together. Do not
train tiers sequentially within the phase.
**Batch**: Include both tiers in each batch for balanced learning.

---

## Checkpoint Evaluation

Run the full eval benchmark (see `04_evaluation_benchmark.md`) after
Phase A training completes. Key metrics:

| Metric | Target |
|--------|--------|
| Board print accuracy (exact match) | > 90% |
| Board → FEN accuracy (exact match) | > 85% |
| Piece ID accuracy | > 90% |
| Piece count accuracy | > 95% |
| State tracking (1 move, exact FEN) | > 80% |
| Legal move gen accuracy (exact set) | > 85% |
| Legality check accuracy (binary) | > 90% |
| Check/checkmate detection | > 85% |
| Chess960 legal moves | > 75% |
| Chess960 castling rules | > 70% |

### Pass Criteria Before Phase B

All of the following must be met:

- Board print: > 90% exact match
- Legal moves: > 85% exact set match
- Legality check: > 90% accuracy
- State tracking (1 move): > 80% exact match
- No Tier 1-2 metric below 65%

If criteria are not met, extend training for 1-2 additional epochs or
investigate data quality issues before proceeding.

---

## Checklist

- [ ] Generate Task 1.1 data (FEN → Board, ~80K)
- [ ] Generate Task 1.2 data (Board → FEN, ~80K)
- [ ] Generate Task 1.3 data (Piece Identification, ~100K)
- [ ] Generate Task 1.4 data (Piece Counting, ~80K)
- [ ] Generate Task 1.5 data (State Tracking, ~100K)
- [ ] Generate Task 1.6 data (Square Lookup, ~100K)
- [ ] Generate Task 1.7 data (Rank Lookup, ~80K)
- [ ] Generate Task 1.8 data (Move Square Edits, ~100K)
- [ ] Generate Task 2.1 data (Legal Move Gen, ~100K)
- [ ] Generate Task 2.2 data (Piece-Specific Moves, ~80K)
- [ ] Generate Task 2.3 data (Legality Verification, ~80K)
- [ ] Generate Task 2.4 data (Check/Mate Detection, ~60K)
- [ ] Generate Task 2.5 data (Special Rules, ~50K)
- [ ] Validate Chess960 mix ratio (~20%)
- [ ] Run python-chess validation on all generated data
- [ ] Confirm no eval set contamination
- [ ] Train 2-3 epochs
- [ ] Run Phase A checkpoint evaluation
- [ ] Confirm pass criteria met before Phase B
