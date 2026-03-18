# Phase B: Understanding — Tactics, Evaluation, Openings, Endgames (Tiers 3-6)

**Phase**: SFT Stage 1, Phase B
**Prerequisites**: Phase A pass criteria met
**Data volume**: ~485K new (Tiers 3-6) + ~243K review (30% of Tiers 1-2)
**Training**: 2-3 epochs
**Focus**: Tactical awareness, position evaluation, opening/endgame theory

---

## Overview

Phase B teaches the model to understand what's happening on the board.
After this phase, the model should recognize tactical patterns, assess
who is winning and why, identify openings and their plans, and understand
basic endgame technique. The 30% Tier 1-2 review prevents catastrophic
forgetting of foundation skills.

---

## Tier 3: Tactical Awareness — "What's happening?"

**Purpose**: Teach pattern recognition for captures, threats, and tactical motifs.

### Task 3.1: Available Captures

List all captures for a specified color.

**Templates**: 5 variants
**Output**: UCI captures + what is captured
**Volume**: ~60K
**Source**: Positions from games where captures are available

### Task 3.2: Threats

Identify what a side is threatening to do next move.

**Templates**: 6 variants
**Output**: Description of 1-3 main threats
**Volume**: ~50K
**Source**: Positions where Stockfish identifies significant threats
(not quiet positions with no clear threats)

### Task 3.3: Attacked & Defended Squares

Determine what squares a piece attacks or whether a square/piece is safe.

**Templates**: 7 variants across subtask types
**Subtask types**:
- "What squares does {piece} on {square} attack?"
- "Is {square} attacked by {color}?"
- "Is the {piece} on {square} defended?"

**Volume**: ~60K

### Task 3.4: Tactical Patterns

Identify specific tactical motifs. Only positions WITH patterns.

**Templates**: 6 variants
**Patterns covered**: pin, fork, skewer, discovered attack, discovered check,
double check, back rank mate threat, removal of the guard, overloaded piece,
deflection, decoy, interference, X-ray
**Volume**: ~50K
**Source**: Lichess puzzles tagged with specific themes, validated by Stockfish

### Task 3.5: Hanging Pieces

Identify undefended pieces that can be captured for free.

**Templates**: 5 variants
**Output**: List of hanging pieces with what can capture them
**Volume**: ~40K
**Source**: Positions from games where blunders occurred (Stockfish eval
swings > 200cp indicate hanging piece situations)

### Tactical Data from Position Evaluations Dataset

The Lichess position evaluations dataset enhances Tier 3 in two ways:

1. **Mate-in-N positions**: All rows where `mate` is not null provide
   forcing/tactical positions. Filter for small `|mate|` values (1-5)
   for direct tactical pattern training.

2. **High eval swings**: When the same FEN has evaluations where different
   candidate moves show large cp differences, the position likely contains
   a tactical element (one move is much better than alternatives).

---

## Tier 4: Evaluation — "Who's winning?"

**Purpose**: Teach positional and material assessment calibrated to
Stockfish centipawn evaluations.

### Task 4.1: Material Balance

Compute exact material advantage using standard piece values.

**Templates**: 6 variants
**Piece values**: P=1, N=3, B=3, R=5, Q=9
**Output**: Piece-by-piece count, total points, net advantage
**Volume**: ~50K
**Source**: Positions sampled across the full material spectrum

### Task 4.2: Position Evaluation

Holistic position assessment with Stockfish evaluation as ground truth.

**Templates**: 7 variants
**Output**: Assessment covering material, king safety, center control,
piece activity, pawn structure. Final verdict calibrated to centipawns:

| Eval Range | Label |
|-----------|-------|
| \|eval\| < 50cp | "Roughly equal" |
| 50-150cp | "Slight edge for {color}" |
| 150-300cp | "Clear advantage for {color}" |
| 300-600cp | "Winning for {color}" |
| 600cp+ | "Decisive advantage" |

**Volume**: ~60K
**Source**: Diverse positions with Stockfish evaluation at depth 20

**Position Evaluations dataset integration**: The 342M position dataset
provides pre-computed centipawn evaluations, eliminating the need for
live Stockfish annotation on most training positions. Filter for depth ≥ 20
and map cp values to the bucket labels above. This is the primary source
for Task 4.2 training data.

### Task 4.3: Pawn Structure Analysis

Identify structural features.

**Templates**: 5 variants
**Features to identify**: Isolated pawns, doubled pawns, backward pawns,
passed pawns, pawn chains, pawn islands, open/half-open files, pawn tension.
**Volume**: ~40K

---

## Tier 5: Opening Theory — "What's the plan from here?"

**Purpose**: Teach named opening recognition and typical plans.

### Task 5.1: Opening Identification

Given a move sequence or position, name the opening.

**Templates**: 6 variants (by move sequence, by FEN, by partial sequence)
**Output**: Opening name, ECO code, brief characterization
**Volume**: ~30K
**Source**: Lichess/chess-openings dataset, with paraphrased descriptions

### Task 5.2: Opening Continuation

Given a named opening position, suggest main line continuations.

**Templates**: 6 variants
**Output**: 2-4 candidate continuations with brief rationale
**Volume**: ~25K
**Source**: Lichess openings + Polyglot book weights for popularity ranking

### Task 5.3: Opening Principles & Plans

Given an early-game position, explain what each side should aim for.

**Templates**: 6 variants
**Output**: Key ideas, typical plans, pawn structure character, what to
watch for (drawing from the opening character taxonomy)
**Volume**: ~20K
**Source**: Generated from opening database + Stockfish analysis +
opening character taxonomy (see `00_data_preparation.md`)

### Opening Insight Generation

```python
def generate_opening_insights(eco, name, uci_moves, epd):
    """Generate training data about an opening's character."""
    board = chess.Board()
    moves = uci_moves.split()
    for m in moves:
        board.push(chess.Move.from_uci(m))

    eval = stockfish.evaluate(board)
    pawn_structure = analyze_pawns(board)
    center_control = evaluate_center(board)

    return {
        "eco": eco,
        "name": name,
        "fen": board.fen(),
        "eval_cp": eval,
        "character": classify_opening_character(pawn_structure),
        "key_ideas_white": generate_plan(board, chess.WHITE),
        "key_ideas_black": generate_plan(board, chess.BLACK),
        "pawn_structure": pawn_structure,
        "typical_plans": get_typical_plans(eco),
    }
```

---

## Tier 6: Endgame Theory — "How do I convert this?"

**Purpose**: Teach endgame technique with tablebase-backed perfect knowledge.

### Task 6.1: Endgame Classification

Identify the endgame type and expected result.

**Templates**: 5 variants
**Output**: Endgame type name, expected result, key technique name
**Volume**: ~30K
**Source**: Syzygy-backed positions + Lichess endgame positions

### Task 6.2: Endgame Evaluation (Tablebase-Backed)

For ≤5 piece positions: is this won, drawn, or lost?

**Templates**: 6 variants
**Output**: WDL result + approximate DTZ + technique description
**Volume**: ~40K
**Source**: Syzygy tablebases, sampling across all common material configs.
Ground truth is PERFECT (mathematical proof, not estimation).

### Task 6.3: Endgame Best Move (Tablebase-Backed)

Find the optimal move in an endgame position.

**Templates**: 6 variants
**Output**: Best move in UCI + explanation of the technique
**Volume**: ~40K
**Source**: Syzygy DTZ-optimal moves with Stockfish analysis for explanation

### Task 6.4: Endgame Principles

Teach named endgame concepts with illustrative positions.

**Templates**: 6 variants
**Concepts covered**: Opposition (direct, distant, diagonal), key squares,
Lucena position, Philidor position, triangulation, zugzwang, fortress,
wrong bishop + rook pawn draw, two bishops mate, bishop + knight mate,
queen vs 7th-rank pawn, active rook principle, cutting off the king
**Volume**: ~30K
**Source**: Known theoretical positions + Syzygy verification

### Endgame Data from Position Evaluations Dataset

The position evaluations dataset provides additional endgame training data:

- Filter for FENs with ≤ 10 pieces (few-piece positions)
- Use cp/mate evaluations as ground truth for endgame assessment
- PV lines in endgame positions show correct technique
- Cross-reference with Syzygy for positions with ≤ 5 pieces to verify

---

## Tier 1-2 Review Data

To prevent catastrophic forgetting, 30% of Phase A data (~243K examples)
is mixed into Phase B training. Sample uniformly from all Tier 1-2 tasks.

**Mix ratio**: For each batch, ~33% Tier 1-2 review, ~67% Tier 3-6 new.

---

## Training Configuration

**Data**: ~728K total (485K new Tier 3-6 + 243K Tier 1-2 review)
**Epochs**: 2-3
**Mixing**: Shuffle all tiers together. Each batch should contain
examples from multiple tiers.
**Starting checkpoint**: Phase A best checkpoint

---

## Checkpoint Evaluation

Run the full eval benchmark after Phase B. All Phase A metrics are
re-measured to detect regression.

### Phase A Metrics (Regression Check)

All Phase A metrics must remain within 5% of their Phase A values.
If any drops more than 5%, increase Tier 1-2 review ratio and retrain.

### New Phase B Metrics

| Metric | Target |
|--------|--------|
| Capture identification accuracy | > 85% |
| Threat detection F1 | > 70% |
| Tactical pattern recognition | > 65% |
| Hanging piece detection | > 75% |
| Material balance (exact) | > 95% |
| Eval bucket accuracy (5 buckets) | > 65% |
| Pawn structure feature detection | > 60% |
| Opening name accuracy | > 70% |
| Opening continuation (SF rank) | top-3 in > 60% |
| Endgame WDL accuracy (≤5 pieces) | > 85% |
| Endgame best move (DTZ-optimal) | > 60% |
| Endgame type classification | > 80% |

### Pass Criteria Before Phase C

- All Phase A metrics maintained (< 5% regression)
- Material balance: > 95% exact
- Endgame WDL: > 85% accuracy on ≤5 piece positions
- No Phase B metric below 55%

---

## Checklist

- [ ] Generate Task 3.1 data (Captures, ~60K)
- [ ] Generate Task 3.2 data (Threats, ~50K)
- [ ] Generate Task 3.3 data (Attacked/Defended, ~60K)
- [ ] Generate Task 3.4 data (Tactical Patterns, ~50K)
- [ ] Generate Task 3.5 data (Hanging Pieces, ~40K)
- [ ] Generate Task 4.1 data (Material Balance, ~50K)
- [ ] Generate Task 4.2 data (Position Evaluation, ~60K) — using Position Evals dataset
- [ ] Generate Task 4.3 data (Pawn Structure, ~40K)
- [ ] Generate Task 5.1 data (Opening ID, ~30K)
- [ ] Generate Task 5.2 data (Continuation, ~25K)
- [ ] Generate Task 5.3 data (Principles, ~20K)
- [ ] Generate Task 6.1 data (Endgame Classification, ~30K)
- [ ] Generate Task 6.2 data (Endgame Eval, ~40K)
- [ ] Generate Task 6.3 data (Endgame Best Move, ~40K)
- [ ] Generate Task 6.4 data (Endgame Principles, ~30K)
- [ ] Sample 30% Tier 1-2 review data (~243K)
- [ ] Validate Chess960 mix ratio (~15% for T3, ~5-10% for T4-6)
- [ ] Run python-chess validation on all generated data
- [ ] Confirm no eval set contamination
- [ ] Train 2-3 epochs from Phase A checkpoint
- [ ] Run Phase B checkpoint evaluation
- [ ] Verify < 5% regression on Phase A metrics
- [ ] Confirm pass criteria met before Phase C
