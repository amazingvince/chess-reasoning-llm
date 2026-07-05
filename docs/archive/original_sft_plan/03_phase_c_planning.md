# Phase C: Planning — Best Move Selection + Full Curriculum (Tier 7)

**Phase**: SFT Stage 1, Phase C (final SFT phase)
**Prerequisites**: Phase B pass criteria met
**Data volume**: All tiers (~1.6M) with emphasis on Tier 7, 20% Tier 1-4 review
**Training**: 2-3 epochs
**Focus**: Move selection, reasoning quality, chain-of-thought traces

---

## Overview

Phase C is the culmination of SFT. The model learns to select strong
moves with structured reasoning, solve tactical puzzles, and analyze
move consequences. This phase introduces the `<think>` → `<move>` format
that carries directly into SDPO. The full curriculum trains on all tiers
simultaneously, with heavy emphasis on Tier 7 and a review mix from
earlier tiers to maintain all skills.

---

## Tier 7: Planning — "What should I play?"

**Purpose**: Best move selection and puzzle solving with chain-of-thought
reasoning. This tier bridges directly into SDPO.

### Task 7.1: Best Move Selection

The core move-selection task with structured reasoning.

**Templates**: 7 variants
**Input**: FEN + side to move + legal move list
**Output**: `<think>` reasoning trace + `<move>` UCI move
**Volume**: ~80K
**Source**: Positions from Lichess games + Position Evaluations dataset,
Stockfish best move as ground truth, reasoning traces generated from
templates + Stockfish analysis + MATE dataset annotations

**Position Evaluations dataset integration**: For Task 7.1, the position
evaluations dataset is a primary data source. Use positions with depth ≥ 30
for reliable best-move labels. The `line` field's first move is the best
move, and the full PV line provides the expected continuation for reasoning
trace generation.

```python
def generate_best_move_from_eval(row):
    """Generate a best-move training example from position evaluations."""
    fen = row["fen"]
    board = chess.Board(fen)
    best_move = row["pv_line"].split()[0]
    pv = row["pv_line"].split()
    cp = row["cp"]
    mate = row["mate"]

    legal_moves = [m.uci() for m in board.legal_moves]

    # Generate reasoning trace from PV and evaluation
    if mate is not None:
        trace = generate_mate_trace(board, best_move, mate, pv)
    elif abs(cp) > 200:
        trace = generate_tactical_trace(board, best_move, cp, pv)
    else:
        trace = generate_positional_trace(board, best_move, cp, pv)

    return {
        "fen": fen,
        "legal_moves": legal_moves,
        "think": trace,
        "move": best_move,
    }
```

### Task 7.2: Puzzle Solving

Tactical puzzles from Lichess (preprocessed per `00_data_preparation.md`).

**Templates**: 6 variants
**Input**: Preprocessed puzzle FEN + legal moves + context hint
  ("This is a tactical puzzle" or theme hint like "Find the fork")
**Output**: `<think>` reasoning + `<move>` first solution move
**Volume**: ~50K
**Source**: Lichess puzzles, filtered by rating 800-2500

### Task 7.3: Move Consequence Analysis

Analyze what happens after a specific move using real game continuations.

**Templates**: 6 variants
**Input**: FEN + specific move to analyze
**Output**: 2-4 likely responses, resulting position character, evaluation
**Volume**: ~40K
**Source**: Lichess games — sample positions, show the actual game continuation.
Also supplemented by Position Evaluations dataset PV lines which show
Stockfish's expected continuation.

---

## MATE Dataset Integration (Tier 7)

### Purpose 1: Reasoning Trace Generation

The strategy/tactic annotations from MATE-ST are converted into our
`<think>` trace format. Expert explanations become the reasoning the
model should learn to produce:

```
MATE-ST example:
  Position: [FEN]
  Move A: d2d4 (better)
  Move B: e2e3
  Strategy: "White should seize central space and open lines for
    the dark-squared bishop"
  Tactic: "d2d4 attacks the e5 pawn and gains a tempo on the knight"

Converted to our format:
  <think>
  Looking at this position, White needs to develop actively.
  d2d4 seizes central space and opens lines for the dark-squared
  bishop. It also attacks the e5 pawn, potentially gaining a tempo
  on the Black knight. e2e3 is too passive — it doesn't challenge
  the center or create threats.
  </think>
  <move>d2d4</move>
```

### Purpose 2: Binary Choice Evaluation

The two-candidate-move format is a natural eval task. Hold out a subset
for the benchmark (see `04_evaluation_benchmark.md`).

---

## Reasoning Trace Format

### Structure

All Tier 7 tasks use a structured `<think>` → `<move>` format:

```
<think>
[Structured reasoning, 50-150 tokens for 0.8B model]
</think>
<move>[UCI move]</move>
```

### Reasoning Templates by Position Type

**For tactical positions** (puzzles, sharp positions):
```
<think>
Looking for forcing moves: checks, captures, threats.
[Identify the key pattern: fork/pin/skewer/etc.]
[Evaluate the main candidate: move → response → outcome]
[Confirm: does this work?]
</think>
```

**For positional positions** (quiet middlegame, endgame):
```
<think>
[Material status: equal / advantage / disadvantage]
[Key positional feature: pawn structure / piece activity / king safety]
[Candidate moves: 2-3 options with brief assessment]
[Selection: best move because...]
</think>
```

**For endgame positions**:
```
<think>
[Endgame type identification]
[Key technique or principle that applies]
[Concrete calculation if needed]
[Best move and why it makes progress]
</think>
```

### Length Calibration

For a 0.8B model, shorter is better. Long reasoning chains degrade quality
at this parameter count.

| Position Type | Token Range |
|---------------|-------------|
| Obvious moves (mate in 1, only legal move) | 20-40 tokens |
| Tactical puzzles | 50-100 tokens |
| Positional decisions | 80-150 tokens |
| Maximum (complex multi-candidate) | 200 tokens |

### Generating Reasoning Traces

Traces are generated programmatically from Stockfish analysis:

```python
def generate_reasoning_trace(fen, best_move, eval_cp, top_moves, themes):
    """Generate a reasoning trace from Stockfish analysis."""
    board = chess.Board(fen)

    if is_tactical(eval_cp, top_moves):
        return generate_tactical_trace(board, best_move, themes)
    elif is_endgame(board):
        return generate_endgame_trace(board, best_move, eval_cp)
    else:
        return generate_positional_trace(board, best_move, eval_cp, top_moves)

def generate_tactical_trace(board, best_move, themes):
    parts = []
    parts.append("Looking for forcing moves.")

    if "fork" in themes:
        target_squares = find_fork_targets(board, best_move)
        parts.append(f"{move_piece_name(board, best_move)} to "
                     f"{best_move[2:4]} attacks "
                     f"{describe_targets(target_squares)}.")
    elif "pin" in themes:
        parts.append(describe_pin(board, best_move))
    # ... etc for other themes

    parts.append(f"Best move: {best_move}.")
    return " ".join(parts)
```

For richer traces, MATE-ST annotations are converted into the `<think>`
format as shown above.

---

## Full Curriculum Training

### Data Mix

Phase C trains on ALL tiers simultaneously:

| Source | Volume | % of Training |
|--------|--------|---------------|
| Tier 7 (new) | ~170K | ~11% |
| Tiers 3-6 (from Phase B) | ~485K | ~30% |
| Tiers 1-2 (review, 20% of original) | ~162K | ~10% |
| **Tier 7 emphasis** (duplicated/upsampled) | ~780K | ~49% |

**Upsampling strategy**: Tier 7 examples are upsampled ~4-5x to form
roughly half the training data. This heavy emphasis ensures the model
spends most of its learning on the target task (move selection with
reasoning) while maintaining earlier skills.

### Why Full Curriculum

Training on all tiers simultaneously in Phase C serves two purposes:

1. **Skill maintenance**: The model retains perception, rules, and
   evaluation capabilities from earlier phases.
2. **Skill integration**: Tier 7 reasoning draws on ALL lower-tier
   skills. Seeing those skills alongside planning tasks helps the
   model connect them.

---

## Training Configuration

**Data**: ~1.6M total (see mix above)
**Epochs**: 2-3
**Starting checkpoint**: Phase B best checkpoint
**Mixing**: Full shuffle across all tiers per batch

---

## Checkpoint Evaluation

Run the full eval benchmark after Phase C. All metrics from Phases A
and B are re-measured.

### Prior Metrics (Regression Check)

All Phase A and Phase B metrics must remain within 5% of their
best values. Tier 7 upsampling should not degrade foundation skills.

### New Phase C Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Puzzle pass@1 | > 20% | Baseline — SDPO will improve |
| Puzzle pass@8 | > 40% | k=8 independent samples |
| ACPL (avg centipawn loss) | < 200 | Baseline — SDPO will improve |
| Format compliance | > 95% | Valid `<think>`/`<move>` structure |
| Legal move rate | > 95% | `<move>` output is legal in position |
| MATE binary choice accuracy | > 55% | Better than random (50%) |

### Exit Criteria (Ready for SDPO)

All of the following must be met before proceeding to SDPO:

- **Legal move rate: > 95%** — SDPO can't optimize move quality if
  the model can't produce legal moves
- **Format compliance: > 95%** — SDPO expects `<think>`/`<move>` format
- **Puzzle pass@1: > 20%** — baseline tactical capability exists
- **ACPL: < 200** — model plays somewhat reasonable chess
- **All Tier 1-2 metrics maintained** (< 5% regression from Phase A)

If exit criteria are not met, options include:
1. Additional training epochs on Phase C data
2. Increasing Tier 7 upsampling ratio
3. Investigating specific failure modes (e.g., format issues → more
   format-focused training data)

---

## Checklist

- [ ] Generate Task 7.1 data (Best Move, ~80K) — using Position Evals + Lichess games
- [ ] Generate Task 7.2 data (Puzzle Solving, ~50K)
- [ ] Generate Task 7.3 data (Move Consequence, ~40K)
- [ ] Convert MATE-ST annotations to `<think>` trace format
- [ ] Generate reasoning traces for all Tier 7 examples
- [ ] Calibrate trace lengths (20-200 tokens)
- [ ] Prepare full curriculum mix (all tiers + Tier 7 upsampling)
- [ ] Sample 20% Tier 1-2 review data
- [ ] Include all Tier 3-6 data from Phase B
- [ ] Validate all `<think>`/`<move>` outputs parse correctly
- [ ] Confirm no eval set contamination
- [ ] Train 2-3 epochs from Phase B checkpoint
- [ ] Run Phase C checkpoint evaluation
- [ ] Verify < 5% regression on Phase A/B metrics
- [ ] Confirm exit criteria met for SDPO handoff
