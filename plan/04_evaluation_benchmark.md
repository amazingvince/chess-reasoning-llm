# Evaluation Benchmark

**When**: Built during Stage 0, BEFORE any training data is generated
**Status**: Frozen after creation — never modified
**Total size**: ~13,000 held-out examples

---

## Overview

The evaluation benchmark is the single source of truth for measuring
progress across all SFT phases and into SDPO. It is created FIRST,
before any training data generation, and its FEN positions are stored
in a blocklist that prevents any training example from sharing a FEN
with any eval example.

---

## Design Principles

- **Held-out from training**: Eval data is separated BEFORE data generation
- **Stratified**: Each tier has its own eval split
- **Versioned**: Benchmark is frozen — never modified after initial creation
- **Automated**: All metrics computable without human judgment
- **Fast**: Full eval runs in < 1 hour on single GPU

---

## Benchmark Splits

| Split | Source | Size | Key Metrics |
|-------|--------|------|-------------|
| **Perception** | Random FEN + Chess960 | 2,000 | Board print accuracy, Board→FEN accuracy, piece ID, piece count, state tracking |
| **Rules** | Random FEN + Chess960 | 2,000 | Legal move gen (exact set), legality check, check/mate detection, special rules |
| **Tactics** | Lichess puzzles (by theme) | 2,000 | Capture ID, threat detection F1, pattern recognition, hanging piece detection |
| **Evaluation** | Stockfish-evaluated positions | 1,500 | Material balance, eval bucket accuracy (5 buckets), pawn structure features |
| **Openings** | Lichess openings (held-out ECOs) | 500 | Opening name accuracy, continuation quality (Stockfish rank) |
| **Endgames** | Syzygy-backed positions | 1,500 | WDL accuracy, best move (DTZ-optimal), endgame type classification |
| **Planning** | Lichess puzzles + engine-evaluated positions | 2,000 | Puzzle pass@1, pass@8, ACPL, format compliance, legal move rate |
| **Chess960** | Chess960 positions only | 500 | Legal move gen, castling rule accuracy |
| **MATE** | MATE dataset (held-out) | 1,000 | Binary choice accuracy (with/without explanations) |
| **Total** | | **13,000** | |

### Evaluation Data from Position Evaluations Dataset

The position evaluations dataset provides high-quality ground truth for
the Evaluation split. Use positions with depth ≥ 40 for near-perfect
centipawn evaluations. These supplement or replace live Stockfish
annotation for eval benchmark construction.

For the Planning split, positions with depth ≥ 30 provide reliable
best-move labels for ACPL calculation.

---

## Metric Definitions

### Exact Match Accuracy

Output exactly matches ground truth after normalization (whitespace,
ordering for move lists).

### Legal Move Generation Accuracy

The set of generated moves exactly matches the set from `board.legal_moves`.
Partial credit: Jaccard similarity (intersection / union of move sets).

### Eval Bucket Accuracy

Model's qualitative assessment (equal/slight/clear/winning/decisive)
matches the bucket derived from Stockfish centipawn eval.

| Bucket | Eval Range |
|--------|-----------|
| Equal | \|cp\| < 50 |
| Slight edge | 50-150cp |
| Clear advantage | 150-300cp |
| Winning | 300-600cp |
| Decisive | 600cp+ |

### ACPL (Average Centipawn Loss)

For each position, compute
`max(0, stockfish_best_eval - stockfish_eval_of_model_move)` in
centipawns. Average across all positions. Lower is better.

Reference: Top human GMs average 20-30 ACPL.

### Puzzle pass@k

Generate k independent samples (temperature > 0). Puzzle is "passed"
if ANY sample contains the correct first solution move. Report for
k=1 and k=8.

### Format Compliance

Output contains a valid `<think>...</think>` block followed by a valid
`<move>...</move>` block with a parseable UCI move.

### Legal Move Rate

Of all outputs that contain a `<move>` tag, what percentage contain a
move that is legal in the given position.

---

## Eval Split Generation

```python
import random

def generate_eval_split(task_name, source_data, n_eval, seed=42):
    """Generate a held-out eval split before training data."""
    rng = random.Random(seed)
    eval_indices = rng.sample(range(len(source_data)), n_eval)
    eval_data = [source_data[i] for i in eval_indices]
    train_data = [source_data[i] for i in range(len(source_data))
                  if i not in set(eval_indices)]
    return eval_data, train_data
```

**Critical**: Eval splits are generated FIRST and their FEN positions
are stored in a blocklist. No training example may share a FEN with
any eval example.

```python
# Build global blocklist
eval_fens = set()
for split in all_eval_splits:
    for example in split:
        eval_fens.add(example["fen"])

# During training data generation
def is_allowed_for_training(fen):
    return fen not in eval_fens
```

---

## Reporting Format

After each training phase, run the full benchmark and report in this
format:

```
=== Phase A Evaluation ===
Perception:
  Board print accuracy:    92.3%
  Board → FEN accuracy:    88.7%
  Piece ID accuracy:       95.1%
  State tracking (1-2m):   83.4%
  State tracking (3-5m):   71.2%
Rules:
  Legal move gen accuracy: 87.5%
  Legality check accuracy: 91.0%
  Check/mate detection:    89.3%
  Chess960 castling:       78.4%
...

=== Phase B Evaluation ===
[All Phase A metrics + regression deltas]
Tactics:
  Capture ID accuracy:     87.2%
  Threat detection F1:     72.1%
  ...
Evaluation:
  Material balance:        96.3%
  Eval bucket accuracy:    68.5%
  ...

=== Phase C Evaluation ===
[All prior metrics + regression deltas]
Planning:
  Puzzle pass@1:           23.4%
  Puzzle pass@8:           44.1%
  ACPL:                    178
  Format compliance:       97.2%
  Legal move rate:         96.8%
MATE:
  Binary choice accuracy:  58.3%
```

### Regression Tracking

For each metric, report the delta from its best historical value:

```
Legal move gen accuracy: 86.2% (Δ -1.3% from Phase A best: 87.5%)
```

Flag any metric with > 5% regression in red.

---

## Benchmark Versioning

The benchmark is assigned a version identifier (e.g., `chess-sft-eval-v1`)
at creation time. All training runs reference this version. If the
benchmark ever needs updating (e.g., a bug is found in ground truth),
a new version is created and all metrics are re-run on both versions
to establish comparability.

---

## Checklist

- [ ] Generate Perception eval split (2,000 examples)
- [ ] Generate Rules eval split (2,000 examples)
- [ ] Generate Tactics eval split (2,000 examples from Lichess puzzles)
- [ ] Generate Evaluation eval split (1,500 examples with SF evals)
- [ ] Generate Openings eval split (500 held-out ECOs)
- [ ] Generate Endgames eval split (1,500 Syzygy-backed positions)
- [ ] Generate Planning eval split (2,000 examples)
- [ ] Generate Chess960 eval split (500 positions)
- [ ] Generate MATE eval split (1,000 held-out examples)
- [ ] Build global FEN blocklist from all eval splits
- [ ] Verify no FEN overlap between eval splits
- [ ] Implement all metric computation functions
- [ ] Validate benchmark against ground truth (100% accuracy on oracle)
- [ ] Freeze benchmark with version identifier
- [ ] Implement automated eval runner (< 1 hour on single GPU)
- [ ] Implement reporting template with regression tracking
