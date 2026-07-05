# SDPO Handoff — From SFT to Reinforcement Learning

**Phase**: Post-SFT (Stage 2)
**Prerequisites**: Phase C exit criteria met
**Framework**: SDPO (Self-Distilled Policy Optimization)
**Reference**: Hübotter et al., arXiv:2601.20802, https://github.com/lasgroup/SDPO

---

## Overview

After completing all three SFT phases, the model enters SDPO reinforcement
learning. This document specifies what the SFT model provides to SDPO,
how the reward function works, and how the rich feedback loop leverages
the model's SFT-trained understanding.

SDPO can only refine what already exists. SFT builds the foundation;
SDPO optimizes the reasoning quality on top of it.

---

## What SFT Provides to SDPO

| Capability | Source Tier | SDPO Benefit |
|-----------|------------|--------------|
| FEN fluency | T1 | Can parse any position in SDPO prompts |
| Board → FEN / FEN → Board | T1 | Internal board representation |
| Legal move mastery | T2 | Doesn't waste reward signal on legality |
| Special rules knowledge | T2 | Handles castling, en passant correctly |
| Tactical vocabulary | T3 | Understands Stockfish feedback about tactics |
| Evaluation calibration | T4 | Produces eval-calibrated reasoning traces |
| Opening knowledge | T5 | Plays principled openings in self-play |
| Endgame technique | T6 | Can convert won endgames |
| CoT reasoning traces | T7 | SDPO refines quality, not format |

The key insight: SDPO's reward signal is spent on making the model
play BETTER chess, not on teaching it chess fundamentals. Without SFT,
SDPO would waste most of its signal on legality and format compliance.

---

## SDPO Reward Components

SDPO uses multi-component rewards from Stockfish:

```python
def sdpo_reward(fen, model_output, legal_moves, stockfish):
    reward = 0.0

    # Format compliance (from SFT → should be ~1.0)
    if has_valid_think_move_format(model_output):
        reward += 1.0
    else:
        reward -= 2.0
        return reward  # Early exit

    move = extract_move(model_output)

    # Legality (from SFT → should be ~1.0)
    if move in legal_moves:
        reward += 0.5
    else:
        reward -= 1.5
        return reward  # Early exit

    # Move quality (THIS is what SDPO optimizes)
    best_eval = stockfish.evaluate(fen, stockfish.best_move(fen))
    move_eval = stockfish.evaluate(fen, move)
    cp_diff = best_eval - move_eval
    quality_reward = max(-1.0, 1.0 - (cp_diff / 200))
    reward += quality_reward

    # Bonus for finding the actual best move
    if move == stockfish.best_move(fen):
        reward += 0.5

    return reward
```

### Reward Breakdown

| Component | Range | What it rewards | SFT contribution |
|-----------|-------|-----------------|------------------|
| Format | -2.0 to +1.0 | Valid `<think>`/`<move>` | Phase C: > 95% compliance |
| Legality | -1.5 to +0.5 | Legal move in position | Phase A: > 95% legal rate |
| Quality | -1.0 to +1.0 | Move strength (cp diff) | Phase C baseline; SDPO optimizes |
| Best move bonus | 0.0 or +0.5 | Finding SF's top move | Phase C baseline; SDPO optimizes |

Because SFT ensures format compliance (~95%) and legality (~95%), SDPO's
learning signal is concentrated on move quality — the hardest and most
impactful component.

---

## SDPO Rich Feedback

When SDPO generates a suboptimal move, Stockfish provides structured
feedback that the SDPO self-distillation mechanism uses:

```
Your move {model_move} scores {eval_cp}cp.
The best move is {best_move} scoring {best_cp}cp.
{best_move} is better because: {stockfish_pv_explanation}
```

Because the SFT-trained model understands:
- **Tactical vocabulary** (Tier 3) — it knows what "fork", "pin", "discovered check" mean
- **Evaluation language** (Tier 4) — it understands centipawn differences
- **Position assessment** (Tier 7) — it can process explanations about WHY a move is better

...it can *comprehend* this feedback and use it to improve its reasoning.
A model without SFT would not understand the feedback at all.

---

## Position Evaluations Dataset in SDPO

The Lichess position evaluations dataset can also serve SDPO directly:

1. **Training positions**: The 342M unique positions provide a massive
   pool of diverse positions for SDPO training episodes, far beyond what
   Lichess games alone offer.

2. **Pre-computed ground truth**: For positions that appear in the dataset,
   the best move and evaluation are already known at high depth, reducing
   the need for live Stockfish calls during SDPO training.

3. **PV-based feedback**: The principal variation lines provide rich
   context for the feedback mechanism — showing not just the best move
   but the expected line of play.

---

## Expected Trajectory

### At SFT Completion (Phase C Exit)

| Metric | Expected Value |
|--------|----------------|
| Format compliance | > 95% |
| Legal move rate | > 95% |
| Puzzle pass@1 | ~20-25% |
| ACPL | ~150-200 |
| Elo estimate | ~800-1200 |

### After SDPO (Expected Improvement)

| Metric | Expected Range | Improvement Source |
|--------|---------------|-------------------|
| Format compliance | ~99% | Reward signal |
| Legal move rate | ~99% | Reward signal |
| Puzzle pass@1 | ~35-50% | Quality reward + self-distillation |
| ACPL | ~80-120 | Quality reward |
| Elo estimate | ~1400-1800 | Overall improvement |

These are estimates based on Chess-R1 results and scaled for model size.
A 0.8B model will not reach GM-level play, but should achieve solid
intermediate club player strength.

---

## Stage 3: Self-Play Refinement (Future)

After SDPO, iterative self-play against Stockfish at calibrated strength
levels provides additional improvement. This is out of scope for the
SFT specification but is the planned next stage.

---

## Checklist

- [ ] Confirm Phase C exit criteria met
- [ ] Set up SDPO training infrastructure (verl-based)
- [ ] Configure Stockfish for SDPO reward computation
- [ ] Implement reward function with all components
- [ ] Implement rich feedback template
- [ ] Prepare position pool for SDPO training episodes
- [ ] Integrate Position Evaluations dataset for pre-computed ground truth
- [ ] Run SDPO training
- [ ] Evaluate on full benchmark after SDPO
- [ ] Compare SFT vs SDPO metrics
