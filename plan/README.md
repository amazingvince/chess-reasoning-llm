# Chess LLM SFT Project

## Training Qwen3.5-0.8B to Reason About Chess

**Version**: 1.1
**Date**: March 2026
**Status**: Design Specification
**Target Model**: Qwen/Qwen3.5-0.8B
**RL Framework**: SDPO (Self-Distilled Policy Optimization)

---

## What Is This?

This project trains a small language model (0.8B parameters) to reason about
chess through a multi-stage pipeline:

1. **SFT (Supervised Fine-Tuning)** — build chess literacy from the ground up
2. **SDPO Reinforcement Learning** — refine reasoning quality with Stockfish rewards
3. **Self-Play Refinement** — iterative improvement via play against Stockfish

This repository contains the specification broken into manageable phases.
Each phase is a self-contained chunk of work with clear inputs, outputs,
and success criteria.

---

## Why SFT Matters

Research consistently shows that RL alone cannot overcome a deficit in chess
knowledge from pretraining. Chess-R1 (Hwang et al., 2025) demonstrated that
even with dense reward signals, LLMs plateau far below expert levels because
they lack fundamental chess understanding. SDPO can only refine what already
exists — SFT builds the foundation.

---

## Pipeline Overview

```
Stage 0: Data Preparation          ← 00_data_preparation.md
  └─> Preprocess all data sources into unified format

Stage 1: SFT
  ├─ Phase A: Perception + Rules   ← 01_phase_a_foundation.md
  ├─ Phase B: Tactics + Eval +     ← 02_phase_b_understanding.md
  │           Openings + Endgames
  └─ Phase C: Full curriculum      ← 03_phase_c_planning.md
              with Planning

Evaluation Benchmark               ← 04_evaluation_benchmark.md
  └─> Held-out eval splits, metrics, reporting

Stage 2: SDPO Handoff              ← 05_sdpo_handoff.md
  └─> What SFT provides, reward design, rich feedback
```

---

## Spec Documents

| Document | Contents | Approx. Scope |
|----------|----------|---------------|
| **`00_data_preparation.md`** | All data sources, preprocessing, FEN pool, Stockfish annotation, validation, output format | ~1-2 weeks |
| **`01_phase_a_foundation.md`** | Tiers 1-2: Perception + Rules tasks, prompt templates, training config, pass criteria | ~810K examples |
| **`02_phase_b_understanding.md`** | Tiers 3-6: Tactics, Evaluation, Openings, Endgames tasks, training config, pass criteria | ~485K new + 243K review |
| **`03_phase_c_planning.md`** | Tier 7: Best move selection, puzzle solving, reasoning traces, full curriculum, exit criteria | ~1.6M total |
| **`04_evaluation_benchmark.md`** | Eval split design, 13K held-out examples, all metrics, reporting format | Frozen before any training |
| **`05_sdpo_handoff.md`** | Connection to SDPO, capability mapping, reward function, rich feedback loop | Post-SFT |

**Read order**: Start with this README, then `00_data_preparation.md`, then
phases A → B → C in order. The evaluation benchmark (`04`) should be built
alongside Stage 0, before any training begins. The SDPO handoff (`05`) is
reference material for planning.

---

## Design Principles

These principles apply across ALL phases. Individual phase specs reference
back to these.

### UCI Only

All moves across all tasks use UCI notation: `e2e4`, `g1f3`, `a7a8q`.
No SAN (Standard Algebraic Notation) anywhere in the training data. UCI is
unambiguous, position-independent, and reduces the model's parsing burden.

### Clean Correct Patterns

SFT training data contains only correct outputs. No self-correction, no
"I was wrong, let me reconsider" patterns. The model learns what right
looks like. Error detection and recovery emerge naturally during SDPO,
where the self-distillation mechanism learns from the model's own mistakes.

### Prompt Variety

Every task has 5-10 prompt templates, randomly selected per example during
data generation. This prevents the model from pattern-matching on prompt
wording instead of learning the underlying task.

### Real Game Data

State tracking, move consequences, and opening tasks draw from actual
Lichess games (2000+ Elo). This exposes the model to realistic position
distributions rather than synthetic/random positions.

### Chess960 for Generalization

~15-20% of Tier 1-3 training examples use Chess960 starting positions,
forcing the model to learn piece movement rules rather than memorizing
patterns tied to the standard starting position.

### Evaluation-First

Every task tier has a held-out evaluation split defined BEFORE data
generation begins. We measure progress tier-by-tier, not just on the
final move-selection task.

---

## System Prompt (Shared Across All Phases)

```
<|system|>You are a chess reasoning engine. You understand chess positions
in FEN notation and express all moves in UCI notation (e.g., e2e4, g1f3,
a7a8q for promotion). When analyzing positions, think step by step.<|end|>
```

This system prompt is included in every training example and carried
forward into SDPO.

---

## Tools & Libraries

| Tool | Purpose | Install |
|------|---------|---------|
| **python-chess** | Board manipulation, legal moves, FEN parsing, Polyglot books, Syzygy probing, Chess960, Stockfish interface | `pip install python-chess` |
| **Stockfish 17+** | Position evaluation, best move, PV lines, blunder detection | Binary install |
| **Syzygy Tablebases** | Perfect endgame truth (3-5 piece: ~1GB) | Download from syzygy-tables.info |
| **HuggingFace datasets** | Loading Lichess datasets | `pip install datasets` |

---

## Volume Summary

| Tier | Focus | Examples | % of Total |
|------|-------|----------|-----------|
| T1: Perception | Board, pieces, state | ~440K | 27% |
| T2: Rules | Legal moves, special rules | ~370K | 23% |
| T3: Tactics | Captures, threats, patterns | ~260K | 16% |
| T4: Evaluation | Material, position, pawns | ~150K | 9% |
| T5: Openings | Names, plans, continuations | ~9K | 1% |
| T6: Endgames | Classification, technique | ~140K | 9% |
| T7: Planning | Best move, puzzles, analysis | ~170K | 11% |
| **Total** | | **~1.5M** | 100% |

---

## References

### Research Papers

- **SDPO**: Hübotter et al., "Reinforcement Learning via Self-Distillation" (2026). arXiv:2601.20802. https://github.com/lasgroup/SDPO
- **Chess-R1**: Hwang et al., "Can LLMs Develop Strategic Reasoning? Post-training Insights from Learning Chess" (2025). arXiv:2507.00726. https://github.com/krafton-ai/Chess-R1
- **VAM**: Zhang et al., "Verbalized Action Masking for Controllable Exploration in RL Post-Training" (2026). arXiv:2602.16833.
- **MATE**: Wang et al., "Explore the Reasoning Capability of LLMs in the Chess Testbed" (NAACL 2025). arXiv:2411.06655.
- **ChessLLM**: Zhang et al., "Complete Chess Games Enable LLM Become A Chess Master" (2025). arXiv:2501.17186.
- **Xiangqi-R1**: "Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning" (2025). arXiv:2507.12215.
- **ChessGPT**: Feng et al., "Bridging Policy Learning and Language Modeling" (NeurIPS 2023).
- **ChessQA**: "Evaluating Large Language Models for Chess Understanding" (2025). arXiv:2510.23948.
- **MAV**: "Mastering Board Games by External and Internal Planning with Language Models" (2024). arXiv:2412.12119.
- **Global Chess Challenge 2025**: AIcrowd competition. https://www.aicrowd.com/challenges/global-chess-challenge-2025

### Datasets

- Lichess Standard Chess Games: https://huggingface.co/datasets/Lichess/standard-chess-games
- Lichess Chess Puzzles: https://huggingface.co/datasets/Lichess/chess-puzzles
- Lichess Chess Openings: https://huggingface.co/datasets/Lichess/chess-openings
- Lichess Chess Position Evaluations: https://huggingface.co/datasets/Lichess/chess-position-evaluations
- MATE Dataset: https://huggingface.co/datasets/OutFlankShu/MATE_DATASET

### Model

- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
