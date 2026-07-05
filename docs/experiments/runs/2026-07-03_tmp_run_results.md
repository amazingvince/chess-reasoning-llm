# What the .tmp / WSL Training Runs Actually Show (2026-07-03)

Analysis of every run artifact on disk: the Windows micro-runs under `.tmp/`
(Jun 29, 10–200 optimizer steps, 100–1000 rows/task, transformers eval,
`max_new_tokens=128`) and the WSL real runs under
`\\wsl.localhost\Ubuntu-24.04-CUDA\home\amazi\chess_sft_checkpoints\` (shakedown,
25k one-pass rehearsal, decomp-pack1024, prelaunch packing smokes; vLLM sidecar
eval, `max_new_tokens=192`). Failure examples below are verbatim from
`eval_predictions.jsonl`; scoring claims were checked by re-running the repo's own
`fen_exact_match` over the stored predictions.

## TL;DR

1. **The famous zeros are half measurement artifact, half real.**
   `state_tracking=0.0` and `legal_moves_by_piece=0.0` are substantially caused by
   the eval generation budget being smaller than the gold answer (128/192 tokens vs
   multi-move traces and median-743-char enumerations). `board_to_fen=0.0` and
   `square_lookup≈0.14` in the tiny shakedown were genuine — but they were cured by
   scale, not by any of the mix tweaks that were A/B'd at micro scale.
2. **The curriculum works.** The 25k one-pass rehearsal (708k rows, 22,137 steps,
   228M tokens, 5h) went: state_tracking **0 → 91.4%**, board_to_fen **→ 100%**,
   square_lookup **→ 100%**, perception overall **92.6%**. The checkpoint trend
   shows the phase transition: state 14.3% @1k → 78.6% @5k → 85.7% @10k → plateau.
3. **Micro-runs (100–200 steps) cannot see any of this.** Every micro-run stayed at
   state_tracking 0.0 regardless of mix, and continued fine-tunes produced
   catastrophic interference (check_detection 0.86→0.14→0.98, special_rules
   0.92→0.08 across consecutive runs). Curriculum conclusions drawn from them were
   mostly noise.
4. **The real remaining bottleneck is legal-move enumeration**, which persists at
   scale (legal_moves 36.1%, piece_legal_filter 26% in decomp-pack1024): the model
   slides rooks through blockers and has learned `Rejected: none` as a cheap default.

## Run-by-run trajectory

### Windows micro-runs (.tmp, Jun 29, eval max_new_tokens=128)

| # | Run | Change | What moved |
|---|-----|--------|-----------|
| 1 | smoke_10step, v100_50step | — | everything ~0; board_print first to rise (0.5) |
| 2 | v500_balanced / weighted_150step | task weighting | board_print 1.0; rules 0.28→0.57; board_to_fen & state still 0.0 |
| 3 | sampling_v500_200step | sampling fix | board_to_fen 0→1.0 (small n); piece_id 0→0.30; state 0.0 |
| 4 | state_trace / state_rank_v2_200step | trace-style state targets | piece_id →0.60; **state still 0.0** |
| 5 | lookup_v1000_100step_sp | square/rank lookup tasks added | board_to_fen 0.875; square_lookup 0.52-ish; state 0.0; legal_moves jaccard 0.084 |
| 6 | lookup_v1000_stateup8 | state upsampled 8× at same budget | **regression everywhere**: piece_id 0.20→0.075, material 0.25→0.025, board_to_fen 0.875→0.75; state gained nothing |
| 7 | mech_eval_v1000 | mechanics eval tasks | square_lookup 0.52, rank_lookup 0.28, move_square_edits 0.24; state 0.0 |
| 8 | mechfocus_cont | continued FT on mechanics | rank_lookup 0.28→0.92; state 0.0 |
| 9 | oneply_v1000 | one-ply data | **check_detection collapsed 0.86→0.14**; state 0.0 |
| 10 | oneply_statefocus_cont | continued FT, state focus | move_square_edits 0.84, rank_lookup 0.96, check_detection 0.98; **special_rules collapsed 0.92→0.08**; legal_moves degenerated to the startpos move list 49/50 (with off-board squares `d1i6 d1j7 d1k8`); state 0/25 despite every answer containing a well-formed `Result FEN:` line |

Reading of rows 6, 9, 10: at 100–200 steps with narrow mixes, each continued run
overwrites whichever task families the new mix underrepresents. The 8× upsampling
experiment (row 6) is the cleanest demonstration that **upsampling weak tasks at
fixed compute buys nothing and starves everything else** — the state zeros were
truncation + genuine edit errors, neither of which upsampling addresses.

### WSL real runs

- **[A] phase_a shakedown** (Jun 30, 2k rows, 3 epochs, 222s) — the run quoted in
  the experiment log: board_to_fen 0.0, state_tracking 0.0, square_lookup 0.1428,
  check_detection 0.0 (predicted `Check.` 14/14). Format learned before mapping.
- **[B] phase-a-t12-v25000-20260630 one-pass** (708k rows, 22,137 steps, 228M
  tokens, 5h, loss 0.0493): perception 92.6%, board_to_fen 100%, square_lookup
  100%, state_tracking 91.4%. Weak: legal_moves 26.1%, material_count 30.6%.
  **The knob that fixed the zeros was unique-data volume + steps, not mix tweaks.**
- **[C] phase-a-t12-v25000-20260701-decomp-pack1024** (decomposition tasks, packing
  on @1024, 7,049 packed steps, 225M tokens, 10.6h, loss 0.0879): legal_moves
  26.1→36.1% (decomposition helped); side_piece_inventory 99%, king_safety_filter
  88%; but piece_legal_filter 26% and legal_moves_by_piece 0.0% (output-budget
  artifact — see below). Slight regressions vs [B]: state 91.4→82.1,
  fen_row_application 94.3→75.0. Packing also cut throughput (5,897 vs 12,677
  tok/s).
- **[D] prelaunch packing comparison** (Jul 2, 100 steps): packing off 5,087 tok/s
  vs packing on 1,201 tok/s on the RTX 5090 → launch decision: sdpa, packing off,
  max-length 1024. vLLM wrapper auto-export fixed the sidecar load failure for
  Qwen3.5 text-only checkpoints.

## Failure modes, with receipts

### 1. Eval budget < gold answer (high confidence — this is capping two metrics at 0)

Micro-runs eval at `max_new_tokens=128`; in lookup_v1000_sp, 17/40 state_tracking
items are multi-move and every one is cut mid-trace before the `Result FEN:` line:

> PROMPT: `Starting FEN: rnbq1rk1/... After the moves d1d4 b8a6, what is the resulting position?`
> PRED: `Move 1: white king d1d4.\nLookup: d1=white king; d4=empty.\n...Move 2: black rook b8a6.\n...Ranks: rank 8 b r->1; rank 6 a` *(ends mid-line at ~128 tokens — automatic 0)*

The decomp run evals at 192 tokens while `legal_moves_by_piece` golds have
**median 743 chars** (35/50 golds > 550 chars); 41/50 predictions are cut before
the scored content. The scorer is *not* the problem — `fen_exact_match` re-run over
stored predictions correctly extracts and canonicalizes `Result FEN:` lines.
Also, training logs contain **no truncation/skipped-row warnings** — the truncation
is at eval generation, not in data packing.

### 2. Right format, wrong edits (real capability gap at low steps)

0/40 and 0/25 state_tracking answers correct in the two audited micro-runs; 11/25
`Result FEN:` strings structurally invalid (rank-width errors). Signature example
(shakedown):

> GOLD: `Result FEN: r1bqkbnr/pppp1ppp/2n5/4p3/8/1P3N2/PBPPPPPP/RN1QKB1R b KQkq - 3 3`
> PRED: `Result FEN: r1bqkbnr/pppp1ppp/2n5/4p3/8/1P6/PBPPPPPP/RN1QKBNR b KQkq - 3 3`
> *(input FEN copied verbatim, only the counters flipped — the move was never applied)*

### 3. Majority-class collapse on starved tasks

Undertrained tasks emit the modal answer: `h4=empty` for square lookups (8/12
wrong answers say "empty"); `Check.` 14/14 in the shakedown; the starting-position
move list for legal_moves 49/50 regardless of input FEN; `Castling available:
kingside.` 50/50 in oneply_statefocus; the 16-piece starting-material inventory for
material_count. square_lookup≈0.14 in the shakedown is roughly the base rate of the
modal answer being right.

### 4. Pseudo-legal over-generation + "Rejected: none" collapse (persists at scale)

decomp-pack1024, rules_00019 — rook slides straight through its own rook on d2:

> GOLD: `Pseudo-legal from d1: d1a1 d1b1 d1c1 d1e1 d1f1.` …
> PRED: `Pseudo-legal from d1: d1a1 d1b1 d1c1 d1d2 d1d3 d1d4 d1d5 d1d6 d1d7 d1d8 d1e1 d1f1 d1g1.` … `Rejected: none.`

rules_00030 — a king given rook-like moves, check-filtering skipped:

> GOLD: `Pseudo-legal from a1: a1b1.\nLegal: none.\nRejected: a1b1.`
> PRED: `Pseudo-legal from a1: a1b1 a1c1 a1d1 a1e1.\nLegal: a1b1 a1c1 a1d1 a1e1.\nRejected: none.`

The training data rarely makes `Rejected: none` wrong (few sampled positions have
rejections), so the model learned it as a default. This is consistent with the data
audit: the 2.3 legality task's negatives were 0% king-safety in the oneply set and
7.6% in the current set.

### 5. Minor: Qwen thinking-template bleed

100% of decomp-run raw predictions begin `<think>\n\n</think>\n\n` (empty block,
~5 tokens). Cosmetic; `normalize_prediction()` strips it before scoring.

## What to do (ordered)

1. **Raise eval `max_new_tokens` to ≥512 and re-score the existing decomp-pack1024
   checkpoint** — the true legal_moves_by_piece number is free.
2. **Partial-credit scoring for long enumerative tasks** (per-piece-line set-F1 /
   jaccard), keeping exact match secondary. A structurally-perfect answer with one
   extra move currently scores the same as garbage.
3. **No capability decisions from <1k-step runs.** Use ≥1k-step probes with the
   sidecar checkpoint-trend; the phase transition sits ~5k steps.
4. **Don't upsample weak tasks at fixed compute** — scale unique rows (stateup8 is
   the counterexample on record).
5. **Attack the legal-move bottleneck with contrast-heavy data**: oversample pins,
   blocked rays, in-check positions so `Rejected: none` is frequently wrong; add a
   ray-walk/blocker decomposition task; gate legal_moves_by_piece reintroduction on
   piece_legal_filter ≳70%.
6. **Material count**: decomposed subtasks score 78–82% but composed material_count
   is 25% — add a composed trace task chaining inventory → counts → values →
   verdict.
7. **Launch config**: keep sdpa + packing off + max-length 1024 (packing was 4.2×
   slower on the 5090); trainer eval off with vLLM sidecar on GPU 2; re-baseline
   against the 25k one-pass final eval, not the decomp run (the decomp run traded
   state/FEN points for rules gains).

## Caveat on the 90%+ numbers

The code review (see `docs/reviews/2026-07-03_deep_code_review.md`) confirmed the
eval split is drawn from the same per-ply position chains as training with only
exact-position blocklisting, so eval positions' 1-ply neighbors are almost always
in train. The 25k-run headline numbers are therefore optimistic for true
generalization. The relative trends (phase transition, task orderings, bottlenecks)
survive this caveat; absolute numbers should be re-read after game-scoped
splitting lands.

## Operational warning

Everything that fed real runs — datasets, checkpoints, launcher scripts — lives
only in gitignored `.tmp/` and the WSL filesystem
(`/home/amazi/chess_sft_data/…`, `/home/amazi/chess_sft_checkpoints/…`);
`chess_sft_data/` in the repo is empty. One cleanup pass erases the experimental
record. Back up or commit at least the launcher scripts and
`eval_predictions.results.json` files.
