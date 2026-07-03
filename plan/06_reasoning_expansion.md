# 06 — Reasoning Expansion: Seeding Chess Reasoning in a 0.8B Model

Status: design proposal (2026-07-03). Companion to `03_phase_c_planning.md`
(what tier 7 becomes), `05_sdpo_handoff.md` and `sdpo.md` (what consumes the
trace channel), and `docs/experiments/2026-07-03_tmp_run_results.md` (the
empirical constraints). This document is the data-side answer to "where do
reasoning traces come from and how do we know they're working."

---

## 1. What we're trying to buy

"Reasoning" here means: given an arbitrary position, the model produces a
short, verifiable derivation — candidates considered, lines calculated,
rejections with reasons — whose conclusion is the move it plays. We want this
for three reasons, in priority order:

1. **Process-supervised SDPO needs a trace channel.** The sdpo.md program
   (teacher = same model under richer conditioning) is dramatically stronger
   when the judge can score *steps*, not just final moves — chess step
   verification costs microseconds via python-chess/Stockfish.
2. **Sample efficiency out of distribution.** A 0.8B model that can execute a
   calculation procedure should degrade more gracefully on weird positions
   than one that memorized a policy surface.
3. **Explainability** — a stated project goal.

The honest counterweight: searchless-chess distillation (270M, action-value
targets, no traces) reached grandmaster blitz. Traces are NOT the efficient
path to raw strength; they're the path to (1)–(3). The plan below hedges
accordingly (see the value-anchoring task, R4).

## 2. What the literature says (and how it changes the design)

Six findings, in decreasing order of how much they should shape us:

**(a) Multi-move trajectories beat single best-move data — and beat verbose
search verbalization too.** The most on-point study (fine-tuning Qwen2.5-7B on
chess, SFT→RL; arXiv:2604.05134) compared Best Move, Best Line (4–6-ply PV +
final centipawn eval), factual QA, verbalized alpha-beta, teacher-explained
moves, and rejection-sampled traces. Best Move won raw benchmark rank but
became an **unfaithful reasoner** under RL (answers disconnected from traces)
with unstable RL dynamics. **Best Line kept faithful reasoning, trained
stably, and tripled OOD generalization** (14.8% vs 5.3% on unseen checkmate
problems). Verbalized tree search did *not* win. Their interpretation: n-step
trajectories (2–3 steps emphasized) make the model internalize transition +
value functions — a world model — while policy-only data leaves reasoning as
decoration. **Implication for us: the workhorse reasoning dataset should be
dense PV-trajectory data, not long verbalized search.**

**(b) Dense beats verbose; triviality is poison.** Same study: datasets
lacking prose outperformed wordy ones; post-SFT "predictive complexity" of the
data correlated with downstream performance (their factual-QA set became 62%
trivial; Best Line stayed 24% complex). Blending dataset types beat any single
type. Our fixed-grammar decomposition style (labeled lines, no filler) is the
right register — extend it upward rather than switching to prose.

**(c) Hallucination rate in traces is the leading indicator.** Their strongest
SFT-time predictor of final RL performance was **referenced-move accuracy** —
whether moves mentioned inside the trace are legal/real in context. We can
compute this today with `core/legality` + `formats/answers`. Measure it on
every checkpoint before committing to RL scaling.

**(d) Search traces work but carry a verbosity trap.** Stream of Search /
Searchformer / GSoS show models trained on serialized search (exploration,
pruning, backtracking as text) do learn to search and even discover better
strategies — but SFT on search traces "can induce suboptimal or verbose
traversal behaviors, limiting performance unless refined via RL"
(arXiv:2404.03683 and successors, e.g. self-backtracking, ASTRO). So: include
explicit-search data as a **bounded, minority slice**, and plan for the RL
stage to compress it.

**(e) LLM planning is measurably myopic — train the specific deficits.**
Search trees extracted from frontier-model chess traces (arXiv:2605.06840)
show shallow depth, narrow breadth, and almost no backtracking. If we want
those behaviors, the data must contain them explicitly and the eval must
measure them (depth/breadth/backtrack counts are parseable from our
fixed-grammar traces).

**(f) Engine-grounded, engine-filtered traces are the current recipe at small
scale.** C1-4B ("Master Distillation", arXiv:2603.20510) pairs moves with
intermediate board states, uses Stockfish to generate/verify, and keeps only
traces aligning with engine assessments. Step-level (process) supervision
beats outcome-only across the 2025–26 verification literature, with automated
process labels replacing human ones (V-STaR derives preference pairs from
correct AND incorrect traces; reward-hacking of learned verifiers is the
noted failure mode — our verifier is programmatic, which sidesteps most of it).

## 3. Design principles (0.8B edition)

1. **Grounded and re-derivable**: every trace line machine-checkable by the
   validation.py pattern (generator, validator, benchmark share one formatter).
2. **Dense, fixed grammar**: labeled lines, no narrative filler. Trace token
   budget scales with position sharpness, hard cap ~350 tokens (eval budget
   512; train max_length 1024).
3. **Trajectory-first, search-second**: PV-trajectory data is the volume
   workhorse (a); explicit candidate-scan traces are a smaller slice (d).
4. **Faithfulness is a metric from day one** (c): referenced-move accuracy +
   trace-causality tracked per checkpoint, before any RL.
5. **Verification before generation**: corruption-detection tasks teach the
   judge-skill SDPO needs, cheaply.
6. **Blend** (b): trajectories + scans + verification + factual anchors in one
   mix, not sequential silos.

## 4. The data portfolio (proposed tier-7 revamp + additions)

All new benchmark task types start DIAGNOSTIC. Volumes sized against the
~2.3M-row full curriculum; traces are token-expensive, so counts are modest.

### R1 — `7.5_best_line` (workhorse, ~80k)
Best-Line-style trajectories: position → engine PV (2–6 plies, weighted toward
2–3 per finding (a)) → final assessment. Dense format:

```
Line: d4e5 f6e5 d1d8 e8d8
State after: <FEN>
Eval: +180 (White has a clear advantage)
Move: d4e5
```

`State after` reuses the exact state-tracking skill (1.5/1.19) — this is where
the perception curriculum cashes in. Eval text reuses the 4.2 cp-bucket
contract. Sources: `lichess_evals` PVs (already legality-validated) + a new
multipv Stockfish annotation pass for depth. Scoring: move exact-match primary;
diagnostics for PV-legality, state-FEN match, eval-bucket match.

### R2 — `7.6_candidate_scan` (bounded search, ~40k)
The explicit-search slice, with backtracking made textual (e, d):

```
Candidates: checks: c7c8q; captures: d4e5; threats: f3g5
Try c7c8q: c8q d8c8 -> eval +520. Best so far: c7c8q (+520).
Try d4e5: e5 f6e5 -> hangs the queen. Reject (loses_material). Back to c7c8q.
Try f3g5: g5 h7g5 -> eval +40. Worse than best. Back to c7c8q.
Move: c7c8q
```

Grammar rules: candidate scan is always checks→captures→threats (a calculation
discipline, and a parseable breadth signal); 3–5 candidates; lines 1–3 plies;
every `Try` line ends in an eval or a taxonomized rejection
(`loses_material`, `allows_mate_in_N`, `hangs_piece`, `worse_than_best`);
explicit `Back to <move>` markers (the backtracking token (e)). Generated from
multipv + shallow re-search; every claim engine-derivable. Sampling ladder:
mate-in-1 → mate-in-2 → tactical puzzles by Lichess rating band → quiet
positions last (unique-answer positions first, ambiguous ones after the scan
discipline is learned).

### R3 — `7.7_trace_verification` (~30k)
Verification before generation. Take a correct R1/R2 trace, mechanically
corrupt exactly one line (illegal move in a PV, flipped eval sign, false
rejection reason, wrong resulting FEN — our judge/legality machinery generates
all of these), ask for the broken line:

```
Faulty step: line 3 — d4e5 does not hang the queen; after f6e5, e5 is
defended by the d-pawn. Correct assessment: eval +150.
```

Or "Verdict: sound." for uncorrupted traces (balanced classes). This is the
cheapest reasoning data we can make, it directly trains the skill SDPO's judge
asks of the model, and V-STaR-style preference pairs fall out of it for free.

### R4 — `7.8_candidate_ratings` (value anchor, ~40k)
The searchless-chess hedge: "Rate these 5 candidate moves (centipawns,
side-to-move POV)" with multipv gold. Teaches a calibrated evaluation head the
scans can lean on, at a fraction of a trace's token cost. Scored with bucketed
partial credit (reuse the cp-bucket + set-overlap machinery).

### R5 — retrofit `7.1`/`7.2` traces
Replace the remaining templated narrations with R2-format scans (short form:
1–2 candidates for puzzles, since the solution is forced). The gives-check
guard fix already made them truthful; this makes them procedural.

### What deliberately stays out
Long free-prose "grandmaster commentary" style data (b: verbose loses), and
full verbalized alpha-beta over deep trees (a: didn't win even at 7B; worse at
0.8B).

## 5. The bootstrap loop (after the SFT foundation)

Stage numbers continue the phase plan; all infra named below exists as of
2026-07-03.

1. **Harvest**: `chess_llm.autodata.selfplay` produces on-policy positions +
   judged move artifacts.
2. **Sample**: k traces per position at temperature from the current
   checkpoint (vLLM batch path).
3. **Filter (the critical design point)**: keep a trace only if
   (i) final move regret < threshold (stockfish_judge),
   (ii) every referenced move is legal in its context (referenced-move
   accuracy — finding (c)),
   (iii) stated evals/rejections verify within tolerance (step verification),
   (iv) position survives the eval-split blocklist (already enforced in
   sft_refresh).
   Answer-only filtering (i alone) is explicitly not enough — it breeds
   right-answer-wrong-reasoning traces that poison the channel.
4. **Refresh**: survivors → SFT refresh mix (existing pipeline).
5. **Prefer**: V-STaR-style pairs for SDPO — chosen = fully-verified trace;
   rejected = same-position trace failing (i) or (iii). The asymmetric-teacher
   variant from sdpo.md drops in naturally: the *teacher* is the same model
   conditioned on privileged engine analysis (multipv table in-context)
   writing the trace; the *student* gets the bare FEN. That is the
   "guided synthetic" arm of finding (a)'s study, upgraded to on-policy.
6. **Iterate** 2–3 rounds; the literature says gains flatten quickly and the
   RL stage (SDPO/GRPO-style) should take over compression of verbose scans (d).

## 6. Metrics and diagnostics (build these first)

- **Referenced-move accuracy** (per checkpoint, per task): fraction of moves
  mentioned anywhere in a trace that are legal in their stated context.
  Leading indicator for RL success (c). Implement in prediction_analysis.
- **Trace-causality**: accuracy conditioned on a verified-correct trace prefix
  vs overall accuracy. If conditioning doesn't help, traces are decorative —
  stop paying for them.
- **Myopia panel** (e): mean candidates scanned, mean line depth, backtrack
  count — parsed from the fixed grammar, logged per eval run.
- **Step accuracy**: fraction of trace lines that verify (the benchmark can
  score this as a diagnostic secondary, same pattern as set_f1).
- **Faithfulness canary** (a): during any future RL, monitor divergence
  between trace conclusion and emitted move; rising divergence = the Best-Move
  failure mode reappearing — rebalance toward trajectory data.

## 7. Infrastructure mapping

Exists (reuse): decomposition formatter/validation pattern; cp-bucket and
state-FEN contracts; stockfish judge + regret; selfplay harvest + artifact
triple; sft_refresh with blocklist; benchmark diagnostic wiring; V-STaR-ready
preference schema in sdpo plan.

To build:
1. **Multipv annotation source** (`sft/sources/` or extend `annotation.py`):
   position → top-k moves, PVs, evals at fixed depth; cached like lichess
   evals. The single biggest new component; everything in §4 consumes it.
2. **Trace synthesizer** (`core/traces.py`): R1/R2 formatters, single-sourced.
3. **Corruption engine** for R3 (mutate one line, record which).
4. **STaR driver** (`autodata/star_refresh.py`): steps 2–3–4 of §5 as a CLI,
   composing existing pieces.
5. **Metrics**: referenced-move accuracy + myopia panel in
   prediction_analysis; step-accuracy secondary in benchmark.

## 8. Sequencing and experiments

Phase R-0 (with the next data regeneration): R4 + R5 (cheap, low-risk), plus
the metrics (§6). Phase R-1: R1 + R3 at full volume, R2 at half volume; train
via the schedule mode with tier-7 late-ramp; gate nothing on the new tasks.
Phase R-2: the §5 loop once a Phase-A/B checkpoint exists.

Ablation matrix worth running at 25k-row scale (the proven probe size):
R1-only vs R1+R2 vs R1+R2+R3, measuring §6 metrics + planning split — the
study in (a) predicts R1-heavy blends win on faithfulness at equal tokens;
verify it holds at 0.8B before scaling. Also a trace-length ablation on R2
(3 vs 5 candidates) — capacity may bind earlier than at 7B.

## 9. Risks

- **Verbosity trap** (d): R2 volume capped; SDPO stage owns compression.
- **Capacity**: 0.8B may not sustain both the scan discipline and the value
  head; R4 is the fallback that preserves strength if traces stall.
- **Filter hacking**: our verifiers are programmatic (not learned), which
  removes classic PRM reward hacking, but threshold gaming is still possible
  (e.g., model learns to mention only its final move → referenced-move
  accuracy trivially 1.0). Pair each filter with a coverage floor (scan must
  name ≥3 candidates on sharp positions).
- **Token cost**: R1–R5 as specified ≈ +190k rows but ~2–3× average answer
  length of tier 1–2 rows; budget ~15–20% of total training tokens for the
  reasoning slice, consistent with the blend finding (b).

## Sources

- How Reasoning Evolves from Post-Training Data: An Empirical Study Using
  Chess — https://arxiv.org/html/2604.05134v2
- Grounded Chess Reasoning in Language Models via Master Distillation (C1-4B)
  — https://arxiv.org/pdf/2603.20510
- Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning —
  https://arxiv.org/pdf/2605.06840
- Stream of Search: Learning to Search in Language —
  https://arxiv.org/pdf/2404.03683
- Self-Backtracking for Boosting Reasoning — https://arxiv.org/html/2502.04404v1
- ASTRO: Reflecting and Backtracking In-Context — https://arxiv.org/html/2507.00417v1
- To Backtrack or Not to Backtrack (COLM 2025) — https://arxiv.org/pdf/2504.07052
- Trust but Verify: Verification Design for Test-time Scaling (survey) —
  https://arxiv.org/html/2508.16665v3
- Taming Imperfect Process Verifiers — https://arxiv.org/pdf/2510.03149
- Verified Critical Step Optimization — https://arxiv.org/html/2602.03412
- ChessArena — https://arxiv.org/html/2509.24239v4
- LLM Chess benchmark — https://arxiv.org/html/2512.01992v1
