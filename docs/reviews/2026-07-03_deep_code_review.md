# Deep Code Review — Generation + SFT Training + Run Outputs (2026-07-03)

> **Status update (2026-07-03, later the same day):** the code defects below have
> been FIXED in the working tree (45 fixes across 40 source files, +new
> regression tests; full root suite green at 712 passed). File/line references
> in this document describe the PRE-fix code. Not yet addressed: the
> data-design items that need regeneration or new task design (2.2 empty-case
> coverage, composed material-count trace task, a dedicated A-vs-B task for
> MATE pairwise rows, mixer full-materialization performance), and everything
> in the run-outputs writeup that is about training procedure rather than code.

Multi-agent review of `src/chess_llm/` (generation pipeline, tier generators, sources,
formats, training loop, mixer, eval/gating, eval harness, autodata) plus an empirical
audit of real generated JSONL data and every training run's eval artifacts.

**Method.** 12 slice reviewers + 1 training-outputs analyst ran in parallel; every code
finding was then adversarially verified (3 independent lenses for high severity). 42
findings were machine-verified by verifier agents; the remaining 39 were verified
directly by runtime reproduction against the actual code and data (all reproduced; one
with a nuance, noted inline). 0 of 81 findings were refuted.

---

## 1. What the training outputs actually show (read this first)

The headline zeros in the experiment log (**board_to_fen 0.0, state_tracking 0.0,
square_lookup 0.14**) decompose into three separate causes, and only one of them is a
model problem:

1. **Eval generation budget is smaller than the gold answer.** Micro-runs evaluate with
   `max_new_tokens=128`, the vLLM sidecar with `192`. Multi-move state_tracking items
   truncate mid-trace before the `Result FEN:` line (17/40 items in the lookup_sp run);
   `legal_moves_by_piece` golds have **median 743 chars** vs a 192-token budget, so
   41/50 predictions are cut before the scored content. These tasks are **hard-capped at
   0 today regardless of model quality.** The scorer itself is fine — `fen_exact_match`
   was re-run over stored predictions and correctly canonicalizes both sides.
2. **Genuine near-miss mechanics failure at low step counts.** Where answers do fit the
   budget, failures are real: input FEN copied verbatim with only counters flipped
   (move never applied), rank-width-invalid FENs (11/25 in oneply_statefocus), broken
   empty-count arithmetic. Format is learned orders of magnitude faster than
   square→rank-index mapping.
3. **Majority-class collapse on starved tasks.** Undertrained tasks emit the modal
   answer: `h4=empty`, `Check.` 14/14, the starting-position move list 49/50 (with
   off-board squares like `d1i6`), `Castling available: kingside.` 50/50.

**The decisive positive result:** the 25k one-pass rehearsal (22,137 steps, 228M tokens)
took state_tracking 0→**91.4%**, board_to_fen→**100%**, square_lookup→**100%**
(checkpoint trend 14.3%@1k → 78.6%@5k → plateau@10k). The mechanics curriculum works;
the phase transition sits around ~5k steps. Every 100–200-step micro-run is below it,
so **micro-run A/B comparisons of curriculum ideas were mostly measuring noise and
catastrophic interference** (special_rules 0.92→0.08, check_detection 0.86→0.14 swings
between continued fine-tunes; 8× state upsampling at fixed compute regressed everything
else and bought nothing).

**Remaining true bottleneck at scale:** legal-move enumeration / pseudo-legal filtering
(legal_moves 36.1%, piece_legal_filter 26% in decomp-pack1024). The model slides rooks
through blockers and has learned `Rejected: none` as a cheap default because sampled
positions rarely reject anything.

### Immediate actions, in order

1. Raise eval `--max-new-tokens` to ≥512 (or per-task gold-length + margin) and
   **re-score the existing decomp-pack1024 checkpoint** before drawing any more
   conclusions from it.
2. Add partial-credit scoring (per-line set-F1/jaccard) for long enumerative tasks;
   keep exact match as secondary.
3. Stop making curriculum decisions from <1k-step runs.
4. For legal moves: generate contrast-heavy `piece_legal_filter` data oversampling pins,
   blocked rays, and in-check positions, with per-rejection reasons — and fix the
   legality-reason label bugs below first, since they poison exactly this data.

---

## 2. Gate-breaking and leakage bugs (fix before the full Phase A run)

These directly corrupt the phase gate / benchmark signal you steer by.

- **`phase_gate.py:160` (floor) and `:193` (regression) treat count-valued metrics as
  accuracies.** A perfect model with `legal_moves_by_piece_illegal_extra_count = 0.0`…
  fine, but any count metric > 0.35 fails the T1-2 floor, and a model that *reduces*
  its illegal-move count by >0.05 fails the regression check because
  `drop = baseline - current` is positive for improvements on lower-is-better metrics.
  Worse, `train.py:1124` merges historical baselines with per-metric **max** ("best
  ever"), which for count metrics stores the *worst* ever. Exclude count metrics (or
  add direction metadata) in both checks and in the baseline merge.
- **`sft_refresh.py:272`: the autodata refresh loop re-injects frozen gate-benchmark
  positions and their gold answers into training data with no blocklist check.** Direct
  benchmark contamination the moment you turn the eval-failure→refresh loop on.
- **Eval-split near-duplication (two confirmed findings + one structural).**
  `lichess_games.py:190` yields every consecutive ply of every game into one FEN pool
  (500k cap ≈ only ~7,700 games); `eval_split.py:57` samples individual positions and
  blocklists only the exact FEN. So virtually every eval position's 1-ply neighbors are
  in train. Benchmark scores are inflated by near-duplication (the 25k-run 90%+ numbers
  overstate generalization). Block by *game id*, not position.
- **`mixer.py:118`: eval FEN keys are re-drawn per phase**, so Phase B/C `eval_loss`
  (which drives `load_best_model_at_end`) is computed largely on FENs trained during
  Phase A. Persist the eval key set across phases.
- **`mixer.py:55`: smallest-FEN-group-first eval selection** can leave entire tasks
  (board_to_fen included) with zero trainer-eval representation, since all tier-1
  generators share one fen_pool prefix.
- **`pipeline.py:418`: per-tier eval-split rebuild truncates the blocklist and deletes
  other tiers' frozen benchmark splits** in the documented per-tier A→B→C workflow.
- **`run_eval_split.py:67`**: `freeze_existing_eval_splits` freezes any stale split file
  on disk, including ones the blocklist doesn't cover.
- **`decontamination.py:49`**: only the top-level start FEN is checked; answer/metadata
  FENs escape decontamination.
- **Benchmark blind spot (measured):** the gate's check_detection split is 195 none /
  5 check / **0 checkmate / 0 stalemate** (91–97% majority class) because benchmark FENs
  come from openings/evals pools where terminal positions are absent — while training
  2.4 is a 4-way task with ~19% mate/stalemate. The gate literally cannot see the
  classes the task exists to teach.

## 3. Training-loop findings

- **`training_args.py:171`: `warmup_steps` is computed from the raw example count, not
  the packed step count** — packed runs can spend a large fraction (potentially all) of
  training in warmup.
- **`train.py:566`: full fine-tune runs in pure bf16** (`torch_dtype="auto"` +
  `bf16=True`, no fp32 master weights). At lr ≤ 2e-5, per-step Adam updates fall below
  bf16 resolution; small-update tasks learn slower than they should. Load fp32 and let
  autocast handle compute, or use an optimizer with fp32 state.
- **`train.py:780`: any `--max-steps` value forces eval/save every ≤50 optimizer
  steps** — smoke-run cadence leaking into long bounded runs (hours of checkpoint churn).
- `train.py:628`: `_register_trl_flash_attention_variant` registers *any* selected
  backend (sdpa/eager) as a flash variant, silencing TRL's cross-example packing
  contamination guard.
- `train.py:478`: `--eval-only` on an untrained phase stamps base-model metrics as that
  phase's historical baseline.
- **`evaluate.py:558`: `enable_thinking=False` is keyed on `"qwen" in
  tokenizer.name_or_path`** — i.e., on whether your *checkpoint directory name* happens
  to contain "qwen". Some run dirs do (`.tmp/qwen35_*`), the documented WSL default
  (`chess_sft_checkpoints/`) doesn't, so baseline vs checkpoint evals silently differ.
  Key on the chat template / model config instead. *(Nuance: your recent runs' paths
  contained "qwen", so the mitigation did apply there by luck.)*
- `evaluate.py:624`: `--pass-k > 1` with `--temperature 0.0` generates k−1 identical
  greedy samples (pass@k ≡ pass@1 at k× cost, no warning).
- `evaluate.py:1692`: planning `legal_move_rate` drops tag-less predictions from the
  denominator, inflating the Phase C gate metric.
- `mixer.py:116`: every phase build materializes all rows of all tiers into memory with
  ~650µs of `chess.Board` parsing per row, on every launch/resume. `mixer.py:134`:
  `int()` truncation can select 0 rows from a small tier. `mixer.py:127`: review tiers
  keep 100% of eval rows, overweighting tiers 1–2 in later-phase trainer eval.
  `loader.py:24`: sanitized-JSONL cache key omits a sanitizer version.

## 4. Label-correctness bugs in generated data

Verified by running the actual generators/classifiers. These bake wrong or
contradictory chess supervision into training answers **and** benchmark gold.

- **`core/legality.py:86`: promotion check runs before pseudo-legality**, so impossible
  pawn moves (push onto occupied promotion square; diagonal "capture" to an empty
  square) are labeled `missing_or_invalid_promotion` — and
  `_missing_promotion_candidates` systematically generates exactly these for 2.3.
- **`core/legality.py:97`: castling through check is labeled
  `illegal_piece_movement_or_blocked_path`** (path is clear; the real reason is king
  safety).
- **2.4 check_detection answers a literal yes/no question with a 4-way state label**
  (measured: 382+372 rows in the 25k set answer "Is the king in check?" with
  `Stalemate.` — where the literal answer is *no*). Non-responsive supervision on
  question semantics.
- **`tier1_perception.py:1231` (confirmed earlier): 1.13 fen_rank_cell_edit teaches
  post-move rank rows that contradict 1.8/1.10** and are physically impossible for
  castling (~22% of moves).
- **En-passant FEN convention split:** 1.2 serializes `en_passant="fen"` while 1.5/1.9/
  1.10 and all benchmark golds use the default `"legal"` (`tier1_perception.py:575`).
  Same board state → two different ep fields across tasks.
- **`reasoning_traces.py:48`: mate-themed traces unconditionally claim "The {piece}
  delivers check from {sq}"** — false for quiet first moves of mates (repro: quiet KRK
  mate-in-2 key move, `gives_check == False`). And **`tier7_planning.py:79` tags every
  mate-scored eval row `["mate"]` regardless of sign**, so the *defending* side's move
  is narrated as a mating pattern.
- **`tier7_planning.py:114`: MATE `better_move` (better of exactly two candidates) is
  supervised as the answer to "What is the best move?"** — frequently not the best
  move, and can contradict a depth-30 eval example of the same position in the same task.
- **`tier4_evaluation.py:30` + `benchmark.py:1157`: `_cp_to_bucket` produces
  ungrammatical, side-attributed labels** ("White has a equal", "White has a winning");
  cp in ±1..49 gets a side-attributed "equal" chosen by engine-noise sign while cp=0
  gets bare "equal"; and the eval_bucket scorer ignores the side entirely, so a
  wrong-side answer scores 1.0.
- **`tier6_endgames.py:40`: KQKP principle is chess-false** ("bishop/center pawn may
  draw" — center pawns lose; the drawing pawns are bishop *and rook* pawns; Syzygy
  probe confirms).
- **`tier3_tactics.py:236`: all raw Lichess puzzle themes — including bookkeeping tags
  (`short`, `long`, `crushing`, `masterVsMaster`) — are rendered as "The tactic is …"**;
  `pipeline.py:171` applies no theme filter.
- `chess960.py:14`: ID 518 (= standard start) is sampled as Chess960 with no variant
  marker in any prompt (`templates.py` contains no "960"), producing contradictory
  castling-UCI supervision (`e1h1` vs `e1g1`) for identical prompts.
- **2.3 negatives are trivially illegal** (measured, oneply set: 53% from-empty-square,
  24% opponent's piece, 23% wrong pattern, **0% king-safety**; current set: 7.6%
  king-safety). The binary legality task never teaches the hard class — consistent with
  the persistent legal-move bottleneck at scale.
- 2.2 piece_specific_moves: 1000/1000 rows query an own piece that has moves; the
  empty-square/opponent/no-moves cases are never trained (measured).

## 5. Eval-harness / diagnosis bugs

Your failure-bucket diagnostics have bugs that would misdirect the next data iteration:

- **`prediction_analysis.py:340`: any output containing `Result FEN:` is bucketed
  `square_edit_trace`** — the canonical state_tracking answer format. The `result_fen`
  family is dead code and correct-format answers get flagged as format bleed (repro'd).
- **`run_benchmark.py:42`: CLI format_compliance is computed on think-stripped text**
  (`raw_prediction` is never read; `validate_think_move_format` requires the `<think>`
  block that normalization removed) → systematically 0.
- **`benchmark.py:686`: `'no check'` substring-matches inside `'no checkmate'`**, so
  "no checkmate here, but the king is in check" scores as `normal` (repro'd).
- **`benchmark.py:1062` + `run_benchmark.py:260`: splits with no predictions are scored
  0% with full ACPL penalties** instead of being reported uncovered.
- `formats/answers.py:112` + `benchmark.py:58`: the UCI regex matches FEN rows like
  `b2b4`/`b4b2`, so FEN-echoing outputs get phantom "moves" extracted (corrupts move
  scoring, ACPL, jaccard, and format-family histograms; repro'd).
- `benchmark.py:2381`: terminal-position legal_moves gold uses a sentinel ("No legal
  moves available.") never seen in training; the trained `none` convention scores 0
  (and 2.1 training skips terminal positions entirely, `tier2_rules.py:356`).
- `benchmark.py:1430`: terminal positions freeze a degenerate "After the moves , …"
  state_tracking prompt; a non-canonical raw FEN there would abort the whole freeze.
- `batch_judge.py:96`: `--split` filtering hard-fails on the first out-of-split row.
- Also confirmed earlier: `evaluate.py` think-stripping etc. — see §3; and
  `stockfish_judge.py:89` (self-correction rows), `judge.py:51` (bad FENs bucketed as
  model illegal-move failures).

## 6. Pipeline / sources — determinism, sampling bias, robustness

- **`tier2_rules.py:992` (confirmed high): `_select_bucketed_rows` fills quotas by
  cycling the same rows** → exact duplicates that the pipeline's identity scan then
  counts as errors and *regenerates on every resume*. Same effect from
  **tier-5 identity collision** (`identity.py:13`: no 5.x entries in
  `TASK_IDENTITY_FIELDS`, so 4–5 variants per opening share one identity —
  repro: 5 variants → 1 identity; extend path would drop 75–80% of variants).
- `identity.py:15`: 1.4 identity hashes unused random color/piece fields → byte-identical
  rows pass dedup (repro'd). Measured: 13.2% exact-dup rows in 1.13, 7.1% in 2.5 (25k
  set).
- **`lichess_evals.py:169`: stream_evals is all-or-nothing** — nothing yields and the
  dedup DB isn't flushed until the full scan completes; a mid-stream network error
  silently discards the entire scan (pipeline catches and continues with stale cache).
- **Sampling bias trio:** `lichess_games.py:137` strict first-N games (pool ≈ earliest
  ~7.7k games of the 2013-01 shard); `lichess_evals.py:182` `ORDER BY depth DESC` +
  first-N consumption (training sees only the most-analyzed/theory positions);
  `tier1_perception.py:533` (confirmed) ~12 tier-1 tasks all consume the same unshuffled
  pool prefix.
- `lichess_games.py:79`: any PGN error discards the entire game; the `_parse_token_moves`
  fallback is unreachable (repro: recoverable game → 0 positions).
- `fen_pool.py:44`: `__contains__`/`get_tags`/`remove` compute the key without stored
  tags — chess960-tagged entries whose FEN is standard-valid are stored under `960:` but
  looked up under `std:` (repro: contains=False, get_tags={}, remove no-op).
- `pipeline.py:253`: polyglot weights from different `.bin` books summed on incompatible
  per-book scales, distorting "top book move" ordering.
- Confirmed earlier (§gen-pipeline): extension/resume paths reuse one RNG per tier
  (`pipeline.py:701`, `:787`) so data depends on which tasks were skipped; `--volume`
  changes the source pool so `--eval-split-volume` extension always invalidates the
  manifest (`:134`); eval-split manifest re-fingerprints every row (~8 min) on every
  invocation (`:499`); plus `settings.py:164` CWD-relative output root,
  `annotation.py:147` depth-ignoring cache, `hub_upload.py:168` silent empty benchmark
  upload.
- **Operational:** `chess_sft_data/{output,eval_splits,benchmark,pool}` are all empty —
  every dataset that fed real runs lives in gitignored `.tmp/` or the WSL filesystem,
  including the run launcher scripts. One `rm -rf .tmp` loses the actual experimental
  record.

## 7. Autodata loop (confirmed by verifiers)

- `sft_refresh.py:272` benchmark re-injection (see §2 — the big one).
- `sft_refresh.py:384`: repair rows reference a "previous answer" absent from the prompt.
- `sft_refresh.py:320`: run-level chess960 flag ignored at refresh time.
- `stockfish_judge.py:89`: depth-mismatched regret can "correct" a move to itself.
- `judge.py:51`: unparseable prompt FENs mislabeled as model ILLEGAL_MOVE failures.

---

## Verification ledger

- 12 reviewer agents + outputs analyst; 81 findings total.
- 42 findings: adversarially verified by 1–3 independent verifier agents (code-trace /
  reproduce / impact lenses, majority vote). 0 refuted.
- 39 findings (verifiers hit session limit): verified directly via runtime repro and
  targeted code reads on 2026-07-03. All reproduced. Data-quality measurements
  spot-checked 3-for-3 exact ([33] 195/5/0/0, [34] answer cross-tab, [37] 25 dups);
  [36]/[38] accepted on the validated methodology.
- One nuance: [22] `enable_thinking` — mechanism confirmed; impact is path-dependent
  (recent `.tmp/qwen35_*` runs contained "qwen" and did get the mitigation).
