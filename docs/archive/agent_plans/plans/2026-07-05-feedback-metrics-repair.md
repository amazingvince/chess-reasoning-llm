# Feedback Metrics Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix misleading Phase C move-quality metrics by making ACPL bounded and adding distribution telemetry.

**Architecture:** Shared ACPL constants and clamping live in `chess_llm.evals.benchmark` so the standalone benchmark CLI and training eval path cannot diverge. Existing result dictionaries keep mean ACPL under the existing `acpl` keys and add median/p90/p95 keys alongside them.

**Tech Stack:** Python, python-chess, pytest, existing `chess_llm.evals` and `chess_llm.training.evaluate` modules.

---

### Task 1: Shared ACPL Clamp Policy

**Files:**
- Modify: `src/chess_llm/evals/benchmark.py`
- Modify: `src/chess_llm/evals/run_benchmark.py`
- Modify: `src/chess_llm/training/evaluate.py`
- Test: `tests/test_evals_benchmark.py`
- Test: `tests/test_evals_cli.py`
- Test: `tests/test_training_eval_reliability.py`

- [x] **Step 1: Write failing benchmark clamp test**

Add this test near `test_package_scoring_metrics_cover_move_protocol_and_acpl` in `tests/test_evals_benchmark.py`:

```python
def test_package_centipawn_loss_clamps_tail_values():
    assert packaged.ACPL_CP_LOSS_CLAMP == 1000.0
    assert packaged.ACPL_INVALID_MOVE_PENALTY == 1000.0
    assert packaged.centipawn_loss(12000.0, -5000.0) == 1000.0
    assert packaged.centipawn_loss(120.0, 20.0) == 100.0
    assert packaged.centipawn_loss(-20.0, 120.0) == 0.0
```

- [x] **Step 2: Run benchmark clamp test and confirm RED**

Run:

```powershell
python -m pytest tests/test_evals_benchmark.py::test_package_centipawn_loss_clamps_tail_values -q
```

Expected: fail because `ACPL_CP_LOSS_CLAMP` and `ACPL_INVALID_MOVE_PENALTY` are not exported yet, or because `centipawn_loss` is still uncapped.

- [x] **Step 3: Implement shared constants and clamped `centipawn_loss`**

In `src/chess_llm/evals/benchmark.py`, add constants near the scoring helpers:

```python
ACPL_CP_LOSS_CLAMP = 1000.0
ACPL_INVALID_MOVE_PENALTY = ACPL_CP_LOSS_CLAMP
```

Change `centipawn_loss` to:

```python
def centipawn_loss(
    gold_cp: float,
    predicted_cp: float,
    *,
    clamp: float = ACPL_CP_LOSS_CLAMP,
) -> float:
    """Bounded centipawn loss when both scores use the same perspective."""
    loss = max(0.0, gold_cp - predicted_cp)
    return min(float(clamp), loss)
```

- [x] **Step 4: Run benchmark clamp test and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_evals_benchmark.py::test_package_centipawn_loss_clamps_tail_values -q
```

Expected: pass.

### Task 2: Invalid Move ACPL Penalty Consistency

**Files:**
- Modify: `src/chess_llm/evals/run_benchmark.py`
- Modify: `src/chess_llm/training/evaluate.py`
- Test: `tests/test_evals_cli.py`
- Test: `tests/test_training_eval_reliability.py`

- [x] **Step 1: Write failing standalone benchmark invalid penalty test**

Add this test after `test_run_benchmark_package_acpl_accepts_chess960_id` in `tests/test_evals_cli.py`:

```python
def test_run_benchmark_acpl_invalid_move_uses_shared_clamp():
    from chess_llm.evals import run_benchmark
    from chess_llm.evals.benchmark import ACPL_INVALID_MOVE_PENALTY

    class FakeEngine:
        def analyse(self, _board, _limit):
            raise AssertionError("invalid moves should not call Stockfish")

    example = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={"cp": 23},
    )

    scores = run_benchmark.compute_acpl(
        FakeEngine(),
        [example],
        {"planning_00000": "no move tag"},
        depth=1,
    )

    assert scores["planning_00000"] == ACPL_INVALID_MOVE_PENALTY
```

- [x] **Step 2: Write failing training eval invalid/tail test**

Add this test near the existing ACPL/WPD tests in `tests/test_training_eval_reliability.py`:

```python
def test_compute_acpl_uses_shared_clamp_for_invalid_and_tail_losses(monkeypatch):
    from chess_llm.evals.benchmark import ACPL_INVALID_MOVE_PENALTY
    from chess_llm.training import evaluate

    invalid = BenchmarkExample(
        example_id="planning_00000",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )
    tail = BenchmarkExample(
        example_id="planning_00001",
        split="planning",
        task_type="best_move",
        fen=STARTING_FEN,
        prompt="FEN: ...",
        gold_answer="e2e4",
        metric_type="move_extraction",
        metadata={},
    )

    monkeypatch.setattr(evaluate, "_evaluate_position", lambda *_args, **_kwargs: 10000)
    monkeypatch.setattr(evaluate, "_evaluate_predicted_move", lambda *_args, **_kwargs: -10000)

    scores = evaluate.compute_acpl(
        object(),
        [invalid, tail],
        {
            invalid.example_id: "no move tag",
            tail.example_id: "<move>e2e4</move>",
        },
        depth=1,
    )

    assert scores[invalid.example_id] == ACPL_INVALID_MOVE_PENALTY
    assert scores[tail.example_id] == ACPL_INVALID_MOVE_PENALTY
```

- [x] **Step 3: Run invalid/tail tests and confirm RED**

Run:

```powershell
python -m pytest tests/test_evals_cli.py::test_run_benchmark_acpl_invalid_move_uses_shared_clamp tests/test_training_eval_reliability.py::test_compute_acpl_uses_shared_clamp_for_invalid_and_tail_losses -q
```

Expected: fail because invalid moves still use 150cp or metadata-limited penalties.

- [x] **Step 4: Implement shared penalty usage**

In `src/chess_llm/evals/run_benchmark.py`, import `ACPL_INVALID_MOVE_PENALTY` from `chess_llm.evals.benchmark`, delete the local `_ACPL_INVALID_MOVE_PENALTY`, and replace both invalid branches with:

```python
acpl_scores[example.example_id] = ACPL_INVALID_MOVE_PENALTY
```

In `src/chess_llm/training/evaluate.py`, import `ACPL_INVALID_MOVE_PENALTY` from `chess_llm.evals.benchmark`, delete the local `_ACPL_INVALID_MOVE_PENALTY`, and replace invalid branches with the shared constant.

- [x] **Step 5: Run invalid/tail tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/test_evals_cli.py::test_run_benchmark_acpl_invalid_move_uses_shared_clamp tests/test_training_eval_reliability.py::test_compute_acpl_uses_shared_clamp_for_invalid_and_tail_losses -q
```

Expected: pass.

### Task 3: ACPL Distribution Aggregation

**Files:**
- Modify: `src/chess_llm/evals/benchmark.py`
- Test: `tests/test_evals_benchmark.py`

- [x] **Step 1: Write failing distribution aggregation test**

Add this test after `test_package_scoring_metrics_cover_move_protocol_and_acpl`:

```python
def test_package_score_split_reports_acpl_distribution_stats():
    examples = [
        _example(example_id=f"planning_{index:05d}")
        for index in range(5)
    ]
    prediction = "<think>control center</think><move>e2e4</move>"

    aggregate = packaged.score_split(
        examples,
        {example.example_id: prediction for example in examples},
        acpl_scores={
            examples[0].example_id: 0.0,
            examples[1].example_id: 10.0,
            examples[2].example_id: 100.0,
            examples[3].example_id: 400.0,
            examples[4].example_id: 1000.0,
        },
    )

    assert aggregate["best_move_acpl"] == 302.0
    assert aggregate["best_move_acpl_median"] == 100.0
    assert aggregate["best_move_acpl_p90"] == 1000.0
    assert aggregate["best_move_acpl_p95"] == 1000.0
    assert aggregate["acpl"] == 302.0
    assert aggregate["acpl_median"] == 100.0
    assert aggregate["acpl_p90"] == 1000.0
    assert aggregate["acpl_p95"] == 1000.0
```

- [x] **Step 2: Run distribution test and confirm RED**

Run:

```powershell
python -m pytest tests/test_evals_benchmark.py::test_package_score_split_reports_acpl_distribution_stats -q
```

Expected: fail because median/p90/p95 keys do not exist.

- [x] **Step 3: Implement percentile helpers and aggregation**

In `src/chess_llm/evals/benchmark.py`, add:

```python
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    index = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[index]


def _append_acpl_summary(result: dict[str, float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    result[prefix] = _mean(values)
    result[f"{prefix}_median"] = _nearest_rank(values, 50.0)
    result[f"{prefix}_p90"] = _nearest_rank(values, 90.0)
    result[f"{prefix}_p95"] = _nearest_rank(values, 95.0)
```

Replace direct ACPL mean assignment in `score_split` with `_append_acpl_summary`.

- [x] **Step 4: Run distribution test and focused benchmark tests**

Run:

```powershell
python -m pytest tests/test_evals_benchmark.py::test_package_score_split_reports_acpl_distribution_stats tests/test_evals_benchmark.py::test_package_scoring_metrics_cover_move_protocol_and_acpl -q
```

Expected: pass.

### Task 4: Full Verification And Commit

**Files:**
- All files modified above.

- [x] **Step 1: Run focused metric suite**

Run:

```powershell
python -m pytest tests/test_evals_benchmark.py tests/test_evals_cli.py tests/test_training_eval_reliability.py -q
```

Expected: pass.

- [x] **Step 2: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: pass.

- [x] **Step 3: Inspect git diff**

Run:

```powershell
git diff --stat
git diff -- docs/archive/agent_plans/specs/2026-07-05-feedback-metrics-design.md docs/archive/agent_plans/plans/2026-07-05-feedback-metrics-repair.md src/chess_llm/evals/benchmark.py src/chess_llm/evals/run_benchmark.py src/chess_llm/training/evaluate.py tests/test_evals_benchmark.py tests/test_evals_cli.py tests/test_training_eval_reliability.py
```

Expected: only metric-slice files changed.

- [x] **Step 4: Commit metric repair**

Run:

```powershell
git add docs/archive/agent_plans/specs/2026-07-05-feedback-metrics-design.md docs/archive/agent_plans/plans/2026-07-05-feedback-metrics-repair.md src/chess_llm/evals/benchmark.py src/chess_llm/evals/run_benchmark.py src/chess_llm/training/evaluate.py tests/test_evals_benchmark.py tests/test_evals_cli.py tests/test_training_eval_reliability.py
git commit -m "fix: bound acpl metric reporting"
```

Expected: commit succeeds.
