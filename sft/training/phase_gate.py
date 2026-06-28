"""Phase gate criteria checking — extracted for testability.

This module has no heavy dependencies (no torch, transformers, etc.)
and can be safely imported in the test environment.
"""

from __future__ import annotations


def check_phase_criteria(
    split_results: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]] | None = None,
    *,
    phase: str | None = None,
    has_acpl: bool = False,
) -> int:
    """Check all phase criteria and return the number of failures.

    Implements:
    - Per-phase absolute thresholds (from plan docs)
    - "No metric below X%" floor checks per phase
    - ACPL exit criterion for Phase C (MANDATORY when phase="c")
    - Regression checks (< 5% drop) against best-historical baseline

    Parameters
    ----------
    phase : str | None
        Current training phase ("a", "b", "c"). When set, only checks
        relevant to that phase and all prior phases are enforced.
        Phase C makes ACPL a hard requirement.
    has_acpl : bool
        Whether ACPL was actually computed in this run.
    """
    print(f"\n{'=' * 60}")
    print("  Phase Criteria Check")
    if phase:
        print(f"  Active phase: {phase.upper()}")
    print(f"{'=' * 60}")

    failures = 0

    # Which phases' criteria are active?
    active_phases: set[str] = {"A", "B", "C"}
    if phase:
        phase_order = ["A", "B", "C"]
        cutoff = phase_order.index(phase.upper()) + 1
        active_phases = set(phase_order[:cutoff])

    # --- Absolute threshold criteria ---
    criteria = [
        ("perception", "board_print", 0.90, "Board print > 90%", "A"),
        ("rules", "legal_moves", 0.85, "Legal moves > 85%", "A"),
        ("rules", "legality_check", 0.90, "Legality check > 90%", "A"),
        ("perception", "state_tracking", 0.80, "State tracking (1-2 moves) > 80%", "A"),
        ("evaluation", "material_balance", 0.95, "Material balance > 95%", "B"),
        ("endgames", "endgame_wdl", 0.85, "Endgame WDL > 85%", "B"),
        ("planning", "format_compliance", 0.95, "Format compliance > 95%", "C"),
        ("planning", "legal_move_rate", 0.95, "Legal move rate > 95%", "C"),
        ("planning", "puzzle_solve", 0.20, "Puzzle pass@1 > 20%", "C"),
    ]

    print("\n  Absolute thresholds:")
    for split, metric, threshold, label, phase_tag in criteria:
        if phase_tag not in active_phases:
            continue
        if split not in split_results:
            if phase:
                # When running a specific phase, missing required splits are failures
                failures += 1
                print(f"  [FAIL] [{phase_tag}] {label}  (split '{split}' not found)")
            else:
                print(f"  [ -- ] [{phase_tag}] {label}  (split '{split}' not found)")
            continue
        value = split_results[split].get(metric)
        if value is None:
            if phase:
                failures += 1
                print(f"  [FAIL] [{phase_tag}] {label}  (metric '{metric}' not found)")
            else:
                print(f"  [ -- ] [{phase_tag}] {label}  (metric '{metric}' not found)")
            continue
        passed = value >= threshold
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        print(f"  [{status}] [{phase_tag}] {label}: {value:.1%}")

    # --- ACPL check (Phase C exit criterion — planning split only) ---
    if "C" in active_phases:
        print("\n  ACPL (Phase C exit criterion):")
        planning_acpl = split_results.get("planning", {}).get("acpl")
        if planning_acpl is not None:
            passed = planning_acpl < 200.0
            status = "PASS" if passed else "FAIL"
            if not passed:
                failures += 1
            print(f"  [{status}] [C] ACPL < 200: {planning_acpl:.1f} cp")
        elif phase and phase.lower() == "c":
            failures += 1
            if not has_acpl:
                print(
                    f"  [FAIL] [C] ACPL < 200: Stockfish not available. "
                    f"ACPL is required for Phase C exit. "
                    f"Provide --stockfish-path or set STOCKFISH_PATH."
                )
            else:
                print(
                    f"  [FAIL] [C] ACPL < 200: no ACPL data in planning split "
                    f"(benchmark examples may lack 'cp' metadata)"
                )
        else:
            print(f"  [ -- ] [C] ACPL < 200: not computed")

    # --- Floor checks ---
    floor_checks = [
        (["perception", "rules", "chess960"], 0.65, "No T1-2 metric below 65%", "A"),
        (["tactics", "evaluation", "openings", "endgames"], 0.55, "No T3-6 metric below 55%", "B"),
    ]

    print("\n  Floor checks:")
    for split_names, floor, label, phase_tag in floor_checks:
        if phase_tag not in active_phases:
            continue
        worst_metric = None
        worst_value = 1.0
        for split_name in split_names:
            if split_name not in split_results:
                continue
            for metric_name, value in split_results[split_name].items():
                if metric_name in ("overall", "acpl") or not isinstance(value, float):
                    continue
                if "acpl" in metric_name:
                    continue
                if value < worst_value:
                    worst_value = value
                    worst_metric = f"{split_name}/{metric_name}"

        if worst_metric is None:
            print(f"  [ -- ] [{phase_tag}] {label}  (no metrics found)")
            continue

        passed = worst_value >= floor
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        print(f"  [{status}] [{phase_tag}] {label}: worst={worst_metric} @ {worst_value:.1%}")

    # --- Regression checks ---
    if baseline:
        print("\n  Regression checks (< 5% drop from best-historical baseline):")
        max_regression = 0.05
        regression_failures = 0
        for split_name, baseline_metrics in baseline.items():
            if split_name not in split_results:
                continue
            current = split_results[split_name]
            for metric_name, baseline_val in baseline_metrics.items():
                if not isinstance(baseline_val, (int, float)):
                    continue
                if "acpl" in metric_name:
                    continue
                current_val = current.get(metric_name)
                if current_val is None:
                    continue
                drop = baseline_val - current_val
                if drop > max_regression:
                    regression_failures += 1
                    print(
                        f"  [FAIL] {split_name}/{metric_name}: "
                        f"{baseline_val:.1%} -> {current_val:.1%} "
                        f"(dropped {drop:.1%}, max allowed {max_regression:.0%})"
                    )
        failures += regression_failures
        if regression_failures == 0:
            print("  [PASS] All metrics within 5% of best-historical baseline")
    else:
        print("\n  Regression checks: skipped (no --baseline provided)")

    # --- Summary ---
    print(f"\n  {'=' * 40}")
    if failures == 0:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print(f"  RESULT: {failures} CHECK(S) FAILED")
    print(f"  {'=' * 40}\n")

    return failures
