import io
import sys


def _capture_gate(
    split_results,
    baseline=None,
    phase=None,
    has_acpl=False,
    allow_missing_criteria=False,
):
    from chess_llm.training.phase_gate import check_phase_criteria

    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        failures = check_phase_criteria(
            split_results,
            baseline,
            phase=phase,
            has_acpl=has_acpl,
            allow_missing_criteria=allow_missing_criteria,
        )
    finally:
        sys.stdout = old
    return failures, buf.getvalue()


def test_package_phase_gate_checks_phase_c_acpl_requirement():
    results = {
        "perception": {"board_print": 0.95, "state_tracking": 0.85},
        "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        "planning": {
            "format_compliance": 0.99,
            "legal_move_rate": 0.99,
            "puzzle_solve": 0.30,
        },
        "evaluation": {"material_balance": 0.99},
        "endgames": {"endgame_wdl": 0.90},
    }

    failures, output = _capture_gate(results, phase="c", has_acpl=False)

    assert failures >= 1
    assert "Stockfish not available" in output


def test_package_phase_gate_can_skip_missing_criteria_for_capped_soft_eval():
    results = {
        "perception": {"board_print": 0.95},
        "rules": {"legal_moves": 0.90},
    }

    failures, output = _capture_gate(
        results,
        phase="a",
        allow_missing_criteria=True,
    )

    assert failures == 0
    assert "[ -- ] [A] Legality check > 90%" in output
    assert "[ -- ] [A] State tracking (1 move) > 80%" in output


def test_package_phase_gate_excludes_diagnostic_mechanics_from_floor_checks():
    results = {
        "perception": {
            "board_print": 0.95,
            "state_tracking": 0.85,
            "square_lookup": 0.0,
            "rank_lookup": 0.0,
            "move_square_edits": 0.0,
            "fen_assembly": 0.0,
            "material_inventory": 0.0,
            "material_piece_counts": 0.0,
            "material_value_totals": 0.0,
            "material_balance_trace": 0.0,
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "legality_check_legality_reason_accuracy": 0.0,
            "side_piece_inventory": 0.0,
            "piece_legal_moves": 0.0,
            "piece_pseudo_legal_moves": 0.0,
            "piece_legal_filter": 0.0,
            "king_safety_filter": 0.0,
            "legal_moves_by_piece": 0.0,
        },
    }

    failures, output = _capture_gate(results, phase="a")

    assert failures == 0
    assert "perception/square_lookup" not in output
    assert "perception/rank_lookup" not in output
    assert "perception/move_square_edits" not in output
    assert "perception/fen_assembly" not in output
    assert "perception/material_inventory" not in output
    assert "perception/material_balance_trace" not in output
    assert "rules/legality_check_legality_reason_accuracy" not in output
    assert "rules/side_piece_inventory" not in output
    assert "rules/piece_legal_moves" not in output
    assert "rules/piece_pseudo_legal_moves" not in output
    assert "rules/king_safety_filter" not in output


def test_package_phase_gate_excludes_diagnostic_mechanics_from_regression_checks():
    results = {
        "perception": {
            "board_print": 0.95,
            "state_tracking": 0.85,
            "square_lookup": 0.0,
            "fen_assembly": 0.0,
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "legality_check_legality_reason_accuracy": 0.0,
            "side_piece_inventory": 0.0,
            "piece_legal_moves": 0.0,
        },
    }
    baseline = {
        "perception": {
            "board_print": 0.95,
            "state_tracking": 0.85,
            "square_lookup": 1.0,
            "fen_assembly": 1.0,
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "legality_check_legality_reason_accuracy": 1.0,
            "side_piece_inventory": 1.0,
            "piece_legal_moves": 1.0,
        },
    }

    failures, output = _capture_gate(results, baseline=baseline, phase="a")

    assert failures == 0
    assert "perception/square_lookup" not in output
    assert "perception/fen_assembly" not in output
    assert "rules/legality_check_legality_reason_accuracy" not in output
    assert "rules/side_piece_inventory" not in output
    assert "rules/piece_legal_moves" not in output


def test_package_phase_gate_excludes_new_diagnostic_metric_names_from_floor_checks():
    results = {
        "perception": {
            "board_print": 0.95,
            "state_tracking": 0.85,
            "multi_state_tracking": 0.0,
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "ray_walk": 0.0,
            "legal_filter_trace": 0.0,
            "legal_filter_trace_final_jaccard": 0.0,
            "piece_legal_filter_set_f1": 0.0,
            "legal_moves_by_piece_set_f1": 0.0,
        },
    }

    failures, output = _capture_gate(results, phase="a")

    assert failures == 0
    assert "perception/multi_state_tracking" not in output
    assert "rules/ray_walk" not in output
    assert "rules/legal_filter_trace" not in output
    assert "rules/legal_filter_trace_final_jaccard" not in output
    assert "rules/piece_legal_filter_set_f1" not in output
    assert "rules/legal_moves_by_piece_set_f1" not in output


def test_package_phase_gate_excludes_new_diagnostic_metric_names_from_regression_checks():
    results = {
        "perception": {
            "board_print": 0.95,
            "state_tracking": 0.85,
            "multi_state_tracking": 0.0,
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "ray_walk": 0.0,
            "legal_filter_trace_final_jaccard": 0.0,
        },
    }
    baseline = {
        "perception": {
            "board_print": 0.95,
            "state_tracking": 0.85,
            "multi_state_tracking": 1.0,
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "ray_walk": 1.0,
            "legal_filter_trace_final_jaccard": 1.0,
        },
    }

    failures, output = _capture_gate(results, baseline=baseline, phase="a")

    assert failures == 0
    assert "perception/multi_state_tracking" not in output
    assert "rules/ray_walk" not in output
    assert "rules/legal_filter_trace_final_jaccard" not in output


def test_package_phase_gate_identifies_count_metrics_by_name_pattern():
    from chess_llm.training.phase_gate import is_count_metric

    assert is_count_metric("legal_moves_by_piece_illegal_extra_count") is True
    assert is_count_metric("missing_move_count") is True
    assert is_count_metric("material_piece_counts") is False
    assert is_count_metric("board_print") is False


def test_package_phase_gate_excludes_count_metrics_from_floor_checks():
    results = {
        "perception": {"board_print": 0.95, "state_tracking": 0.85},
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            # Perfect count outcomes (0 extras / 0 misses) must not be read
            # as 0% accuracies and fail the floor.
            "legal_moves_by_piece_illegal_extra_count": 0.0,
            "missing_move_count": 0.0,
        },
    }

    failures, output = _capture_gate(results, phase="a")

    assert failures == 0
    assert "rules/legal_moves_by_piece_illegal_extra_count" not in output
    assert "rules/missing_move_count" not in output


def test_package_phase_gate_reports_count_metrics_as_informational_in_regression():
    results = {
        "perception": {"board_print": 0.95, "state_tracking": 0.85},
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "missing_move_count": 2.0,
            "legal_moves_by_piece_illegal_extra_count": 40.0,
        },
    }
    baseline = {
        "perception": {"board_print": 0.95, "state_tracking": 0.85},
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            # Improvement (40 -> 2) used to be flagged as a >5% accuracy drop.
            "missing_move_count": 40.0,
            # Worsening (2 -> 40) is reported but not gated.
            "legal_moves_by_piece_illegal_extra_count": 2.0,
        },
    }

    failures, output = _capture_gate(results, baseline=baseline, phase="a")

    assert failures == 0
    assert (
        "[INFO] rules/missing_move_count: count metric "
        "baseline=40 current=2 (informational, not gated)"
    ) in output
    assert (
        "[INFO] rules/legal_moves_by_piece_illegal_extra_count: count metric "
        "baseline=2 current=40 (informational, not gated)"
    ) in output
    assert "[FAIL] rules/missing_move_count" not in output
    assert "[PASS] All metrics within 5% of best-historical baseline" in output


def test_legacy_phase_gate_reexports_package_function():
    import importlib.util
    from pathlib import Path

    phase_gate_path = (
        Path(__file__).resolve().parents[1] / "sft" / "training" / "phase_gate.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_phase_gate", phase_gate_path)
    assert spec is not None
    assert spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    from chess_llm.training.phase_gate import check_phase_criteria

    assert legacy.check_phase_criteria is check_phase_criteria
