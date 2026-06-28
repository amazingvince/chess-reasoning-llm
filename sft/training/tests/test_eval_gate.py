"""Tests for phase_gate.check_phase_criteria — phase gating logic."""

from __future__ import annotations

import io
import sys

import pytest

from phase_gate import check_phase_criteria as _check_phase_criteria


def _capture_gate(split_results, baseline=None, phase=None, has_acpl=False):
    """Run _check_phase_criteria, return (failure_count, stdout_text)."""
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        failures = _check_phase_criteria(
            split_results, baseline, phase=phase, has_acpl=has_acpl,
        )
    finally:
        sys.stdout = old
    return failures, buf.getvalue()


# -----------------------------------------------------------------------

class TestAbsoluteThresholds:

    PASSING_A = {
        "perception": {"board_print": 0.95, "state_tracking": 0.85},
        "rules": {"legal_moves": 0.90, "legality_check": 0.95},
    }

    def test_phase_a_all_pass(self):
        failures, output = _capture_gate(self.PASSING_A, phase="a")
        assert failures == 0
        assert "ALL CHECKS PASSED" in output

    def test_phase_a_board_print_fail(self):
        results = {
            "perception": {"board_print": 0.80, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(results, phase="a")
        assert failures >= 1
        assert "Board print" in output

    def test_phase_c_checks_all_prior_phases(self):
        results = {
            "perception": {"board_print": 0.50, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
            "planning": {"format_compliance": 0.99, "legal_move_rate": 0.99, "puzzle_solve": 0.30},
            "evaluation": {"material_balance": 0.99},
            "endgames": {"endgame_wdl": 0.90},
        }
        failures, output = _capture_gate(results, phase="c", has_acpl=True)
        assert failures >= 1
        assert "Board print" in output


class TestACPLGating:

    def test_phase_c_acpl_missing_is_failure(self):
        results = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
            "planning": {"format_compliance": 0.99, "legal_move_rate": 0.99, "puzzle_solve": 0.30},
            "evaluation": {"material_balance": 0.99},
            "endgames": {"endgame_wdl": 0.90},
        }
        failures, output = _capture_gate(results, phase="c", has_acpl=False)
        assert failures >= 1
        assert "Stockfish not available" in output

    def test_phase_c_acpl_passing(self):
        results = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
            "planning": {
                "format_compliance": 0.99, "legal_move_rate": 0.99,
                "puzzle_solve": 0.30, "acpl": 150.0,
            },
            "evaluation": {"material_balance": 0.99},
            "endgames": {"endgame_wdl": 0.90},
        }
        failures, output = _capture_gate(results, phase="c", has_acpl=True)
        assert failures == 0
        assert "ACPL < 200: 150.0 cp" in output

    def test_phase_c_acpl_too_high(self):
        results = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
            "planning": {
                "format_compliance": 0.99, "legal_move_rate": 0.99,
                "puzzle_solve": 0.30, "acpl": 250.0,
            },
            "evaluation": {"material_balance": 0.99},
            "endgames": {"endgame_wdl": 0.90},
        }
        failures, output = _capture_gate(results, phase="c", has_acpl=True)
        assert failures >= 1
        assert "ACPL < 200: 250.0 cp" in output

    def test_phase_a_acpl_not_required(self):
        results = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(results, phase="a", has_acpl=False)
        assert failures == 0

    def test_acpl_uses_planning_split_only(self):
        results = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
            "planning": {
                "format_compliance": 0.99, "legal_move_rate": 0.99,
                "puzzle_solve": 0.30,
            },
            "endgames": {"endgame_wdl": 0.90, "acpl": 100.0},
            "evaluation": {"material_balance": 0.99},
        }
        failures, output = _capture_gate(results, phase="c", has_acpl=True)
        assert failures >= 1
        assert "no ACPL data in planning split" in output


class TestMissingMetrics:
    """Missing required metrics should be failures when phase is specified."""

    def test_missing_metric_fails_with_phase(self):
        """Phase A with board_print missing should fail."""
        results = {
            "perception": {"state_tracking": 0.85},  # no board_print
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(results, phase="a")
        assert failures >= 1
        assert "metric 'board_print' not found" in output
        assert "FAIL" in output

    def test_missing_split_fails_with_phase(self):
        """Phase A with perception split missing should fail."""
        results = {
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(results, phase="a")
        assert failures >= 1
        assert "split 'perception' not found" in output

    def test_missing_metric_skips_without_phase(self):
        """Without --phase, missing metrics should be informational skips."""
        results = {
            "perception": {"state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(results, phase=None)
        assert "[ -- ]" in output
        # Should not count as failure without explicit phase
        # (board_print is missing but we're not enforcing)


class TestFloorChecks:
    def test_floor_violation_counted(self):
        results = {
            "perception": {"board_print": 0.95, "state_tracking": 0.60},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(results, phase="a")
        assert failures >= 1
        assert "below 65%" in output


class TestRegressionChecks:
    def test_regression_detected(self):
        current = {
            "perception": {"board_print": 0.80, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        baseline = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(current, baseline=baseline, phase="b")
        assert failures >= 1
        assert "dropped" in output

    def test_no_regression_passes(self):
        current = {
            "perception": {"board_print": 0.94, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
            "evaluation": {"material_balance": 0.99},
            "endgames": {"endgame_wdl": 0.90},
        }
        baseline = {
            "perception": {"board_print": 0.95, "state_tracking": 0.85},
            "rules": {"legal_moves": 0.90, "legality_check": 0.95},
        }
        failures, output = _capture_gate(current, baseline=baseline, phase="b")
        assert "within 5%" in output
