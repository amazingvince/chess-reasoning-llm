from __future__ import annotations

from argparse import Namespace

import evaluate


def _args(**overrides) -> Namespace:
    defaults = {
        "no_acpl": False,
        "full_acpl_report": False,
        "phase": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_acpl_defaults_to_phase_c_planning_only():
    splits = {"planning": [object()], "rules": [object()]}

    assert evaluate._select_acpl_splits(_args(phase="c"), splits) == {"planning"}


def test_acpl_skips_non_phase_c_by_default():
    splits = {"planning": [object()], "rules": [object()]}

    assert evaluate._select_acpl_splits(_args(phase="b"), splits) == set()
    assert evaluate._select_acpl_splits(_args(), splits) == set()


def test_full_acpl_report_selects_all_splits():
    splits = {"planning": [object()], "rules": [object()]}

    assert evaluate._select_acpl_splits(_args(full_acpl_report=True), splits) == {
        "planning",
        "rules",
    }


def test_no_acpl_overrides_full_report():
    splits = {"planning": [object()], "rules": [object()]}

    assert evaluate._select_acpl_splits(
        _args(no_acpl=True, full_acpl_report=True, phase="c"),
        splits,
    ) == set()


def test_soft_gate_returns_success_even_with_failures():
    assert evaluate._phase_gate_return_code(n_failures=3, soft_gate=True) == 0


def test_hard_gate_returns_failure_when_checks_fail():
    assert evaluate._phase_gate_return_code(n_failures=3, soft_gate=False) == 1
