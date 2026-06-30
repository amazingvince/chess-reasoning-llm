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
        },
        "rules": {
            "legal_moves": 0.90,
            "legality_check": 0.95,
            "legality_check_legality_reason_accuracy": 0.0,
            "side_piece_inventory": 0.0,
            "piece_legal_moves": 0.0,
        },
    }

    failures, output = _capture_gate(results, phase="a")

    assert failures == 0
    assert "perception/square_lookup" not in output
    assert "perception/rank_lookup" not in output
    assert "perception/move_square_edits" not in output
    assert "perception/fen_assembly" not in output
    assert "rules/legality_check_legality_reason_accuracy" not in output
    assert "rules/side_piece_inventory" not in output
    assert "rules/piece_legal_moves" not in output


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
