"""Tests for config/phases.py — phase configuration and checkpoint resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.phases import (
    PHASE_A, PHASE_B, PHASE_C,
    PHASES,
    BASE_MODEL,
    PHASE_PASSED_SENTINEL,
    PHASE_READY_SENTINEL,
    PhaseConfig, TierMix,
    resolve_checkpoint,
)


class TestPhaseConfigs:
    """Verify the three phase constant definitions."""

    def test_phase_a_basics(self):
        assert PHASE_A.name == "a"
        assert PHASE_A.resume_from is None
        assert PHASE_A.epochs == 3
        assert PHASE_A.learning_rate == 2e-5

    def test_phase_b_basics(self):
        assert PHASE_B.name == "b"
        assert PHASE_B.resume_from == "a"
        assert PHASE_B.epochs == 3
        assert PHASE_B.learning_rate == 1e-5

    def test_phase_c_basics(self):
        assert PHASE_C.name == "c"
        assert PHASE_C.resume_from == "b"
        assert PHASE_C.epochs == 3
        assert PHASE_C.learning_rate == 5e-6

    def test_lr_decreases_across_phases(self):
        """Learning rate should decrease: A > B > C."""
        assert PHASE_A.learning_rate > PHASE_B.learning_rate > PHASE_C.learning_rate

    def test_phase_a_tiers(self):
        """Phase A uses only tiers 1 and 2 at 100%."""
        tiers = {tm.tier for tm in PHASE_A.tier_mix}
        assert tiers == {1, 2}
        for tm in PHASE_A.tier_mix:
            assert tm.fraction == 1.0
            assert tm.upsample == 1

    def test_phase_b_includes_review(self):
        """Phase B should include T1-T2 at 30% and T3-T6 at 100%."""
        tier_map = {tm.tier: tm for tm in PHASE_B.tier_mix}
        assert tier_map[1].fraction == 0.3
        assert tier_map[2].fraction == 0.3
        assert tier_map[3].fraction == 1.0
        assert tier_map[6].fraction == 1.0

    def test_phase_c_tier7_upsampled(self):
        """Phase C should upsample tier 7 by 5x."""
        tier_map = {tm.tier: tm for tm in PHASE_C.tier_mix}
        assert tier_map[7].upsample == 5
        assert tier_map[7].fraction == 1.0

    def test_phase_c_includes_all_tiers(self):
        """Phase C should include tiers 1-7."""
        tiers = {tm.tier for tm in PHASE_C.tier_mix}
        assert tiers == {1, 2, 3, 4, 5, 6, 7}

    def test_phase_c_review_fractions(self):
        """Phase C should have T1-T2 at 20%."""
        tier_map = {tm.tier: tm for tm in PHASE_C.tier_mix}
        assert tier_map[1].fraction == 0.2
        assert tier_map[2].fraction == 0.2

    def test_phases_dict_complete(self):
        """PHASES dict should have all three phases."""
        assert set(PHASES.keys()) == {"a", "b", "c"}
        assert PHASES["a"] is PHASE_A
        assert PHASES["b"] is PHASE_B
        assert PHASES["c"] is PHASE_C


class TestTierMix:
    """Verify TierMix dataclass behavior."""

    def test_defaults(self):
        tm = TierMix(tier=1)
        assert tm.fraction == 1.0
        assert tm.upsample == 1

    def test_frozen(self):
        tm = TierMix(tier=1)
        with pytest.raises(AttributeError):
            tm.fraction = 0.5  # type: ignore[misc]


def _make_passed_phase(tmp_path: Path, phase_name: str) -> Path:
    """Create a phase directory with both best/ checkpoint and PASSED sentinel."""
    phase_dir = tmp_path / f"phase_{phase_name}"
    (phase_dir / "best").mkdir(parents=True)
    (phase_dir / PHASE_PASSED_SENTINEL).write_text("passed\n")
    return phase_dir


def _make_ready_phase(tmp_path: Path, phase_name: str) -> Path:
    """Create a phase directory with a best/ checkpoint but no PASSED sentinel."""
    phase_dir = tmp_path / f"phase_{phase_name}"
    (phase_dir / "best").mkdir(parents=True)
    (phase_dir / PHASE_READY_SENTINEL).write_text("ready\n")
    return phase_dir


class TestResolveCheckpoint:
    """Verify checkpoint resolution logic."""

    def test_phase_a_returns_base_model(self, tmp_path: Path):
        result = resolve_checkpoint(PHASE_A, tmp_path)
        assert result == BASE_MODEL

    def test_phase_b_requires_phase_a_checkpoint(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="phase a"):
            resolve_checkpoint(PHASE_B, tmp_path)

    def test_phase_b_finds_passed_checkpoint(self, tmp_path: Path):
        _make_passed_phase(tmp_path, "a")
        checkpoint = tmp_path / "phase_a" / "best"
        result = resolve_checkpoint(PHASE_B, tmp_path)
        assert result == str(checkpoint)

    def test_phase_b_accepts_checkpoint_without_sentinel_by_default(self, tmp_path: Path):
        """best/ exists but PASSED does not — should fail."""
        (tmp_path / "phase_a" / "best").mkdir(parents=True)
        checkpoint = tmp_path / "phase_a" / "best"

        result = resolve_checkpoint(PHASE_B, tmp_path)

        assert result == str(checkpoint)

    def test_phase_b_requires_passed_checkpoint_in_strict_mode(self, tmp_path: Path):
        """Hard-gate mode can still require the PASSED sentinel."""
        (tmp_path / "phase_a" / "best").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="passed.*benchmark evaluation"):
            resolve_checkpoint(PHASE_B, tmp_path, require_passed=True)

    def test_phase_c_requires_phase_b_checkpoint(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="phase b"):
            resolve_checkpoint(PHASE_C, tmp_path)

    def test_phase_c_finds_passed_checkpoint(self, tmp_path: Path):
        _make_passed_phase(tmp_path, "b")
        checkpoint = tmp_path / "phase_b" / "best"
        result = resolve_checkpoint(PHASE_C, tmp_path)
        assert result == str(checkpoint)

    def test_ready_checkpoint_allows_next_phase_by_default(self, tmp_path: Path):
        """Simulates --skip-eval: best/ exists but no PASSED sentinel."""
        _make_ready_phase(tmp_path, "a")
        checkpoint = tmp_path / "phase_a" / "best"

        result = resolve_checkpoint(PHASE_B, tmp_path)

        assert result == str(checkpoint)

    def test_custom_phase_no_resume(self, tmp_path: Path):
        custom = PhaseConfig(
            name="x", display_name="X", tier_mix=(),
            epochs=1, learning_rate=1e-4,
            warmup_ratio=0.01, weight_decay=0.0,
            resume_from=None,
        )
        assert resolve_checkpoint(custom, tmp_path) == BASE_MODEL
