"""Phase configuration for the SFT curriculum."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TierMix:
    """Specifies how to sample one tier for a training phase."""

    tier: int
    fraction: float = 1.0
    upsample: int = 1


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for one training phase."""

    name: str
    display_name: str
    tier_mix: tuple[TierMix, ...]
    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    resume_from: str | None


PHASE_A = PhaseConfig(
    name="a",
    display_name="Phase A: Foundation (Tiers 1-2)",
    tier_mix=(
        TierMix(tier=1, fraction=1.0),
        TierMix(tier=2, fraction=1.0),
    ),
    epochs=3,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    resume_from=None,
)

PHASE_B = PhaseConfig(
    name="b",
    display_name="Phase B: Understanding (Tiers 3-6 + review)",
    tier_mix=(
        TierMix(tier=1, fraction=0.3),
        TierMix(tier=2, fraction=0.3),
        TierMix(tier=3, fraction=1.0),
        TierMix(tier=4, fraction=1.0),
        TierMix(tier=5, fraction=1.0),
        TierMix(tier=6, fraction=1.0),
    ),
    epochs=3,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    resume_from="a",
)

PHASE_C = PhaseConfig(
    name="c",
    display_name="Phase C: Planning (full curriculum + T7 emphasis)",
    tier_mix=(
        TierMix(tier=1, fraction=0.2),
        TierMix(tier=2, fraction=0.2),
        TierMix(tier=3, fraction=1.0),
        TierMix(tier=4, fraction=1.0),
        TierMix(tier=5, fraction=1.0),
        TierMix(tier=6, fraction=1.0),
        TierMix(tier=7, fraction=1.0, upsample=5),
    ),
    epochs=3,
    learning_rate=5e-6,
    warmup_ratio=0.03,
    weight_decay=0.01,
    resume_from="b",
)

PHASES: dict[str, PhaseConfig] = {
    "a": PHASE_A,
    "b": PHASE_B,
    "c": PHASE_C,
}

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_MODEL = os.environ.get("CHESS_SFT_BASE_MODEL") or DEFAULT_BASE_MODEL

PHASE_READY_SENTINEL = "READY"
PHASE_PASSED_SENTINEL = "PASSED"


def resolve_checkpoint(
    phase: PhaseConfig,
    output_root: Path,
    *,
    require_passed: bool = False,
) -> str:
    """Return the base model or previous phase checkpoint for ``phase``."""
    if phase.resume_from is None:
        return BASE_MODEL

    prev_dir = output_root / f"phase_{phase.resume_from}"
    checkpoint = prev_dir / "best"
    passed_sentinel = prev_dir / PHASE_PASSED_SENTINEL

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Phase {phase.name} requires checkpoint from phase {phase.resume_from} "
            f"at {checkpoint}, but it does not exist. "
            f"Train phase {phase.resume_from} first."
        )

    if require_passed and not passed_sentinel.exists():
        raise FileNotFoundError(
            f"Phase {phase.name} requires phase {phase.resume_from} to have passed "
            f"its benchmark evaluation, but {passed_sentinel} does not exist. "
            f"Phase {phase.resume_from} either failed evaluation or was run with --skip-eval. "
            f"Re-run phase {phase.resume_from} evaluation before proceeding."
        )

    return str(checkpoint)


__all__ = [
    "BASE_MODEL",
    "DEFAULT_BASE_MODEL",
    "PHASE_A",
    "PHASE_B",
    "PHASE_C",
    "PHASE_PASSED_SENTINEL",
    "PHASE_READY_SENTINEL",
    "PHASES",
    "PhaseConfig",
    "TierMix",
    "resolve_checkpoint",
]
