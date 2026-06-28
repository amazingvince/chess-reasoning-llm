"""Phase configuration for three-phase SFT curriculum.

Phase A: Foundation (Tiers 1-2) — perception + rules
Phase B: Understanding (Tiers 3-6 + 30% T1-2 review) — tactics, eval, openings, endgames
Phase C: Planning (all tiers, T7 upsampled 5x) — move selection with reasoning
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TierMix:
    """Specifies how to sample a tier's data for a training phase."""

    tier: int
    fraction: float = 1.0  # 1.0 = all, 0.3 = 30% sample
    upsample: int = 1      # multiply after sampling


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for one training phase."""

    name: str               # "a", "b", "c"
    display_name: str
    tier_mix: tuple[TierMix, ...]
    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    resume_from: str | None  # phase name to resume from, or None for base model


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

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
        TierMix(tier=1, fraction=0.3),   # review
        TierMix(tier=2, fraction=0.3),   # review
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
        TierMix(tier=1, fraction=0.2),    # review
        TierMix(tier=2, fraction=0.2),    # review
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

BASE_MODEL = os.environ.get("CHESS_SFT_BASE_MODEL") or "Qwen/Qwen3-0.6B"


PHASE_READY_SENTINEL = "READY"
PHASE_PASSED_SENTINEL = "PASSED"


def resolve_checkpoint(
    phase: PhaseConfig,
    output_root: Path,
    *,
    require_passed: bool = False,
) -> str:
    """Return model path for a phase: base model for A, or previous best checkpoint.

    By default, phase chaining is training-first: a previous ``best/``
    checkpoint is enough to continue. Pass ``require_passed=True`` to enforce
    the older hard-gate behavior where the previous phase must also have a
    ``PASSED`` benchmark sentinel.

    Parameters
    ----------
    phase : PhaseConfig
        The phase about to train.
    output_root : Path
        Root directory containing ``phase_{name}/best/`` subdirectories.

    Returns
    -------
    str
        HuggingFace model ID or local path to checkpoint.
    """
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
