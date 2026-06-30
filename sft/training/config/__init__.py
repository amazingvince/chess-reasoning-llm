"""Training configuration — phase definitions and SFTConfig builder."""

from .phases import PhaseConfig, TierMix, PHASE_A, PHASE_B, PHASE_C, resolve_checkpoint

__all__ = [
    "PhaseConfig",
    "TierMix",
    "PHASE_A",
    "PHASE_B",
    "PHASE_C",
    "resolve_checkpoint",
]
