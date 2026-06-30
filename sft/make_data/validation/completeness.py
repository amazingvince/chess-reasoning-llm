"""Compatibility wrapper for ``chess_llm.sft.completeness``."""

from __future__ import annotations

from pathlib import Path

from chess_llm.sft.completeness import expected_task_path

try:
    from config.settings import VOLUMES
except ModuleNotFoundError:
    from sft.make_data.config.settings import VOLUMES


def audit_output_completeness(
    output_dir: str | Path,
    expected_volume_override: int | None = None,
) -> dict[str, str]:
    """Return missing/underfilled output files keyed by relative path."""
    from chess_llm.sft.completeness import audit_output_completeness as _audit

    return _audit(
        output_dir,
        expected_volume_override=expected_volume_override,
        volumes=VOLUMES,
    )


__all__ = ["VOLUMES", "audit_output_completeness", "expected_task_path"]
