"""Source availability reporting for SFT data generation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_llm.sft.context import raw_is_chess960


SOURCE_KEYS: tuple[str, ...] = (
    "fen_pool",
    "fen_pool_standard",
    "fen_pool_chess960",
    "game_positions",
    "puzzles",
    "openings",
    "position_evals",
    "best_move_evals",
    "consequence_evals",
    "book_moves",
    "endgame_positions",
    "mate_rows",
)

TIER_REQUIRED_SOURCES: dict[int, dict[str, tuple[str, ...]]] = {
    1: {
        "fen_pool": ("tier1",),
        "game_positions": ("tier1",),
    },
    2: {
        "fen_pool": ("tier2",),
    },
    3: {
        "fen_pool": ("tier3",),
        "puzzles": ("tier3",),
    },
    4: {
        "fen_pool": ("tier4",),
        "position_evals": ("tier4",),
    },
    5: {
        "openings": ("tier5",),
        "book_moves": ("tier5",),
    },
    6: {
        "endgame_positions": ("tier6",),
    },
    7: {
        "best_move_evals": ("tier7",),
        "consequence_evals": ("tier7",),
        "puzzles": ("tier7",),
    },
}

TIER_OPTIONAL_SOURCES: dict[int, dict[str, tuple[str, ...]]] = {
    7: {
        "mate_rows": ("tier7_optional",),
    },
}

STRICT_EVAL_REQUIRED_SOURCES: dict[str, tuple[str, ...]] = {
    "fen_pool": ("eval:perception", "eval:rules"),
    "fen_pool_chess960": ("eval:chess960",),
    "puzzles": ("eval:tactics", "eval:planning"),
    "openings": ("eval:openings",),
    "position_evals": ("eval:evaluation",),
    "best_move_evals": ("eval:planning",),
    "endgame_positions": ("eval:endgames",),
    "mate_rows": ("eval:mate",),
}

EVAL_SPLITS_BY_TIER: dict[int, tuple[str, ...]] = {
    1: ("perception",),
    2: ("rules",),
    3: ("tactics",),
    4: ("evaluation",),
    5: ("openings",),
    6: ("endgames",),
    7: ("planning", "mate"),
}


@dataclass(frozen=True)
class SourceReadinessIssue:
    """One missing or weak SFT source family."""

    source_key: str
    severity: str
    message: str
    required_by: tuple[str, ...]
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "severity": self.severity,
            "message": self.message,
            "required_by": list(self.required_by),
            "count": self.count,
        }


@dataclass(frozen=True)
class SourceReadinessReport:
    """Tier-aware source counts and generation blockers."""

    selected_tiers: tuple[int, ...]
    strict_eval_splits: bool
    counts: dict[str, int]
    required_sources: tuple[str, ...]
    optional_sources: tuple[str, ...]
    missing_required: tuple[SourceReadinessIssue, ...]
    optional_gaps: tuple[SourceReadinessIssue, ...]
    artifact_type: str = "sft_source_readiness"
    schema_version: str = "1.0"

    @property
    def ok(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "selected_tiers": list(self.selected_tiers),
            "strict_eval_splits": self.strict_eval_splits,
            "ok": self.ok,
            "counts": dict(sorted(self.counts.items())),
            "required_sources": list(self.required_sources),
            "optional_sources": list(self.optional_sources),
            "missing_required": [
                issue.to_dict() for issue in self.missing_required
            ],
            "optional_gaps": [issue.to_dict() for issue in self.optional_gaps],
        }

    def format_lines(self) -> list[str]:
        status = "ok" if self.ok else "missing required sources"
        lines = [
            f"SFT source readiness: {status}",
            "Source counts:",
        ]
        for key, count in sorted(self.counts.items()):
            lines.append(f"  {key}: {count}")
        if self.missing_required:
            lines.append("Missing required sources:")
            for issue in self.missing_required:
                required_by = ", ".join(issue.required_by)
                lines.append(
                    f"  {issue.source_key}: {issue.message} "
                    f"(required by {required_by})"
                )
        if self.optional_gaps:
            lines.append("Optional source gaps:")
            for issue in self.optional_gaps:
                required_by = ", ".join(issue.required_by)
                lines.append(
                    f"  {issue.source_key}: {issue.message} "
                    f"(used by {required_by})"
                )
        return lines


def build_source_readiness_report(
    config: Mapping[str, Any],
    *,
    tiers: list[int] | tuple[int, ...],
    strict_eval_splits: bool,
) -> SourceReadinessReport:
    """Build a tier-aware source-readiness report from loaded source config."""
    selected_tiers = tuple(sorted(set(tiers)))
    counts = source_counts(config)
    required = required_sources_for_tiers(selected_tiers)
    optional = optional_sources_for_tiers(selected_tiers)

    if strict_eval_splits:
        _merge_requirement_map(
            required,
            strict_eval_required_sources_for_tiers(selected_tiers),
        )

    missing_required = tuple(
        _missing_issue(
            source_key,
            required_by,
            counts,
            severity="error",
        )
        for source_key, required_by in sorted(required.items())
        if counts.get(source_key, 0) <= 0
    )
    optional_gaps = tuple(
        _missing_issue(
            source_key,
            required_by,
            counts,
            severity="warning",
        )
        for source_key, required_by in sorted(optional.items())
        if counts.get(source_key, 0) <= 0
    )

    return SourceReadinessReport(
        selected_tiers=selected_tiers,
        strict_eval_splits=strict_eval_splits,
        counts=counts,
        required_sources=tuple(sorted(required)),
        optional_sources=tuple(sorted(optional)),
        missing_required=missing_required,
        optional_gaps=optional_gaps,
    )


def source_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return source-family counts, including derived FEN-pool variant counts."""
    counts = {key: _count_value(config.get(key)) for key in SOURCE_KEYS}

    fen_pool = config.get("fen_pool", [])
    standard = 0
    chess960 = 0
    if isinstance(fen_pool, (list, tuple)):
        for row in fen_pool:
            if isinstance(row, Mapping) and raw_is_chess960(row):
                chess960 += 1
            else:
                standard += 1
    counts["fen_pool_standard"] = standard
    counts["fen_pool_chess960"] = chess960
    counts["fen_pool"] = counts.get("fen_pool", 0)
    return counts


def required_sources_for_tiers(tiers: tuple[int, ...]) -> dict[str, tuple[str, ...]]:
    """Return source requirements for the selected generation tiers."""
    required: dict[str, tuple[str, ...]] = {}
    for tier in tiers:
        _merge_requirement_map(required, TIER_REQUIRED_SOURCES.get(tier, {}))
    return required


def optional_sources_for_tiers(tiers: tuple[int, ...]) -> dict[str, tuple[str, ...]]:
    """Return non-blocking source gaps worth surfacing for selected tiers."""
    optional: dict[str, tuple[str, ...]] = {}
    for tier in tiers:
        _merge_requirement_map(optional, TIER_OPTIONAL_SOURCES.get(tier, {}))
    return optional


def eval_splits_for_tiers(tiers: tuple[int, ...] | list[int] | None) -> tuple[str, ...]:
    """Return benchmark split names required by a selected tier set.

    An empty tier set means an eval-only/full benchmark context, so every known
    strict eval split remains in scope.
    """
    selected_tiers = tuple(sorted(set(tiers or ())))
    if not selected_tiers or set(selected_tiers) == set(EVAL_SPLITS_BY_TIER):
        return _all_strict_eval_splits()

    selected: set[str] = set()
    for tier in selected_tiers:
        selected.update(EVAL_SPLITS_BY_TIER.get(tier, ()))

    return tuple(split for split in _all_strict_eval_splits() if split in selected)


def strict_eval_required_sources_for_tiers(
    tiers: tuple[int, ...] | list[int] | None,
) -> dict[str, tuple[str, ...]]:
    """Return strict eval source requirements scoped to selected tiers."""
    selected_splits = {f"eval:{split}" for split in eval_splits_for_tiers(tiers)}
    required: dict[str, tuple[str, ...]] = {}
    for source_key, required_by in STRICT_EVAL_REQUIRED_SOURCES.items():
        refs = tuple(ref for ref in required_by if ref in selected_splits)
        if refs:
            required[source_key] = refs
    return required


def write_source_readiness_report(
    report: SourceReadinessReport,
    output_path: str | Path,
) -> None:
    """Write a JSON source-readiness manifest."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _missing_issue(
    source_key: str,
    required_by: tuple[str, ...],
    counts: Mapping[str, int],
    *,
    severity: str,
) -> SourceReadinessIssue:
    return SourceReadinessIssue(
        source_key=source_key,
        severity=severity,
        message=f"loaded 0 rows for {source_key}",
        required_by=tuple(sorted(required_by)),
        count=counts.get(source_key, 0),
    )


def _merge_requirement_map(
    target: dict[str, tuple[str, ...]],
    incoming: Mapping[str, tuple[str, ...]],
) -> None:
    for source_key, required_by in incoming.items():
        existing = set(target.get(source_key, ()))
        existing.update(required_by)
        target[source_key] = tuple(sorted(existing))


def _count_value(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Sized):
        return len(value)
    return 0


def _all_strict_eval_splits() -> tuple[str, ...]:
    seen: set[str] = set()
    splits: list[str] = []
    for required_by in STRICT_EVAL_REQUIRED_SOURCES.values():
        for ref in required_by:
            split = ref.removeprefix("eval:")
            if split not in seen:
                seen.add(split)
                splits.append(split)
    return tuple(splits)


__all__ = [
    "SOURCE_KEYS",
    "STRICT_EVAL_REQUIRED_SOURCES",
    "EVAL_SPLITS_BY_TIER",
    "TIER_OPTIONAL_SOURCES",
    "TIER_REQUIRED_SOURCES",
    "SourceReadinessIssue",
    "SourceReadinessReport",
    "build_source_readiness_report",
    "eval_splits_for_tiers",
    "optional_sources_for_tiers",
    "required_sources_for_tiers",
    "source_counts",
    "strict_eval_required_sources_for_tiers",
    "write_source_readiness_report",
]
