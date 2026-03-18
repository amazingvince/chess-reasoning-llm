"""Pipeline statistics tracking and reporting."""

from __future__ import annotations

from collections import defaultdict

from config.settings import VOLUMES, CHESS960_RATIOS


class PipelineStats:
    """Accumulates counts per task and Chess960 flag during generation."""

    def __init__(self) -> None:
        self._total: dict[str, int] = defaultdict(int)
        self._chess960: dict[str, int] = defaultdict(int)
        self._passed: dict[str, int] = defaultdict(int)
        self._failed: dict[str, int] = defaultdict(int)

    def record(
        self,
        task_id: str,
        is_chess960: bool = False,
        passed_validation: bool = True,
    ) -> None:
        self._total[task_id] += 1
        if is_chess960:
            self._chess960[task_id] += 1
        if passed_validation:
            self._passed[task_id] += 1
        else:
            self._failed[task_id] += 1

    def report(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"{'Task':<30} {'Target':>8} {'Total':>8} {'Pass':>8} {'Fail':>8} {'960%':>6}",
            "-" * 76,
        ]
        for task_id in sorted(self._total):
            target = VOLUMES.get(task_id, 0)
            total = self._total[task_id]
            passed = self._passed[task_id]
            failed = self._failed[task_id]
            c960_pct = (
                f"{self._chess960[task_id] / total * 100:.1f}%"
                if total > 0
                else "n/a"
            )
            lines.append(
                f"{task_id:<30} {target:>8} {total:>8} {passed:>8} {failed:>8} {c960_pct:>6}"
            )
        return "\n".join(lines)

    def verify_chess960_mix(self) -> dict[str, dict[str, float]]:
        """Per-tier actual vs. target Chess960 ratios.

        Returns ``{tier: {target, actual, delta}}``.
        """
        tier_totals: dict[int, int] = defaultdict(int)
        tier_960: dict[int, int] = defaultdict(int)

        for task_id, total in self._total.items():
            tier = int(task_id.split(".")[0])
            tier_totals[tier] += total
            tier_960[tier] += self._chess960.get(task_id, 0)

        result: dict[str, dict[str, float]] = {}
        for tier in sorted(tier_totals):
            target = CHESS960_RATIOS.get(tier, 0.0)
            total = tier_totals[tier]
            actual = tier_960[tier] / total if total else 0.0
            result[f"tier_{tier}"] = {
                "target": target,
                "actual": round(actual, 4),
                "delta": round(actual - target, 4),
            }
        return result
