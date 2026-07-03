"""Trainer callback tracking the tier-mix schedule during one long run.

This module imports transformers, so it must only be imported lazily inside
``train.main`` — the training entrypoints have to keep importing without
heavy GPU dependencies.

All numeric logic delegates to :mod:`chess_llm.training.schedule`. W&B sees
the injected ``schedule/*`` log keys because ``train.main`` repositions this
callback ahead of the report_to integration callbacks (chosen over having
the callback call ``wandb.log`` directly so the metrics stay on the
Trainer's own step axis).
"""

from __future__ import annotations

import logging
from typing import Sequence

from transformers import TrainerCallback

from chess_llm.training.schedule import (
    SegmentPlan,
    replay_fraction_t12,
    segment_index_at_step,
)

logger = logging.getLogger(__name__)


class MixingScheduleCallback(TrainerCallback):
    """Log schedule telemetry and force eval/save at segment boundaries."""

    def __init__(
        self,
        plans: Sequence[SegmentPlan],
        boundary_steps: Sequence[int],
        save_at_boundaries: bool = True,
        evaluate_at_boundaries: bool = True,
    ) -> None:
        if len(plans) != len(boundary_steps):
            raise ValueError(
                f"plans ({len(plans)}) and boundary_steps ({len(boundary_steps)}) "
                "must have the same length"
            )
        self.plans = list(plans)
        self.boundary_steps = list(boundary_steps)
        self.save_at_boundaries = save_at_boundaries
        self.evaluate_at_boundaries = evaluate_at_boundaries
        self._all_tiers = sorted(
            {tier for plan in self.plans for tier, _ in plan.tier_weights}
        )
        # The final boundary coincides with the end of training, where the
        # Trainer already saves/evaluates through its own end-of-train path.
        self._interior_boundaries = set(self.boundary_steps[:-1])

    def on_train_begin(self, args, state, control, **kwargs):
        last_boundary = self.boundary_steps[-1] if self.boundary_steps else 0
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps and last_boundary and max_steps != last_boundary:
            logger.warning(
                "Schedule drift: the planned final boundary is step %d but the "
                "Trainer reports max_steps=%d. Logged segment indices and "
                "boundary saves may not line up with the planned data order "
                "(check effective batch size and max_steps overrides).",
                last_boundary,
                max_steps,
            )
        self._update_wandb_config()
        return control

    def _update_wandb_config(self) -> None:
        try:
            import wandb
        except Exception:
            return
        run = getattr(wandb, "run", None)
        if run is None:
            return
        try:
            run.config.update(
                {
                    "schedule/segment_names": [plan.name for plan in self.plans],
                    "schedule/boundary_steps": list(self.boundary_steps),
                    "schedule/total_rows": (
                        self.plans[-1].end_row if self.plans else 0
                    ),
                },
                allow_val_change=True,
            )
        except Exception:
            logger.warning(
                "Could not record the schedule plan in the W&B run config",
                exc_info=True,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        index = segment_index_at_step(step, self.boundary_steps)
        plan = self.plans[index]
        weights = dict(plan.tier_weights)
        logs["schedule/segment_index"] = index
        for tier in self._all_tiers:
            logs[f"schedule/tier_weight_{tier}"] = float(weights.get(tier, 0.0))
        logs["schedule/replay_fraction_t12"] = replay_fraction_t12(plan.tier_weights)
        logs["schedule/segment_progress"] = self._segment_progress(step, index)
        return control

    def _segment_progress(self, step: int, index: int) -> float:
        start = self.boundary_steps[index - 1] if index > 0 else 0
        span = self.boundary_steps[index] - start
        if span <= 0:
            return 1.0
        return min(1.0, max(0.0, (step - start) / span))

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if step in self._interior_boundaries:
            index = self.boundary_steps.index(step)
            next_name = (
                self.plans[index + 1].name
                if index + 1 < len(self.plans)
                else "<end>"
            )
            logger.info(
                "Schedule boundary at step %d: segment %r complete, entering %r",
                step,
                self.plans[index].name,
                next_name,
            )
            if self.save_at_boundaries:
                control.should_save = True
            if self.evaluate_at_boundaries:
                control.should_evaluate = True
        return control


__all__ = ["MixingScheduleCallback"]
