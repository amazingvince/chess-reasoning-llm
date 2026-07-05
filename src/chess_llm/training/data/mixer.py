"""Phase-aware SFT dataset mixing and train/eval splitting."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Mapping, Protocol

from datasets import Dataset, concatenate_datasets

from chess_llm.core.board import variant_fen_key
from chess_llm.sft.context import raw_fen_identity_key
from chess_llm.training.data.loader import (
    TrainingDataTransformConfig,
    load_tier_data,
)
from chess_llm.training.schedule import (
    ScheduleConfig,
    SegmentPlan,
    boundary_steps,
    plan_segments,
    schedule_tiers,
)

logger = logging.getLogger(__name__)

_DROPPED_METADATA_COLUMNS = ("fen", "metadata", "is_chess960", "chess960_id")

_EVAL_FEN_KEYS_FILENAME = "eval_fen_keys.json"


class _TierMixLike(Protocol):
    tier: int
    fraction: float
    upsample: int


class _PhaseLike(Protocol):
    name: str
    display_name: str
    tier_mix: tuple[_TierMixLike, ...]


def _fen_key(row: str | Mapping[str, Any]) -> str:
    """Normalize FEN for split identity by ignoring move counters."""
    if isinstance(row, Mapping):
        return raw_fen_identity_key(row)
    return variant_fen_key(row)


def _dataset_fen_keys(
    ds: Dataset,
    cache: dict[tuple[str, bool], str],
) -> list[str]:
    """Derive per-row FEN keys from stored columns without re-reading rows.

    Sanitized loader rows already mirror Chess960 markers into
    ``is_chess960``/``chess960_id``, so identical (fen, variant) pairs can
    share one memoized key instead of re-parsing a board per row.
    """
    n_rows = len(ds)
    columns = ds.column_names
    fens = ds["fen"] if "fen" in columns else [""] * n_rows
    flags = ds["is_chess960"] if "is_chess960" in columns else [False] * n_rows
    ids = ds["chess960_id"] if "chess960_id" in columns else [None] * n_rows

    keys: list[str] = []
    for fen, flag, chess960_id in zip(fens, flags, ids):
        chess960 = bool(flag) or chess960_id is not None
        memo_key = (str(fen), chess960)
        key = cache.get(memo_key)
        if key is None:
            key = variant_fen_key(str(fen), chess960=chess960)
            cache[memo_key] = key
        keys.append(key)
    return keys


def _select_eval_fen_keys_for_tasks(
    keys: list[str],
    tasks: list[str],
    eval_fraction: float,
    seed: int,
    existing_keys: frozenset[str] = frozenset(),
) -> set[str]:
    """Select eval-side FEN keys stratified per task.

    Each task with data gets an eval share proportional to its row count
    (minimum one row), while all rows of a chosen FEN stay on the eval side.
    Keys in ``existing_keys`` count toward each task's target but are not
    re-selected.
    """
    if eval_fraction <= 0.0:
        return set()

    rng = random.Random(seed)
    task_key_counts: dict[str, dict[str, int]] = {}
    for key, task in zip(keys, tasks):
        counts = task_key_counts.setdefault(task, {})
        counts[key] = counts.get(key, 0) + 1

    selected: set[str] = set()
    for task in sorted(task_key_counts):
        counts = task_key_counts[task]
        n_task_rows = sum(counts.values())
        n_eval_target = max(1, int(n_task_rows * eval_fraction))
        n_eval_so_far = sum(
            count
            for key, count in counts.items()
            if key in existing_keys or key in selected
        )

        groups = [
            (key, count)
            for key, count in counts.items()
            if key not in existing_keys and key not in selected
        ]
        rng.shuffle(groups)
        groups.sort(key=lambda group: group[1])

        for key, count in groups:
            overshoot = (n_eval_so_far + count) - n_eval_target
            undershoot = n_eval_target - n_eval_so_far

            if undershoot > 0 and overshoot <= undershoot:
                selected.add(key)
                n_eval_so_far += count

        if n_eval_so_far == 0 and len(groups) >= 2:
            smallest_key, _count = min(groups, key=lambda group: group[1])
            selected.add(smallest_key)

    return selected


def _select_eval_fen_keys(
    rows: list[str | Mapping[str, Any]],
    eval_fraction: float,
    seed: int,
) -> set[str]:
    """Select eval-side FEN keys for row-like inputs, stratified per task."""
    keys = [_fen_key(row) for row in rows]
    tasks = [
        str(row.get("task", "")) if isinstance(row, Mapping) else ""
        for row in rows
    ]
    return _select_eval_fen_keys_for_tasks(keys, tasks, eval_fraction, seed)


def _load_or_extend_eval_fen_keys(
    data_root: Path,
    tier_rows: list[tuple[int, list[str], list[str]]],
    eval_fraction: float,
    seed: int,
) -> set[str]:
    """Load the persisted eval FEN key set, selecting keys for unseen tiers.

    The selection is persisted under ``data_root`` on first build so a FEN
    that was ever assigned to eval stays on the eval side in later phase
    builds; tiers not seen before only add keys.
    """
    state_path = data_root / _EVAL_FEN_KEYS_FILENAME
    state: dict[str, Any] = {}
    if state_path.exists():
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)

    tiers_seen = {int(tier) for tier in state.get("tiers_seen", [])}
    eval_keys = set(state.get("eval_keys", []))

    new_tiers: list[int] = []
    new_keys: list[str] = []
    new_tasks: list[str] = []
    for tier, keys, tasks in tier_rows:
        if tier in tiers_seen:
            continue
        new_tiers.append(tier)
        new_keys.extend(keys)
        new_tasks.extend(tasks)

    if not new_tiers:
        return eval_keys

    added = _select_eval_fen_keys_for_tasks(
        new_keys,
        new_tasks,
        eval_fraction,
        seed,
        existing_keys=frozenset(eval_keys),
    )
    eval_keys |= added
    tiers_seen.update(new_tiers)

    payload = {
        "version": 1,
        "seed": seed,
        "eval_fraction": eval_fraction,
        "tiers_seen": sorted(tiers_seen),
        "eval_keys": sorted(eval_keys),
    }
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp_path.replace(state_path)
    logger.info(
        "Eval FEN keys: %d total (%d added for tiers %s) -> %s",
        len(eval_keys),
        len(added),
        new_tiers,
        state_path,
    )
    return eval_keys


def _sampled_count(n_rows: int, fraction: float) -> int:
    """Rounded sample size that keeps at least one row when fraction > 0."""
    if n_rows <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return n_rows
    return max(1, min(n_rows, round(n_rows * fraction)))


def _split_by_fen(
    ds: Dataset,
    eval_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a dataset so no FEN appears on both sides."""
    if eval_fraction <= 0.0:
        return ds, ds.select([])

    rows = list(ds)
    eval_keys = _select_eval_fen_keys(rows, eval_fraction, seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for idx, row in enumerate(rows):
        if _fen_key(row) in eval_keys:
            eval_indices.append(idx)
        else:
            train_indices.append(idx)

    return ds.select(train_indices), ds.select(eval_indices)


def build_phase_dataset(
    phase: _PhaseLike,
    data_root: Path,
    eval_fraction: float = 0.02,
    seed: int = 42,
    task_upsample: Mapping[str, int] | None = None,
    data_transform_config: TrainingDataTransformConfig | None = None,
) -> tuple[Dataset, Dataset]:
    """Build train/eval datasets for one curriculum phase."""
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []
    task_upsample = _normalize_task_upsample(task_upsample)

    fen_key_cache: dict[tuple[str, bool], str] = {}
    loaded_tiers: list[tuple[_TierMixLike, Dataset, list[str]]] = []
    tier_rows: list[tuple[int, list[str], list[str]]] = []
    for tier_mix in phase.tier_mix:
        ds = load_tier_data(
            tier_mix.tier,
            data_root,
            transform_config=data_transform_config,
        )
        if eval_fraction <= 0.0:
            loaded_tiers.append((tier_mix, ds, []))
            continue
        keys = _dataset_fen_keys(ds, fen_key_cache)
        tasks = (
            [str(task) for task in ds["task"]]
            if "task" in ds.column_names
            else [""] * len(ds)
        )
        loaded_tiers.append((tier_mix, ds, keys))
        tier_rows.append((tier_mix.tier, keys, tasks))

    if eval_fraction <= 0.0:
        phase_eval_keys: set[str] = set()
    else:
        phase_eval_keys = _load_or_extend_eval_fen_keys(
            data_root,
            tier_rows,
            eval_fraction,
            seed,
        )

    for tier_mix, ds, keys in loaded_tiers:
        if eval_fraction <= 0.0:
            tier_train, tier_eval = ds, ds.select([])
        else:
            train_indices: list[int] = []
            eval_indices: list[int] = []
            for idx, key in enumerate(keys):
                if key in phase_eval_keys:
                    eval_indices.append(idx)
                else:
                    train_indices.append(idx)
            tier_train, tier_eval = ds.select(train_indices), ds.select(eval_indices)

        if tier_mix.fraction < 1.0:
            n_samples = _sampled_count(len(tier_train), tier_mix.fraction)
            tier_train = tier_train.shuffle(seed=seed).select(range(n_samples))
            n_eval_samples = _sampled_count(len(tier_eval), tier_mix.fraction)
            tier_eval = tier_eval.shuffle(seed=seed).select(range(n_eval_samples))
            logger.info(
                "Tier %d: sampled %d train / %d eval (%.0f%% of splits)",
                tier_mix.tier,
                n_samples,
                n_eval_samples,
                tier_mix.fraction * 100,
            )

        tier_train = _apply_task_upsampling(
            tier_train,
            task_upsample,
            tier=tier_mix.tier,
        )

        if tier_mix.upsample > 1 and len(tier_train) > 0:
            original_len = len(tier_train)
            tier_train = concatenate_datasets([tier_train] * tier_mix.upsample)
            logger.info(
                "Tier %d: upsampled %dx (%d -> %d)",
                tier_mix.tier,
                tier_mix.upsample,
                original_len,
                len(tier_train),
            )

        if len(tier_train) > 0:
            train_parts.append(tier_train)
        if len(tier_eval) > 0:
            eval_parts.append(tier_eval)

    if not train_parts:
        raise ValueError(
            f"Phase {phase.name!r} has no training examples after data transforms."
        )
    train_ds = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_ds = concatenate_datasets(eval_parts) if eval_parts else train_ds.select([])

    train_drop_cols = [
        name
        for name in ("fen", "metadata", "is_chess960", "chess960_id")
        if name in train_ds.column_names
    ]
    eval_drop_cols = [
        name
        for name in ("fen", "metadata", "is_chess960", "chess960_id")
        if name in eval_ds.column_names
    ]
    if train_drop_cols:
        train_ds = train_ds.remove_columns(train_drop_cols)
    if eval_drop_cols:
        eval_ds = eval_ds.remove_columns(eval_drop_cols)

    logger.info(
        "Phase %s (%s): %d train, %d eval",
        phase.name,
        phase.display_name,
        len(train_ds),
        len(eval_ds),
    )

    return train_ds, eval_ds


def summarize_phase_data(
    phase: _PhaseLike,
    data_root: Path,
    task_upsample: Mapping[str, int] | None = None,
    data_transform_config: TrainingDataTransformConfig | None = None,
) -> dict[str, int]:
    """Compute expected example counts per tier without building train/eval splits."""
    summary: dict[str, int] = {}
    total = 0
    task_upsample = _normalize_task_upsample(task_upsample)

    for tier_mix in phase.tier_mix:
        ds = load_tier_data(
            tier_mix.tier,
            data_root,
            transform_config=data_transform_config,
        )
        raw_count = len(ds)

        sampled = _sampled_count(raw_count, tier_mix.fraction)
        task_extra = _estimated_task_upsample_extra(
            ds,
            task_upsample,
            fraction=tier_mix.fraction,
        )
        final = (sampled + task_extra) * tier_mix.upsample
        summary[f"tier_{tier_mix.tier}"] = final
        total += final

        logger.info(
            "Tier %d: %d raw -> %d sampled (%.0f%%) + %d task-upsample -> %d final (%dx upsample)",
            tier_mix.tier,
            raw_count,
            sampled,
            tier_mix.fraction * 100,
            task_extra,
            final,
            tier_mix.upsample,
        )

    summary["total"] = total
    return summary


class _IndexCycler:
    """Seeded shuffled index stream that only repeats after pool exhaustion.

    Draws are sequential across segments; the pool is reshuffled and reused
    only once every index has been handed out, so duplicates appear only
    when total demand exceeds the pool size.
    """

    def __init__(self, n_rows: int, rng: random.Random) -> None:
        self._n_rows = n_rows
        self._rng = rng
        self._order = list(range(n_rows))
        self._rng.shuffle(self._order)
        self._position = 0

    def draw(self, count: int) -> list[int]:
        indices: list[int] = []
        while len(indices) < count:
            if self._position >= self._n_rows:
                self._rng.shuffle(self._order)
                self._position = 0
            indices.append(self._order[self._position])
            self._position += 1
        return indices


def _drop_metadata_columns(ds: Dataset) -> Dataset:
    """Drop split-identity columns, keeping task and tier for telemetry."""
    drop_cols = [name for name in _DROPPED_METADATA_COLUMNS if name in ds.column_names]
    return ds.remove_columns(drop_cols) if drop_cols else ds


def build_schedule_dataset(
    schedule: ScheduleConfig,
    data_root: Path,
    eval_fraction: float = 0.02,
    seed: int = 42,
    task_upsample: Mapping[str, int] | None = None,
    total_examples: int | None = None,
    data_transform_config: TrainingDataTransformConfig | None = None,
) -> tuple[Dataset, Dataset, list[SegmentPlan]]:
    """Build one sequential train dataset following the schedule's segments.

    The eval FEN keys are selected ONCE for every tier the schedule will
    ever touch, so the eval split covers all tiers from step 0 and later
    segments cannot leak eval FENs into training.

    The returned train dataset is already in schedule order (per-segment
    tier draws, shuffled within each segment); it must be trained with a
    sequential sampler and never reshuffled.
    """
    task_upsample = _normalize_task_upsample(task_upsample)
    tiers_needed = schedule_tiers(schedule.segments)

    fen_key_cache: dict[tuple[str, bool], str] = {}
    loaded_tiers: list[tuple[int, Dataset, list[str]]] = []
    tier_rows_for_state: list[tuple[int, list[str], list[str]]] = []
    for tier in tiers_needed:
        ds = load_tier_data(
            tier,
            data_root,
            transform_config=data_transform_config,
        )
        if eval_fraction <= 0.0:
            loaded_tiers.append((tier, ds, []))
            continue
        keys = _dataset_fen_keys(ds, fen_key_cache)
        tasks = (
            [str(task) for task in ds["task"]]
            if "task" in ds.column_names
            else [""] * len(ds)
        )
        loaded_tiers.append((tier, ds, keys))
        tier_rows_for_state.append((tier, keys, tasks))

    if eval_fraction <= 0.0:
        eval_keys: set[str] = set()
    else:
        eval_keys = _load_or_extend_eval_fen_keys(
            data_root,
            tier_rows_for_state,
            eval_fraction,
            seed,
        )

    train_pools: list[tuple[int, Dataset]] = []
    eval_parts: list[Dataset] = []
    for tier, ds, keys in loaded_tiers:
        if eval_fraction <= 0.0:
            tier_train, tier_eval = ds, ds.select([])
        else:
            train_indices: list[int] = []
            eval_indices: list[int] = []
            for idx, key in enumerate(keys):
                if key in eval_keys:
                    eval_indices.append(idx)
                else:
                    train_indices.append(idx)
            tier_train, tier_eval = ds.select(train_indices), ds.select(eval_indices)

        tier_train = _apply_task_upsampling(tier_train, task_upsample, tier=tier)
        train_pools.append((tier, tier_train))
        if len(tier_eval) > 0:
            eval_parts.append(tier_eval)

    pool_sizes = {tier: len(pool) for tier, pool in train_pools}
    total = total_examples if total_examples is not None else sum(pool_sizes.values())
    plans = plan_segments(schedule.segments, total)

    demand: dict[int, int] = {}
    for plan in plans:
        for tier, n_rows in plan.tier_rows:
            demand[tier] = demand.get(tier, 0) + n_rows
    for tier, n_rows in sorted(demand.items()):
        if n_rows > 0 and pool_sizes.get(tier, 0) == 0:
            raise ValueError(
                f"Schedule {schedule.name!r} needs {n_rows} train rows from "
                f"tier {tier}, but its train pool is empty after the eval split."
            )

    offsets: dict[int, int] = {}
    running = 0
    for tier, pool in train_pools:
        offsets[tier] = running
        running += len(pool)
    cyclers = {
        tier: _IndexCycler(len(pool), random.Random(seed * 31 + tier))
        for tier, pool in train_pools
    }

    order: list[int] = []
    for plan in plans:
        segment_indices: list[int] = []
        for tier, n_rows in plan.tier_rows:
            if n_rows <= 0:
                continue
            offset = offsets[tier]
            segment_indices.extend(
                offset + idx for idx in cyclers[tier].draw(n_rows)
            )
        random.Random(seed * 1009 + plan.index).shuffle(segment_indices)
        order.extend(segment_indices)
        logger.info(
            "Segment %d (%s): rows %d..%d, tiers %s",
            plan.index,
            plan.name,
            plan.start_row,
            plan.end_row,
            dict(plan.tier_rows),
        )

    nonempty_train_pools = [pool for _, pool in train_pools if len(pool) > 0]
    if not nonempty_train_pools:
        raise ValueError(
            f"Schedule {schedule.name!r} has no training examples after data transforms."
        )
    all_train = concatenate_datasets(nonempty_train_pools)
    # Segment-ordered selection; no final shuffle — the order IS the schedule.
    train_ds = all_train.select(order)
    eval_ds = (
        concatenate_datasets(eval_parts) if eval_parts else all_train.select([])
    )

    train_ds = _drop_metadata_columns(train_ds)
    eval_ds = _drop_metadata_columns(eval_ds)

    logger.info(
        "Schedule %s (%s): %d train (budget %d), %d eval across tiers %s",
        schedule.name,
        schedule.display_name,
        len(train_ds),
        total,
        len(eval_ds),
        tiers_needed,
    )
    return train_ds, eval_ds, plans


def summarize_schedule_data(
    schedule: ScheduleConfig,
    data_root: Path,
    task_upsample: Mapping[str, int] | None = None,
    total_examples: int | None = None,
    data_transform_config: TrainingDataTransformConfig | None = None,
) -> dict[str, Any]:
    """Dry-run summary: per-segment per-tier row counts plus boundary steps."""
    task_upsample = _normalize_task_upsample(task_upsample)

    pool_sizes: dict[int, int] = {}
    for tier in schedule_tiers(schedule.segments):
        ds = load_tier_data(
            tier,
            data_root,
            transform_config=data_transform_config,
        )
        task_extra = _estimated_task_upsample_extra(ds, task_upsample, fraction=1.0)
        pool_sizes[tier] = len(ds) + task_extra

    total = total_examples if total_examples is not None else sum(pool_sizes.values())
    plans = plan_segments(schedule.segments, total)
    boundaries = boundary_steps(plans)

    return {
        "total_examples": total,
        "tier_pool_sizes": {f"tier_{tier}": size for tier, size in pool_sizes.items()},
        "boundary_steps": boundaries,
        "segments": [
            {
                "index": plan.index,
                "name": plan.name,
                "start_row": plan.start_row,
                "end_row": plan.end_row,
                "rows": plan.end_row - plan.start_row,
                "tier_rows": {
                    f"tier_{tier}": n_rows for tier, n_rows in plan.tier_rows
                },
            }
            for plan in plans
        ],
    }


def _normalize_task_upsample(
    task_upsample: Mapping[str, int] | None,
) -> dict[str, int]:
    """Keep only task upsample factors that add copies."""
    if not task_upsample:
        return {}
    return {
        task: int(factor)
        for task, factor in task_upsample.items()
        if task and int(factor) > 1
    }


def _apply_task_upsampling(
    ds: Dataset,
    task_upsample: Mapping[str, int],
    *,
    tier: int,
) -> Dataset:
    """Upsample selected task rows on the train split only."""
    if not task_upsample or "task" not in ds.column_names:
        return ds

    extra_parts: list[Dataset] = []
    for task, factor in sorted(task_upsample.items()):
        indices = [
            idx
            for idx, row_task in enumerate(ds["task"])
            if row_task == task
        ]
        if not indices:
            continue
        selected = ds.select(indices)
        extra_parts.extend([selected] * (factor - 1))
        logger.info(
            "Tier %d: task %s upsampled %dx (%d -> %d train rows)",
            tier,
            task,
            factor,
            len(indices),
            len(indices) * factor,
        )

    if not extra_parts:
        return ds
    return concatenate_datasets([ds, *extra_parts])


def _estimated_task_upsample_extra(
    ds: Dataset,
    task_upsample: Mapping[str, int],
    *,
    fraction: float,
) -> int:
    """Estimate extra rows added by train-only task upsampling."""
    if not task_upsample or "task" not in ds.column_names:
        return 0
    counts: dict[str, int] = {}
    for task in ds["task"]:
        counts[task] = counts.get(task, 0) + 1
    extra = 0
    for task, factor in task_upsample.items():
        task_count = counts.get(task, 0)
        sampled_count = (
            int(task_count * fraction)
            if fraction < 1.0
            else task_count
        )
        extra += sampled_count * (factor - 1)
    return extra


__all__ = [
    "_EVAL_FEN_KEYS_FILENAME",
    "_fen_key",
    "_apply_task_upsampling",
    "_dataset_fen_keys",
    "_estimated_task_upsample_extra",
    "_load_or_extend_eval_fen_keys",
    "_normalize_task_upsample",
    "_sampled_count",
    "_select_eval_fen_keys",
    "_select_eval_fen_keys_for_tasks",
    "_split_by_fen",
    "build_phase_dataset",
    "build_schedule_dataset",
    "summarize_phase_data",
    "summarize_schedule_data",
]
