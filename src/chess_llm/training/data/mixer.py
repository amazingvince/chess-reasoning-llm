"""Phase-aware SFT dataset mixing and train/eval splitting."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Mapping, Protocol

from datasets import Dataset, concatenate_datasets

from chess_llm.core.board import variant_fen_key
from chess_llm.sft.context import raw_fen_identity_key
from chess_llm.training.data.loader import load_tier_data

logger = logging.getLogger(__name__)


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


def _select_eval_fen_keys(
    rows: list[str | Mapping[str, Any]],
    eval_fraction: float,
    seed: int,
) -> set[str]:
    """Select eval-side FEN keys using a group-sizing heuristic."""
    if eval_fraction <= 0.0:
        return set()

    rng = random.Random(seed)
    key_to_count: dict[str, int] = {}
    for row in rows:
        key = _fen_key(row)
        key_to_count[key] = key_to_count.get(key, 0) + 1

    groups = list(key_to_count.items())
    rng.shuffle(groups)
    groups.sort(key=lambda group: group[1])

    n_total = len(rows)
    n_eval_target = int(n_total * eval_fraction)
    eval_keys: set[str] = set()
    n_eval_so_far = 0

    for key, count in groups:
        overshoot = (n_eval_so_far + count) - n_eval_target
        undershoot = n_eval_target - n_eval_so_far

        if undershoot > 0 and overshoot <= undershoot:
            eval_keys.add(key)
            n_eval_so_far += count

    if not eval_keys and n_eval_target >= 1 and len(groups) >= 2:
        smallest_key, _count = min(groups, key=lambda group: group[1])
        eval_keys.add(smallest_key)

    return eval_keys


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
) -> tuple[Dataset, Dataset]:
    """Build train/eval datasets for one curriculum phase."""
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []
    task_upsample = _normalize_task_upsample(task_upsample)

    loaded_tiers: list[tuple[_TierMixLike, Dataset]] = []
    all_rows: list[dict] = []
    for tier_mix in phase.tier_mix:
        ds = load_tier_data(tier_mix.tier, data_root)
        loaded_tiers.append((tier_mix, ds))
        all_rows.extend(list(ds))

    phase_eval_keys = _select_eval_fen_keys(all_rows, eval_fraction, seed)

    for tier_mix, ds in loaded_tiers:
        if eval_fraction <= 0.0:
            tier_train, tier_eval = ds, ds.select([])
        else:
            train_indices: list[int] = []
            eval_indices: list[int] = []
            for idx, row in enumerate(ds):
                if _fen_key(row) in phase_eval_keys:
                    eval_indices.append(idx)
                else:
                    train_indices.append(idx)
            tier_train, tier_eval = ds.select(train_indices), ds.select(eval_indices)

        if tier_mix.fraction < 1.0:
            n_samples = int(len(tier_train) * tier_mix.fraction)
            tier_train = tier_train.shuffle(seed=seed).select(range(n_samples))
            logger.info(
                "Tier %d: sampled %d (%.0f%% of train split)",
                tier_mix.tier,
                n_samples,
                tier_mix.fraction * 100,
            )

        tier_train = _apply_task_upsampling(
            tier_train,
            task_upsample,
            tier=tier_mix.tier,
        )

        if tier_mix.upsample > 1:
            original_len = len(tier_train)
            tier_train = concatenate_datasets([tier_train] * tier_mix.upsample)
            logger.info(
                "Tier %d: upsampled %dx (%d -> %d)",
                tier_mix.tier,
                tier_mix.upsample,
                original_len,
                len(tier_train),
            )

        train_parts.append(tier_train)
        if len(tier_eval) > 0:
            eval_parts.append(tier_eval)

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
) -> dict[str, int]:
    """Compute expected example counts per tier without building train/eval splits."""
    summary: dict[str, int] = {}
    total = 0
    task_upsample = _normalize_task_upsample(task_upsample)

    for tier_mix in phase.tier_mix:
        ds = load_tier_data(tier_mix.tier, data_root)
        raw_count = len(ds)

        sampled = int(raw_count * tier_mix.fraction) if tier_mix.fraction < 1.0 else raw_count
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
    "_fen_key",
    "_apply_task_upsampling",
    "_estimated_task_upsample_extra",
    "_normalize_task_upsample",
    "_select_eval_fen_keys",
    "_split_by_fen",
    "build_phase_dataset",
    "summarize_phase_data",
]
