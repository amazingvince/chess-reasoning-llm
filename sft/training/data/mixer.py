"""Phase-aware data mixing: sampling, upsampling, and train/eval splitting.

Eval split strategy (prevents leakage):
  1. Each tier is split into train/eval on *unique FENs* before any
     sampling or upsampling.  All examples sharing a FEN go to the same
     side, so no position appears in both train and eval.
  2. Fractional sampling and upsampling are applied only to the train
     portion.
  3. All tier train portions are concatenated and shuffled; likewise
     for eval.
  4. The ``fen`` column is dropped from both outputs — it is not
     needed by the trainer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from config.phases import PhaseConfig
from data.loader import load_tier_data

logger = logging.getLogger(__name__)


def _fen_key(fen: str) -> str:
    """Normalize FEN for split identity by ignoring move counters."""
    parts = fen.strip().split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return fen.strip()


def _select_eval_fen_keys(
    fens: list[str],
    eval_fraction: float,
    seed: int,
) -> set[str]:
    """Select eval-side FEN keys using the same group-sizing heuristic."""
    if eval_fraction <= 0.0:
        return set()

    import random

    rng = random.Random(seed)
    key_to_count: dict[str, int] = {}
    for fen in fens:
        key = _fen_key(fen)
        key_to_count[key] = key_to_count.get(key, 0) + 1

    groups = list(key_to_count.items())
    rng.shuffle(groups)
    groups.sort(key=lambda g: g[1])

    n_total = len(fens)
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
        smallest_key, _count = min(groups, key=lambda g: g[1])
        eval_keys.add(smallest_key)

    return eval_keys


def _split_by_fen(
    ds: Dataset,
    eval_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a dataset so no FEN appears on both sides.

    Groups rows by ``fen``, then assigns whole groups to eval until
    the target row count is reached.  Groups are sorted smallest-first
    and shuffled within each size bucket so that the eval set stays
    close to the requested fraction even when some FENs have many rows.

    The assignment stops when adding the *next* group would overshoot
    the target by more than it would undershoot by skipping it.
    """
    if eval_fraction <= 0.0:
        return ds, ds.select([])

    fens: list[str] = ds["fen"]
    eval_keys = _select_eval_fen_keys(fens, eval_fraction, seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for idx, fen in enumerate(fens):
        if _fen_key(fen) in eval_keys:
            eval_indices.append(idx)
        else:
            train_indices.append(idx)

    return ds.select(train_indices), ds.select(eval_indices)


def build_phase_dataset(
    phase: PhaseConfig,
    data_root: Path,
    eval_fraction: float = 0.02,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Build train/eval datasets for a phase.

    For each ``TierMix`` in ``phase.tier_mix``:
      1. Load tier data (retains ``fen`` column for dedup)
      2. Split into train/eval on unique FENs (no position leaks)
      3. Sample ``fraction`` of the *train* portion (if < 1.0)
      4. Upsample the *train* portion (if upsample > 1)

    All tier train subsets are concatenated and shuffled; likewise for
    eval.  The ``fen`` column is dropped from both outputs.

    Parameters
    ----------
    phase : PhaseConfig
        Phase configuration with tier mix specifications.
    data_root : Path
        Root directory containing ``tier{N}/`` subdirectories.
    eval_fraction : float
        Fraction of data to hold out for evaluation (default 2%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[Dataset, Dataset]
        (train_dataset, eval_dataset)
    """
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []

    loaded_tiers: list[tuple[object, Dataset]] = []
    all_fens: list[str] = []
    for tier_mix in phase.tier_mix:
        ds = load_tier_data(tier_mix.tier, data_root)
        loaded_tiers.append((tier_mix, ds))
        all_fens.extend(ds["fen"])

    phase_eval_keys = _select_eval_fen_keys(all_fens, eval_fraction, seed)

    for tier_mix, ds in loaded_tiers:
        if eval_fraction <= 0.0:
            tier_train, tier_eval = ds, ds.select([])
        else:
            train_indices: list[int] = []
            eval_indices: list[int] = []
            for idx, fen in enumerate(ds["fen"]):
                if _fen_key(fen) in phase_eval_keys:
                    eval_indices.append(idx)
                else:
                    train_indices.append(idx)
            tier_train, tier_eval = ds.select(train_indices), ds.select(eval_indices)

        # 2. Sample fraction (train only)
        if tier_mix.fraction < 1.0:
            n_samples = int(len(tier_train) * tier_mix.fraction)
            tier_train = tier_train.shuffle(seed=seed).select(range(n_samples))
            logger.info(
                "Tier %d: sampled %d (%.0f%% of train split)",
                tier_mix.tier, n_samples, tier_mix.fraction * 100,
            )

        # 3. Upsample (train only)
        if tier_mix.upsample > 1:
            original_len = len(tier_train)
            tier_train = concatenate_datasets([tier_train] * tier_mix.upsample)
            logger.info(
                "Tier %d: upsampled %dx (%d -> %d)",
                tier_mix.tier, tier_mix.upsample, original_len, len(tier_train),
            )

        train_parts.append(tier_train)
        if len(tier_eval) > 0:
            eval_parts.append(tier_eval)

    train_ds = concatenate_datasets(train_parts).shuffle(seed=seed)
    eval_ds = concatenate_datasets(eval_parts) if eval_parts else train_ds.select([])

    # Drop fen — not needed by the trainer
    if "fen" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["fen"])
    if "fen" in eval_ds.column_names:
        eval_ds = eval_ds.remove_columns(["fen"])

    logger.info(
        "Phase %s (%s): %d train, %d eval",
        phase.name, phase.display_name, len(train_ds), len(eval_ds),
    )

    return train_ds, eval_ds


def summarize_phase_data(
    phase: PhaseConfig,
    data_root: Path,
) -> dict[str, int]:
    """Compute expected example counts per tier without loading full data.

    Returns a dict mapping ``"tier_{N}"`` to expected count after
    sampling and upsampling. Useful for ``--dry-run``.
    """
    from data.loader import load_tier_data

    summary: dict[str, int] = {}
    total = 0

    for tier_mix in phase.tier_mix:
        ds = load_tier_data(tier_mix.tier, data_root)
        raw_count = len(ds)

        if tier_mix.fraction < 1.0:
            sampled = int(raw_count * tier_mix.fraction)
        else:
            sampled = raw_count

        final = sampled * tier_mix.upsample
        summary[f"tier_{tier_mix.tier}"] = final
        total += final

        logger.info(
            "Tier %d: %d raw -> %d sampled (%.0f%%) -> %d final (%dx upsample)",
            tier_mix.tier, raw_count, sampled,
            tier_mix.fraction * 100, final, tier_mix.upsample,
        )

    summary["total"] = total
    return summary
