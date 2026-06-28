"""Load tier JSONL files into HuggingFace Datasets.

Each JSONL line from the data pipeline has the schema:
    {task, tier, fen, is_chess960, messages, metadata}

where ``messages`` is:
    [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

This module loads those files and returns a Dataset with the ``messages``
column in TRL's expected chat format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

_KEEP_COLS = ("messages", "task", "tier", "fen")


def _sanitized_jsonl_path(jsonl_file: Path, cache_root: Path) -> Path:
    """Return cache path for a source JSONL, keyed by file identity."""
    stat = jsonl_file.stat()
    cache_name = f"{jsonl_file.stem}.{stat.st_size}.{stat.st_mtime_ns}.stripped.jsonl"
    return cache_root / cache_name


def _write_sanitized_jsonl(src: Path, dst: Path) -> None:
    """Write a stripped JSONL containing only training-relevant fields."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    with open(src, encoding="utf-8") as in_fh, open(tmp_path, "w", encoding="utf-8") as out_fh:
        for line_no, line in enumerate(in_fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stripped = {key: row[key] for key in _KEEP_COLS if key in row}
            missing = [key for key in _KEEP_COLS if key not in stripped]
            if missing:
                raise ValueError(
                    f"{src} line {line_no} is missing required keys: {', '.join(missing)}"
                )
            out_fh.write(json.dumps(stripped, ensure_ascii=False) + "\n")
    tmp_path.replace(dst)


def load_tier_data(tier: int, data_root: Path) -> Dataset:
    """Load all JSONL files for tier N into a HuggingFace Dataset.

    Scans ``data_root/tier{N}/*.jsonl``. Returns a Dataset with columns:
    ``messages`` (list[dict]), ``task`` (str), ``tier`` (int).

    The ``messages`` column is already in TRL's expected chat format:
    ``[{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]``

    Uses ``datasets.load_dataset("json")`` which creates memory-mapped Arrow
    cache automatically — critical for Phase C's ~1.5M examples.

    Parameters
    ----------
    tier : int
        Tier number (1-7).
    data_root : Path
        Root directory containing ``tier{N}/`` subdirectories.

    Returns
    -------
    Dataset
        HuggingFace Dataset with at least ``messages``, ``task``, ``tier`` columns.

    Raises
    ------
    FileNotFoundError
        If no JSONL files are found for the tier.
    """
    tier_dir = data_root / f"tier{tier}"
    jsonl_files = sorted(tier_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found in {tier_dir}. "
            f"Run the data generation pipeline first."
        )

    logger.info(
        "Loading tier %d: %d file(s) from %s",
        tier, len(jsonl_files), tier_dir,
    )

    # Sanitize each source file to a stripped JSONL before Arrow sees it.
    # This avoids schema inference failures from heterogeneous metadata payloads,
    # both across files and within a single file.
    cache_root = data_root / ".training_cache" / f"tier{tier}"
    datasets: list[Dataset] = []
    for jsonl_file in jsonl_files:
        sanitized_file = _sanitized_jsonl_path(jsonl_file, cache_root)
        if not sanitized_file.exists():
            logger.info("Sanitizing %s -> %s", jsonl_file.name, sanitized_file.name)
            _write_sanitized_jsonl(jsonl_file, sanitized_file)
        file_ds = load_dataset(
            "json",
            data_files=str(sanitized_file),
            split="train",
        )
        datasets.append(file_ds)

    ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    logger.info("Tier %d: %d examples loaded", tier, len(ds))
    return ds
