"""Load generated tier JSONL files into HuggingFace Datasets."""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from chess_llm.sft.context import normalize_chess_variant_metadata

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ("messages", "task", "tier", "fen")
_KEEP_COLS = ("messages", "task", "tier", "fen", "is_chess960", "chess960_id")

# Bump whenever sanitization output changes (normalization logic, _KEEP_COLS)
# so stale caches written by older code are not reused.
_SANITIZER_VERSION = 2


def _sanitized_jsonl_path(jsonl_file: Path, cache_root: Path) -> Path:
    """Return cache path for a source JSONL, keyed by file and sanitizer identity."""
    stat = jsonl_file.stat()
    digest = _file_digest(jsonl_file)
    cache_name = (
        f"{jsonl_file.stem}.v{_SANITIZER_VERSION}.{stat.st_size}."
        f"{stat.st_mtime_ns}.{digest}.stripped.jsonl"
    )
    return cache_root / cache_name


def _file_digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_sanitized_jsonl(src: Path, dst: Path) -> None:
    """Write a stripped JSONL containing only training-relevant fields."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    with open(src, encoding="utf-8") as in_fh, open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as out_fh:
        for line_no, line in enumerate(in_fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [key for key in _REQUIRED_COLS if key not in row]
            if missing:
                raise ValueError(
                    f"{src} line {line_no} is missing required keys: "
                    f"{', '.join(missing)}"
                )
            normalized = normalize_chess_variant_metadata(row)
            stripped = {key: normalized[key] for key in _KEEP_COLS if key in normalized}
            stripped.setdefault("is_chess960", False)
            stripped.setdefault("chess960_id", None)
            out_fh.write(json.dumps(stripped, ensure_ascii=False) + "\n")
    tmp_path.replace(dst)


def _load_sanitized_jsonl(path: Path) -> Dataset:
    """Load sanitized JSONL through HuggingFace's local Arrow JSON loader."""
    return Dataset.from_json(str(path))


def load_tier_data(tier: int, data_root: Path) -> Dataset:
    """Load all JSONL files for one tier into a HuggingFace Dataset."""
    tier_dir = data_root / f"tier{tier}"
    jsonl_files = sorted(tier_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found in {tier_dir}. "
            f"Run the data generation pipeline first."
        )

    logger.info(
        "Loading tier %d: %d file(s) from %s",
        tier,
        len(jsonl_files),
        tier_dir,
    )

    cache_root = data_root / ".training_cache" / f"tier{tier}"
    datasets: list[Dataset] = []
    for jsonl_file in jsonl_files:
        sanitized_file = _sanitized_jsonl_path(jsonl_file, cache_root)
        if not sanitized_file.exists():
            logger.info("Sanitizing %s -> %s", jsonl_file.name, sanitized_file.name)
            _write_sanitized_jsonl(jsonl_file, sanitized_file)
        datasets.append(_load_sanitized_jsonl(sanitized_file))

    ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    logger.info("Tier %d: %d examples loaded", tier, len(ds))
    return ds


__all__ = [
    "_KEEP_COLS",
    "_REQUIRED_COLS",
    "_SANITIZER_VERSION",
    "_file_digest",
    "_load_sanitized_jsonl",
    "_sanitized_jsonl_path",
    "_write_sanitized_jsonl",
    "load_tier_data",
]
