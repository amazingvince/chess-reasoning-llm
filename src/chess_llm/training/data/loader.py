"""Load generated tier JSONL files into HuggingFace Datasets."""

from __future__ import annotations

import json
import logging
import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value, concatenate_datasets

from chess_llm.sft.context import normalize_chess_variant_metadata

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ("messages", "task", "tier", "fen")
_KEEP_COLS = ("messages", "task", "tier", "fen", "is_chess960", "chess960_id")

# Bump whenever sanitization output changes (normalization logic, _KEEP_COLS)
# so stale caches written by older code are not reused.
_SANITIZER_VERSION = 3

_SANITIZED_FEATURES = Features(
    {
        "messages": [
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ],
        "task": Value("string"),
        "tier": Value("int64"),
        "fen": Value("string"),
        "is_chess960": Value("bool"),
        "chess960_id": Value("int64"),
    }
)

_MOVE_TAG_RE = re.compile(
    r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>",
    re.IGNORECASE,
)

_MOVE_ONLY_THINK_TEXT = "Choose the final legal move."


@dataclass(frozen=True)
class TrainingDataTransformConfig:
    """Optional row-level data controls applied before training sanitization."""

    task_include: frozenset[str] | None = None
    task_exclude: frozenset[str] = frozenset()
    move_only_tasks: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        task_include = _normalize_optional_task_set(self.task_include)
        task_exclude = _normalize_task_set(self.task_exclude)
        move_only_tasks = _normalize_task_set(self.move_only_tasks)
        if task_include is not None:
            overlap = task_include & task_exclude
            if overlap:
                tasks = ", ".join(sorted(overlap))
                raise ValueError(f"Tasks cannot be both included and excluded: {tasks}")
        object.__setattr__(self, "task_include", task_include)
        object.__setattr__(self, "task_exclude", task_exclude)
        object.__setattr__(self, "move_only_tasks", move_only_tasks)

    @property
    def is_default(self) -> bool:
        return (
            self.task_include is None
            and not self.task_exclude
            and not self.move_only_tasks
        )

    def cache_suffix(self) -> str:
        if self.is_default:
            return ""
        payload = {
            "task_include": sorted(self.task_include) if self.task_include else None,
            "task_exclude": sorted(self.task_exclude),
            "move_only_tasks": sorted(self.move_only_tasks),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        digest = hashlib.blake2b(encoded, digest_size=8).hexdigest()
        return f".xf{digest}"


def _normalize_task_set(values: object) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        candidates = [values]
    else:
        candidates = list(values)
    return frozenset(str(value).strip() for value in candidates if str(value).strip())


def _normalize_optional_task_set(values: object) -> frozenset[str] | None:
    normalized = _normalize_task_set(values)
    return normalized or None


def _sanitized_jsonl_path(
    jsonl_file: Path,
    cache_root: Path,
    *,
    transform_config: TrainingDataTransformConfig | None = None,
) -> Path:
    """Return cache path for a source JSONL, keyed by file and sanitizer identity."""
    stat = jsonl_file.stat()
    digest = _file_digest(jsonl_file)
    config = transform_config or TrainingDataTransformConfig()
    cache_name = (
        f"{jsonl_file.stem}.v{_SANITIZER_VERSION}{config.cache_suffix()}."
        f"{stat.st_size}."
        f"{stat.st_mtime_ns}.{digest}.stripped.jsonl"
    )
    return cache_root / cache_name


def _file_digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_sanitized_jsonl(
    src: Path,
    dst: Path,
    *,
    transform_config: TrainingDataTransformConfig | None = None,
) -> int:
    """Write a stripped JSONL containing only training-relevant fields."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    config = transform_config or TrainingDataTransformConfig()
    rows_written = 0
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dst.parent,
            prefix=f"{dst.name}.tmp.",
            delete=False,
        ) as out_fh:
            tmp_path = Path(out_fh.name)
            with open(src, encoding="utf-8") as in_fh:
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
                    row = _transform_training_row(row, config)
                    if row is None:
                        continue
                    normalized = normalize_chess_variant_metadata(row)
                    stripped = {
                        key: normalized[key] for key in _KEEP_COLS if key in normalized
                    }
                    stripped.setdefault("is_chess960", False)
                    stripped.setdefault("chess960_id", None)
                    out_fh.write(json.dumps(stripped, ensure_ascii=False) + "\n")
                    rows_written += 1
        tmp_path.replace(dst)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise
    return rows_written


def _transform_training_row(
    row: dict[str, Any],
    config: TrainingDataTransformConfig,
) -> dict[str, Any] | None:
    task = str(row.get("task", ""))
    if config.task_include is not None and task not in config.task_include:
        return None
    if task in config.task_exclude:
        return None
    if task in config.move_only_tasks:
        return _rewrite_move_only_row(row, task)
    return row


def _rewrite_move_only_row(row: dict[str, Any], task: str) -> dict[str, Any]:
    target = _move_only_target(row)
    if target is None:
        raise ValueError(f"Cannot build move-only target for task {task!r}")

    messages = []
    replaced = False
    for message in row.get("messages", []):
        rewritten = dict(message)
        if rewritten.get("role") == "assistant":
            rewritten["content"] = (
                f"<think>{_MOVE_ONLY_THINK_TEXT}</think>\n<move>{target}</move>"
            )
            replaced = True
        messages.append(rewritten)
    if not replaced:
        raise ValueError(f"Cannot build move-only target for task {task!r}: no assistant message")

    rewritten_row = dict(row)
    rewritten_row["messages"] = messages
    return rewritten_row


def _move_only_target(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("target_move", "best_move", "solution_first_move"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        match = _MOVE_TAG_RE.search(content)
        if match:
            return match.group(1).lower()
    return None


def _load_sanitized_jsonl(path: Path) -> Dataset:
    """Load sanitized JSONL through HuggingFace's local Arrow JSON loader."""
    if path.stat().st_size == 0:
        return Dataset.from_dict(
            {name: [] for name in _SANITIZED_FEATURES},
            features=_SANITIZED_FEATURES,
        )
    return Dataset.from_json(str(path), features=_SANITIZED_FEATURES)


def load_tier_data(
    tier: int,
    data_root: Path,
    *,
    transform_config: TrainingDataTransformConfig | None = None,
) -> Dataset:
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
    config = transform_config or TrainingDataTransformConfig()
    for jsonl_file in jsonl_files:
        sanitized_file = _sanitized_jsonl_path(
            jsonl_file,
            cache_root,
            transform_config=config,
        )
        if not sanitized_file.exists():
            logger.info("Sanitizing %s -> %s", jsonl_file.name, sanitized_file.name)
            _write_sanitized_jsonl(
                jsonl_file,
                sanitized_file,
                transform_config=config,
            )
        datasets.append(_load_sanitized_jsonl(sanitized_file))

    ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    logger.info("Tier %d: %d examples loaded", tier, len(ds))
    return ds


__all__ = [
    "_KEEP_COLS",
    "TrainingDataTransformConfig",
    "_REQUIRED_COLS",
    "_SANITIZED_FEATURES",
    "_SANITIZER_VERSION",
    "_file_digest",
    "_load_sanitized_jsonl",
    "_sanitized_jsonl_path",
    "_write_sanitized_jsonl",
    "load_tier_data",
]
