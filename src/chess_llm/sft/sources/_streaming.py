"""Utilities for cleaning up external streaming dataset iterables."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable

DEFAULT_STREAMING_SHUTDOWN_WAIT_SECONDS = 0.0
STREAMING_SHUTDOWN_WAIT_ENV = "CHESS_SFT_STREAMING_SHUTDOWN_WAIT_SECONDS"


def close_if_possible(value: object) -> None:
    """Close a dataset iterable or iterator when the backend exposes cleanup."""
    close = getattr(value, "close", None)
    if callable(close):
        close()


def close_streaming_dataset(
    row_iter: object,
    rows: object,
    *,
    collect: Callable[[], object] = gc.collect,
    sleep: Callable[[float], object] = time.sleep,
    shutdown_wait_seconds: float = 0.0,
) -> None:
    """Close a streaming dataset and collect parquet/fsspec finalizers."""
    close_if_possible(row_iter)
    if rows is not row_iter:
        close_if_possible(rows)
    collect()
    if shutdown_wait_seconds > 0:
        sleep(shutdown_wait_seconds)


def streaming_shutdown_wait_seconds() -> float:
    """Return the configured HF streaming shutdown drain interval."""
    raw = os.environ.get(STREAMING_SHUTDOWN_WAIT_ENV)
    if raw is None:
        return DEFAULT_STREAMING_SHUTDOWN_WAIT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_STREAMING_SHUTDOWN_WAIT_SECONDS


__all__ = [
    "DEFAULT_STREAMING_SHUTDOWN_WAIT_SECONDS",
    "STREAMING_SHUTDOWN_WAIT_ENV",
    "close_if_possible",
    "close_streaming_dataset",
    "streaming_shutdown_wait_seconds",
]
