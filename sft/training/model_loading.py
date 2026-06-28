"""Shared model-loading helpers for attention backend selection."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

ATTENTION_IMPLEMENTATION_CHOICES = ("auto", "flash_attention_2", "sdpa", "eager")


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def attention_candidates(requested: str = "auto") -> list[str | None]:
    """Return attention implementations to try, ordered by preference.

    ``None`` means leaving ``attn_implementation`` unspecified so Transformers
    can use the model/config default. Explicit user choices do not silently
    fall back; only ``auto`` tries multiple options.
    """
    if requested not in ATTENTION_IMPLEMENTATION_CHOICES:
        raise ValueError(
            f"Unknown attention implementation {requested!r}; "
            f"expected one of {ATTENTION_IMPLEMENTATION_CHOICES}",
        )

    if requested != "auto":
        return [requested]

    candidates: list[str | None] = []
    if _flash_attn_available():
        candidates.append("flash_attention_2")
    if _cuda_available():
        candidates.append("sdpa")
    candidates.append(None)
    return candidates


def load_causal_lm_with_attention(
    model_cls: Any,
    model_path: str,
    base_kwargs: Mapping[str, Any],
    *,
    requested_attn: str = "auto",
    logger: logging.Logger | None = None,
) -> tuple[Any, str | None]:
    """Load a causal LM with explicit attention selection and auto fallback."""
    log = logger or logging.getLogger(__name__)
    candidates = attention_candidates(requested_attn)
    last_exc: Exception | None = None

    for idx, attn_impl in enumerate(candidates):
        model_kwargs = dict(base_kwargs)
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        try:
            if attn_impl is None:
                log.info("Loading model with default attention implementation")
            else:
                log.info("Loading model with attn_implementation=%s", attn_impl)
            model = model_cls.from_pretrained(model_path, **model_kwargs)
            return model, attn_impl
        except Exception as exc:
            last_exc = exc
            is_last = idx == len(candidates) - 1
            if requested_attn != "auto" or is_last:
                raise
            log.warning(
                "Model load failed with attn_implementation=%s (%s: %s); retrying",
                attn_impl or "<default>",
                type(exc).__name__,
                exc,
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No attention implementation candidates were available")
