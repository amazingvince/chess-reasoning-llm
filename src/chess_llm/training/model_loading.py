"""Shared model-loading helpers for attention backend selection."""

from __future__ import annotations

import logging
import os
from importlib import util as importlib_util
from collections.abc import Mapping
from importlib import import_module
from typing import Any

HF_FLASH_ATTN2_KERNEL_REVISION = "bcc70a66fbe0445d4484f56167d706134d53633d"
HF_FLASH_ATTN2_KERNEL = (
    f"kernels-community/flash-attn2@{HF_FLASH_ATTN2_KERNEL_REVISION}"
)
HF_FLASH_ATTN2_ALIASES = (
    "hf_flash_attention_2",
    "kernels-community/flash-attn2",
)

ATTENTION_IMPLEMENTATION_CHOICES = (
    "auto",
    *HF_FLASH_ATTN2_ALIASES,
    HF_FLASH_ATTN2_KERNEL,
    "flash_attention_4",
    "flash_attention_3",
    "flash_attention_2",
    "sdpa",
    "eager",
)

_QWEN35_TEXT_ONLY_UNUSED_KEY_PREFIXES = {
    "model.visual.": "model.visual.*",
    "mtp.": "mtp.*",
}
_QWEN35_TEXT_ONLY_LOAD_NOTED: set[str] = set()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def qwen35_text_only_unused_key_summary(keys: list[str] | tuple[str, ...]) -> dict[str, int]:
    """Summarize Qwen3.5 checkpoint keys expected to be unused for text-only LM loads."""
    summary: dict[str, int] = {}
    for key in keys:
        for prefix, label in _QWEN35_TEXT_ONLY_UNUSED_KEY_PREFIXES.items():
            if key.startswith(prefix):
                summary[label] = summary.get(label, 0) + 1
                break
    return summary


def _looks_like_qwen35_base_model(model_path: str) -> bool:
    normalized = model_path.replace("\\", "/").lower()
    return normalized in {
        "qwen/qwen3.5-0.8b",
    }


def _log_qwen35_text_only_load_note(model_path: str, log: logging.Logger) -> None:
    """Explain the expected unused-key warning before loading Qwen3.5 as CausalLM."""
    if not _looks_like_qwen35_base_model(model_path):
        return
    if model_path in _QWEN35_TEXT_ONLY_LOAD_NOTED:
        return
    _QWEN35_TEXT_ONLY_LOAD_NOTED.add(model_path)
    log.info(
        "Qwen3.5 text-only causal-LM load: Transformers may report unused "
        "checkpoint keys for %s and %s because chess SFT loads "
        "AutoModelForCausalLM, not the multimodal/MTP wrapper.",
        "model.visual.*",
        "mtp.*",
    )


def _flash_attn4_available() -> bool:
    try:
        from transformers.utils.import_utils import is_flash_attn_4_available

        return bool(is_flash_attn_4_available())
    except Exception:
        pass

    try:
        from importlib.metadata import packages_distributions

        distributions = {
            distribution.replace("_", "-")
            for distribution in packages_distributions().get("flash_attn", [])
        }
        return "flash-attn-4" in distributions
    except Exception:
        return False


def _flash_attn4_auto_enabled() -> bool:
    return _truthy_env("CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO")


def _hf_flash_attn2_auto_enabled() -> bool:
    return _truthy_env("CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO") and not _truthy_env(
        "CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO"
    )


def _hf_flash_attn2_available() -> bool:
    return importlib_util.find_spec("kernels") is not None


def _canonical_attention_implementation(requested: str) -> str:
    if requested in HF_FLASH_ATTN2_ALIASES:
        return HF_FLASH_ATTN2_KERNEL
    return requested


def _supported_attention_implementation(requested: str) -> bool:
    if requested in ATTENTION_IMPLEMENTATION_CHOICES:
        return True
    return requested.startswith("kernels-community/flash-attn2@")


def _flash_attn3_available() -> bool:
    try:
        from transformers.utils.import_utils import is_flash_attn_3_available

        return bool(is_flash_attn_3_available())
    except Exception:
        pass

    try:
        import flash_attn_interface  # noqa: F401

        return True
    except Exception:
        return False


def _flash_attn3_supported() -> bool:
    """Return whether the visible CUDA device can execute FlashAttention 3."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return (major, minor) == (9, 0)


def _flash_attn_available() -> bool:
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except Exception:
        pass

    try:
        from importlib.metadata import packages_distributions

        distributions = {
            distribution.replace("_", "-")
            for distribution in packages_distributions().get("flash_attn", [])
        }
        if "flash-attn" not in distributions:
            return False
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


def _visible_cuda_device_capability() -> tuple[int, int] | None:
    """Return the capability for the first visible CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return None
    return major, minor


def _import_qwen35_modeling_module() -> Any | None:
    """Import the Transformers Qwen3.5 modeling module when available."""
    try:
        return import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except Exception:
        return None


def _qwen35_fast_path_smoke_passes(qwen35_module: Any) -> bool:
    """Return whether Qwen3.5 optional CUDA fast-path kernels are safe to use.

    Optional 1D-conv kernels can import successfully even when the wheel lacks
    code for the visible GPU architecture. Use a tiny launch as the source of
    truth so Hopper, Blackwell, and future builds are handled by capability,
    not by a hard-coded architecture allow/deny list.
    """
    if not bool(
        getattr(qwen35_module, "causal_conv1d_fn", None)
        and getattr(qwen35_module, "causal_conv1d_update", None)
        and getattr(qwen35_module, "chunk_gated_delta_rule", None)
        and getattr(qwen35_module, "fused_recurrent_gated_delta_rule", None)
    ):
        return False

    return _qwen35_causal_conv1d_smoke_passes(qwen35_module)


def _qwen35_causal_conv1d_smoke_passes(qwen35_module: Any) -> bool:
    """Return whether Qwen3.5 causal-conv1d kernels launch on the visible GPU."""
    causal_conv1d_fn = getattr(qwen35_module, "causal_conv1d_fn", None)
    causal_conv1d_update = getattr(qwen35_module, "causal_conv1d_update", None)
    if not callable(causal_conv1d_fn) or not callable(causal_conv1d_update):
        return False

    try:
        import torch

        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda")
        dtype = torch.float16
        batch, channels, seq_len, width = 1, 4, 8, 3
        x = torch.randn(batch, channels, seq_len, device=device, dtype=dtype)
        weight = torch.randn(channels, width, device=device, dtype=dtype)
        bias = torch.randn(channels, device=device, dtype=dtype)
        causal_conv1d_fn(x, weight, bias, activation="silu")
        conv_state = torch.zeros(batch, channels, width, device=device, dtype=dtype)
        x_step = torch.randn(batch, channels, device=device, dtype=dtype)
        causal_conv1d_update(x_step, conv_state, weight, bias, activation="silu")
        torch.cuda.synchronize()
        return True
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        return False


def _disable_qwen35_fast_path(qwen35_module: Any) -> None:
    """Force Qwen3.5 modeling code onto its pure-PyTorch fallback kernels."""
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        if hasattr(qwen35_module, name):
            setattr(qwen35_module, name, None)
    if hasattr(qwen35_module, "is_fast_path_available"):
        setattr(qwen35_module, "is_fast_path_available", False)


def prepare_qwen35_fast_path_for_visible_device(
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Disable Qwen3.5 optional kernels when the visible CUDA device cannot run them.

    Returns ``True`` when the fast path was disabled.
    """
    if not _cuda_available():
        return False

    qwen35_module = _import_qwen35_modeling_module()
    if qwen35_module is None:
        return False

    fast_path_names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    )
    if not any(getattr(qwen35_module, name, None) is not None for name in fast_path_names):
        return False

    if _qwen35_fast_path_smoke_passes(qwen35_module):
        return False

    _disable_qwen35_fast_path(qwen35_module)
    log = logger or logging.getLogger(__name__)
    log.warning(
        "Disabled Qwen3.5 optional fast-path kernels for the visible CUDA device; "
        "falling back to Transformers torch implementations.",
    )
    return True


def attention_candidates(requested: str = "auto") -> list[str | None]:
    """Return attention implementations to try, ordered by preference.

    ``None`` means leaving ``attn_implementation`` unspecified so Transformers
    can use the model/config default. Explicit user choices do not silently
    fall back; only ``auto`` tries multiple options.
    """
    if not _supported_attention_implementation(requested):
        raise ValueError(
            f"Unknown attention implementation {requested!r}; "
            f"expected one of {ATTENTION_IMPLEMENTATION_CHOICES}",
        )

    if requested != "auto":
        return [_canonical_attention_implementation(requested)]

    candidates: list[str | None] = []
    cuda_available = _cuda_available()
    if cuda_available and _flash_attn4_auto_enabled() and _flash_attn4_available():
        candidates.append("flash_attention_4")
    if cuda_available and _flash_attn3_available() and _flash_attn3_supported():
        candidates.append("flash_attention_3")
    if cuda_available and _hf_flash_attn2_auto_enabled() and _hf_flash_attn2_available():
        candidates.append(HF_FLASH_ATTN2_KERNEL)
    if cuda_available and _flash_attn_available():
        candidates.append("flash_attention_2")
    if cuda_available:
        candidates.append("sdpa")
    candidates.append("eager")
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
    _log_qwen35_text_only_load_note(model_path, log)

    for idx, attn_impl in enumerate(candidates):
        model_kwargs = dict(base_kwargs)
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        try:
            prepare_qwen35_fast_path_for_visible_device(logger=log)
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


__all__ = [
    "ATTENTION_IMPLEMENTATION_CHOICES",
    "attention_candidates",
    "load_causal_lm_with_attention",
    "prepare_qwen35_fast_path_for_visible_device",
    "qwen35_text_only_unused_key_summary",
]
