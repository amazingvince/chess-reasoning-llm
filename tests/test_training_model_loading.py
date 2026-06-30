import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


def test_package_model_loading_import_is_light(monkeypatch):
    sys.modules.pop("chess_llm.training.model_loading", None)
    sys.modules.pop("torch", None)
    sys.modules.pop("transformers", None)

    original_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "transformers", "flash_attn", "flash_attn_3", "flash_attn_4"}:
            raise AssertionError(f"heavy import at model_loading import time: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    module = importlib.import_module("chess_llm.training.model_loading")

    assert module.__name__ == "chess_llm.training.model_loading"


def test_package_attention_candidates_auto_prefers_stable_flash_then_sdpa(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setattr(model_loading, "_flash_attn4_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_supported", lambda: True)
    monkeypatch.setattr(model_loading, "_hf_flash_attn2_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn_available", lambda: True)
    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)

    assert model_loading.attention_candidates("auto") == [
        "flash_attention_3",
        "flash_attention_2",
        "sdpa",
        "eager",
        None,
    ]


def test_package_attention_candidates_auto_can_opt_into_flash4(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setenv("CHESS_SFT_ENABLE_FLASH_ATTENTION_4_AUTO", "1")
    monkeypatch.setattr(model_loading, "_flash_attn4_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_supported", lambda: True)
    monkeypatch.setattr(model_loading, "_hf_flash_attn2_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn_available", lambda: True)
    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)

    assert model_loading.attention_candidates("auto") == [
        "flash_attention_4",
        "flash_attention_3",
        "flash_attention_2",
        "sdpa",
        "eager",
        None,
    ]


def test_package_attention_candidates_auto_skips_flash3_when_gpu_unsupported(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setattr(model_loading, "_flash_attn4_available", lambda: False)
    monkeypatch.setattr(model_loading, "_flash_attn3_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_supported", lambda: False)
    monkeypatch.setattr(model_loading, "_hf_flash_attn2_available", lambda: False)
    monkeypatch.setattr(model_loading, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)

    assert model_loading.attention_candidates("auto") == [
        "sdpa",
        "eager",
        None,
    ]


def test_package_attention_candidates_auto_can_opt_into_hf_flash2(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setenv("CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO", "1")
    monkeypatch.setattr(model_loading, "_flash_attn4_available", lambda: False)
    monkeypatch.setattr(model_loading, "_flash_attn3_available", lambda: False)
    monkeypatch.setattr(model_loading, "_flash_attn3_supported", lambda: False)
    monkeypatch.setattr(model_loading, "_hf_flash_attn2_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)

    assert model_loading.attention_candidates("auto") == [
        model_loading.HF_FLASH_ATTN2_KERNEL,
        "sdpa",
        "eager",
        None,
    ]


def test_package_attention_candidates_auto_disable_overrides_hf_flash2_opt_in(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setenv("CHESS_SFT_ENABLE_HF_FLASH_ATTN2_AUTO", "1")
    monkeypatch.setenv("CHESS_SFT_DISABLE_HF_FLASH_ATTN2_AUTO", "1")
    monkeypatch.setattr(model_loading, "_flash_attn4_available", lambda: False)
    monkeypatch.setattr(model_loading, "_flash_attn3_available", lambda: False)
    monkeypatch.setattr(model_loading, "_flash_attn3_supported", lambda: False)
    monkeypatch.setattr(model_loading, "_hf_flash_attn2_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)

    assert model_loading.attention_candidates("auto") == ["sdpa", "eager", None]


def test_package_attention_candidates_auto_requires_cuda_for_fast_backends(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setattr(model_loading, "_flash_attn4_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn3_supported", lambda: True)
    monkeypatch.setattr(model_loading, "_hf_flash_attn2_available", lambda: True)
    monkeypatch.setattr(model_loading, "_flash_attn_available", lambda: True)
    monkeypatch.setattr(model_loading, "_cuda_available", lambda: False)

    assert model_loading.attention_candidates("auto") == ["eager", None]


def test_package_attention_candidates_explicit_and_invalid():
    from chess_llm.training.model_loading import HF_FLASH_ATTN2_KERNEL, attention_candidates

    assert attention_candidates("eager") == ["eager"]
    assert attention_candidates("hf_flash_attention_2") == [HF_FLASH_ATTN2_KERNEL]
    assert attention_candidates("kernels-community/flash-attn2") == [HF_FLASH_ATTN2_KERNEL]
    assert attention_candidates(HF_FLASH_ATTN2_KERNEL) == [HF_FLASH_ATTN2_KERNEL]
    assert attention_candidates("flash_attention_4") == ["flash_attention_4"]
    assert attention_candidates("flash_attention_3") == ["flash_attention_3"]
    with pytest.raises(ValueError, match="Unknown attention implementation"):
        attention_candidates("unknown")


def test_package_load_causal_lm_auto_falls_back_to_default(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setattr(
        model_loading,
        "attention_candidates",
        lambda _requested: ["flash_attention_2", None],
    )

    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append((model_path, kwargs))
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise RuntimeError("flash unavailable")
            return {"model_path": model_path, "kwargs": kwargs}

    model, selected = model_loading.load_causal_lm_with_attention(
        FakeModel,
        "model-id",
        {"torch_dtype": "auto"},
        requested_attn="auto",
    )

    assert selected is None
    assert model["model_path"] == "model-id"
    assert calls == [
        ("model-id", {"torch_dtype": "auto", "attn_implementation": "flash_attention_2"}),
        ("model-id", {"torch_dtype": "auto"}),
    ]


def test_package_prepare_qwen35_fast_path_disables_broken_cuda_kernels(monkeypatch):
    from types import SimpleNamespace

    from chess_llm.training import model_loading

    qwen_module = SimpleNamespace(
        causal_conv1d_fn=object(),
        causal_conv1d_update=object(),
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule=object(),
        FusedRMSNormGated=object(),
        is_fast_path_available=True,
        torch_causal_conv1d_update=object(),
        torch_chunk_gated_delta_rule=object(),
        torch_recurrent_gated_delta_rule=object(),
    )

    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        model_loading,
        "_import_qwen35_modeling_module",
        lambda: qwen_module,
    )
    monkeypatch.setattr(
        model_loading,
        "_qwen35_fast_path_smoke_passes",
        lambda _module: False,
    )

    disabled = model_loading.prepare_qwen35_fast_path_for_visible_device()

    assert disabled is True
    assert qwen_module.causal_conv1d_fn is None
    assert qwen_module.causal_conv1d_update is None
    assert qwen_module.chunk_gated_delta_rule is None
    assert qwen_module.fused_recurrent_gated_delta_rule is None
    assert qwen_module.FusedRMSNormGated is None
    assert qwen_module.is_fast_path_available is False


def test_package_prepare_qwen35_fast_path_keeps_working_kernels(monkeypatch):
    from types import SimpleNamespace

    from chess_llm.training import model_loading

    sentinel = object()
    qwen_module = SimpleNamespace(
        causal_conv1d_fn=sentinel,
        causal_conv1d_update=sentinel,
        chunk_gated_delta_rule=sentinel,
        fused_recurrent_gated_delta_rule=sentinel,
        FusedRMSNormGated=sentinel,
        is_fast_path_available=True,
    )

    monkeypatch.setattr(model_loading, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        model_loading,
        "_import_qwen35_modeling_module",
        lambda: qwen_module,
    )
    monkeypatch.setattr(
        model_loading,
        "_qwen35_fast_path_smoke_passes",
        lambda _module: True,
    )

    disabled = model_loading.prepare_qwen35_fast_path_for_visible_device()

    assert disabled is False
    assert qwen_module.causal_conv1d_fn is sentinel
    assert qwen_module.causal_conv1d_update is sentinel
    assert qwen_module.chunk_gated_delta_rule is sentinel
    assert qwen_module.fused_recurrent_gated_delta_rule is sentinel
    assert qwen_module.FusedRMSNormGated is sentinel
    assert qwen_module.is_fast_path_available is True


def test_package_qwen35_fast_path_allows_blackwell_when_conv1d_smoke_passes(monkeypatch):
    from types import SimpleNamespace

    from chess_llm.training import model_loading

    qwen_module = SimpleNamespace(
        causal_conv1d_fn=object(),
        causal_conv1d_update=object(),
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule=object(),
    )

    monkeypatch.setattr(model_loading, "_visible_cuda_device_capability", lambda: (12, 0))
    monkeypatch.setattr(
        model_loading,
        "_qwen35_causal_conv1d_smoke_passes",
        lambda _module: True,
        raising=False,
    )

    assert model_loading._qwen35_fast_path_smoke_passes(qwen_module) is True


def test_package_qwen35_fast_path_disables_hopper_when_conv1d_smoke_fails(monkeypatch):
    from types import SimpleNamespace

    from chess_llm.training import model_loading

    qwen_module = SimpleNamespace(
        causal_conv1d_fn=object(),
        causal_conv1d_update=object(),
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule=object(),
    )

    monkeypatch.setattr(model_loading, "_visible_cuda_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        model_loading,
        "_qwen35_causal_conv1d_smoke_passes",
        lambda _module: False,
        raising=False,
    )

    assert model_loading._qwen35_fast_path_smoke_passes(qwen_module) is False


def test_package_load_causal_lm_prepares_qwen35_fast_path(monkeypatch):
    from chess_llm.training import model_loading

    monkeypatch.setattr(model_loading, "attention_candidates", lambda _requested: ["sdpa"])
    prepared = []
    monkeypatch.setattr(
        model_loading,
        "prepare_qwen35_fast_path_for_visible_device",
        lambda logger=None: prepared.append(logger),
    )

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return {"model_path": model_path, "kwargs": kwargs}

    model, selected = model_loading.load_causal_lm_with_attention(
        FakeModel,
        "model-id",
        {"torch_dtype": "auto"},
        requested_attn="auto",
    )

    assert selected == "sdpa"
    assert model["kwargs"] == {"torch_dtype": "auto", "attn_implementation": "sdpa"}
    assert len(prepared) == 1


def test_package_qwen35_text_only_key_classifier_identifies_expected_unused_keys():
    from chess_llm.training import model_loading

    keys = [
        "model.visual.patch_embed.proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]

    assert model_loading.qwen35_text_only_unused_key_summary(keys) == {
        "model.visual.*": 1,
        "mtp.*": 1,
    }


def test_package_load_causal_lm_logs_qwen35_text_only_note_once(monkeypatch, caplog):
    import logging

    from chess_llm.training import model_loading

    monkeypatch.setattr(model_loading, "attention_candidates", lambda _requested: ["sdpa"])
    monkeypatch.setattr(
        model_loading,
        "prepare_qwen35_fast_path_for_visible_device",
        lambda logger=None: False,
    )

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return {"model_path": model_path, "kwargs": kwargs}

    with caplog.at_level(logging.INFO):
        model_loading.load_causal_lm_with_attention(
            FakeModel,
            "Qwen/Qwen3.5-0.8B",
            {"torch_dtype": "auto"},
            requested_attn="sdpa",
        )

    assert caplog.text.count("text-only causal-LM load") == 1
    assert "model.visual.*" in caplog.text
    assert "mtp.*" in caplog.text


def test_legacy_model_loading_reexports_package_objects():
    legacy_path = (
        Path(__file__).resolve().parents[1] / "sft" / "training" / "model_loading.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_model_loading", legacy_path)
    assert spec is not None
    assert spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    from chess_llm.training import model_loading as package

    assert legacy.ATTENTION_IMPLEMENTATION_CHOICES is package.ATTENTION_IMPLEMENTATION_CHOICES
    assert legacy.attention_candidates is package.attention_candidates
    assert legacy.load_causal_lm_with_attention is package.load_causal_lm_with_attention


def test_legacy_model_loading_import_alias_preserves_monkeypatches(monkeypatch):
    legacy_dir = Path(__file__).resolve().parents[1] / "sft" / "training"
    monkeypatch.syspath_prepend(str(legacy_dir))
    sys.modules.pop("model_loading", None)

    legacy = importlib.import_module("model_loading")
    from chess_llm.training import model_loading as package

    assert legacy is package

    monkeypatch.setattr(legacy, "attention_candidates", lambda _requested: [None])

    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append((model_path, kwargs))
            return object()

    legacy.load_causal_lm_with_attention(
        FakeModel,
        "model-id",
        {"torch_dtype": "auto"},
        requested_attn="eager",
    )

    assert calls == [("model-id", {"torch_dtype": "auto"})]
