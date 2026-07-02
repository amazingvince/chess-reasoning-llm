from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_qwen35_text_checkpoint_exports_wrapper_index(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    from chess_llm.training import vllm_export

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    _write_json(
        base_dir / "config.json",
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_5"},
        },
    )
    (base_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    base_weight_name = "model.safetensors-00001-of-00001.safetensors"
    safetensors_torch.save_file(
        {
            "model.language_model.embed_tokens.weight": torch.ones(1),
            "model.visual.blocks.0.norm1.weight": torch.zeros(1),
        },
        base_dir / base_weight_name,
    )
    _write_json(
        base_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 2},
            "weight_map": {
                "model.language_model.embed_tokens.weight": base_weight_name,
                "model.visual.blocks.0.norm1.weight": base_weight_name,
            },
        },
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_json(
        checkpoint_dir / "config.json",
        {
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_text",
        },
    )
    (checkpoint_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    safetensors_torch.save_file(
        {"model.language_model.embed_tokens.weight": torch.full((1,), 2.0)},
        checkpoint_dir / "model.safetensors",
    )

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda _model: str(base_dir),
    )

    output_dir = vllm_export.export_qwen35_text_checkpoint_for_vllm(
        checkpoint_dir,
        base_model="Qwen/Qwen3.5-0.8B",
    )

    assert output_dir == checkpoint_dir / "vllm_qwen35_wrapper"
    assert json.loads((output_dir / "config.json").read_text(encoding="utf-8"))[
        "model_type"
    ] == "qwen3_5"
    exported_index = json.loads(
        (output_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert exported_index["weight_map"] == {
        "model.language_model.embed_tokens.weight": "sft_model.safetensors",
        "model.visual.blocks.0.norm1.weight": f"base_{base_weight_name}",
    }
    assert (output_dir / "sft_model.safetensors").exists()
    assert (output_dir / f"base_{base_weight_name}").exists()
    assert (output_dir / "preprocessor_config.json").exists()
    assert json.loads(
        (output_dir / "vllm_export_manifest.json").read_text(encoding="utf-8")
    )["format"] == "qwen35_text_vllm_wrapper_v1"


def test_prepare_model_for_vllm_returns_original_for_nonlocal_model():
    from chess_llm.training.vllm_export import prepare_model_for_vllm

    assert prepare_model_for_vllm("Qwen/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"
