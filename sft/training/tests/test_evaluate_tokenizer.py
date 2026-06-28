from __future__ import annotations

from pathlib import Path

import pytest

import evaluate


class _DummyTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path
        self.pad_token = None
        self.eos_token = "<|im_end|>"
        self.padding_side = "right"
        self.chat_template = None


def test_load_prompt_tokenizer_falls_back_for_local_checkpoint(monkeypatch, tmp_path: Path):
    checkpoint_dir = tmp_path / "phase_b_best"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")

    calls: list[str] = []

    def fake_from_pretrained(model_path: str, trust_remote_code: bool = True):
        calls.append(model_path)
        if Path(model_path) == checkpoint_dir:
            raise AttributeError("'list' object has no attribute 'keys'")
        return _DummyTokenizer(model_path)

    monkeypatch.setenv("CHESS_SFT_BASE_MODEL", "Qwen/Qwen3-0.6B")
    monkeypatch.setattr(evaluate.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    tokenizer = evaluate.load_prompt_tokenizer(str(checkpoint_dir))

    assert calls == [str(checkpoint_dir), "Qwen/Qwen3-0.6B"]
    assert tokenizer.name_or_path == "Qwen/Qwen3-0.6B"
    assert tokenizer.chat_template == "{{ messages }}"
    assert tokenizer.pad_token == "<|im_end|>"
    assert tokenizer.padding_side == "left"
