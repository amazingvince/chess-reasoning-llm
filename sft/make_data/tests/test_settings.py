import os
import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from pathlib import Path

import pytest


def test_legacy_settings_are_backed_by_package_settings():
    from config import settings as legacy
    from chess_llm.sft.settings import (
        DEFAULT_HF_CACHE_DIR,
        DEFAULT_HF_DATASETS,
        SftDataSettings,
    )

    assert isinstance(legacy.SETTINGS, SftDataSettings)
    assert DEFAULT_HF_CACHE_DIR
    assert legacy.HF_CACHE_DIR == os.environ["HF_HOME"]
    assert legacy.HF_DATASETS == DEFAULT_HF_DATASETS
    assert legacy.OUTPUT_DIR == Path(os.environ["CHESS_SFT_OUTPUT"])
    assert legacy.POOL_DIR == legacy.OUTPUT_DIR / "pool"
    assert legacy.TIER_OUTPUT_DIR == legacy.OUTPUT_DIR / "output"
    assert legacy.BENCHMARK_DIR == legacy.OUTPUT_DIR / "benchmark"
    assert os.environ["HF_HOME"]


class _RecordingHfLoader(importlib.abc.Loader):
    def __init__(self, records):
        self.records = records

    def create_module(self, spec):
        return ModuleType(spec.name)

    def exec_module(self, module):
        self.records[module.__name__] = os.environ.get("HF_HOME")
        if module.__name__ == "datasets":
            module.load_dataset = lambda *args, **kwargs: ()
        elif module.__name__ == "huggingface_hub":
            module.hf_hub_download = lambda *args, **kwargs: ""


class _RecordingHfFinder(importlib.abc.MetaPathFinder):
    def __init__(self, records):
        self.records = records

    def find_spec(self, fullname, path, target=None):
        if fullname not in {"datasets", "huggingface_hub"}:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _RecordingHfLoader(self.records),
        )


@pytest.mark.parametrize(
    ("source_module", "hf_module"),
    [
        ("sources.lichess_games", "datasets"),
        ("sources.lichess_puzzles", "datasets"),
        ("sources.lichess_openings", "datasets"),
        ("sources.lichess_evals", "datasets"),
        ("sources.mate_dataset", "huggingface_hub"),
    ],
)
def test_hf_cache_default_is_applied_before_hf_library_import(
    monkeypatch,
    tmp_path,
    source_module,
    hf_module,
):
    from chess_llm.sft.settings import DEFAULT_HF_CACHE_DIR

    records = {}
    finder = _RecordingHfFinder(records)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    for module_name in [
        source_module,
        "config.settings",
        "datasets",
        "huggingface_hub",
    ]:
        sys.modules.pop(module_name, None)

    try:
        module = importlib.import_module(source_module)
        if source_module == "sources.lichess_games":
            list(module.stream_games(max_games=1))
        elif source_module == "sources.lichess_puzzles":
            list(module.load_puzzles(max_puzzles=1))
        elif source_module == "sources.lichess_openings":
            list(module.load_openings(max_openings=0))
        elif source_module == "sources.lichess_evals":
            list(module.stream_evals(max_rows=1, dedup_db_path=tmp_path / "evals.db"))
        elif source_module == "sources.mate_dataset":
            list(module.load_mate(max_rows=1))
    finally:
        for module_name in [source_module, "datasets", "huggingface_hub"]:
            sys.modules.pop(module_name, None)

    assert records[hf_module] == DEFAULT_HF_CACHE_DIR
