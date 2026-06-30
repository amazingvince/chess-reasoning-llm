import importlib
import sys
import types


def _fresh_import(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_legacy_make_data_modules_import_from_repo_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CHESS_SFT_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    module_names = [
        "sft.make_data.sources.lichess_games",
        "sft.make_data.sources.lichess_evals",
        "sft.make_data.sources.lichess_puzzles",
        "sft.make_data.sources.lichess_openings",
        "sft.make_data.sources.mate_dataset",
        "sft.make_data.sources.stockfish_engine",
        "sft.make_data.pool.eval_split",
        "sft.make_data.generators.tier1_perception",
        "sft.make_data.output.writer",
        "sft.make_data.validation.benchmark",
        "sft.make_data.validation.completeness",
    ]

    for module_name in module_names:
        assert _fresh_import(module_name)


def test_legacy_training_config_and_data_packages_import_from_repo_root(monkeypatch):
    fake_datasets = types.SimpleNamespace(
        Dataset=object,
        concatenate_datasets=lambda *args, **kwargs: None,
        load_dataset=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    try:
        assert _fresh_import("sft.training.config")
        assert _fresh_import("sft.training.data")
    finally:
        for module_name in (
            "sft.training.data",
            "sft.training.data.loader",
            "sft.training.data.mixer",
            "chess_llm.training.data",
            "chess_llm.training.data.loader",
            "chess_llm.training.data.mixer",
        ):
            sys.modules.pop(module_name, None)
