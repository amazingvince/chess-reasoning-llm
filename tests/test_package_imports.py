import importlib


def test_initial_subpackages_import():
    modules = [
        "chess_llm",
        "chess_llm.core",
        "chess_llm.artifacts",
        "chess_llm.formats",
        "chess_llm.sft",
        "chess_llm.evals",
        "chess_llm.autodata",
        "chess_llm.external",
        "chess_llm.preference",
        "chess_llm.sdpo",
        "chess_llm.inference",
    ]

    for module in modules:
        assert importlib.import_module(module)
