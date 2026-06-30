import importlib

import chess_llm
from chess_llm import artifacts


def test_initial_subpackages_import():
    modules = [
        "chess_llm",
        "chess_llm.core",
        "chess_llm.artifacts",
        "chess_llm.formats",
        "chess_llm.sft",
        "chess_llm.sft.annotation",
        "chess_llm.sft.decontamination",
        "chess_llm.sft.completeness",
        "chess_llm.sft.download_tablebases",
        "chess_llm.sft.eval_split",
        "chess_llm.sft.extract_polyglot_books",
        "chess_llm.sft.fen_pool",
        "chess_llm.sft.generators",
        "chess_llm.sft.generators.tier1_perception",
        "chess_llm.sft.hub_upload",
        "chess_llm.sft.output",
        "chess_llm.sft.preflight",
        "chess_llm.sft.run_eval_split",
        "chess_llm.sft.source_preparation",
        "chess_llm.sft.source_readiness",
        "chess_llm.sft.sources",
        "chess_llm.sft.templates",
        "chess_llm.sft.validate_outputs",
        "chess_llm.sft.sources.chess960",
        "chess_llm.sft.sources.lichess_openings",
        "chess_llm.sft.sources.polyglot_books",
        "chess_llm.sft.sources.syzygy_probing",
        "chess_llm.training",
        "chess_llm.training.evaluate",
        "chess_llm.training.model_loading",
        "chess_llm.training.phase_gate",
        "chess_llm.training.run_curriculum",
        "chess_llm.training.train",
        "chess_llm.evals",
        "chess_llm.evals.prediction_analysis",
        "chess_llm.autodata",
        "chess_llm.external",
        "chess_llm.preference",
        "chess_llm.sdpo",
        "chess_llm.inference",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_package_exposes_version():
    assert chess_llm.__version__ == "0.1.0"


def test_artifacts_exports_jsonl_helpers():
    assert artifacts.read_jsonl
    assert artifacts.write_jsonl


def test_sft_exports_data_generation_helpers():
    from chess_llm import sft
    from chess_llm.sft import generators

    assert sft.FENPool
    assert sft.build_eval_split_sources
    assert sft.build_source_readiness_report
    assert generators.FENRowApplication


def test_evals_exports_prediction_analysis_helpers():
    from chess_llm import evals

    assert evals.write_prediction_analysis_report
