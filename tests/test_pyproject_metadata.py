from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def test_pyproject_declares_runtime_dependency_and_package_clis():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert '"python-chess>=1.10.0"' in text
    assert '"huggingface_hub>=0.24.0"' in text
    assert 'chess-llm-batch-judge = "chess_llm.evals.batch_judge:main"' in text
    assert 'chess-llm-download-tablebases = "chess_llm.sft.download_tablebases:main"' in text
    assert 'chess-llm-evaluate = "chess_llm.training.evaluate:main"' in text
    assert (
        'chess-llm-extract-polyglot-books = '
        '"chess_llm.sft.extract_polyglot_books:main"'
        in text
    )
    assert (
        'chess-llm-freeze-benchmark = "chess_llm.evals.freeze_benchmark:main"'
        in text
    )
    assert 'chess-llm-run-benchmark = "chess_llm.evals.run_benchmark:main"' in text
    assert 'chess-llm-run-curriculum = "chess_llm.training.run_curriculum:main"' in text
    assert 'chess-llm-run-eval-split = "chess_llm.sft.run_eval_split:main"' in text
    assert 'chess-llm-make-data = "chess_llm.sft.pipeline:cli"' in text
    assert 'chess-llm-preflight = "chess_llm.sft.preflight:main"' in text
    assert 'chess-llm-upload-data = "chess_llm.sft.hub_upload:main"' in text
    assert (
        'chess-llm-run-eval-harness = "chess_llm.evals.run_eval_harness:main"'
        in text
    )
    assert 'chess-llm-sft-refresh = "chess_llm.autodata.sft_refresh:main"' in text
    assert 'chess-llm-train = "chess_llm.training.train:main"' in text
    assert 'chess-llm-validate-outputs = "chess_llm.sft.validate_outputs:main"' in text


def test_data_extra_contains_archive_dependency_used_by_polyglot_extractor():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]

    assert any(dep.startswith("py7zr") for dep in optional["data"])
    assert any(dep.startswith("py7zr") for dep in optional["dev"])
    assert any(dep.startswith("Pillow") for dep in optional["data"])
    assert any(dep.startswith("Pillow") for dep in optional["dev"])


def test_ui_extra_contains_backend_server_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]

    assert any(dep.startswith("fastapi") for dep in optional["ui"])
    assert any(dep.startswith("uvicorn") for dep in optional["ui"])
