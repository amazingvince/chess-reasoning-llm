from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_built_wheel_exposes_package_and_ui_backend(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    wheelhouse = tmp_path / "wheelhouse"
    install_target = tmp_path / "site"
    outside_cwd = tmp_path / "outside"
    wheelhouse.mkdir()
    outside_cwd.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
            str(repo_root),
        ],
        check=True,
        cwd=outside_cwd,
    )
    wheels = sorted(wheelhouse.glob("chess_llm-*.whl"))
    assert len(wheels) == 1

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_target),
            str(wheels[0]),
        ],
        check=True,
        cwd=outside_cwd,
    )

    script = """
import importlib
import importlib.metadata as metadata
import importlib.resources as resources
import pathlib
import sys

target = pathlib.Path(sys.argv[1]).resolve()
chess_llm = importlib.import_module("chess_llm")
chess_llm_ui = importlib.import_module("chess_llm_ui")
for module in (chess_llm, chess_llm_ui):
    module_path = pathlib.Path(module.__file__).resolve()
    if not module_path.is_relative_to(target):
        raise AssertionError(f"{module.__name__} imported from {module_path}, not {target}")

if metadata.version("chess_llm") != chess_llm.__version__:
    raise AssertionError("installed metadata version does not match package version")
if not resources.files("chess_llm").joinpath("py.typed").is_file():
    raise AssertionError("py.typed missing from installed package")

entry_points = {
    entry.name: entry.value
    for entry in metadata.distribution("chess_llm").entry_points
    if entry.group == "console_scripts"
}
expected = {
    "chess-llm-make-data": "chess_llm.sft.pipeline:cli",
    "chess-llm-run-eval-split": "chess_llm.sft.run_eval_split:main",
    "chess-llm-upload-data": "chess_llm.sft.hub_upload:main",
    "chess-llm-train": "chess_llm.training.train:main",
    "chess-llm-evaluate": "chess_llm.training.evaluate:main",
    "chess-llm-export-vllm": "chess_llm.training.vllm_export:main",
    "chess-llm-run-curriculum": "chess_llm.training.run_curriculum:main",
}
missing = {
    name: value
    for name, value in expected.items()
    if entry_points.get(name) != value
}
if missing:
    raise AssertionError(f"missing or wrong entry points: {missing}")
for name in expected:
    loaded = metadata.distribution("chess_llm").entry_points.select(
        group="console_scripts",
        name=name,
    )[0].load()
    if not callable(loaded):
        raise AssertionError(f"entry point {name} did not load to a callable")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_target)
    subprocess.run(
        [sys.executable, "-c", script, str(install_target)],
        check=True,
        cwd=outside_cwd,
        env=env,
    )
