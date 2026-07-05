from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_speed_probe_builds_single_gpu_batch_command(tmp_path: Path):
    module = _load_script("tune_eval_speed.py")

    command = module.build_eval_command(
        model="/models/best",
        benchmark_dir="/data/benchmark",
        output=tmp_path / "probe.jsonl",
        batch_size=64,
        cuda_visible_devices="0",
        split="planning",
        max_examples_per_split=100,
        pass_k=8,
        max_new_tokens=256,
        acpl_depth=12,
        no_acpl=False,
    )

    assert command.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert command.argv[:3] == ["python", "-m", "chess_llm.training.evaluate"]
    assert ["--batch-size", "64"] == command.argv[
        command.argv.index("--batch-size"): command.argv.index("--batch-size") + 2
    ]
    assert ["--split", "planning"] == command.argv[
        command.argv.index("--split"): command.argv.index("--split") + 2
    ]
    assert "--no-acpl" not in command.argv


def test_eval_speed_probe_summarizes_nvidia_smi_csv(tmp_path: Path):
    module = _load_script("tune_eval_speed.py")
    csv_path = tmp_path / "nvidia.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp, index, utilization.gpu [%], memory.used [MiB], memory.total [MiB]",
                "now, 0, 10 %, 1000 MiB, 81559 MiB",
                "now, 1, 0 %, 0 MiB, 81559 MiB",
                "later, 0, 50 %, 1500 MiB, 81559 MiB",
                "later, 1, 5 %, 500 MiB, 81559 MiB",
            ]
        ),
        encoding="utf-8",
    )

    summary = module.summarize_nvidia_csv(csv_path)

    assert summary["0"]["samples"] == 2
    assert summary["0"]["avg_gpu_util"] == 30.0
    assert summary["0"]["max_gpu_util"] == 50.0
    assert summary["0"]["max_memory_mib"] == 1500.0
    assert summary["1"]["avg_gpu_util"] == 2.5


def test_eval_speed_probe_parses_cuda_device_sets():
    module = _load_script("tune_eval_speed.py")

    assert module.parse_cuda_visible_devices("0;0,1;1") == ["0", "0,1", "1"]


def test_stockfish_probe_builds_worker_thread_matrix():
    module = _load_script("tune_stockfish_speed.py")

    matrix = module.build_probe_matrix(
        worker_counts="1,4",
        threads_per_worker="1,2",
    )

    assert [item.as_dict() for item in matrix] == [
        {"workers": 1, "threads_per_worker": 1},
        {"workers": 1, "threads_per_worker": 2},
        {"workers": 4, "threads_per_worker": 1},
        {"workers": 4, "threads_per_worker": 2},
    ]


def test_stockfish_probe_extracts_unique_fens_from_jsonl(tmp_path: Path):
    module = _load_script("tune_stockfish_speed.py")
    path = tmp_path / "planning.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"fen": "fen a", "task_type": "best_move"}',
                '{"fen": "fen a", "task_type": "best_move"}',
                '{"metadata": {"fen": "fen b"}, "task": "7.2_puzzle_solving"}',
                '{"prompt": "no fen here"}',
            ]
        ),
        encoding="utf-8",
    )

    assert module.load_fens_from_jsonl(path, limit=10) == ["fen a", "fen b"]
