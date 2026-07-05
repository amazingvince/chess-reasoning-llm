#!/usr/bin/env python3
"""Benchmark chess-llm eval throughput across GPU/batch settings.

This is an operator probe, not a trainer entrypoint. It runs bounded eval jobs,
captures wall time, optionally samples nvidia-smi, and writes a small summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import NamedTuple


class ProbeCommand(NamedTuple):
    argv: list[str]
    env: dict[str, str]


def _parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in parsed):
        raise ValueError("all integers must be > 0")
    return parsed


def parse_cuda_visible_devices(value: str) -> list[str]:
    """Parse semicolon-separated CUDA_VISIBLE_DEVICES probe values."""
    parsed = [item.strip() for item in value.split(";") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one value")
    return parsed


def build_eval_command(
    *,
    model: str,
    benchmark_dir: str | Path,
    output: str | Path,
    batch_size: int,
    cuda_visible_devices: str,
    split: str | None = "planning",
    phase: str = "c",
    max_examples_per_split: int = 100,
    pass_k: int = 8,
    max_new_tokens: int = 256,
    attn_implementation: str = "flash_attention_3",
    acpl_depth: int = 12,
    no_acpl: bool = True,
    python_executable: str = "python",
) -> ProbeCommand:
    """Build a deterministic eval command for one probe setting."""
    argv = [
        python_executable,
        "-m",
        "chess_llm.training.evaluate",
        "--model",
        str(model),
        "--benchmark-dir",
        str(benchmark_dir),
        "--output",
        str(output),
        "--phase",
        phase,
        "--pass-k",
        str(pass_k),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--attn-implementation",
        attn_implementation,
        "--max-examples-per-split",
        str(max_examples_per_split),
        "--acpl-depth",
        str(acpl_depth),
        "--report-only",
        "--no-wandb",
    ]
    if split:
        argv.extend(["--split", split])
    if no_acpl:
        argv.append("--no-acpl")
    return ProbeCommand(
        argv=argv,
        env={"CUDA_VISIBLE_DEVICES": str(cuda_visible_devices)},
    )


def summarize_nvidia_csv(path: str | Path) -> dict[str, dict[str, float | int]]:
    """Summarize nvidia-smi CSV produced by this script."""
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}

    by_gpu: dict[str, dict[str, list[float]]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index = str(row.get(" index", row.get("index", ""))).strip()
            util_text = str(
                row.get(" utilization.gpu [%]", row.get("utilization.gpu [%]", "0"))
            )
            memory_text = str(
                row.get(" memory.used [MiB]", row.get("memory.used [MiB]", "0"))
            )
            try:
                util = float(util_text.strip().rstrip("%"))
                memory = float(memory_text.strip().split()[0])
            except (ValueError, IndexError):
                continue
            bucket = by_gpu.setdefault(index, {"util": [], "memory": []})
            bucket["util"].append(util)
            bucket["memory"].append(memory)

    summary: dict[str, dict[str, float | int]] = {}
    for index, values in sorted(by_gpu.items()):
        util = values["util"]
        memory = values["memory"]
        if not util:
            continue
        summary[index] = {
            "samples": len(util),
            "avg_gpu_util": round(sum(util) / len(util), 2),
            "max_gpu_util": round(max(util), 2),
            "max_memory_mib": round(max(memory), 2) if memory else 0.0,
        }
    return summary


def _start_gpu_monitor(path: Path) -> subprocess.Popen[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
            "--format=csv",
            "-l",
            "1",
        ],
        stdout=path.open("w", encoding="utf-8"),
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_probe(command: ProbeCommand, *, log_path: Path) -> dict[str, object]:
    """Run one eval command and return a compact result payload."""
    env = os.environ.copy()
    env.update(command.env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command.argv,
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "argv": command.argv,
        "env": command.env,
        "log_path": str(log_path),
    }


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Eval Speed Probe",
        "",
        "| cuda | batch | no_acpl | elapsed_s | rc | max_mem_mib | avg_gpu_util |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        gpu_summary = row.get("gpu_summary") or {}
        max_memory = 0.0
        avg_utils: list[float] = []
        if isinstance(gpu_summary, dict):
            for payload in gpu_summary.values():
                if isinstance(payload, dict):
                    max_memory = max(max_memory, float(payload.get("max_memory_mib", 0)))
                    avg_utils.append(float(payload.get("avg_gpu_util", 0)))
        avg_gpu = sum(avg_utils) / len(avg_utils) if avg_utils else 0.0
        lines.append(
            "| {cuda} | {batch} | {no_acpl} | {elapsed} | {rc} | {mem:.0f} | {util:.1f} |".format(
                cuda=row.get("cuda_visible_devices"),
                batch=row.get("batch_size"),
                no_acpl=row.get("no_acpl"),
                elapsed=row.get("elapsed_seconds"),
                rc=row.get("returncode"),
                mem=max_memory,
                util=avg_gpu,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune chess-llm eval throughput")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--cuda-visible-devices",
        default="0;0,1",
        help="Semicolon-separated CUDA_VISIBLE_DEVICES values, for example '0;0,1'",
    )
    parser.add_argument("--batch-sizes", default="16,64,128", help="Comma-separated batch sizes")
    parser.add_argument("--split", default="planning", help="Benchmark split; use empty string for phase defaults")
    parser.add_argument("--phase", default="c")
    parser.add_argument("--max-examples-per-split", type=int, default=100)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default="flash_attention_3")
    parser.add_argument("--acpl-depth", type=int, default=12)
    parser.add_argument("--with-acpl", action="store_true", help="Enable ACPL/WPD during probes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cuda_values = parse_cuda_visible_devices(args.cuda_visible_devices)
    batch_sizes = _parse_csv_ints(args.batch_sizes)
    split = args.split.strip() or None

    for cuda_value in cuda_values:
        for batch_size in batch_sizes:
            safe_cuda = cuda_value.replace(",", "_").replace(" ", "")
            stem = f"cuda{safe_cuda}_bs{batch_size}_{'acpl' if args.with_acpl else 'noacpl'}"
            output_path = args.output_dir / f"{stem}.jsonl"
            log_path = args.output_dir / f"{stem}.log"
            monitor_path = args.output_dir / f"{stem}.nvidia.csv"
            command = build_eval_command(
                model=args.model,
                benchmark_dir=args.benchmark_dir,
                output=output_path,
                batch_size=batch_size,
                cuda_visible_devices=cuda_value,
                split=split,
                phase=args.phase,
                max_examples_per_split=args.max_examples_per_split,
                pass_k=args.pass_k,
                max_new_tokens=args.max_new_tokens,
                attn_implementation=args.attn_implementation,
                acpl_depth=args.acpl_depth,
                no_acpl=not args.with_acpl,
                python_executable=sys.executable,
            )
            monitor = _start_gpu_monitor(monitor_path)
            try:
                result = run_probe(command, log_path=log_path)
            finally:
                _stop_process(monitor)
            result.update(
                {
                    "cuda_visible_devices": cuda_value,
                    "batch_size": batch_size,
                    "no_acpl": not args.with_acpl,
                    "output_path": str(output_path),
                    "gpu_summary": summarize_nvidia_csv(monitor_path),
                }
            )
            rows.append(result)
            print(
                f"cuda={cuda_value} batch={batch_size} "
                f"elapsed={result['elapsed_seconds']}s rc={result['returncode']}"
            )

    summary_path = args.output_dir / "eval_speed_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(rows, args.output_dir / "eval_speed_summary.md")
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
