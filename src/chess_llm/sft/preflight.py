#!/usr/bin/env python3
"""Preflight check: verify external dependencies before running the pipeline.

Exit code 0 = all pass, 1 = any fail.

Usage:
    python scripts/preflight_check.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chess_llm.sft.settings import (
    SftDataSettings,
    apply_hf_cache_env,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MAKE_DATA_ROOT = _REPO_ROOT / "sft" / "make_data"
_SETTINGS_ROOT = _LEGACY_MAKE_DATA_ROOT if _LEGACY_MAKE_DATA_ROOT.exists() else Path.cwd()
SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)

HF_DATASETS = dict(SETTINGS.hf_datasets)
OUTPUT_DIR = SETTINGS.output_dir
POLYGLOT_DIR = SETTINGS.polyglot_dir
STOCKFISH_PATH = SETTINGS.stockfish_path
SYZYGY_PATH = SETTINGS.syzygy_path

def check_python_deps() -> tuple[bool, str]:
    """Check that required Python packages are importable."""
    missing = []
    for pkg in ("chess", "datasets", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All required packages importable"


def check_stockfish() -> tuple[bool, str]:
    """Check that Stockfish binary exists and responds to UCI."""
    path = Path(STOCKFISH_PATH)
    if not path.exists():
        return False, f"Binary not found: {path}"
    try:
        proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = proc.communicate(input=b"uci\nquit\n", timeout=5)
        if b"uciok" in stdout:
            return True, f"Stockfish OK at {path}"
        return False, "Stockfish did not respond with 'uciok'"
    except subprocess.TimeoutExpired:
        proc.kill()
        return False, "Stockfish timed out (5s)"
    except Exception as exc:
        return False, f"Stockfish error: {exc}"


def check_syzygy() -> tuple[bool, str]:
    """Check that Syzygy tablebase path exists and can probe WDL/DTZ."""
    path = Path(SYZYGY_PATH)
    if not path.exists():
        return False, f"Directory not found: {path}"
    rtbw_files = list(path.glob("*.rtbw"))
    if not rtbw_files:
        return False, f"No .rtbw files in {path}"
    rtbz_files = list(path.glob("*.rtbz"))
    if not rtbz_files:
        return False, f"No .rtbz files in {path}"

    # Try probing a KRK position
    tb = None
    try:
        import chess
        import chess.syzygy

        tb = chess.syzygy.open_tablebase(str(path))
        board = chess.Board("8/8/8/4k3/8/8/8/4K2R w - - 0 1")
        wdl = tb.probe_wdl(board)
        dtz = tb.probe_dtz(board)
        if wdl == 2:
            return (
                True,
                "Syzygy OK "
                f"({len(rtbw_files)} .rtbw, {len(rtbz_files)} .rtbz, "
                f"KRK probe=Win, DTZ={dtz})",
            )
        return False, f"KRK probe returned WDL={wdl}, expected 2"
    except Exception as exc:
        return False, (
            "Syzygy files present "
            f"({len(rtbw_files)} .rtbw, {len(rtbz_files)} .rtbz) "
            f"but probe failed: {exc}"
        )
    finally:
        if tb is not None:
            try:
                tb.close()
            except Exception:
                pass


def check_hf_datasets() -> tuple[bool, str]:
    """Check that HuggingFace datasets are accessible (streaming, 1 row)."""
    apply_hf_cache_env(SETTINGS)
    try:
        from datasets import load_dataset
    except ImportError:
        return False, "datasets package not installed"

    failures = []
    for key, repo_id in HF_DATASETS.items():
        try:
            ds = load_dataset(repo_id, split="train", streaming=True)
            row = next(iter(ds))
            if not row:
                failures.append(f"{key}: empty first row")
        except Exception as exc:
            failures.append(f"{key}: {exc}")

    if failures:
        return False, "HF dataset issues:\n  " + "\n  ".join(failures)
    return True, f"All {len(HF_DATASETS)} HF datasets accessible"


def check_polyglot() -> tuple[bool, str]:
    """Check that Polyglot opening books directory has .bin files."""
    path = Path(POLYGLOT_DIR)
    if not path.exists():
        return False, f"Directory not found: {path}"
    bins = list(path.glob("*.bin"))
    if not bins:
        return False, f"No .bin files in {path}"
    return True, f"Polyglot OK ({len(bins)} .bin files)"


def check_output_dir() -> tuple[bool, str]:
    """Check that the output directory is writable."""
    path = Path(OUTPUT_DIR)
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".preflight_test"
        test_file.write_text("ok")
        test_file.unlink()
        return True, f"Output dir writable: {path}"
    except Exception as exc:
        return False, f"Output dir not writable ({path}): {exc}"


CHECKS = [
    ("Python deps", check_python_deps),
    ("Stockfish", check_stockfish),
    ("Syzygy tables", check_syzygy),
    ("HF datasets", check_hf_datasets),
    ("Polyglot books", check_polyglot),
    ("Output dir", check_output_dir),
]


def main() -> int:
    any_fail = False
    for name, fn in CHECKS:
        ok, detail = fn()
        status = "[PASS]" if ok else "[FAIL]"
        if not ok:
            any_fail = True
        print(f"  {status} {name}: {detail}")

    print()
    if any_fail:
        print("Some checks failed. Fix issues above before running the pipeline.")
        return 1
    else:
        print("All preflight checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CHECKS",
    "HF_DATASETS",
    "OUTPUT_DIR",
    "POLYGLOT_DIR",
    "STOCKFISH_PATH",
    "SYZYGY_PATH",
    "check_hf_datasets",
    "check_output_dir",
    "check_polyglot",
    "check_python_deps",
    "check_stockfish",
    "check_syzygy",
    "main",
]
