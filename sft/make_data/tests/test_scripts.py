"""Tests for make_data command-line helper modules."""

from __future__ import annotations

import sys
from pathlib import Path


def test_validate_outputs_fails_when_required_task_files_are_missing(monkeypatch, tmp_path: Path):
    from scripts import validate_outputs

    monkeypatch.setattr(sys, "argv", ["validate_outputs.py", "--output-dir", str(tmp_path)])

    assert validate_outputs.main() == 1


def test_syzygy_preflight_fails_when_probe_raises(monkeypatch, tmp_path: Path):
    import chess.syzygy
    from scripts import preflight_check

    (tmp_path / "dummy.rtbw").write_text("not a real tablebase")
    (tmp_path / "dummy.rtbz").write_text("not a real tablebase")
    monkeypatch.setattr(preflight_check, "SYZYGY_PATH", str(tmp_path))

    def raise_probe_error(_path):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(chess.syzygy, "open_tablebase", raise_probe_error)

    ok, detail = preflight_check.check_syzygy()

    assert ok is False
    assert "probe failed" in detail
