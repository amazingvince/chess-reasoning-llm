from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _clear_legacy_preflight_modules() -> None:
    for module_name in (
        "scripts.preflight_check",
        "sft.make_data.scripts.preflight_check",
    ):
        sys.modules.pop(module_name, None)


def test_package_preflight_imports_without_legacy_script_module():
    _clear_legacy_preflight_modules()

    module = importlib.import_module("chess_llm.sft.preflight")

    assert module.check_syzygy
    assert module.main
    assert "scripts.preflight_check" not in sys.modules
    assert "sft.make_data.scripts.preflight_check" not in sys.modules


def test_package_preflight_syzygy_reports_probe_errors(monkeypatch, tmp_path: Path):
    import chess.syzygy
    from chess_llm.sft import preflight

    (tmp_path / "dummy.rtbw").write_text("not a real tablebase", encoding="utf-8")
    (tmp_path / "dummy.rtbz").write_text("not a real tablebase", encoding="utf-8")
    monkeypatch.setattr(preflight, "SYZYGY_PATH", str(tmp_path))

    def raise_probe_error(_path):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(chess.syzygy, "open_tablebase", raise_probe_error)

    ok, detail = preflight.check_syzygy()

    assert ok is False
    assert "probe failed" in detail


def test_package_preflight_main_returns_failure_when_any_check_fails(monkeypatch, capsys):
    from chess_llm.sft import preflight

    monkeypatch.setattr(
        preflight,
        "CHECKS",
        [
            ("Good", lambda: (True, "ok")),
            ("Bad", lambda: (False, "broken")),
        ],
    )

    assert preflight.main() == 1
    output = capsys.readouterr().out
    assert "[PASS] Good: ok" in output
    assert "[FAIL] Bad: broken" in output


def test_legacy_preflight_import_aliases_package_module(monkeypatch):
    package_module = importlib.import_module("chess_llm.sft.preflight")
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    _clear_legacy_preflight_modules()

    legacy_module = importlib.import_module("scripts.preflight_check")

    assert legacy_module is package_module
