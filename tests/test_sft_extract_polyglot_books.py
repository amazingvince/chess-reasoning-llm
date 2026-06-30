from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path


def _clear_legacy_extract_modules() -> None:
    for module_name in (
        "config.settings",
        "scripts.extract_polyglot_books",
        "sft.make_data.scripts.extract_polyglot_books",
    ):
        sys.modules.pop(module_name, None)


def test_package_extract_polyglot_imports_without_legacy_modules():
    _clear_legacy_extract_modules()

    module = importlib.import_module("chess_llm.sft.extract_polyglot_books")

    assert module.extract_zip
    assert module.extract_7z
    assert module.extract_polyglot_books
    assert "config.settings" not in sys.modules
    assert "scripts.extract_polyglot_books" not in sys.modules


def test_extract_zip_flattens_bin_files_and_skips_non_bins(tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_zip

    archive = tmp_path / "books.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/book.bin", b"polyglot")
        zf.writestr("nested/readme.txt", b"ignore me")
        zf.writestr("../escape.bin", b"safe basename")

    count = extract_zip(archive, tmp_path)

    assert count == 2
    assert (tmp_path / "book.bin").read_bytes() == b"polyglot"
    assert (tmp_path / "escape.bin").read_bytes() == b"safe basename"
    assert not (tmp_path / "nested").exists()
    assert not (tmp_path.parent / "escape.bin").exists()


def test_extract_zip_flattens_backslash_member_names(tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_zip

    archive = tmp_path / "books.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested\\windows.bin", b"book")

    assert extract_zip(archive, tmp_path) == 1
    assert (tmp_path / "windows.bin").read_bytes() == b"book"
    assert not (tmp_path / "nested\\windows.bin").exists()


def test_extract_polyglot_books_reports_missing_directory(tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_polyglot_books

    result = extract_polyglot_books(tmp_path / "missing")

    assert result.exit_code == 1
    assert result.extracted_count == 0
    assert result.total_bin_files == 0


def test_extract_polyglot_books_processes_zip_archives(tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_polyglot_books

    archive = tmp_path / "collection.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("book.bin", b"book")
        zf.writestr("notes.md", b"notes")

    result = extract_polyglot_books(tmp_path)

    assert result.exit_code == 0
    assert result.extracted_count == 1
    assert result.total_bin_files == 1
    assert result.processed_archives == [archive]


def test_extract_polyglot_books_main_accepts_argv(tmp_path: Path):
    from chess_llm.sft import extract_polyglot_books

    archive = tmp_path / "collection.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("book.bin", b"book")

    assert extract_polyglot_books.main(["--polyglot-dir", str(tmp_path)]) == 0


def test_extract_7z_skips_when_py7zr_missing(monkeypatch, tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_7z

    archive = tmp_path / "books.7z"
    archive.write_bytes(b"not a real archive")
    monkeypatch.setitem(sys.modules, "py7zr", None)

    assert extract_7z(archive, tmp_path) == 0


def test_extract_7z_uses_keyword_targets_and_flattens_from_temp_dir(tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_7z

    calls: list[dict] = []

    class FakeSevenZip:
        def __init__(self, archive: Path, mode: str):
            self.archive = archive
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def getnames(self):
            return ["nested/book.bin", "notes.txt"]

        def extract(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            extraction_dir = Path(kwargs["path"])
            (extraction_dir / "nested").mkdir(parents=True)
            (extraction_dir / "nested" / "book.bin").write_bytes(b"book")

    archive = tmp_path / "books.7z"
    archive.write_bytes(b"fake")

    assert extract_7z(archive, tmp_path, seven_zip_cls=FakeSevenZip) == 1
    assert calls[0]["args"] == ()
    assert calls[0]["kwargs"]["targets"] == ["nested/book.bin"]
    assert Path(calls[0]["kwargs"]["path"]) != tmp_path
    assert (tmp_path / "book.bin").read_bytes() == b"book"
    assert not (tmp_path / "nested").exists()


def test_extract_7z_does_not_pass_traversal_targets_to_extractor(tmp_path: Path):
    from chess_llm.sft.extract_polyglot_books import extract_7z

    calls: list[dict] = []

    class FakeSevenZip:
        def __init__(self, archive: Path, mode: str):
            self.archive = archive
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def getnames(self):
            return ["../escape.bin", "/abs.bin", "safe/book.bin"]

        def extract(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            extraction_dir = Path(kwargs["path"])
            (extraction_dir / "safe").mkdir(parents=True)
            (extraction_dir / "safe" / "book.bin").write_bytes(b"book")

    archive = tmp_path / "books.7z"
    archive.write_bytes(b"fake")

    assert extract_7z(archive, tmp_path, seven_zip_cls=FakeSevenZip) == 1
    assert calls[0]["kwargs"]["targets"] == ["safe/book.bin"]
    assert not (tmp_path.parent / "escape.bin").exists()
    assert not (tmp_path / "abs.bin").exists()


def test_legacy_extract_polyglot_import_aliases_package_module(monkeypatch):
    package_module = importlib.import_module("chess_llm.sft.extract_polyglot_books")
    make_data_root = Path(__file__).resolve().parents[1] / "sft" / "make_data"
    monkeypatch.syspath_prepend(str(make_data_root))
    _clear_legacy_extract_modules()

    legacy_module = importlib.import_module("scripts.extract_polyglot_books")

    assert legacy_module is package_module
