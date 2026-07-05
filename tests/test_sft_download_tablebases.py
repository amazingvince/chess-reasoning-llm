from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_source_dirs_uses_current_lichess_layout(monkeypatch):
    from chess_llm.sft import download_tablebases

    monkeypatch.setattr(download_tablebases, "_list_rtb_files", lambda _url: [])
    monkeypatch.setattr(download_tablebases, "_list_dirs", lambda _url: [])

    assert download_tablebases._resolve_source_dirs(
        "https://tablebase.lichess.ovh/tables/standard",
        [3, 4, 5, 6],
        include_wdl=True,
        include_dtz=True,
    ) == [
        "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/",
        "https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/",
        "https://tablebase.lichess.ovh/tables/standard/6-wdl/",
        "https://tablebase.lichess.ovh/tables/standard/6-dtz/",
    ]


def test_resolve_source_dirs_treats_leaf_url_as_source(monkeypatch):
    from chess_llm.sft import download_tablebases

    monkeypatch.setattr(
        download_tablebases,
        "_list_rtb_files",
        lambda _url: ["KQvK.rtbw"],
    )

    assert download_tablebases._resolve_source_dirs(
        "https://example.test/tables",
        [3],
        include_wdl=True,
        include_dtz=True,
    ) == ["https://example.test/tables/"]


def test_filter_files_returns_matching_url_name_tuples():
    from chess_llm.sft.download_tablebases import filter_files

    files = [
        ("https://example.test/KQvK.rtbw", "KQvK.rtbw"),
        ("https://example.test/KQQvK.rtbz", "KQQvK.rtbz"),
        ("https://example.test/readme.txt", "readme.txt"),
    ]

    assert filter_files(files, [3]) == [("https://example.test/KQvK.rtbw", "KQvK.rtbw")]


def test_download_tablebases_dry_run_limits_without_downloading(monkeypatch, tmp_path: Path):
    from chess_llm.sft import download_tablebases

    monkeypatch.setattr(
        download_tablebases,
        "_resolve_source_dirs",
        lambda *_args, **_kwargs: ["https://example.test/3-4-5-wdl/"],
    )
    monkeypatch.setattr(
        download_tablebases,
        "_list_rtb_files",
        lambda _url: ["KQvK.rtbw", "KQRvK.rtbw"],
    )
    downloaded: list[Path] = []

    result = download_tablebases.download_tablebases(
        output_dir=tmp_path,
        base_url="https://example.test/",
        pieces=[3],
        include_wdl=True,
        include_dtz=False,
        max_files=1,
        dry_run=True,
        downloader=lambda _url, dest: downloaded.append(dest),
    )

    assert result.exit_code == 0
    assert result.files == [("https://example.test/3-4-5-wdl/KQvK.rtbw", "KQvK.rtbw")]
    assert result.downloaded == []
    assert downloaded == []


def test_download_tablebases_skips_existing_and_downloads_missing(
    monkeypatch,
    tmp_path: Path,
):
    from chess_llm.sft import download_tablebases

    (tmp_path / "KQvK.rtbw").write_bytes(b"existing")
    monkeypatch.setattr(
        download_tablebases,
        "_resolve_source_dirs",
        lambda *_args, **_kwargs: ["https://example.test/3-4-5-wdl/"],
    )
    monkeypatch.setattr(
        download_tablebases,
        "_list_rtb_files",
        lambda _url: ["KQvK.rtbw", "KRvK.rtbw"],
    )

    def fake_downloader(_url: str, dest: Path) -> None:
        dest.write_bytes(b"downloaded")

    result = download_tablebases.download_tablebases(
        output_dir=tmp_path,
        base_url="https://example.test/",
        pieces=[3],
        include_wdl=True,
        include_dtz=False,
        downloader=fake_downloader,
    )

    assert result.exit_code == 0
    assert result.skipped == [tmp_path / "KQvK.rtbw"]
    assert result.downloaded == [tmp_path / "KRvK.rtbw"]
    assert (tmp_path / "KRvK.rtbw").read_bytes() == b"downloaded"


def test_discover_tablebase_files_ignores_unsafe_or_external_hrefs(monkeypatch):
    from chess_llm.sft import download_tablebases

    monkeypatch.setattr(
        download_tablebases,
        "_resolve_source_dirs",
        lambda *_args, **_kwargs: ["https://example.test/3-4-5-wdl/"],
    )
    monkeypatch.setattr(
        download_tablebases,
        "_list_rtb_files",
        lambda _url: [
            "KQvK.rtbw",
            "../escape.rtbw",
            "/absolute.rtbw",
            "nested/KRPK.rtbw",
            "https://evil.test/KQvK.rtbw",
        ],
    )

    _source_dirs, files = download_tablebases.discover_tablebase_files(
        base_url="https://example.test/",
        pieces=[3],
        include_wdl=True,
        include_dtz=False,
    )

    assert files == [("https://example.test/3-4-5-wdl/KQvK.rtbw", "KQvK.rtbw")]


def test_download_tablebases_uses_atomic_temp_file_on_failure(monkeypatch, tmp_path: Path):
    from chess_llm.sft import download_tablebases

    monkeypatch.setattr(
        download_tablebases,
        "_resolve_source_dirs",
        lambda *_args, **_kwargs: ["https://example.test/3-4-5-wdl/"],
    )
    monkeypatch.setattr(
        download_tablebases,
        "_list_rtb_files",
        lambda _url: ["KQvK.rtbw"],
    )

    def failing_downloader(_url: str, dest: Path) -> None:
        dest.write_bytes(b"partial")
        raise RuntimeError("network interrupted")

    with pytest.raises(RuntimeError, match="network interrupted"):
        download_tablebases.download_tablebases(
            output_dir=tmp_path,
            base_url="https://example.test/",
            pieces=[3],
            include_wdl=True,
            include_dtz=False,
            downloader=failing_downloader,
        )

    assert not (tmp_path / "KQvK.rtbw").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_main_returns_usage_error_for_invalid_piece_list():
    from chess_llm.sft import download_tablebases

    assert download_tablebases.main(["--pieces", "abc"]) == 2


def test_main_returns_usage_error_for_negative_max_files():
    from chess_llm.sft import download_tablebases

    assert download_tablebases.main(["--pieces", "3", "--max-files", "-1"]) == 2


def test_main_returns_failure_for_empty_piece_list():
    from chess_llm.sft import download_tablebases

    assert download_tablebases.main(["--pieces", ""]) == 1
