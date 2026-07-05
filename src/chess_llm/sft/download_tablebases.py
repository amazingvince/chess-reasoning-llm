"""Download Syzygy tablebase files from an HTTP index."""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from chess_llm.sft.settings import SftDataSettings

DEFAULT_BASE_URL = "https://tablebase.lichess.ovh/tables/standard/"
DEFAULT_TIMEOUT_SECONDS = 60

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_ROOT = _REPO_ROOT
SETTINGS = SftDataSettings.from_env(_SETTINGS_ROOT)
SYZYGY_PATH = SETTINGS.syzygy_path

Downloader = Callable[[str, Path], None]


@dataclass(frozen=True)
class TablebaseDownloadResult:
    """Structured result from a tablebase download run."""

    exit_code: int
    source_dirs: list[str]
    files: list[tuple[str, str]]
    downloaded: list[Path]
    skipped: list[Path]


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def _fetch_html(url: str) -> str:
    with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="ignore")


def _list_hrefs(url: str) -> list[str]:
    html = _fetch_html(url)
    return sorted(set(re.findall(r'href="([^"]+)"', html)))


def _list_dirs(url: str) -> list[str]:
    return [
        safe
        for href in _list_hrefs(url)
        if (safe := _safe_dir_href(href)) is not None
    ]


def _list_rtb_files(url: str) -> list[str]:
    return [
        safe
        for href in _list_hrefs(url)
        if (safe := _safe_rtb_href(href)) is not None
    ]


def _safe_dir_href(href: str) -> str | None:
    """Return a safe single-level directory href, or None."""
    parsed = urllib.parse.urlsplit(href)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None
    path = urllib.parse.unquote(parsed.path)
    if not path.endswith("/") or path in ("../", "/"):
        return None
    dirname = path.rstrip("/")
    if not dirname or dirname in (".", ".."):
        return None
    if dirname.startswith("/") or "/" in dirname or "\\" in dirname:
        return None
    return f"{dirname}/"


def _safe_rtb_href(href: str) -> str | None:
    """Return a safe basename for a tablebase href, or None."""
    parsed = urllib.parse.urlsplit(href)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None
    path = urllib.parse.unquote(parsed.path)
    if "/" in path or "\\" in path or path.startswith((".", "/")):
        return None
    if not (path.endswith(".rtbw") or path.endswith(".rtbz")):
        return None
    return path


def _resolve_source_dirs(
    base_url: str,
    pieces: Iterable[int],
    include_wdl: bool,
    include_dtz: bool,
) -> list[str]:
    """Resolve tablebase index directories for piece-count/file-type filters."""
    base_url = _ensure_trailing_slash(base_url)

    if _list_rtb_files(base_url):
        return [base_url]

    pieces_set = {int(piece) for piece in pieces}
    source_dirs: list[str] = []

    if pieces_set & {3, 4, 5}:
        if include_wdl:
            source_dirs.append(urllib.parse.urljoin(base_url, "3-4-5-wdl/"))
        if include_dtz:
            source_dirs.append(urllib.parse.urljoin(base_url, "3-4-5-dtz/"))

    if 6 in pieces_set:
        if include_wdl:
            source_dirs.append(urllib.parse.urljoin(base_url, "6-wdl/"))
        if include_dtz:
            source_dirs.append(urllib.parse.urljoin(base_url, "6-dtz/"))

    if 7 in pieces_set:
        seven_url = urllib.parse.urljoin(base_url, "7/")
        for subdir in _list_dirs(seven_url):
            source_dirs.append(urllib.parse.urljoin(seven_url, subdir))

    if not source_dirs:
        for subdir in _list_dirs(base_url):
            candidate = urllib.parse.urljoin(base_url, subdir)
            if _list_rtb_files(candidate):
                source_dirs.append(candidate)

    return source_dirs


def piece_count(name: str) -> int:
    """Return the number of pieces encoded in a Syzygy filename stem."""
    return sum(1 for ch in name if ch.isalpha() and ch.isupper())


def filter_files(
    files: Iterable[tuple[str, str]],
    pieces: Iterable[int],
) -> list[tuple[str, str]]:
    """Return ``(url, name)`` pairs matching requested piece counts."""
    wanted = {int(piece) for piece in pieces}
    results: list[tuple[str, str]] = []
    for url, name in files:
        if piece_count(Path(name).stem) in wanted:
            results.append((url, name))
    return results


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_atomically(url: str, dest: Path, downloader: Downloader) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_name(f"{dest.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        downloader(url, temp_path)
        temp_path.replace(dest)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def download_file(url: str, dest: Path) -> None:
    """Download one file to ``dest`` through a temporary file."""
    _download_atomically(url, dest, _stream_download)


def discover_tablebase_files(
    *,
    base_url: str,
    pieces: Sequence[int],
    include_wdl: bool,
    include_dtz: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return source directories and filtered tablebase file URLs."""
    source_dirs = _resolve_source_dirs(
        base_url,
        pieces,
        include_wdl=include_wdl,
        include_dtz=include_dtz,
    )
    index_files: list[tuple[str, str]] = []
    for folder_url in source_dirs:
        folder_url = _ensure_trailing_slash(folder_url)
        for href in _list_rtb_files(folder_url):
            name = _safe_rtb_href(href)
            if name is None:
                continue
            if href.endswith(".rtbw") and not include_wdl:
                continue
            if href.endswith(".rtbz") and not include_dtz:
                continue
            index_files.append((urllib.parse.urljoin(folder_url, name), name))
    return source_dirs, filter_files(index_files, pieces)


def download_tablebases(
    *,
    output_dir: str | Path = SYZYGY_PATH,
    base_url: str = DEFAULT_BASE_URL,
    pieces: Sequence[int] = (3, 4, 5),
    include_wdl: bool = True,
    include_dtz: bool = True,
    max_files: int = 0,
    dry_run: bool = False,
    downloader: Downloader = download_file,
) -> TablebaseDownloadResult:
    """Discover and optionally download Syzygy tablebase files."""
    output_path = Path(output_dir)
    source_dirs, filtered = discover_tablebase_files(
        base_url=base_url,
        pieces=pieces,
        include_wdl=include_wdl,
        include_dtz=include_dtz,
    )
    if not source_dirs:
        print("No suitable tablebase directories found at that URL.")
        return TablebaseDownloadResult(1, [], [], [], [])

    if max_files and max_files > 0:
        filtered = filtered[:max_files]

    if dry_run:
        print(f"Found {len(filtered)} files:")
        for _, name in filtered:
            print(f"  {name}")
        return TablebaseDownloadResult(0, source_dirs, filtered, [], [])

    downloaded: list[Path] = []
    skipped: list[Path] = []
    for url, name in filtered:
        dest = output_path / name
        if dest.exists():
            print(f"Skipping existing {dest}")
            skipped.append(dest)
            continue
        print(f"Downloading {url} -> {dest}")
        _download_atomically(url, dest, downloader)
        downloaded.append(dest)

    print("Done.")
    return TablebaseDownloadResult(0, source_dirs, filtered, downloaded, skipped)


def _parse_pieces(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Syzygy tablebases.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(SYZYGY_PATH),
        help="Directory to store tablebase files",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for the tablebase index",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default="3,4,5",
        help="Comma-separated piece counts to download (e.g. 3,4,5)",
    )
    parser.add_argument("--wdl", action="store_true", help="Download WDL files (.rtbw)")
    parser.add_argument("--dtz", action="store_true", help="Download DTZ files (.rtbz)")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of files to download (0 = no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be downloaded without downloading",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    include_wdl = args.wdl or not args.dtz
    include_dtz = args.dtz or not args.wdl

    try:
        pieces = _parse_pieces(args.pieces)
    except ValueError as exc:
        print(f"Invalid --pieces value: {exc}")
        return 2
    if not pieces:
        print("No piece counts provided.")
        return 1
    if args.max_files < 0:
        print("--max-files must be greater than or equal to 0.")
        return 2

    print(f"Fetching index from {args.base_url}")
    result = download_tablebases(
        output_dir=args.output_dir,
        base_url=args.base_url,
        pieces=pieces,
        include_wdl=include_wdl,
        include_dtz=include_dtz,
        max_files=args.max_files,
        dry_run=args.dry_run,
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BASE_URL",
    "SYZYGY_PATH",
    "TablebaseDownloadResult",
    "_ensure_trailing_slash",
    "_fetch_html",
    "_list_dirs",
    "_list_hrefs",
    "_list_rtb_files",
    "_resolve_source_dirs",
    "_safe_dir_href",
    "_safe_rtb_href",
    "build_arg_parser",
    "discover_tablebase_files",
    "download_file",
    "download_tablebases",
    "filter_files",
    "main",
    "piece_count",
]
