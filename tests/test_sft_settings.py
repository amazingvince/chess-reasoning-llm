import os
from pathlib import Path

from chess_llm.sft.settings import (
    DEFAULT_VOLUME_LICHESS_GAME_DATA_FILES,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HF_DATASETS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STOCKFISH_PATH,
    SftDataSettings,
    apply_hf_cache_env,
)


def test_sft_data_settings_from_env_does_not_mutate_environment(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    project_root = Path("C:/repo/sft/make_data")

    settings = SftDataSettings.from_env(
        project_root,
        env={"CHESS_SFT_OUTPUT": "D:/data", "STOCKFISH_PATH": "D:/sf.exe"},
    )

    assert os.environ.get("HF_HOME") is None
    assert settings.hf_cache_dir == DEFAULT_HF_CACHE_DIR
    assert settings.output_dir == Path("D:/data")
    assert settings.pool_dir == Path("D:/data") / "pool"
    assert settings.benchmark_dir == Path("D:/data") / "benchmark"
    assert settings.stockfish_path == "D:/sf.exe"
    assert settings.syzygy_path == str(project_root / "data" / "syzygy")
    assert settings.polyglot_dir == str(project_root / "polyglot_opening_books")
    assert settings.hf_datasets == DEFAULT_HF_DATASETS
    assert settings.volume_lichess_game_data_files == DEFAULT_VOLUME_LICHESS_GAME_DATA_FILES


def test_apply_hf_cache_env_preserves_existing_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "Z:/existing")
    settings = SftDataSettings.from_env(Path("C:/repo/sft/make_data"), env={})

    apply_hf_cache_env(settings)

    assert os.environ["HF_HOME"] == "Z:/existing"


def test_apply_hf_cache_env_sets_default_when_missing(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    settings = SftDataSettings.from_env(Path("C:/repo/sft/make_data"), env={})

    apply_hf_cache_env(settings)

    assert os.environ["HF_HOME"] == DEFAULT_HF_CACHE_DIR
    assert settings.output_dir.name == DEFAULT_OUTPUT_DIR
    assert settings.output_dir.is_absolute()


def test_default_paths_are_portable_and_not_machine_specific():
    settings = SftDataSettings.from_env(Path("project"), env={})
    values = [
        DEFAULT_OUTPUT_DIR,
        DEFAULT_STOCKFISH_PATH,
        settings.stockfish_path,
        settings.syzygy_path,
        settings.polyglot_dir,
    ]

    for value in values:
        normalized = str(value).replace("\\", "/")
        assert not normalized.startswith("E:/")
        assert "C:/Users/amazi/" not in normalized
        assert "Downloads/stockfish" not in normalized

    assert Path(DEFAULT_HF_CACHE_DIR).is_absolute()
    assert "chess_sft_data" not in DEFAULT_HF_CACHE_DIR.replace("\\", "/")


def test_default_output_dir_resolves_against_package_project_root():
    import chess_llm.sft.settings as settings_module

    settings = SftDataSettings.from_env(Path("unrelated-project"), env={})

    expected_root = Path(settings_module.__file__).resolve().parents[3]
    assert settings.output_dir == expected_root / DEFAULT_OUTPUT_DIR
    assert settings.pool_dir == expected_root / DEFAULT_OUTPUT_DIR / "pool"


def test_output_dir_env_override_is_used_as_given(tmp_path):
    settings = SftDataSettings.from_env(
        tmp_path / "project",
        env={"CHESS_SFT_OUTPUT": str(tmp_path / "data")},
    )

    assert settings.output_dir == tmp_path / "data"


def test_settings_support_explicit_asset_and_cache_overrides(tmp_path):
    settings = SftDataSettings.from_env(
        tmp_path / "project",
        env={
            "CHESS_SFT_OUTPUT": str(tmp_path / "data"),
            "HF_HOME": str(tmp_path / "hf-cache"),
            "POLYGLOT_DIR": str(tmp_path / "books"),
            "SYZYGY_PATH": str(tmp_path / "syzygy"),
            "STOCKFISH_PATH": str(tmp_path / "stockfish"),
        },
    )

    assert settings.output_dir == tmp_path / "data"
    assert settings.hf_cache_dir == str(tmp_path / "hf-cache")
    assert settings.polyglot_dir == str(tmp_path / "books")
    assert settings.syzygy_path == str(tmp_path / "syzygy")
    assert settings.stockfish_path == str(tmp_path / "stockfish")
