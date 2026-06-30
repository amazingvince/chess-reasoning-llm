def test_legacy_phase_config_reexports_package_objects():
    import importlib.util
    from pathlib import Path

    phase_path = (
        Path(__file__).resolve().parents[1]
        / "sft"
        / "training"
        / "config"
        / "phases.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_training_phases", phase_path)
    assert spec is not None
    assert spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    from chess_llm.training import phases as package

    assert legacy.PhaseConfig is package.PhaseConfig
    assert legacy.TierMix is package.TierMix
    assert legacy.DEFAULT_BASE_MODEL == package.DEFAULT_BASE_MODEL
    assert legacy.PHASES == package.PHASES
    assert legacy.resolve_checkpoint is package.resolve_checkpoint


def test_default_phase_a_base_model_is_qwen35():
    from chess_llm.training.phases import DEFAULT_BASE_MODEL, PHASE_A, resolve_checkpoint

    assert DEFAULT_BASE_MODEL == "Qwen/Qwen3.5-0.8B"
    assert resolve_checkpoint(PHASE_A, output_root=object()) == DEFAULT_BASE_MODEL
