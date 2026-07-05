def test_default_phase_a_base_model_is_qwen35():
    from chess_llm.training.phases import DEFAULT_BASE_MODEL, PHASE_A, resolve_checkpoint

    assert DEFAULT_BASE_MODEL == "Qwen/Qwen3.5-0.8B"
    assert resolve_checkpoint(PHASE_A, output_root=object()) == DEFAULT_BASE_MODEL
