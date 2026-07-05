def test_evaluate_reexports_extracted_helper_boundaries():
    from chess_llm.training import evaluate
    from chess_llm.training import eval_artifacts, eval_config, eval_scoring, eval_wandb

    assert evaluate.EvaluationConfig is eval_config.EvaluationConfig
    assert evaluate.EvaluationResult is eval_config.EvaluationResult
    assert evaluate._evaluation_result is eval_artifacts.evaluation_result
    assert evaluate._write_evaluation_result_artifact is (
        eval_artifacts.write_evaluation_result_artifact
    )
    assert evaluate._finalize_evaluation_result_artifacts is (
        eval_artifacts.finalize_evaluation_result_artifacts
    )
    assert evaluate._build_wandb_payload is eval_wandb.build_wandb_payload
    assert evaluate._maybe_log_to_wandb is eval_wandb.maybe_log_to_wandb
    assert evaluate._score_evaluation_splits is eval_scoring.score_evaluation_splits
