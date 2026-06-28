from chess_llm.artifacts.schemas import (
    ChatMessage,
    FeedbackDistillationArtifact,
    JudgmentArtifact,
    ParsedAnswer,
    PreferencePairArtifact,
    PromptArtifact,
    RolloutArtifact,
)


def test_prompt_artifact_round_trip_preserves_required_fields():
    artifact = PromptArtifact(
        prompt_id="prompt-1",
        messages=[
            ChatMessage(role="system", content="You are a chess engine."),
            ChatMessage(role="user", content="FEN: 8/8/8/8/8/8/4K3/4k3 w - - 0 1"),
        ],
        fen="8/8/8/8/8/8/4K3/4k3 w - - 0 1",
        task_type="best_move",
        metadata={"split": "smoke"},
    )

    restored = PromptArtifact.from_dict(artifact.to_dict())

    assert restored.schema_version == "artifact.v1"
    assert restored.artifact_type == "prompt"
    assert restored.prompt_id == "prompt-1"
    assert restored.messages[1].role == "user"
    assert restored.metadata == {"split": "smoke"}


def test_rollout_artifact_round_trip_preserves_nested_parsed_answer():
    artifact = RolloutArtifact(
        rollout_id="rollout-1",
        prompt_id="prompt-1",
        model_id="Qwen/Qwen3-4B",
        raw_output="<think>Only legal move.</think>\n<move>e2e3</move>",
        parsed_answer=ParsedAnswer(
            raw_text="<think>Only legal move.</think>\n<move>e2e3</move>",
            move_uci="e2e3",
            format_type="move_tag",
        ),
        metadata={"temperature": 0.2},
    )

    restored = RolloutArtifact.from_dict(artifact.to_dict())

    assert restored.artifact_type == "rollout"
    assert restored.parsed_answer.move_uci == "e2e3"
    assert restored.parsed_answer.format_type == "move_tag"
    assert restored.metadata["temperature"] == 0.2


def test_downstream_artifacts_round_trip_preserves_identity_fields():
    judgment = JudgmentArtifact(
        judgment_id="judgment-1",
        rollout_id="rollout-1",
        legal=True,
        regret_cp=34.5,
        failure_bucket=None,
        teacher_move_uci="e2e4",
        feedback="The move is legal but not best.",
    )
    pair = PreferencePairArtifact(
        pair_id="pair-1",
        prompt_id="prompt-1",
        chosen_output="<move>e2e4</move>",
        rejected_output="<move>e2e3</move>",
        chosen_move_uci="e2e4",
        rejected_move_uci="e2e3",
        reason="Lower engine regret.",
    )
    feedback = FeedbackDistillationArtifact(
        example_id="feedback-1",
        prompt_id="prompt-1",
        student_output="<move>e2e3</move>",
        feedback="Best move is e2e4 because it controls the center.",
        teacher_output="<think>Take the center.</think>\n<move>e2e4</move>",
        target_output="<think>Take the center.</think>\n<move>e2e4</move>",
    )

    assert JudgmentArtifact.from_dict(judgment.to_dict()).artifact_type == "judgment"
    assert PreferencePairArtifact.from_dict(pair.to_dict()).chosen_move_uci == "e2e4"
    assert (
        FeedbackDistillationArtifact.from_dict(feedback.to_dict()).artifact_type
        == "feedback_distillation"
    )


def test_artifact_from_dict_rejects_wrong_artifact_type():
    payload = {
        "schema_version": "artifact.v1",
        "artifact_type": "rollout",
        "prompt_id": "prompt-1",
        "messages": [],
    }

    try:
        PromptArtifact.from_dict(payload)
    except ValueError as exc:
        assert "artifact_type" in str(exc)
    else:
        raise AssertionError("wrong artifact_type should be rejected")
