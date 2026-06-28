from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact
from chess_llm.autodata.rollouts import build_rollout


def _prompt() -> PromptArtifact:
    return PromptArtifact(
        prompt_id="prompt-1",
        messages=[ChatMessage(role="user", content="FEN: start")],
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        task_type="best_move",
    )


def test_build_rollout_preserves_raw_output_and_parsed_move():
    rollout = build_rollout(
        prompt=_prompt(),
        model_id="Qwen/Qwen3-4B",
        raw_output="<think>Take the center.</think>\n<move>e2e4</move>",
        rollout_id="rollout-explicit",
        metadata={"temperature": 0.2},
    )

    assert rollout.rollout_id == "rollout-explicit"
    assert rollout.prompt_id == "prompt-1"
    assert rollout.model_id == "Qwen/Qwen3-4B"
    assert rollout.raw_output == "<think>Take the center.</think>\n<move>e2e4</move>"
    assert rollout.parsed_answer.move_uci == "e2e4"
    assert rollout.parsed_answer.format_type == "move_tag"
    assert rollout.metadata == {"temperature": 0.2}


def test_build_rollout_derives_deterministic_id_when_omitted():
    prompt = _prompt()

    rollout_a = build_rollout(prompt, "model-a", "<move>e2e4</move>")
    rollout_b = build_rollout(prompt, "model-a", "<move>e2e4</move>")
    rollout_c = build_rollout(prompt, "model-a", "<move>d2d4</move>")

    assert rollout_a.rollout_id == rollout_b.rollout_id
    assert rollout_a.rollout_id.startswith("rollout-")
    assert rollout_a.rollout_id != rollout_c.rollout_id
