from chess_llm.artifacts.jsonl import read_jsonl, write_jsonl
from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact


def test_jsonl_helpers_preserve_order(tmp_path):
    path = tmp_path / "prompts.jsonl"
    artifacts = [
        PromptArtifact(
            prompt_id="prompt-1",
            messages=[ChatMessage(role="user", content="FEN: one")],
        ),
        PromptArtifact(
            prompt_id="prompt-2",
            messages=[ChatMessage(role="user", content="FEN: two")],
        ),
    ]

    write_jsonl(path, artifacts)
    rows = list(read_jsonl(path, PromptArtifact))

    assert [row.prompt_id for row in rows] == ["prompt-1", "prompt-2"]
