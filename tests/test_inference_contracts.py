from chess_llm.artifacts.schemas import ChatMessage
from chess_llm.inference import InferenceRequest, InferenceResponse


def test_inference_contracts_preserve_messages_and_sampling_metadata():
    request = InferenceRequest(
        model_id="local-model",
        messages=[ChatMessage(role="user", content="FEN: start")],
        temperature=0.2,
        max_tokens=512,
        metadata={"game_id": "game-1"},
    )
    response = InferenceResponse(
        model_id="local-model",
        raw_text="<think>Take space.</think>\n<move>e2e4</move>",
        metadata={"latency_ms": 42},
    )

    assert request.messages[0].content == "FEN: start"
    assert request.metadata["game_id"] == "game-1"
    assert response.raw_text.startswith("<think>")
    assert response.metadata["latency_ms"] == 42
