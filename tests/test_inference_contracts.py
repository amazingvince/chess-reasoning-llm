from chess_llm.artifacts.schemas import ChatMessage
from chess_llm.inference import (
    DeterministicLegalMoveClient,
    InferenceRequest,
    InferenceResponse,
)


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


def test_deterministic_legal_move_client_lives_in_inference_package():
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    client = DeterministicLegalMoveClient()

    response = client.complete(
        InferenceRequest(
            model_id="stub-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=f"FEN: {starting_fen}\nWhat is the best move?",
                )
            ],
        )
    )

    # First sorted legal move from the standard starting position.
    assert "<move>a2a3</move>" in response.raw_text
    assert response.metadata["client"] == "deterministic_legal_stub"
    assert response.metadata["fen"] == starting_fen


def test_ui_llm_module_reexports_deterministic_legal_move_client():
    from chess_llm_ui.llm import DeterministicLegalMoveClient as UiClient

    assert UiClient is DeterministicLegalMoveClient
