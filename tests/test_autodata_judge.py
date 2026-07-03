from chess_llm.artifacts.schemas import ChatMessage, PromptArtifact
from chess_llm.autodata.failure_buckets import (
    ILLEGAL_MOVE,
    INVALID_FEN,
    LEGAL_UNSCORED,
    MISSING_FEN,
    PARSE_FAILURE,
)
from chess_llm.autodata.judge import judge_rollout
from chess_llm.autodata.rollouts import build_rollout


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _prompt(fen: str | None = STARTING_FEN) -> PromptArtifact:
    return PromptArtifact(
        prompt_id="prompt-1",
        messages=[ChatMessage(role="user", content="FEN: start")],
        fen=fen,
        task_type="best_move",
    )


def test_judge_rollout_marks_legal_move_as_legal_unscored():
    prompt = _prompt()
    rollout = build_rollout(prompt, "model-a", "<move>e2e4</move>", "rollout-1")

    judgment = judge_rollout(prompt, rollout, judgment_id="judgment-1")

    assert judgment.judgment_id == "judgment-1"
    assert judgment.rollout_id == "rollout-1"
    assert judgment.legal is True
    assert judgment.failure_bucket == LEGAL_UNSCORED
    assert judgment.regret_cp is None
    assert judgment.teacher_move_uci is None


def test_judge_rollout_marks_illegal_move():
    prompt = _prompt()
    rollout = build_rollout(prompt, "model-a", "<move>e2e5</move>", "rollout-1")

    judgment = judge_rollout(prompt, rollout)

    assert judgment.legal is False
    assert judgment.failure_bucket == ILLEGAL_MOVE
    assert "illegal" in judgment.feedback.lower()


def test_judge_rollout_marks_ambiguous_or_missing_move_as_parse_failure():
    prompt = _prompt()
    ambiguous = build_rollout(
        prompt,
        "model-a",
        "I considered e2e4 and d2d4.",
        "rollout-ambiguous",
    )
    missing = build_rollout(prompt, "model-a", "I cannot find a move.", "rollout-missing")

    ambiguous_judgment = judge_rollout(prompt, ambiguous)
    missing_judgment = judge_rollout(prompt, missing)

    assert ambiguous_judgment.legal is False
    assert ambiguous_judgment.failure_bucket == PARSE_FAILURE
    assert missing_judgment.legal is False
    assert missing_judgment.failure_bucket == PARSE_FAILURE


def test_judge_rollout_marks_missing_fen():
    prompt = _prompt(fen=None)
    rollout = build_rollout(prompt, "model-a", "<move>e2e4</move>", "rollout-1")

    judgment = judge_rollout(prompt, rollout)

    assert judgment.legal is None
    assert judgment.failure_bucket == MISSING_FEN
    assert "missing fen" in judgment.feedback.lower()


def test_judge_rollout_marks_invalid_fen_as_data_error_not_illegal_move():
    prompt = _prompt(fen="8/8/8/8/8/8/8/8 w - - 0 1")  # parseable but kingless
    rollout = build_rollout(prompt, "model-a", "<move>e2e4</move>", "rollout-1")

    judgment = judge_rollout(prompt, rollout)

    assert judgment.legal is None
    assert judgment.failure_bucket == INVALID_FEN
    assert "invalid" in judgment.feedback.lower()


def test_judge_rollout_invalid_fen_takes_precedence_over_parse_failure():
    prompt = _prompt(fen="not a fen at all")
    rollout = build_rollout(prompt, "model-a", "I cannot find a move.", "rollout-1")

    judgment = judge_rollout(prompt, rollout)

    assert judgment.legal is None
    assert judgment.failure_bucket == INVALID_FEN
