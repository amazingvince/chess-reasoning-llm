"""Fixed-grammar helpers for step-verification SFT and eval tasks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


STEP_VERIFICATION_ERROR_TYPES = frozenset(
    {
        "none",
        "illegal_move",
        "malformed_line",
        "wrong_fen",
        "wrong_eval",
        "wrong_bucket",
        "wrong_best",
        "wrong_rejection",
        "wrong_piece_claim",
        "wrong_move_set",
        "wrong_ray",
    }
)

_VERDICT_RE = re.compile(r"^Verdict:\s*(sound|broken)\s*$", re.IGNORECASE)
_FAULTY_LINE_RE = re.compile(
    r"^Faulty line:\s*(none|[1-9][0-9]*)\s*$",
    re.IGNORECASE,
)
_ERROR_TYPE_RE = re.compile(r"^Error type:\s*([a-z_]+)\s*$", re.IGNORECASE)
_CORRECTION_RE = re.compile(r"^Correction:\s*(.+?)\s*$", re.IGNORECASE)
_CANDIDATE_LINE_RE = re.compile(
    r"^Candidate\s+([a-h][1-8][a-h][1-8][qrbn]?):",
    re.IGNORECASE,
)
_CANDIDATE_RATING_LINE_RE = re.compile(
    r"^Candidate\s+(?P<uci>[a-h][1-8][a-h][1-8][qrbn]?):\s*"
    r"(?P<score>[+-]?\d+cp|M-?\d+);\s*Bucket:\s*(?P<bucket>.+?)\s*$",
    re.IGNORECASE,
)
_BEST_LINE_RE = re.compile(
    r"^Best:\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*$",
    re.IGNORECASE,
)
_CP_SCORE_RE = re.compile(r"(?P<sign>[+-])(?P<value>\d+)cp")


@dataclass(frozen=True)
class StepVerificationLabel:
    """Expected answer label for one step-verification prompt."""

    verdict: str
    faulty_line: int | None
    error_type: str
    correction: str


def format_step_verification_answer(label: StepVerificationLabel) -> str:
    """Return the canonical four-line answer for a verifier label."""
    faulty_line = "none" if label.faulty_line is None else str(label.faulty_line)
    return "\n".join(
        [
            f"Verdict: {label.verdict}",
            f"Faulty line: {faulty_line}",
            f"Error type: {label.error_type}",
            f"Correction: {label.correction}",
        ]
    )


def parse_step_verification_answer(text: str) -> StepVerificationLabel | None:
    """Parse the fixed verifier grammar, returning None for malformed text."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) != 4:
        return None

    verdict_match = _VERDICT_RE.match(lines[0])
    faulty_match = _FAULTY_LINE_RE.match(lines[1])
    error_match = _ERROR_TYPE_RE.match(lines[2])
    correction_match = _CORRECTION_RE.match(lines[3])
    if not (verdict_match and faulty_match and error_match and correction_match):
        return None

    verdict = verdict_match.group(1).lower()
    faulty_text = faulty_match.group(1).lower()
    error_type = error_match.group(1).lower()
    correction = correction_match.group(1).strip()
    if error_type not in STEP_VERIFICATION_ERROR_TYPES:
        return None

    faulty_line = None if faulty_text == "none" else int(faulty_text)
    if verdict == "sound" and (faulty_line is not None or error_type != "none"):
        return None
    if verdict == "broken" and (faulty_line is None or error_type == "none"):
        return None
    if verdict == "sound" and correction.lower() != "none":
        return None

    return StepVerificationLabel(
        verdict=verdict,
        faulty_line=faulty_line,
        error_type=error_type,
        correction=correction,
    )


def step_verification_score(prediction: str, gold: str) -> dict[str, float]:
    """Score a step-verification prediction against the fixed answer grammar."""
    expected = parse_step_verification_answer(gold)
    pred = parse_step_verification_answer(prediction)
    if expected is None:
        return {
            "primary": 0.0,
            "verdict_accuracy": 0.0,
            "faulty_line_accuracy": 0.0,
            "error_type_accuracy": 0.0,
            "correction_match": 0.0,
        }
    if pred is None:
        return {
            "primary": 0.0,
            "verdict_accuracy": 0.0,
            "faulty_line_accuracy": 0.0,
            "error_type_accuracy": 0.0,
            "correction_match": 0.0,
        }

    verdict_accuracy = 1.0 if pred.verdict == expected.verdict else 0.0
    faulty_line_accuracy = 1.0 if pred.faulty_line == expected.faulty_line else 0.0
    error_type_accuracy = 1.0 if pred.error_type == expected.error_type else 0.0
    correction_match = (
        1.0
        if _normalize_text(pred.correction) == _normalize_text(expected.correction)
        else 0.0
    )
    parts = [
        verdict_accuracy,
        faulty_line_accuracy,
        error_type_accuracy,
        correction_match,
    ]
    return {
        "primary": sum(parts) / len(parts),
        "verdict_accuracy": verdict_accuracy,
        "faulty_line_accuracy": faulty_line_accuracy,
        "error_type_accuracy": error_type_accuracy,
        "correction_match": correction_match,
    }


def number_trace_lines(answer: str) -> str:
    """Prefix non-empty trace lines with 1-based line numbers."""
    return "\n".join(
        f"{idx}. {line.strip()}"
        for idx, line in enumerate(_non_empty_lines(answer), start=1)
    )


def sound_step_verification_label() -> StepVerificationLabel:
    """Return the canonical label for an uncorrupted trace."""
    return StepVerificationLabel(
        verdict="sound",
        faulty_line=None,
        error_type="none",
        correction="none",
    )


def corrupt_candidate_ratings_best_line(
    answer: str,
) -> tuple[str, StepVerificationLabel] | None:
    """Change the final best move to a different candidate in a candidate-rating trace."""
    lines = _non_empty_lines(answer)
    if len(lines) < 6:
        return None

    best_match = _BEST_LINE_RE.match(lines[-1])
    if best_match is None:
        return None
    best_move = best_match.group(1).lower()

    candidates: list[str] = []
    for line in lines[:-1]:
        match = _CANDIDATE_LINE_RE.match(line)
        if match is not None:
            candidates.append(match.group(1).lower())
    wrong_best = next((move for move in candidates if move != best_move), None)
    if wrong_best is None:
        return None

    corrupted = list(lines)
    corrupted[-1] = f"Best: {wrong_best}"
    label = StepVerificationLabel(
        verdict="broken",
        faulty_line=len(corrupted),
        error_type="wrong_best",
        correction=f"Best should be {best_move}.",
    )
    return "\n".join(corrupted), label


def corrupt_candidate_ratings_bucket_line(
    answer: str,
) -> tuple[str, StepVerificationLabel] | None:
    """Change one candidate's bucket while preserving the rest of the line."""
    lines = _non_empty_lines(answer)
    for index, line in enumerate(lines):
        match = _CANDIDATE_RATING_LINE_RE.match(line)
        if match is None:
            continue
        uci = match.group("uci").lower()
        score = match.group("score")
        bucket = " ".join(match.group("bucket").lower().split())
        wrong_bucket = "decisive" if bucket != "decisive" else "equal"
        corrupted = list(lines)
        corrupted[index] = f"Candidate {uci}: {score}; Bucket: {wrong_bucket}"
        label = StepVerificationLabel(
            verdict="broken",
            faulty_line=index + 1,
            error_type="wrong_bucket",
            correction=f"Candidate {uci} bucket should be {bucket}.",
        )
        return "\n".join(corrupted), label
    return None


def corrupt_candidate_ratings_eval_sign(
    answer: str,
) -> tuple[str, StepVerificationLabel] | None:
    """Flip the sign on one centipawn score in a candidate-rating trace."""
    lines = _non_empty_lines(answer)
    for index, line in enumerate(lines):
        match = _CANDIDATE_RATING_LINE_RE.match(line)
        if match is None:
            continue
        score = match.group("score")
        cp_match = _CP_SCORE_RE.fullmatch(score)
        if cp_match is None or int(cp_match.group("value")) == 0:
            continue
        uci = match.group("uci").lower()
        bucket = match.group("bucket")
        flipped_sign = "-" if cp_match.group("sign") == "+" else "+"
        flipped_score = f"{flipped_sign}{cp_match.group('value')}cp"
        corrupted = list(lines)
        corrupted[index] = f"Candidate {uci}: {flipped_score}; Bucket: {bucket}"
        label = StepVerificationLabel(
            verdict="broken",
            faulty_line=index + 1,
            error_type="wrong_eval",
            correction=f"Candidate {uci} score should be {score}.",
        )
        return "\n".join(corrupted), label
    return None


def corrupt_candidate_ratings_malformed_line(
    answer: str,
) -> tuple[str, StepVerificationLabel] | None:
    """Break one candidate line's fixed grammar without changing its content."""
    lines = _non_empty_lines(answer)
    for index, line in enumerate(lines):
        match = _CANDIDATE_RATING_LINE_RE.match(line)
        if match is None:
            continue
        uci = match.group("uci").lower()
        score = match.group("score")
        bucket = match.group("bucket")
        corrupted = list(lines)
        corrupted[index] = f"Candidate {uci} {score}; Bucket: {bucket}"
        label = StepVerificationLabel(
            verdict="broken",
            faulty_line=index + 1,
            error_type="malformed_line",
            correction=f"Line should be `{line}`.",
        )
        return "\n".join(corrupted), label
    return None


def corrupt_material_balance_line(
    answer: str,
) -> tuple[str, StepVerificationLabel] | None:
    """Replace the material balance line with an impossible fixed claim."""
    return _replace_line_with_prefix(
        answer,
        prefix="Balance:",
        replacement="Balance: White is ahead by 99.",
        error_type="wrong_eval",
    )


def corrupt_final_move_set_line(
    answer: str,
    *,
    prefix: str,
) -> tuple[str, StepVerificationLabel] | None:
    """Replace a final move-set line with ``none`` while preserving its label."""
    return _replace_line_with_prefix(
        answer,
        prefix=prefix,
        replacement=f"{prefix} none",
        error_type="wrong_move_set",
    )


def label_from_metadata(metadata: Mapping[str, Any]) -> StepVerificationLabel | None:
    """Build the expected verifier label from SFT metadata."""
    verdict = metadata.get("verification_verdict")
    error_type = metadata.get("error_type")
    correction = metadata.get("correction")
    faulty_value = metadata.get("faulty_line")
    if (
        not isinstance(verdict, str)
        or not isinstance(error_type, str)
        or not isinstance(correction, str)
    ):
        return None
    if verdict == "sound":
        faulty_line = None
    else:
        try:
            faulty_line = int(faulty_value)
        except (TypeError, ValueError):
            return None
    return StepVerificationLabel(
        verdict=verdict,
        faulty_line=faulty_line,
        error_type=error_type,
        correction=correction,
    )


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _replace_line_with_prefix(
    answer: str,
    *,
    prefix: str,
    replacement: str,
    error_type: str,
) -> tuple[str, StepVerificationLabel] | None:
    lines = _non_empty_lines(answer)
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        if line == replacement:
            return None
        corrupted = list(lines)
        corrupted[index] = replacement
        label = StepVerificationLabel(
            verdict="broken",
            faulty_line=index + 1,
            error_type=error_type,
            correction=f"Line should be `{line}`.",
        )
        return "\n".join(corrupted), label
    return None


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


__all__ = [
    "STEP_VERIFICATION_ERROR_TYPES",
    "StepVerificationLabel",
    "corrupt_candidate_ratings_bucket_line",
    "corrupt_candidate_ratings_best_line",
    "corrupt_candidate_ratings_eval_sign",
    "corrupt_candidate_ratings_malformed_line",
    "corrupt_final_move_set_line",
    "corrupt_material_balance_line",
    "format_step_verification_answer",
    "label_from_metadata",
    "number_trace_lines",
    "parse_step_verification_answer",
    "sound_step_verification_label",
    "step_verification_score",
]
