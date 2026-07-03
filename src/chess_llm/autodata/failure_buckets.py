"""Initial failure buckets for the bootstrap Autodata judge."""

PARSE_FAILURE = "parse_failure"
ILLEGAL_MOVE = "illegal_move"
MISSING_FEN = "missing_fen"
INVALID_FEN = "invalid_fen"
LEGAL_UNSCORED = "legal_unscored"

__all__ = [
    "ILLEGAL_MOVE",
    "INVALID_FEN",
    "LEGAL_UNSCORED",
    "MISSING_FEN",
    "PARSE_FAILURE",
]
