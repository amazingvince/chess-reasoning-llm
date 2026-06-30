"""SFT bootstrap pipeline integration points."""

from chess_llm.sft.context import (
    board_from_raw,
    build_template_context,
    normalize_chess_variant_metadata,
    raw_fen_identity_key,
    raw_is_chess960,
)
from chess_llm.sft.examples import (
    SftExample,
    build_sft_messages,
    build_sft_row,
    write_legacy_sft_jsonl,
)
from chess_llm.sft.fen_pool import FENPool
from chess_llm.sft.output import JSONLWriter, PipelineStats
from chess_llm.sft.completeness import (
    audit_output_completeness,
    expected_task_path,
)
from chess_llm.sft.settings import SftDataSettings, apply_hf_cache_env
from chess_llm.sft.source_preparation import (
    EvalSplitSourcePreparation,
    build_eval_split_sources,
)
from chess_llm.sft.source_readiness import (
    SourceReadinessIssue,
    SourceReadinessReport,
    build_source_readiness_report,
)
from chess_llm.sft.validation import (
    validate_example,
    validate_fen,
    validate_legal_moves,
    validate_move_legal,
    validate_state_tracking,
    validate_template_complete,
    validate_think_move_format,
)

__all__ = [
    "EvalSplitSourcePreparation",
    "FENPool",
    "JSONLWriter",
    "PipelineStats",
    "SftExample",
    "SftDataSettings",
    "SourceReadinessIssue",
    "SourceReadinessReport",
    "apply_hf_cache_env",
    "audit_output_completeness",
    "board_from_raw",
    "build_template_context",
    "build_eval_split_sources",
    "build_source_readiness_report",
    "build_sft_messages",
    "build_sft_row",
    "expected_task_path",
    "raw_is_chess960",
    "normalize_chess_variant_metadata",
    "raw_fen_identity_key",
    "validate_example",
    "validate_fen",
    "validate_legal_moves",
    "validate_move_legal",
    "validate_state_tracking",
    "validate_template_complete",
    "validate_think_move_format",
    "write_legacy_sft_jsonl",
]
