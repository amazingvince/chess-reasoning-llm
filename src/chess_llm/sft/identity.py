"""Stable identity keys for generated SFT examples."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from chess_llm.sft.context import raw_fen_identity_key


TASK_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "1.3_piece_identification": ("query_kind", "square", "color", "piece"),
    "1.4_piece_counting": ("count_kind", "color", "piece"),
    "1.5_state_tracking": ("moves",),
    "1.6_square_lookup": ("square",),
    "1.7_rank_lookup": ("rank",),
    "1.8_move_square_edits": ("move",),
    "1.9_fen_assembly": ("move",),
    "1.10_fen_row_application": ("move",),
    "1.11_square_coordinates": ("square",),
    "1.12_fen_rank_expansion": ("rank", "fen_rank_row"),
    "1.13_fen_rank_cell_edit": ("move", "square", "rank", "file"),
    "1.14_fen_board_edit": ("move",),
    "1.15_material_inventory": (),
    "1.16_material_piece_counts": (),
    "1.17_material_value_totals": (),
    "1.18_material_balance_trace": (),
    "1.19_multi_move_state_tracking": ("moves",),
    "2.0_side_piece_inventory": (),
    "2.1_legal_move_gen": (),
    "2.2_piece_specific_moves": ("source_square", "square"),
    "2.3_move_legality_check": ("tested_move", "move"),
    "2.4_check_detection": (),
    "2.5_special_rules": ("special_rule_bucket", "square"),
    "2.6_piece_pseudo_legal_moves": ("source_square",),
    "2.7_piece_legal_filter": ("source_square",),
    "2.8_king_safety_filter": ("tested_move",),
    "2.9_legal_moves_by_piece": (),
    "2.10_ray_walk": ("source_square",),
    "2.11_legal_filter_trace": (),
    "5.1_opening_identification": ("variant",),
    "5.2_opening_continuation": ("variant",),
    "5.3_opening_principles": ("variant",),
    "7.8_candidate_ratings": ("candidate_moves",),
}


def build_example_identity(task_id: str, raw: Mapping[str, Any]) -> str:
    """Return a stable, variant-aware identity for a generated example."""
    metadata = raw.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    payload: dict[str, Any] = {
        "task": task_id,
        "fen": raw_fen_identity_key(raw),
        "fields": {},
    }
    fields = payload["fields"]
    for field in TASK_IDENTITY_FIELDS.get(task_id, ()):
        value = raw.get(field)
        if value is None or value == "":
            value = metadata_map.get(field)
        if value is not None and value != "":
            fields[field] = value

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"{task_id}:{digest}"


__all__ = ["TASK_IDENTITY_FIELDS", "build_example_identity"]
