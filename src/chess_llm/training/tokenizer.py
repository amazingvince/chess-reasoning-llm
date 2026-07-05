"""Tokenizer utilities shared by training entrypoints."""

from __future__ import annotations

from collections.abc import Sequence


MOVE_SPECIAL_TOKENS: tuple[str, str] = ("<move>", "</move>")
_MOVE_TOKEN_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "<move>": ("<", "move", ">"),
    "</move>": ("</", "move", ">"),
}


def add_move_special_tokens(tokenizer) -> int:
    """Register structural move tags while preserving existing special tokens."""
    try:
        return tokenizer.add_special_tokens(
            {"additional_special_tokens": list(MOVE_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
    except TypeError as exc:
        if "replace_additional_special_tokens" not in str(exc):
            raise

    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    desired = existing + [token for token in MOVE_SPECIAL_TOKENS if token not in existing]
    return tokenizer.add_special_tokens({"additional_special_tokens": desired})


def initialize_added_move_token_embeddings(
    model,
    tokenizer,
    *,
    original_vocab_size: int,
) -> None:
    """Initialize newly added move-tag embeddings from their old token fragments."""
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    for token in MOVE_SPECIAL_TOKENS:
        target_id = _token_id(tokenizer, token)
        if target_id is None or target_id < original_vocab_size:
            continue
        source_ids = _source_token_ids(tokenizer, _MOVE_TOKEN_FRAGMENTS[token])
        if not source_ids:
            continue
        _copy_mean_embedding(input_embeddings.weight, target_id, source_ids)
        if output_embeddings is not None and output_embeddings is not input_embeddings:
            _copy_mean_embedding(output_embeddings.weight, target_id, source_ids)


def resize_and_initialize_move_special_tokens(
    model,
    tokenizer,
    *,
    original_vocab_size: int,
    added_token_count: int,
) -> None:
    """Resize model embeddings and initialize newly added move special tokens."""
    if added_token_count <= 0:
        return
    input_embeddings = model.get_input_embeddings()
    embedding_rows = int(input_embeddings.weight.shape[0])
    tokenizer_size = len(tokenizer)
    if embedding_rows < tokenizer_size:
        model.resize_token_embeddings(tokenizer_size)
    initialize_added_move_token_embeddings(
        model,
        tokenizer,
        original_vocab_size=original_vocab_size,
    )


def _copy_mean_embedding(weight, target_id: int, source_ids: Sequence[int]) -> None:
    import torch

    with torch.no_grad():
        weight[target_id].copy_(weight[list(source_ids)].mean(dim=0))


def _source_token_ids(tokenizer, tokens: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for token in tokens:
        token_id = _token_id(tokenizer, token)
        if token_id is None:
            return []
        ids.append(token_id)
    return ids


def _token_id(tokenizer, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == unk_token_id:
        return None
    return int(token_id)


__all__ = [
    "MOVE_SPECIAL_TOKENS",
    "add_move_special_tokens",
    "initialize_added_move_token_embeddings",
    "resize_and_initialize_move_special_tokens",
]
