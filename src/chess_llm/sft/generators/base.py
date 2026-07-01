"""Abstract base class for task generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from random import Random
from typing import Iterator

from chess_llm.core.board import canonical_fen_key
from chess_llm.formats import render_ascii_board
from chess_llm.formats.prompts import SYSTEM_PROMPT
from chess_llm.sft import (
    build_sft_row,
    build_template_context,
    normalize_chess_variant_metadata,
    raw_fen_identity_key,
    raw_is_chess960,
)
from chess_llm.sft.identity import build_example_identity
from chess_llm.sft.settings import DEFAULT_CHESS960_RATIOS, DEFAULT_VOLUMES
from chess_llm.sft.templates import append_answer_contract, select_template


_board_to_ascii = render_ascii_board


class TaskGenerator(ABC):
    """Base for every task generator.

    Subclasses implement ``task_id``, ``tier``, and ``generate`` at minimum.
    The base class provides template formatting, blocklist checking, and
    Chess960 mix helpers.
    """

    def __init__(
        self,
        config: dict | None = None,
        blocklist: frozenset[str] = frozenset(),
        rng: Random | None = None,
    ) -> None:
        self.config = config or {}
        self.blocklist = blocklist
        self.rng = rng or Random()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def task_id(self) -> str:
        """Return task identifier, e.g. '1.1_fen_to_board'."""

    @abstractmethod
    def tier(self) -> int:
        """Return tier number (1-7)."""

    @abstractmethod
    def generate(self) -> Iterator[dict]:
        """Yield raw example dicts (before template formatting)."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def target_volume(self) -> int:
        """Volume target from config, with optional override."""
        override = self.config.get("volume_override")
        if override is not None:
            return override
        return DEFAULT_VOLUMES.get(self.task_id(), 0)

    def chess960_ratio(self) -> float:
        """Fraction of examples that should use Chess960 positions."""
        return DEFAULT_CHESS960_RATIOS.get(self.tier(), 0.0)

    def is_blocked(self, fen: str | Mapping) -> bool:
        """Return True if a FEN string or row-like object is in the eval blocklist."""
        if isinstance(fen, Mapping):
            raw = fen
            fen_value = str(raw.get("fen", ""))
        else:
            fen_value = str(fen)
            raw = {"fen": fen_value}
        if not fen_value:
            return False
        return (
            fen_value in self.blocklist
            or raw_fen_identity_key(raw) in self.blocklist
            or canonical_fen_key(fen_value, chess960=raw_is_chess960(raw)) in self.blocklist
        )

    def source_row(self, entry: str | Mapping) -> dict:
        """Return a normalized source row from a FEN string or row mapping."""
        if isinstance(entry, Mapping):
            return normalize_chess_variant_metadata(entry)
        return normalize_chess_variant_metadata({"fen": str(entry)})

    def build_template_context(self, raw: dict) -> dict:
        """Augment *raw* with derived prompt fields like board/state text."""
        return build_template_context(raw)

    def render_template(
        self,
        raw: dict,
        template_text: str | None = None,
    ) -> str:
        """Render a template using raw + derived board/state context."""
        if template_text is None:
            template_text = select_template(self.task_id(), self.rng)
        return template_text.format(**self.build_template_context(raw))

    def format_example(
        self,
        raw: dict,
        template_text: str | None = None,
        assistant_content: str = "",
    ) -> dict:
        """Wrap a raw example into the standard messages format.

        Parameters
        ----------
        raw : dict
            Must contain at least ``fen``.  May contain ``is_chess960``
            and ``metadata``.
        template_text : str, optional
            Pre-formatted user prompt.  If *None*, a random template is
            selected and formatted with the keys in *raw*.
        assistant_content : str
            The ground-truth assistant response.

        Returns
        -------
        dict
            ``{task, tier, fen, is_chess960, messages, metadata}``
        """
        raw = normalize_chess_variant_metadata(raw)
        fen = raw["fen"]
        is_chess960 = raw_is_chess960(raw)
        metadata = dict(raw.get("metadata", {}))
        metadata.setdefault("example_identity", build_example_identity(self.task_id(), raw))

        if template_text is None:
            template_text = self.render_template(raw)
        template_text = append_answer_contract(self.task_id(), template_text)

        return build_sft_row(
            task=self.task_id(),
            tier=self.tier(),
            fen=fen,
            user_prompt=template_text,
            assistant_content=assistant_content,
            is_chess960=is_chess960,
            metadata=metadata,
            system_prompt=SYSTEM_PROMPT,
        )
