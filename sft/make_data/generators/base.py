"""Abstract base class for task generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import Iterator

from config.settings import VOLUMES, CHESS960_RATIOS
from config.system_prompt import SYSTEM_PROMPT
from config.templates import select_template


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
        return VOLUMES.get(self.task_id(), 0)

    def chess960_ratio(self) -> float:
        """Fraction of examples that should use Chess960 positions."""
        return CHESS960_RATIOS.get(self.tier(), 0.0)

    def is_blocked(self, fen: str) -> bool:
        """Return True if *fen* is in the eval blocklist."""
        return fen in self.blocklist

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
        fen = raw["fen"]
        is_chess960 = raw.get("is_chess960", False)
        metadata = raw.get("metadata", {})

        if template_text is None:
            tpl = select_template(self.task_id(), self.rng)
            template_text = tpl.format(**raw)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template_text},
            {"role": "assistant", "content": assistant_content},
        ]

        return {
            "task": self.task_id(),
            "tier": self.tier(),
            "fen": fen,
            "is_chess960": is_chess960,
            "messages": messages,
            "metadata": metadata,
        }
