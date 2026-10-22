"""The interface every LLM backend implements"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

Message = dict[str, str]


@dataclass(frozen=True)
class Completion:
    """One LLM answer and its token counts

    The counts are None when a backend does not report them
    """

    text: str
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


class LanguageModel(ABC):
    """Answers chat conversations with one LLM

    Subclasses set PROVIDER and say through CONSTRAINED_DECODING whether they can force replies
    to match a regular expression
    """

    PROVIDER: str
    CONSTRAINED_DECODING: bool

    def __init__(self, model: str) -> None:
        """Store the model id

        Args:
            model: The model id the backend serves
        """
        self.model = model

    @property
    def serving(self) -> dict[str, Any]:
        """Return the backend settings recorded in each lattice and graph file

        Returns:
            The settings by name
        """
        return {}

    @abstractmethod
    def complete(
        self,
        conversations: list[list[Message]],
        temperature: float,
        max_tokens: int,
        regex: str | None = None,
    ) -> list[Completion]:
        """Answer every conversation and force each reply to match regex when one is given

        Backends decide whether to batch the conversations or send them one by one

        Args:
            conversations: The chat conversations to answer
            temperature: The sampling temperature
            max_tokens: The token cap per answer
            regex: The pattern answers must match, if any

        Returns:
            One completion per conversation in the same order
        """
