"""The prompt ContextCite ablates, with concepts that can be switched on and off

A boolean mask picks which concepts appear and everything else in the prompt stays fixed
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

INTRODUCTION = (
    'You are solving a multiple-choice question using only the provided sources as context.\n'
    'Choose exactly one option from ANSWER OPTIONS based on the provided criteria.\n'
    'The criteria and answers do not follow any particular order of importance.\n'
    'Weight the criteria equally when forming your answer.\n'
    'Output must be the exact option text only (no numbering, no punctuation changes).\n'
    'Do not include any explanation, reasoning, or extra words.\n\n'
)


class ConceptPrompt:
    """Renders the question, answer options, kept concepts, and sources"""

    def __init__(
        self, question: str, concepts: Sequence[str], sources: str, answers: Sequence[str]
    ) -> None:
        """Store the prompt's parts with surrounding whitespace removed

        The sources are kept as they are because they arrive as one formatted block

        Args:
            question: The query text
            concepts: The concepts that can be switched off
            sources: The retrieved sources
            answers: The answer options
        """
        self.question = question.strip()
        self.concepts = [concept.strip() for concept in concepts]
        self.sources = sources
        self.answers = [answer.strip() for answer in answers]

    def render(self, mask: NDArray[np.bool_] | None = None) -> str:
        """Render the prompt with the concepts the mask keeps

        No mask means every concept is kept

        Args:
            mask: Which concepts to keep, in the order they were given

        Returns:
            The prompt text
        """
        if mask is None:
            mask = np.ones(len(self.concepts), dtype=bool)
        kept = ', '.join(concept for concept, keep in zip(self.concepts, mask, strict=True) if keep)
        options = '\n'.join(f'{index + 1}. {answer}' for index, answer in enumerate(self.answers))
        return (
            f'{INTRODUCTION}'
            f'QUESTION\n{self.question}\n\n'
            f'ANSWER OPTIONS\n{options or "N/A"}\n\n'
            f'CRITERIA\n{kept}\n\n'
            f'SOURCES\n{self.sources}'
        )
