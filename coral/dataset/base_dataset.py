"""Reads a dataset file into one record per query"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from coral.utilities.files import read_json

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class QueryRecord:
    """One query with its sources, candidate answers, criteria, and pragmas

    Pragmas are flattened to "category: value" strings to match how the prompt shows them
    """

    query_id: int
    question: str
    sources: str
    criteria: tuple[str, ...]
    pragmas: tuple[str, ...]
    answers: tuple[str, ...]

    def concepts(self, concept_set: str) -> tuple[str, ...]:
        """Return the criteria or the pragmas of this query

        Any other name raises a ValueError

        Args:
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            The concepts of that set in dataset order
        """
        if concept_set == 'criteria':
            return self.criteria
        if concept_set == 'pragmas':
            return self.pragmas
        raise ValueError(f'unknown concept set {concept_set!r}')


def read_records(path: Path) -> list[QueryRecord]:
    """Read every query in a dataset file

    Args:
        path: The dataset JSON file

    Returns:
        One record per row in file order
    """
    return [
        QueryRecord(
            query_id=row['query_id'],
            question=row['query'],
            sources=row['sources'],
            criteria=tuple(row['criteria']),
            pragmas=tuple(
                f'{category}: {value}'
                for category, values in row['pragmas'].items()
                for value in values
            ),
            answers=tuple(row['plausible_answers']),
        )
        for row in read_json(path)['rows']
    ]
