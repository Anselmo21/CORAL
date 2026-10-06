"""Reads stored lattices and looks up the ranking at a node"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

from coral.utilities.files import read_json


def concept_key(concepts: Iterable[str]) -> str:
    """Return the sorted comma-joined key of a concept subset

    The empty subset gets the empty string as its key

    Args:
        concepts: The concepts at a node

    Returns:
        The key that node has in the rankings
    """
    return ', '.join(sorted(concepts))


@dataclass(frozen=True)
class Lattice:
    """The LLM's ranking of the answers at every node of one query

    A halted lattice has no larger nodes because its single concepts all agreed on the top answer
    """

    query_id: str
    question: str
    sources: str
    concept_set: str
    concepts: tuple[str, ...]
    answers: tuple[str, ...]
    rankings: dict[str, list[str]]
    halted: bool = False

    @classmethod
    def from_directory(cls, directory: Path) -> Lattice:
        """Read a lattice written by the lattice generator

        The directory name doubles as the query id

        Args:
            directory: The query's lattice directory, holding config.json and answers.json

        Returns:
            The lattice stored there
        """
        configuration = read_json(directory / 'config.json')
        return cls(
            query_id=directory.name,
            question=configuration['question'],
            sources=configuration['sources'],
            concept_set=configuration['concept_set'],
            concepts=tuple(configuration['concepts']),
            answers=tuple(configuration['plausible_answers']),
            rankings=read_json(directory / 'answers.json'),
            halted=configuration['halted'],
        )

    @property
    def is_populated(self) -> bool:
        """Whether the lattice was fully populated

        The full set is the last node asked for and its ranking shows the run got to the end

        Returns:
            True if the lattice did not halt and has a ranking for the full concept set
        """
        return not self.halted and concept_key(self.concepts) in self.rankings

    def ranking(self, concepts: Iterable[str]) -> list[str] | None:
        """Return the ranking at a node or None

        The concepts can come in any order because the key sorts them

        Args:
            concepts: The concepts at the node

        Returns:
            The answers best first, or None if the node has no ranking
        """
        return self.rankings.get(concept_key(concepts))

    def restrict(self, concepts: Sequence[str]) -> Lattice:
        """Cut the lattice down to the nodes over a subset of the concepts

        Each subset's key is built with concept_key instead of splitting the stored keys because
        concepts can contain commas

        Args:
            concepts: The concepts to keep

        Returns:
            A new lattice with those concepts and only the rankings over them
        """
        rankings = {}
        for size in range(len(concepts) + 1):
            for subset in combinations(concepts, size):
                key = concept_key(subset)
                if key in self.rankings:
                    rankings[key] = self.rankings[key]
        return replace(self, concepts=tuple(concepts), rankings=rankings)


def read_lattices(directory: Path) -> dict[str, Lattice]:
    """Read every lattice in a directory

    The folders are read in sorted order so the sampler sees the queries the same way on every run

    Args:
        directory: The directory that holds one folder per query

    Returns:
        The lattices keyed by query id
    """
    return {
        query_directory.name: Lattice.from_directory(query_directory)
        for query_directory in sorted(directory.iterdir())
        if query_directory.is_dir()
    }
