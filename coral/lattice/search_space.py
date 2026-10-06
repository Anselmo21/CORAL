"""Defines which nodes of a lattice a search may visit and in what order

Pinned concepts stay in every node while the direction says whether the search removes concepts from
the full set or adds them to the pinned ones
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations


class Direction(str, Enum):
    """Top-down removes concepts and bottom-up adds them

    Each value also names a folder under the data and result directories
    """

    TOP_DOWN = 'top-down'
    BOTTOM_UP = 'bottom-up'


@dataclass(frozen=True)
class SearchSpace:
    """The nodes between the anchor and the full concept set

    Anchored concepts appear in every node
    """

    concepts: tuple[str, ...]
    anchor: tuple[str, ...] = ()
    direction: Direction = Direction.TOP_DOWN

    def __post_init__(self) -> None:
        """Reject an anchor that holds concepts outside the concept set"""
        if not set(self.anchor) <= set(self.concepts):
            raise ValueError('the anchor must be a subset of the concepts')

    @property
    def initial_node(self) -> tuple[str, ...]:
        """Return the node a search starts from and compares every answer against

        That is the full concept set going top-down and the bare anchor going bottom-up

        Returns:
            The concepts at the initial node
        """
        return self.concepts if self.direction is Direction.TOP_DOWN else self.anchor

    def contains(self, node: Sequence[str]) -> bool:
        """Return whether a node keeps the anchor

        Args:
            node: The concepts at the node

        Returns:
            True if every anchored concept is in the node
        """
        return set(self.anchor) <= set(node)

    def layers(self) -> Iterator[tuple[int, list[tuple[str, ...]]]]:
        """Yield each layer's size and nodes moving away from the initial node

        The initial node is left out

        Yields:
            The node size and the nodes of that size
        """
        free_concepts = [concept for concept in self.concepts if concept not in self.anchor]
        if self.direction is Direction.TOP_DOWN:
            sizes = range(len(free_concepts) - 1, -1, -1)
        else:
            sizes = range(1, len(free_concepts) + 1)
        for size in sizes:
            nodes = []
            for added in combinations(free_concepts, size):
                members = set(self.anchor).union(added)
                nodes.append(tuple(concept for concept in self.concepts if concept in members))
            yield len(self.anchor) + size, nodes
