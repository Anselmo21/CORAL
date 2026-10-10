"""The interface every search strategy implements and the results a search returns

Searches read rankings through LatticeOracle and every read counts as one LLM call
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from coral.lattice.lattice import concept_key

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace

Supports = Mapping[str, frozenset[str]]
"""Each candidate answer mapped to the concepts the graph of thoughts says support it"""


"""Value types"""


@dataclass(frozen=True)
class Counterfactual:
    """A node with a different top answer than the initial node"""

    node: tuple[str, ...]
    answer: str

    @property
    def key(self) -> str:
        """Return the node's lattice key

        Returns:
            The sorted comma-joined key
        """
        return concept_key(self.node)

    @property
    def size(self) -> int:
        """Return the number of concepts at the node

        Returns:
            The concept count
        """
        return len(self.node)


@dataclass(frozen=True)
class SearchResult:
    """The counterfactuals found and the LLM calls spent

    Calls cover the traversal while budget_used covers what was spent before it
    """

    counterfactuals: tuple[Counterfactual, ...]
    calls: int
    budget_used: int = 0
    pruned_nodes: int = 0

    @property
    def counterfactual(self) -> Counterfactual | None:
        """Return the first counterfactual or None

        Returns:
            The first counterfactual found, or None if there were none
        """
        return self.counterfactuals[0] if self.counterfactuals else None

    @property
    def cost(self) -> int:
        """Return the traversal calls plus the budget

        Returns:
            The total LLM calls
        """
        return self.calls + self.budget_used


"""The lattice as a stand-in for the LLM"""


class LatticeOracle:
    """Answers node queries from the lattice and counts the calls"""

    def __init__(self, lattice: Lattice) -> None:
        """Wrap a lattice and start the call count at zero

        Args:
            lattice: The lattice that answers the queries
        """
        self.lattice = lattice
        self.calls = 0

    def ask(self, key: str) -> list[str] | None:
        """Return the ranking at a node and count the call

        A node with no ranking still counts because the request would have been made

        Args:
            key: The node's lattice key

        Returns:
            The answers best first, or None if the lattice has no ranking there
        """
        self.calls += 1
        return self.lattice.rankings.get(key)


"""Strategy interface"""


class Search(ABC):
    """A counterfactual search strategy"""

    STRATEGY: str

    def __init__(self, count: int = 1) -> None:
        """Store how many counterfactuals to look for

        Args:
            count: The number of counterfactuals to look for
        """
        self.count = count

    @abstractmethod
    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Search the space for counterfactuals of the initial answer

        Strategies without a graph ignore the supports while the ones that use a graph raise an
        error without them

        Args:
            lattice: The lattice to search
            space: The search space
            supports: Each answer's supporting concepts from the graph, if the strategy uses them

        Returns:
            The counterfactuals found and what they cost
        """

    def _initial_ranking(self, lattice: Lattice, space: SearchSpace) -> list[str]:
        """Return the ranking at the initial node

        Args:
            lattice: The lattice to read
            space: The search space that names the initial node

        Returns:
            The answers best first
        """
        ranking = lattice.ranking(space.initial_node)
        if not ranking:
            raise ValueError(f'query {lattice.query_id} has no answer at the initial node')
        return ranking
