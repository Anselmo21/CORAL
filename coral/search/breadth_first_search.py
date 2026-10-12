"""Breadth-first search (BFS) checks every node layer by layer until the answer changes"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coral.lattice.lattice import concept_key
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace
    from coral.search.search import Supports


class BreadthFirstSearch(Search):
    """Evaluates every node layer by layer and always finds the minimal counterfactual

    Nodes within a layer are checked in the order the search space gives them
    """

    STRATEGY = 'bfs'

    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Evaluate nodes in order until the counterfactuals are found

        Args:
            lattice: The lattice to search
            space: The search space
            supports: Ignored because BFS uses no graph

        Returns:
            The counterfactuals found and the calls spent
        """
        del supports
        initial_answer = self._initial_ranking(lattice, space)[0]
        oracle = LatticeOracle(lattice)
        counterfactuals = []
        for _size, nodes in space.layers():
            for node in nodes:
                ranking = oracle.ask(concept_key(node))
                if ranking and ranking[0] != initial_answer:
                    counterfactuals.append(Counterfactual(node, ranking[0]))
                    if len(counterfactuals) == self.count:
                        return SearchResult(tuple(counterfactuals), oracle.calls)
        return SearchResult(tuple(counterfactuals), oracle.calls)
