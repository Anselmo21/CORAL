"""Simple rank aggregation (SRA) estimates each node from the single-concept rankings alone

SRA checks only the nodes where the estimate moves the initial answer out of first place and spends
no budget
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coral.lattice.lattice import concept_key
from coral.proxy.aggregator import borda_vector
from coral.proxy.borda_aggregator import BordaAggregator
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace
    from coral.search.search import Supports


class SimpleRankAggregation(Search):
    """Evaluates only the nodes where the single-concept estimate displaces the initial answer"""

    STRATEGY = 'sra'

    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Evaluate the displaced nodes layer by layer

        Within a layer the nodes where the estimate moves the initial answer furthest down go first

        Args:
            lattice: The lattice to search
            space: The search space
            supports: Ignored because SRA uses no graph

        Returns:
            The counterfactuals found and the calls spent
        """
        del supports
        initial_ranking = self._initial_ranking(lattice, space)
        initial_answer = initial_ranking[0]
        oracle = LatticeOracle(lattice)
        singletons = {
            concept: ranking for concept in space.concepts if (ranking := oracle.ask(concept))
        }
        aggregator = BordaAggregator()
        counterfactuals = []
        for _size, nodes in space.layers():
            layer = []
            for node in nodes:
                key = concept_key(node)
                if key in singletons:
                    estimate = singletons[key]
                    displacement = estimate.index(initial_answer)
                elif node:
                    voters = {
                        concept: borda_vector(singletons[concept], initial_ranking)
                        for concept in node
                        if concept in singletons
                    }
                    estimate, _scores = aggregator.aggregate(voters, initial_ranking)
                    displacement = estimate.index(initial_answer)
                else:
                    # The empty node has no voters and is always evaluated
                    displacement = len(initial_ranking)
                layer.append((node, key, displacement))
            layer.sort(key=lambda entry: entry[2], reverse=True)
            for node, key, displacement in layer:
                if displacement == 0:
                    continue
                ranking = singletons[key] if key in singletons else oracle.ask(key)
                if ranking and ranking[0] != initial_answer:
                    counterfactuals.append(Counterfactual(node, ranking[0]))
                    if len(counterfactuals) == self.count:
                        return SearchResult(tuple(counterfactuals), oracle.calls)
        return SearchResult(tuple(counterfactuals), oracle.calls)
