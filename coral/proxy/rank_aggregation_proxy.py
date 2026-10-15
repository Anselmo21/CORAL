"""Builds the proxy's ranking estimates for every node of a search space"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from coral.lattice.lattice import concept_key
from coral.lattice.search_space import Direction
from coral.proxy.aggregator import borda_vector, normalize_max
from coral.proxy.borda_aggregator import BordaAggregator
from coral.proxy.entropy import softmax, softmax_entropy
from coral.proxy.max_jump import max_jump_cutoff

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace
    from coral.proxy.aggregator import Aggregator
    from coral.proxy.entropy import EntropyScorer


@dataclass
class ProxyEstimate:
    """What the proxy knows and estimates once its budget is spent

    A search uses the true rankings in the known field and falls back on the estimates for every
    other node
    """

    known: dict[str, list[str]]
    estimates: dict[str, dict[str, float]]
    entropies: dict[str, float]
    budget_used: int


class RankAggregationProxy:
    """Estimates each node's ranking from the nodes one concept smaller

    After each layer the budget buys the true rankings of the nodes Max Jump finds most uncertain
    """

    def __init__(
        self, aggregator: Aggregator | None = None, entropy_scorer: EntropyScorer = softmax_entropy
    ) -> None:
        """Store how the proxy aggregates and scores its estimates

        Without an aggregator the proxy uses Borda

        Args:
            aggregator: The aggregator to use, or None for Borda
            entropy_scorer: The function that scores how uncertain an estimate is
        """
        self.aggregator = aggregator or BordaAggregator()
        self.entropy_scorer = entropy_scorer

    def estimate(
        self, lattice: Lattice, space: SearchSpace, answers: Sequence[str], budget: int
    ) -> ProxyEstimate:
        """Estimate every node and spend the budget on the most uncertain ones

        True rankings bought in a layer feed the estimates of larger nodes because layers go up from
        size two

        Args:
            lattice: The lattice that supplies true rankings
            space: The search space to estimate
            answers: The answers to score
            budget: The most true rankings to buy

        Returns:
            The known rankings, the estimates with their entropies, and the budget actually spent
        """
        known = {
            concept: lattice.rankings[concept]
            for concept in space.concepts
            if concept in lattice.rankings
        }
        for node in ((), space.initial_node):
            key = concept_key(node)
            if key in lattice.rankings:
                known[key] = lattice.rankings[key]
        estimates: dict[str, dict[str, float]] = {}
        entropies: dict[str, float] = {}
        budget_remaining = budget
        # Only bottom-up needs an estimate for the full set because top-down starts there
        if space.direction is Direction.TOP_DOWN:
            largest = len(space.concepts) - 1
        else:
            largest = len(space.concepts)
        for size in range(2, largest + 1):
            layer = []
            for node in combinations(space.concepts, size):
                key = concept_key(node)
                if key in known:
                    continue
                voters = {}
                for child in combinations(node, size - 1):
                    child_key = concept_key(child)
                    if child_key in known:
                        voters[child_key] = normalize_max(borda_vector(known[child_key], answers))
                    elif child_key in estimates:
                        voters[child_key] = estimates[child_key]
                _ranking, scores = self.aggregator.aggregate(voters, answers)
                estimates[key] = softmax(scores)
                entropies[key] = self.entropy_scorer(scores)
                if space.contains(node):
                    layer.append((key, entropies[key]))
            for key in max_jump_cutoff(layer, budget_remaining):
                if lattice.rankings.get(key):
                    known[key] = lattice.rankings[key]
                budget_remaining -= 1
        return ProxyEstimate(known, estimates, entropies, budget - budget_remaining)
