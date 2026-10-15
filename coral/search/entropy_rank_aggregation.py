"""Entropy-guided rank aggregation (ERA) combines rank aggregation with an entropy cutoff

Nodes the estimate expects to keep the initial answer get skipped unless their entropy says the
estimate is unsure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from coral.lattice.lattice import concept_key
from coral.proxy.aggregator import rank_by_score
from coral.proxy.entropy import softmax_entropy
from coral.proxy.max_jump import max_jump_cutoff
from coral.proxy.rank_aggregation_proxy import RankAggregationProxy
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace
    from coral.proxy.aggregator import Aggregator
    from coral.proxy.entropy import EntropyScorer
    from coral.search.search import Supports


class UnmeasuredNode(NamedTuple):
    """A node the proxy estimated without asking the LLM

    Displacement is the initial answer's rank in the estimate and zero means the proxy expects no
    change
    """

    node: tuple[str, ...]
    key: str
    displacement: int
    entropy: float


class EntropyRankAggregation(Search):
    """Skips nodes the proxy expects to keep the initial answer unless it is unsure about them

    Nodes where the estimate ranks the initial answer lowest go to the LLM first
    """

    STRATEGY = 'era'

    def __init__(
        self,
        budget: int = 16,
        aggregator: Aggregator | None = None,
        entropy_scorer: EntropyScorer = softmax_entropy,
        count: int = 1,
    ) -> None:
        """Store the budget and build the proxy

        The budget caps how many true rankings the proxy buys before the traversal starts

        Args:
            budget: The most true rankings the proxy may buy
            aggregator: The aggregator to use, or None for Borda
            entropy_scorer: The function that scores how uncertain an estimate is
            count: The number of counterfactuals to look for
        """
        super().__init__(count)
        self.budget = budget
        self.proxy = RankAggregationProxy(aggregator, entropy_scorer)

    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Search the space layer by layer using the proxy's estimates

        A node is kept if the estimate displaces the initial answer or Max Jump places it among the
        most uncertain

        Args:
            lattice: The lattice to search
            space: The search space
            supports: Ignored because ERA uses no graph

        Returns:
            The counterfactuals found, the calls and budget spent, and the pruned node count
        """
        del supports
        initial_ranking = self._initial_ranking(lattice, space)
        initial_answer = initial_ranking[0]
        estimate = self.proxy.estimate(lattice, space, initial_ranking, self.budget)
        oracle = LatticeOracle(lattice)
        counterfactuals = []
        pruned_nodes = 0

        def result() -> SearchResult:
            """Package the counterfactuals and call counts so far

            Returns:
                The current search result
            """
            return SearchResult(
                tuple(counterfactuals), oracle.calls, estimate.budget_used, pruned_nodes
            )

        for _size, nodes in space.layers():
            unmeasured = []
            for node in nodes:
                key = concept_key(node)
                if key in estimate.known:
                    ranking = estimate.known[key]
                    if ranking[0] != initial_answer:
                        counterfactuals.append(Counterfactual(node, ranking[0]))
                        if len(counterfactuals) == self.count:
                            return result()
                    continue
                displacement = rank_by_score(estimate.estimates[key]).index(initial_answer)
                unmeasured.append(UnmeasuredNode(node, key, displacement, estimate.entropies[key]))
            if not unmeasured:
                continue
            high_entropy = set(
                max_jump_cutoff(
                    [(entry.key, entry.entropy) for entry in unmeasured], len(unmeasured)
                )
            )
            candidates = [
                entry for entry in unmeasured if entry.displacement > 0 or entry.key in high_entropy
            ]
            pruned_nodes += len(unmeasured) - len(candidates)
            candidates.sort(key=lambda entry: (-entry.displacement, entry.entropy))
            for entry in candidates:
                ranking = oracle.ask(entry.key)
                if ranking and ranking[0] != initial_answer:
                    counterfactuals.append(Counterfactual(entry.node, ranking[0]))
                    if len(counterfactuals) == self.count:
                        return result()
        return result()
