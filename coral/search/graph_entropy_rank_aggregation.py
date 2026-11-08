"""Graph and entropy-guided rank aggregation (GERA) adds the graph of thoughts to ERA

The checking order and the point to stop a layer come from how much more of a rival answer's
graph-linked concepts each node keeps than of the initial answer's
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

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


class ScoredNode(NamedTuple):
    """A node the proxy only estimated, with the scores that order it

    Contrast (how much more of a rival answer's graph-linked concepts the node keeps than of the
    initial answer's) and margin (the initial answer's estimated lead over its closest rival) order
    the nodes
    """

    node: tuple[str, ...]
    key: str
    displacement: int
    entropy: float
    contrast: float
    margin: float


class GraphEntropyRankAggregation(Search):
    """ERA with a graph of thoughts that orders and skips undisplaced nodes by contrast

    Turning off the entropy cutoff gives GRA and lets contrast alone pick among undisplaced nodes
    """

    STRATEGY = 'gera'

    def __init__(
        self,
        budget: int = 16,
        aggregator: Aggregator | None = None,
        entropy_scorer: EntropyScorer = softmax_entropy,
        entropy_cutoff: bool = True,
        count: int = 1,
    ) -> None:
        """Store the budget and build the proxy

        Args:
            budget: The most true rankings the proxy may buy
            aggregator: The aggregator to use, or None for Borda
            entropy_scorer: The function that scores how uncertain an estimate is
            entropy_cutoff: Whether Max Jump on the entropy decides which nodes survive
            count: The number of counterfactuals to look for
        """
        super().__init__(count)
        self.budget = budget
        self.entropy_cutoff = entropy_cutoff
        self.proxy = RankAggregationProxy(aggregator, entropy_scorer)

    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Search the space layer by layer using the proxy and the graph

        Displaced nodes go first, then the undisplaced high-entropy nodes by contrast, and last any
        node with positive contrast that the entropy cutoff dropped

        Args:
            lattice: The lattice to search
            space: The search space
            supports: Each answer's supporting concepts from the graph (required)

        Returns:
            The counterfactuals found, the calls and budget spent, and the pruned node count
        """
        if supports is None:
            raise ValueError(f'{self.STRATEGY} needs the supports of a graph of thoughts')
        initial_ranking = self._initial_ranking(lattice, space)
        initial_answer = initial_ranking[0]
        estimate = self.proxy.estimate(lattice, space, initial_ranking, self.budget)
        oracle = LatticeOracle(lattice)
        counterfactuals = []
        pruned_nodes = 0

        def coverage(node: tuple[str, ...], answer: str) -> float:
            """Return the share of an answer's support that the node keeps

            An answer with no support scores zero

            Args:
                node: The concepts at the node
                answer: The answer to check

            Returns:
                The kept fraction of that answer's support
            """
            support = supports.get(answer)
            return len(set(node) & support) / len(support) if support else 0.0

        def result() -> SearchResult:
            """Package the counterfactuals and call counts so far

            Returns:
                The current search result
            """
            return SearchResult(
                tuple(counterfactuals),
                oracle.calls + estimate.calls,
                estimate.budget_used,
                pruned_nodes,
            )

        for _size, nodes in space.layers():
            unmeasured = []
            for node in nodes:
                key = concept_key(node)
                if key in estimate.known or key not in estimate.estimates:
                    ranking = estimate.known[key] if key in estimate.known else oracle.ask(key)
                    if ranking and ranking[0] != initial_answer:
                        counterfactuals.append(Counterfactual(node, ranking[0]))
                        if len(counterfactuals) == self.count:
                            return result()
                    continue
                probabilities = estimate.estimates[key]
                initial_coverage = coverage(node, initial_answer)
                contrast = max(
                    (
                        coverage(node, answer) - initial_coverage
                        for answer in initial_ranking
                        if answer != initial_answer
                    ),
                    default=0.0,
                )
                rival = max(
                    (
                        probability
                        for answer, probability in probabilities.items()
                        if answer != initial_answer
                    ),
                    default=0.0,
                )
                unmeasured.append(
                    ScoredNode(
                        node,
                        key,
                        rank_by_score(probabilities).index(initial_answer),
                        estimate.entropies[key],
                        contrast,
                        probabilities[initial_answer] - rival,
                    )
                )
            if not unmeasured:
                continue

            if self.entropy_cutoff:
                high_entropy = set(
                    max_jump_cutoff(
                        [(entry.key, entry.entropy) for entry in unmeasured], len(unmeasured)
                    )
                )
            else:
                high_entropy = {entry.key for entry in unmeasured}
            candidates, rescue_pool = [], []
            for entry in unmeasured:
                if entry.displacement > 0 or entry.key in high_entropy:
                    candidates.append(entry)
                elif entry.contrast > 0:
                    rescue_pool.append(entry)
                else:
                    pruned_nodes += 1
            pool_median = (
                float(np.median([entry.contrast for entry in rescue_pool])) if rescue_pool else 0.0
            )
            rescue_pool.sort(key=lambda entry: -entry.contrast)

            displaced = sorted(
                (entry for entry in candidates if entry.displacement > 0),
                key=lambda entry: (entry.margin, entry.entropy),
            )
            uncertain = [entry for entry in candidates if entry.displacement == 0]
            # No uncertain node gets skipped when equal contrasts leave no high-contrast group
            high_contrast: set[str] = set()
            unconfirmed = None
            if len({entry.contrast for entry in uncertain}) > 1:
                high_contrast = set(
                    max_jump_cutoff(
                        [(entry.key, entry.contrast) for entry in uncertain], len(uncertain)
                    )
                )
                unconfirmed = len(high_contrast)
            uncertain.sort(
                key=lambda entry: (
                    entry.key not in high_contrast,
                    -entry.contrast,
                    entry.margin,
                    entry.entropy,
                )
            )

            skip_uncertain = pool_closed = False
            for entry in displaced + uncertain + rescue_pool:
                in_pool = entry.displacement == 0 and entry.key not in high_entropy
                if (
                    skip_uncertain
                    and not in_pool
                    and entry.displacement == 0
                    and entry.key not in high_contrast
                ):
                    pruned_nodes += 1
                    continue
                if pool_closed and in_pool:
                    pruned_nodes += 1
                    continue
                ranking = oracle.ask(entry.key)
                if not ranking:
                    continue
                if ranking[0] != initial_answer:
                    counterfactuals.append(Counterfactual(entry.node, ranking[0]))
                    if len(counterfactuals) == self.count:
                        return result()
                    continue
                if entry.key in high_contrast and unconfirmed is not None:
                    unconfirmed -= 1
                    skip_uncertain = unconfirmed == 0
                if in_pool and entry.contrast >= pool_median:
                    pool_closed = True
        return result()
