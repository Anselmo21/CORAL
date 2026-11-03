"""Creates a search strategy from its abbreviation (bfs, sra, era, gra, gera, or got)

ERA, GRA, and GERA also take a budget, a rank aggregator, and an entropy scorer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coral.proxy.entropy import softmax_entropy
from coral.search.breadth_first_search import BreadthFirstSearch
from coral.search.entropy_rank_aggregation import EntropyRankAggregation
from coral.search.graph_entropy_rank_aggregation import GraphEntropyRankAggregation
from coral.search.graph_of_thoughts_search import GraphOfThoughtsSearch
from coral.search.search import Search
from coral.search.simple_rank_aggregation import SimpleRankAggregation

if TYPE_CHECKING:
    from coral.proxy.aggregator import Aggregator
    from coral.proxy.entropy import EntropyScorer


class SearchFactory:
    """Builds a search from its strategy abbreviation"""

    STRATEGIES = ('bfs', 'sra', 'era', 'gra', 'gera', 'got')

    @staticmethod
    def create(
        strategy: str,
        budget: int = 16,
        count: int = 1,
        aggregator: Aggregator | None = None,
        entropy_scorer: EntropyScorer = softmax_entropy,
    ) -> Search:
        """Return the search for a strategy abbreviation

        LSM is left out because it needs a loaded model

        Args:
            strategy: One of the abbreviations in STRATEGIES
            budget: The budget of the strategies that use one
            count: The number of counterfactuals to look for
            aggregator: The aggregator of ERA, GRA, and GERA, or None for Borda
            entropy_scorer: The entropy scorer of ERA, GRA, and GERA

        Returns:
            The configured search
        """
        match strategy:
            case 'bfs':
                return BreadthFirstSearch(count)
            case 'sra':
                return SimpleRankAggregation(count)
            case 'era':
                return EntropyRankAggregation(budget, aggregator, entropy_scorer, count)
            case 'gra':
                return GraphEntropyRankAggregation(
                    budget, aggregator, entropy_scorer, entropy_cutoff=False, count=count
                )
            case 'gera':
                return GraphEntropyRankAggregation(budget, aggregator, entropy_scorer, count=count)
            case 'got':
                return GraphOfThoughtsSearch(count)
        raise ValueError(f'unknown strategy {strategy!r}, choose from {SearchFactory.STRATEGIES}')
