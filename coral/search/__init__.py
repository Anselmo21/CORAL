"""Strategies that search a concept lattice for a minimal counterfactual

Breadth-first search checks every node while the other strategies estimate rankings or follow a
graph of thoughts to check fewer nodes
"""

from __future__ import annotations

from coral.search.breadth_first_search import BreadthFirstSearch
from coral.search.entropy_rank_aggregation import EntropyRankAggregation
from coral.search.graph_entropy_rank_aggregation import GraphEntropyRankAggregation
from coral.search.graph_of_thoughts_search import GraphOfThoughtsSearch
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult, Supports
from coral.search.search_factory import SearchFactory
from coral.search.simple_rank_aggregation import SimpleRankAggregation

__all__ = [
    'BreadthFirstSearch',
    'Counterfactual',
    'EntropyRankAggregation',
    'GraphEntropyRankAggregation',
    'GraphOfThoughtsSearch',
    'LatticeOracle',
    'Search',
    'SearchFactory',
    'SearchResult',
    'SimpleRankAggregation',
    'Supports',
]
