"""Strategies that search a concept lattice for a minimal counterfactual

Breadth-first search checks every node while simple rank aggregation estimates rankings to check
fewer nodes
"""

from __future__ import annotations

from coral.search.breadth_first_search import BreadthFirstSearch
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult, Supports
from coral.search.simple_rank_aggregation import SimpleRankAggregation

__all__ = [
    'BreadthFirstSearch',
    'Counterfactual',
    'LatticeOracle',
    'Search',
    'SearchResult',
    'SimpleRankAggregation',
    'Supports',
]
