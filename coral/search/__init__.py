"""Strategies that search a concept lattice for a minimal counterfactual

Breadth-first search checks every node layer by layer
"""

from __future__ import annotations

from coral.search.breadth_first_search import BreadthFirstSearch
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult, Supports

__all__ = [
    'BreadthFirstSearch',
    'Counterfactual',
    'LatticeOracle',
    'Search',
    'SearchResult',
    'Supports',
]
