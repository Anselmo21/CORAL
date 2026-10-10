"""Strategies that search a concept lattice for a minimal counterfactual

Every strategy runs over a lattice and a search space and returns the same kind of result
"""

from __future__ import annotations

from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult, Supports

__all__ = [
    'Counterfactual',
    'LatticeOracle',
    'Search',
    'SearchResult',
    'Supports',
]
