"""Estimates the LLM's ranking for a set of concepts without calling the LLM

Each estimate aggregates the rankings of smaller subsets while a limited budget of real LLM calls
goes to the least certain estimates
"""

from __future__ import annotations

from coral.proxy.aggregator import Aggregator
from coral.proxy.borda_aggregator import BordaAggregator
from coral.proxy.entropy import ENTROPY_SCORERS
from coral.proxy.max_jump import max_jump_cutoff

__all__ = [
    'ENTROPY_SCORERS',
    'Aggregator',
    'BordaAggregator',
    'max_jump_cutoff',
]
