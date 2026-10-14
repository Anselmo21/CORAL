"""Estimates the LLM's ranking for a set of concepts without calling the LLM

Each estimate aggregates the rankings of smaller subsets while a limited budget of real LLM calls
goes to the least certain estimates
"""

from __future__ import annotations

from coral.proxy.aggregator import Aggregator
from coral.proxy.borda_aggregator import BordaAggregator

__all__ = [
    'Aggregator',
    'BordaAggregator',
]
