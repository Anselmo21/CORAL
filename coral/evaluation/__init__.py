"""Test cases and the code that scores each search strategy against breadth-first search

A test case is a query's lattice cut down to a subset of its concepts
"""

from __future__ import annotations

from coral.evaluation.case_sampler import CaseSampler
from coral.evaluation.outcome import Outcome
from coral.evaluation.test_case import CasePool, EvaluationCase, TestCase, load_evaluation_cases

__all__ = [
    'CasePool',
    'CaseSampler',
    'EvaluationCase',
    'Outcome',
    'TestCase',
    'load_evaluation_cases',
]
