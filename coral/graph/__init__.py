"""Generates graphs of thoughts with an LLM and reads which concepts support each answer"""

from __future__ import annotations

from coral.graph.graph_generator import GraphGenerator
from coral.graph.graph_of_thoughts import GraphOfThoughts

__all__ = [
    'GraphGenerator',
    'GraphOfThoughts',
]
