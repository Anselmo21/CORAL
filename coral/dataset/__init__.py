"""Loads the query datasets and proposes the answers and concepts of new questions"""

from __future__ import annotations

from coral.dataset.base_dataset import QueryRecord, read_records
from coral.dataset.concept_proposer import ConceptProposer

__all__ = [
    'ConceptProposer',
    'QueryRecord',
    'read_records',
]
