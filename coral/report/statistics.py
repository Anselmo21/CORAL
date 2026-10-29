"""Accuracy, cost, and distance of each strategy from the evaluation records of the test cases

Accuracy is how often a strategy finds a minimal counterfactual, cost is its LLM calls, and distance
is how many concepts a non-minimal result is off by
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from coral.evaluation.outcome import MINIMAL, NOT_MINIMAL
from coral.utilities.files import ResultLayout, read_json

Record = dict[str, Any]


def read_records(
    results: ResultLayout, llm: str, dataset: str, concept_sets: Sequence[str]
) -> list[Record]:
    """Read the records of one LLM and dataset with the concept sets pooled

    Args:
        results: The result layout to read from
        llm: The short name of the LLM
        dataset: The dataset name
        concept_sets: The concept sets to pool

    Returns:
        The evaluation records of all those concept sets in one list
    """
    return [
        record
        for concept_set in concept_sets
        for record in read_json(results.evaluation_file(llm, dataset, concept_set))
    ]


def outcome(record: Record, strategy: str, budget: int | None = None) -> dict[str, Any]:
    """Return a strategy's outcome on a record

    Budgeted strategies keep one outcome per budget while SRA and GoT keep a single one

    Args:
        record: One evaluation record
        strategy: The strategy's abbreviation (sra, lsm, era, gra, gera, or got)
        budget: The budget to read, or None for a strategy that has no budget

    Returns:
        The outcome as its JSON dict
    """
    outcomes = record['outcomes'][strategy]
    return outcomes if budget is None else outcomes[str(budget)]


def has_outcomes(records: Sequence[Record], strategy: str, budget: int) -> bool:
    """Return whether every record has the strategy's outcome at the budget

    Args:
        records: The evaluation records
        strategy: The strategy's abbreviation (sra, lsm, era, gra, gera, or got)
        budget: The budget to look for

    Returns:
        True if there are records and all of them have it
    """
    return bool(records) and all(str(budget) in record['outcomes'][strategy] for record in records)


def accuracy(outcomes: Sequence[dict[str, Any]]) -> float:
    """Return the percentage of outcomes that found the minimal counterfactual

    Args:
        outcomes: The outcomes as JSON dicts

    Returns:
        The percentage from zero to one hundred
    """
    return 100 * float(np.mean([entry['label'] == MINIMAL for entry in outcomes]))


def cost(outcomes: Sequence[dict[str, Any]]) -> float:
    """Return the mean LLM calls including the budget

    Not-found outcomes count too

    Args:
        outcomes: The outcomes as JSON dicts

    Returns:
        The mean number of calls
    """
    return float(np.mean([entry['calls'] + entry['budget_used'] for entry in outcomes]))


def distance(outcomes: Sequence[dict[str, Any]]) -> float | None:
    """Return the mean distance of the non-minimal outcomes or None

    Args:
        outcomes: The outcomes as JSON dicts

    Returns:
        The mean distance, or None if every outcome was minimal or not found
    """
    distances = [entry['distance'] for entry in outcomes if entry['label'] == NOT_MINIMAL]
    return float(np.mean(distances)) if distances else None


def brute_force_cost(records: Sequence[Record]) -> float:
    """Return the mean BFS cost

    This reads the calls saved with the minimal counterfactual because BFS has no outcome of its
    own in the records

    Args:
        records: The evaluation records

    Returns:
        The mean number of calls BFS needed
    """
    return float(np.mean([record['minimal']['calls'] for record in records]))
