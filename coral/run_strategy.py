"""Runs one search strategy with the given settings on the test cases

Accuracy, cost, and distance get printed per LLM and dataset with criteria and pragmas pooled
"""

from __future__ import annotations

import argparse
from itertools import product
from typing import Any

from coral.evaluation.configuration_sweep import AGGREGATORS
from coral.evaluation.outcome import Outcome
from coral.evaluation.test_case import load_evaluation_cases
from coral.lattice.search_space import Direction
from coral.proxy.entropy import ENTROPY_SCORERS
from coral.report.statistics import accuracy, cost, distance
from coral.search.search_factory import SearchFactory
from coral.utilities.arguments import add_selection_arguments
from coral.utilities.files import DataLayout


def main() -> None:
    """Parse the command line and print one line per LLM and dataset"""
    parser = argparse.ArgumentParser(description='Run one search strategy on the test cases')
    add_selection_arguments(parser)
    parser.add_argument('--strategy', required=True, choices=SearchFactory.STRATEGIES)
    parser.add_argument(
        '--budget', type=int, default=16, help='budget of ERA, GRA, and GERA (default: 16)'
    )
    parser.add_argument(
        '--entropy',
        choices=list(ENTROPY_SCORERS),
        default='softmax',
        help='entropy scorer of ERA, GRA, and GERA (default: softmax)',
    )
    parser.add_argument(
        '--aggregator',
        choices=AGGREGATORS,
        default='borda',
        help='rank aggregation method of ERA, GRA, and GERA (default: borda)',
    )
    arguments = parser.parse_args()

    aggregator = None
    if arguments.aggregator != 'borda':
        # PyFlagr is an optional dependency and only these aggregators need it
        from coral.proxy.flagr_aggregator import FlagrAggregator

        aggregator = FlagrAggregator(arguments.aggregator)
    search = SearchFactory.create(
        arguments.strategy,
        arguments.budget,
        aggregator=aggregator,
        entropy_scorer=ENTROPY_SCORERS[arguments.entropy],
    )
    data = DataLayout(arguments.data)
    direction = Direction(arguments.direction)
    for llm, dataset in product(arguments.llms, arguments.datasets):
        outcomes: list[dict[str, Any]] = []
        for concept_set in arguments.concept_sets:
            for evaluation in load_evaluation_cases(data, direction, llm, dataset, concept_set):
                result = search.search(evaluation.lattice, evaluation.space, evaluation.supports)
                outcomes.append(Outcome.from_result(evaluation.minimal, result).to_json())
        mean_distance = distance(outcomes)
        shown = 'none' if mean_distance is None else f'{mean_distance:.1f}'
        print(
            f'{llm} {dataset}: accuracy {accuracy(outcomes):.1f}%, cost {cost(outcomes):.1f}, '
            f'distance {shown}, {len(outcomes)} cases'
        )


if __name__ == '__main__':
    main()
