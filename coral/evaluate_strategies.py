"""Runs every search strategy at every budget on the test cases and writes one record per case

The searches replay the stored lattices instead of calling the LLM
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from coral.evaluation.strategy_suite import StrategySuite
from coral.lattice.search_space import Direction
from coral.utilities.arguments import add_selection_arguments, selected_cells
from coral.utilities.files import DataLayout, ResultLayout, write_json

logger = logging.getLogger(__name__)


def evaluate_cell(arguments: argparse.Namespace, llm: str, dataset: str, concept_set: str) -> Path:
    """Evaluate one LLM, dataset, and concept set and write the records

    Args:
        arguments: The parsed command line
        llm: The short name of the LLM
        dataset: The dataset name
        concept_set: Either 'criteria' or 'pragmas'

    Returns:
        The path of the records file
    """
    data = DataLayout(arguments.data)
    direction = Direction(arguments.direction)
    results = ResultLayout(arguments.results / direction.value)
    records = StrategySuite(data, direction).run(llm, dataset, concept_set)
    path = results.evaluation_file(llm, dataset, concept_set)
    write_json(path, records, indent=None)
    return path


def main() -> None:
    """Evaluate every selected LLM, dataset, and concept set in parallel"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(description='Evaluate the search strategies on the test cases')
    add_selection_arguments(parser)
    parser.add_argument(
        '--results', type=Path, default=Path('results'), help='results directory (default: results)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help='parallel processes (default: all cores)',
    )
    arguments = parser.parse_args()

    cells = selected_cells(arguments)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [pool.submit(evaluate_cell, arguments, *cell) for cell in cells]
        for future in futures:
            logger.info('wrote %s', future.result())


if __name__ == '__main__':
    main()
