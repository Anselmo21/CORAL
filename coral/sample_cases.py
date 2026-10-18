"""Samples the test cases of every selected LLM, dataset, and concept set"""

from __future__ import annotations

import argparse
import logging

from coral.evaluation.case_sampler import CaseSampler
from coral.lattice.lattice import read_lattices
from coral.lattice.search_space import Direction
from coral.utilities.arguments import add_selection_arguments, selected_cells
from coral.utilities.files import DataLayout

logger = logging.getLogger(__name__)


def main() -> None:
    """Sample and write the test cases of every selected LLM, dataset, and concept set"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(description='Sample the test cases')
    add_selection_arguments(parser)
    parser.add_argument(
        '--quota', type=int, default=20, help='cases per concept subset size (default: 20)'
    )
    parser.add_argument('--seed', type=int, default=0, help='seed of the shuffles (default: 0)')
    arguments = parser.parse_args()

    data = DataLayout(arguments.data)
    direction = Direction(arguments.direction)
    sampler = CaseSampler(quota=arguments.quota, seed=arguments.seed, direction=direction)
    for llm, dataset, concept_set in selected_cells(arguments):
        lattices = read_lattices(data.lattice_directory(llm, dataset, concept_set))
        pool = sampler.sample(lattices)
        path = data.case_file(direction.value, llm, dataset, concept_set)
        pool.write(path)
        logger.info('%s %s %s: %d cases -> %s', llm, dataset, concept_set, len(pool.cases), path)


if __name__ == '__main__':
    main()
