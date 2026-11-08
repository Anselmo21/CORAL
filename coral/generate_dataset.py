"""Generates the OpenGen dataset with Claude using ANTHROPIC_API_KEY

A rerun only asks Claude for the queries still missing from the output file
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coral.dataset.opengen_dataset_generator import MODEL, OpenGenDatasetGenerator

logger = logging.getLogger(__name__)


def main() -> None:
    """Generate the queries missing from the dataset file"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(
        description='Generate the OpenGen dataset with Claude (needs ANTHROPIC_API_KEY)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/datasets/opengen.json'),
        help='dataset file to create or extend (default: data/datasets/opengen.json)',
    )
    parser.add_argument('--model', default=MODEL, help=f'Claude model (default: {MODEL})')
    arguments = parser.parse_args()

    rows = OpenGenDatasetGenerator(arguments.model).generate(arguments.output)
    logger.info('%d queries in %s', len(rows), arguments.output)


if __name__ == '__main__':
    main()
