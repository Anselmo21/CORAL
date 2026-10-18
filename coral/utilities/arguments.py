"""Command-line arguments shared by the commands

Commands select what to run with the same flags
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from coral.utilities.catalog import CONCEPT_SETS, DATASETS, LANGUAGE_MODELS


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the data, direction, LLM, dataset, and concept set arguments

    A command run without these flags covers every LLM, dataset, and concept set

    Args:
        parser: The parser to extend
    """
    parser.add_argument(
        '--data', type=Path, default=Path('data'), help='data directory (default: data)'
    )
    parser.add_argument(
        '--direction',
        choices=('top-down', 'bottom-up'),
        default='top-down',
        help='remove concepts from the full set or add them to the empty set (default: top-down)',
    )
    parser.add_argument(
        '--llms',
        nargs='+',
        choices=list(LANGUAGE_MODELS),
        default=list(LANGUAGE_MODELS),
        help='answering LLMs (default: all four)',
    )
    parser.add_argument(
        '--datasets', nargs='+', choices=DATASETS, default=list(DATASETS), help='datasets'
    )
    parser.add_argument(
        '--concept-sets',
        nargs='+',
        choices=CONCEPT_SETS,
        default=list(CONCEPT_SETS),
        help='concept sets',
    )


def selected_cells(arguments: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return every selected combination of LLM, dataset, and concept set

    Args:
        arguments: The parsed command line

    Returns:
        One (llm, dataset, concept set) tuple per cell
    """
    return list(product(arguments.llms, arguments.datasets, arguments.concept_sets))
