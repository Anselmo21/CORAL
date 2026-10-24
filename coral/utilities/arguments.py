"""Command-line arguments shared by the commands"""

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


def add_serving_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the vLLM serving arguments

    Args:
        parser: The parser to extend
    """
    group = parser.add_argument_group('vLLM serving')
    group.add_argument('--tensor-parallel-size', type=int, default=1, help='GPUs per model')
    group.add_argument(
        '--max-model-len', type=int, default=None, help='cap on prompt plus output tokens'
    )
    group.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=None,
        help='share of GPU memory vLLM may reserve',
    )
    group.add_argument('--max-num-seqs', type=int, default=None, help='cap on concurrent sequences')
    group.add_argument(
        '--max-num-batched-tokens', type=int, default=None, help='cap on tokens per engine step'
    )
    group.add_argument('--enforce-eager', action='store_true', help='skip CUDA graph capture')
    group.add_argument(
        '--disable-custom-all-reduce',
        action='store_true',
        help='use NCCL instead of the custom vLLM all-reduce',
    )


def serving_options(arguments: argparse.Namespace) -> dict[str, object]:
    """Return the parsed serving arguments as backend options

    Args:
        arguments: The parsed command line

    Returns:
        The options by name
    """
    return {
        'tensor_parallel_size': arguments.tensor_parallel_size,
        'max_model_len': arguments.max_model_len,
        'gpu_memory_utilization': arguments.gpu_memory_utilization,
        'max_num_seqs': arguments.max_num_seqs,
        'max_num_batched_tokens': arguments.max_num_batched_tokens,
        'enforce_eager': arguments.enforce_eager,
        'disable_custom_all_reduce': arguments.disable_custom_all_reduce,
    }
