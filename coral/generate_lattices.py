"""Asks an LLM to rank the answers under every subset of each query's concepts"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coral.dataset.base_dataset import read_records
from coral.lattice.lattice_generator import LatticeGenerator
from coral.llm.language_model_factory import PROVIDERS, LanguageModelFactory
from coral.utilities.arguments import add_serving_arguments, serving_options
from coral.utilities.catalog import CONCEPT_SETS, DATASETS, LANGUAGE_MODELS
from coral.utilities.files import DataLayout

logger = logging.getLogger(__name__)


def main() -> None:
    """Populate the lattice of every selected query

    A lattice with every node ranked or with single-concept nodes that agree on the top answer
    counts as finished and gets skipped unless --force is set
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(description='Populate the concept lattices with an LLM')
    parser.add_argument(
        '--data', type=Path, default=Path('data'), help='data directory (default: data)'
    )
    parser.add_argument(
        '--llm', required=True, help='short name of the LLM and of its data folders'
    )
    parser.add_argument('--model', default=None, help='model id (default: the catalog id of --llm)')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--concept-set', required=True, choices=CONCEPT_SETS)
    parser.add_argument('--provider', choices=PROVIDERS, default='vllm', help='(default: vllm)')
    parser.add_argument('--base-url', default=None, help='endpoint of an OpenAI-compatible server')
    parser.add_argument('--temperature', type=float, default=0.0, help='(default: 0)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='(default: 4096)')
    parser.add_argument(
        '--max-attempts', type=int, default=3, help='LLM responses per node at most (default: 3)'
    )
    parser.add_argument('--queries', type=int, nargs='+', default=None, help='query ids to run')
    parser.add_argument('--limit', type=int, default=None, help='run at most this many queries')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='keep the nodes of an unfinished lattice and ask only for the missing ones',
    )
    parser.add_argument('--force', action='store_true', help='regenerate finished lattices')
    add_serving_arguments(parser)
    arguments = parser.parse_args()

    model_id = arguments.model or LANGUAGE_MODELS.get(arguments.llm)
    if model_id is None:
        parser.error(f'--llm {arguments.llm} is not in the catalog. Pass --model as well.')
    data = DataLayout(arguments.data)
    records = read_records(data.dataset_file(arguments.dataset))
    if arguments.queries is not None:
        records = [record for record in records if record.query_id in arguments.queries]
    records = records[: arguments.limit]

    if arguments.provider == 'vllm':
        options = serving_options(arguments)
    else:
        options = {'base_url': arguments.base_url}
    model = LanguageModelFactory.create(arguments.provider, model_id, **options)
    generator = LatticeGenerator(
        model, arguments.temperature, arguments.max_tokens, arguments.max_attempts
    )
    directory = data.lattice_directory(arguments.llm, arguments.dataset, arguments.concept_set)
    for index, record in enumerate(records, 1):
        query_directory = directory / str(record.query_id)
        if generator.is_finished(query_directory) and not arguments.force:
            logger.info('[%d/%d] query %s is finished', index, len(records), record.query_id)
            continue
        logger.info('[%d/%d] query %s', index, len(records), record.query_id)
        resume = arguments.resume and (query_directory / 'config.json').exists()
        generator.generate(record, arguments.concept_set, query_directory, resume=resume)


if __name__ == '__main__':
    main()
