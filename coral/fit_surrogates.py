"""Runs the linear surrogate model (LSM) on the test cases with the LLM loaded

The runs of each budget go to their own file under data/surrogate
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from coral.evaluation.outcome import Outcome
from coral.evaluation.test_case import load_evaluation_cases
from coral.lattice.search_space import Direction
from coral.search.linear_surrogate_search import LinearSurrogateSearch
from coral.utilities.arguments import add_serving_arguments, serving_options
from coral.utilities.catalog import CONCEPT_SETS, DATASETS, LANGUAGE_MODELS
from coral.utilities.files import DataLayout, read_json, write_json

logger = logging.getLogger(__name__)


def main() -> None:
    """Run LSM at one budget on the test cases that have no run yet

    The run file is rewritten after every case
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(
        description='Run the linear surrogate model (LSM) on the test cases'
    )
    parser.add_argument(
        '--data', type=Path, default=Path('data'), help='data directory (default: data)'
    )
    parser.add_argument(
        '--direction',
        choices=('top-down', 'bottom-up'),
        default='top-down',
        help='direction of the test cases (default: top-down)',
    )
    parser.add_argument('--llm', required=True, choices=list(LANGUAGE_MODELS))
    parser.add_argument('--model', default=None, help='model id (default: the catalog id of --llm)')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--concept-set', required=True, choices=CONCEPT_SETS)
    parser.add_argument('--budget', type=int, required=True, help='ablations per test case')
    parser.add_argument(
        '--backend',
        choices=('vllm', 'transformers'),
        default='vllm',
        help='serve the LLM with vLLM or with Hugging Face transformers (default: vllm)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='ablations per forward pass with transformers'
    )
    parser.add_argument('--limit', type=int, default=None, help='run at most this many cases')
    add_serving_arguments(parser)
    parser.set_defaults(gpu_memory_utilization=0.85)
    arguments = parser.parse_args()

    data = DataLayout(arguments.data)
    direction = Direction(arguments.direction)
    path = data.surrogate_file(
        direction.value, arguments.llm, arguments.dataset, arguments.concept_set, arguments.budget
    )
    runs = read_json(path) if path.exists() else []
    done = {run['case_id'] for run in runs}
    cases = [
        evaluation
        for evaluation in load_evaluation_cases(
            data, direction, arguments.llm, arguments.dataset, arguments.concept_set, False
        )
        if evaluation.case.case_id not in done
    ][: arguments.limit]
    logger.info('%d cases done, %d to run -> %s', len(done), len(cases), path)

    model_id = arguments.model or LANGUAGE_MODELS[arguments.llm]
    # Backends are optional and load only when chosen
    if arguments.backend == 'vllm':
        from coral.surrogate.vllm_context_citer import VllmContextCiter

        model, tokenizer = VllmContextCiter.load_model(model_id, **serving_options(arguments))
        search = LinearSurrogateSearch(VllmContextCiter, model, tokenizer, arguments.budget)
    else:
        from coral.surrogate.context_citer import ContextCiter

        model, tokenizer = ContextCiter.load_model(model_id)
        search = LinearSurrogateSearch(
            ContextCiter, model, tokenizer, arguments.budget, arguments.batch_size
        )

    for evaluation in cases:
        case = evaluation.case
        start = time.time()
        result = search.search(evaluation.lattice, evaluation.space)
        found = result.counterfactual
        runs.append(
            {
                'case_id': case.case_id,
                'query_id': case.query_id,
                'concepts': list(case.concepts),
                'budget': arguments.budget,
                'key': found.key if found else None,
                'size': found.size if found else None,
                'calls': result.calls,
                'llm_calls': search.usage['llm_calls'],
                'prompt_tokens': search.usage['prompt_tokens'],
                'output_tokens': search.usage['output_tokens'],
                'elapsed': round(time.time() - start, 2),
            }
        )
        write_json(path, sorted(runs, key=lambda run: run['case_id']))
        outcome = Outcome.from_result(evaluation.minimal, result)
        logger.info('case %d: %s in %.1fs', case.case_id, outcome.label, time.time() - start)


if __name__ == '__main__':
    main()
