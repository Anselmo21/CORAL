"""Explains an LLM's answer to a question saved by propose_concepts with a live search

The search asks the LLM only about the concept subsets it checks and caches every ranking in a
folder named after the question file
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coral.dataset.base_dataset import QueryRecord
from coral.evaluation.configuration_sweep import AGGREGATORS
from coral.graph.graph_generator import GraphGenerator
from coral.graph.graph_of_thoughts import GraphOfThoughts
from coral.lattice.lattice import concept_key
from coral.lattice.lattice_generator import LatticeGenerator
from coral.lattice.live_rankings import LiveRankings
from coral.lattice.search_space import Direction, SearchSpace
from coral.proxy.entropy import ENTROPY_SCORERS
from coral.search.search_factory import SearchFactory
from coral.utilities.arguments import add_model_arguments, create_model
from coral.utilities.files import read_json

NAMES = {'bfs': 'BFS', 'sra': 'SRA', 'era': 'ERA', 'gra': 'GRA', 'gera': 'GERA', 'got': 'GoT'}

GRAPH_STRATEGIES = ('gra', 'gera', 'got')


def main() -> None:
    """Parse the command line, run the search, and print the explanation

    Strategies that use a graph of thoughts ask the LLM for one first unless the folder has it
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(description="Explain an LLM's answer to a saved question")
    parser.add_argument('question_file', type=Path, help='JSON file written by propose_concepts')
    parser.add_argument(
        '--strategy', choices=SearchFactory.STRATEGIES, default='gera', help='(default: gera)'
    )
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
    parser.add_argument(
        '--direction',
        choices=('top-down', 'bottom-up'),
        default='top-down',
        help='remove concepts from the full set or add them to the pinned ones (default: top-down)',
    )
    add_model_arguments(parser)
    arguments = parser.parse_args()

    saved = read_json(arguments.question_file)
    pinned = tuple(saved.get('pinned', []))
    concepts = tuple(saved['concepts']) + tuple(c for c in pinned if c not in saved['concepts'])
    record = QueryRecord(0, saved['question'], '', concepts, (), tuple(saved['answers']))
    model = create_model(arguments)
    settings = (arguments.temperature, arguments.max_tokens, arguments.max_attempts)
    directory = arguments.question_file.with_suffix('')
    rankings = LiveRankings(LatticeGenerator(model, *settings), record, 'criteria', directory)
    lattice = rankings.lattice()
    space = SearchSpace(concepts, anchor=pinned, direction=Direction(arguments.direction))
    initial = lattice.ranking(space.initial_node)
    if initial is None:
        raise SystemExit('the LLM gave no parseable ranking at the initial node')

    supports = None
    graph_calls = 0
    if arguments.strategy in GRAPH_STRATEGIES:
        path = directory / 'graph.json'
        saved_graph = read_json(path) if path.exists() else {}
        if not (
            saved_graph.get('graph')
            and saved_graph.get('concepts') == list(concepts)
            and saved_graph.get('plausible_answers') == list(record.answers)
        ):
            # The graph prompt states the answer under every concept
            rankings.get(concept_key(concepts))
            GraphGenerator(model, *settings).generate(record, 'criteria', directory, directory)
            graph_calls = read_json(path)['usage']['llm_calls']
        graph = GraphOfThoughts.from_file(path)
        if graph is None:
            raise SystemExit('the LLM gave no parseable graph of thoughts, try --strategy era')
        supports = graph.supports(concepts, record.answers)

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
    result = search.search(lattice, space, supports)

    name = NAMES[arguments.strategy]
    print(f'Question: {record.question}')
    print(f'Answer: {initial[0]}')
    found = result.counterfactual
    if found is None:
        print(f'{name} found no set of concepts that changes the answer')
    else:
        changed = [c for c in concepts if (c in space.initial_node) != (c in found.node)]
        action = 'Removing' if space.direction is Direction.TOP_DOWN else 'Adding'
        print(f'{action} {", ".join(changed)} changes the answer to: {found.answer}')
    calls = (
        f'LLM calls: {rankings.usage["llm_calls"]} for {len(rankings.asked)} of the '
        f'{len(rankings.nodes)} concept subsets'
    )
    if graph_calls:
        calls += f' and {graph_calls} for the graph of thoughts'
    print(calls)


if __name__ == '__main__':
    main()
