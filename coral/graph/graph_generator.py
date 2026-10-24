"""Asks an LLM for one query's graph of thoughts and stores the reply

The initial answer in the prompt is the LLM's top answer under all the concepts and comes from the
query's lattice when one exists
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

from coral.graph.graph_prompt import graph_prompt, missing_seeds, parse_graph
from coral.lattice.lattice import concept_key
from coral.lattice.ranking_prompt import SYSTEM_MESSAGE, parse_ranking, ranking_prompt
from coral.utilities.files import read_json, write_json

if TYPE_CHECKING:
    from pathlib import Path

    from coral.dataset.base_dataset import QueryRecord
    from coral.llm.language_model import Completion, LanguageModel, Message

logger = logging.getLogger(__name__)


class GraphGenerator:
    """Asks the LLM for one graph per query, naming the concepts and answers it must contain

    A reply that fails to parse or leaves some of them out gets a follow-up in the same conversation
    """

    def __init__(
        self,
        model: LanguageModel,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_attempts: int = 3,
    ) -> None:
        """Store the LLM and its generation settings

        Args:
            model: The LLM backend
            temperature: The sampling temperature of the graph prompt
            max_tokens: The token cap per graph response
            max_attempts: The most responses to request per graph and per initial answer
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    def is_finished(self, directory: Path) -> bool:
        """Return whether the query already has a parsed graph

        Args:
            directory: The query's graph directory

        Returns:
            True if graph.json holds a parsed graph
        """
        path = directory / 'graph.json'
        return path.exists() and read_json(path).get('graph') is not None

    def generate(
        self,
        record: QueryRecord,
        concept_set: str,
        directory: Path,
        lattice_directory: Path | None = None,
    ) -> dict[str, Any] | None:
        """Ask for the query's graph and write it to graph.json

        An earlier failed attempt is kept as graph.failed.json so its raw reply is not lost

        Args:
            record: The query to draw a graph for
            concept_set: Either 'criteria' or 'pragmas'
            directory: The query's graph directory
            lattice_directory: The query's lattice directory to read the initial answer from
                instead of making a live call

        Returns:
            The parsed graph, or None if no response parsed
        """
        concepts = record.concepts(concept_set)
        answers = list(record.answers)
        usage = {'prompt_tokens': 0, 'output_tokens': 0, 'llm_calls': 0}
        finish_reasons: list[str | None] = []

        def ask(messages: list[Message], max_tokens: int, temperature: float) -> Completion:
            """Prompt once and add the cost to the totals

            Args:
                messages: The conversation to send
                max_tokens: The token cap of this response
                temperature: The sampling temperature of this response

            Returns:
                The LLM's completion
            """
            completion = self.model.complete([messages], temperature, max_tokens)[0]
            usage['llm_calls'] += 1
            usage['prompt_tokens'] += completion.prompt_tokens or 0
            usage['output_tokens'] += completion.output_tokens or 0
            finish_reasons.append(completion.finish_reason)
            return completion

        initial_answer = None
        if lattice_directory is not None and (lattice_directory / 'config.json').exists():
            configuration = read_json(lattice_directory / 'config.json')
            rankings = read_json(lattice_directory / 'answers.json')
            if configuration['halted']:
                votes = Counter(rankings[concept][0] for concept in concepts if concept in rankings)
                initial_answer = max(votes, key=votes.get) if votes else None
            elif concept_key(concepts) in rankings:
                initial_answer = rankings[concept_key(concepts)][0]
        source = 'lattice' if initial_answer is not None else 'live'
        if initial_answer is None:
            prompt = ranking_prompt(concept_set, record.question, concepts, record.sources, answers)
            ranking_messages = [
                {'role': 'system', 'content': SYSTEM_MESSAGE},
                {'role': 'user', 'content': prompt},
            ]
            for _attempt in range(self.max_attempts):
                response = ask(ranking_messages, max_tokens=256, temperature=0.0).text
                ranking = parse_ranking(response, answers)
                if ranking is not None:
                    initial_answer = ranking[0]
                    break
                ranking_messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response},
                    {
                        'role': 'user',
                        'content': f'Your previous answer was invalid. Return ONLY a JSON array of '
                        f'exactly {len(answers)} integers, a permutation of 0..{len(answers) - 1}, '
                        f'best first.',
                    },
                ]
            else:
                raise RuntimeError(f'query {record.query_id} got no parseable ranking')

        prompt = graph_prompt(
            record.sources, record.question, concept_set, concepts, answers, initial_answer
        )
        messages: list[Message] = [{'role': 'user', 'content': prompt}]
        graph = None
        response = ''
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            response = ask(messages, self.max_tokens, self.temperature).text
            parsed = parse_graph(response)
            short = missing_seeds(parsed, concepts, answers) if parsed is not None else None
            if parsed is not None:
                # Keep a graph with missing nodes in case no later attempt does better
                graph = parsed
                if short is None:
                    break
            if short:
                follow_up = (
                    f'Your graph left out seeded nodes: {short}. Emit the whole graph '
                    'again with every Layer 3 seed and every Layer 4 answer present, '
                    'inside a single ```json ...``` fenced block at the end.'
                )
            else:
                follow_up = (
                    "Your previous answer did not contain a parseable JSON object with 'nodes' "
                    "and 'edges'. Keep any reasoning brief, then emit the graph inside a single "
                    '```json ...``` fenced block at the end.'
                )
            messages = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': response},
                {'role': 'user', 'content': follow_up},
            ]

        payload: dict[str, Any] = {
            'query_id': record.query_id,
            'query': record.question,
            'concept_set': concept_set,
            'concepts': list(concepts),
            'plausible_answers': answers,
            'baseline_answer': initial_answer,
            'baseline_index': answers.index(initial_answer),
            'baseline_source': source,
            'serving': self.model.serving,
            'model': self.model.model,
            'provider': self.model.PROVIDER,
            'temperature': self.temperature,
            'attempts': attempts,
            'usage': {**usage, 'finish_reasons': finish_reasons},
            'graph': graph,
        }
        if graph is None:
            payload['raw'] = response
            payload['parse_error'] = True
        path = directory / 'graph.json'
        if path.exists() and read_json(path).get('parse_error'):
            path.replace(directory / 'graph.failed.json')
        write_json(path, payload)
        return graph
