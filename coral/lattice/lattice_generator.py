"""Populates a query's lattice by asking the LLM to rank the answers at every node

A query stops after its single-concept nodes when they all give the same top answer
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING, Any

from coral.lattice.lattice import concept_key
from coral.lattice.ranking_prompt import (
    SYSTEM_MESSAGE,
    correction_prompt,
    parse_ranking,
    permutation_regex,
    ranking_prompt,
)
from coral.utilities.files import read_json, write_json

if TYPE_CHECKING:
    from pathlib import Path

    from coral.dataset.base_dataset import QueryRecord
    from coral.llm.language_model import Completion, LanguageModel

logger = logging.getLogger(__name__)


class LatticeGenerator:
    """Asks the LLM to rank the answers under every subset of the concepts"""

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
            temperature: The sampling temperature
            max_tokens: The token cap per response
            max_attempts: The most responses to request per node
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    def is_finished(self, directory: Path) -> bool:
        """Return whether the lattice is complete or halted

        Args:
            directory: The query's lattice directory

        Returns:
            True if config.json says the run halted or reached every node
        """
        path = directory / 'config.json'
        if not path.exists():
            return False
        configuration = read_json(path)
        return (
            configuration['halted'] or configuration['combos_run'] == configuration['combos_total']
        )

    def generate(
        self, record: QueryRecord, concept_set: str, directory: Path, resume: bool = False
    ) -> dict[str, list[str]]:
        """Populate one query's lattice and write it to the directory

        The rankings, token counts, and run settings go into answers.json, tokens.json, and
        config.json

        Args:
            record: The query to populate
            concept_set: Either 'criteria' or 'pragmas'
            directory: The query's lattice directory
            resume: Whether to keep the nodes already on disk and ask only for the rest

        Returns:
            The rankings by node key
        """
        concepts = record.concepts(concept_set)
        nodes = [
            subset for size in range(len(concepts) + 1) for subset in combinations(concepts, size)
        ]
        previous = (
            read_json(directory / 'config.json')
            if resume and (directory / 'config.json').exists()
            else None
        )
        rankings: dict[str, list[str]] = read_json(directory / 'answers.json') if previous else {}
        node_tokens: dict[str, list[int | None]] = (
            read_json(directory / 'tokens.json') if previous else {}
        )
        pending = [node for node in nodes if concept_key(node) not in rankings]
        logger.info(
            'query %s: %d of %d nodes to populate', record.query_id, len(pending), len(nodes)
        )

        usage = {'prompt_tokens': 0, 'output_tokens': 0, 'llm_calls': 0}
        finish_reasons: Counter[str] = Counter()
        constrained: list[str] = []
        halted_reason = None

        def ask(
            conversations: list[list[dict[str, str]]], regex: str | None = None
        ) -> list[Completion]:
            """Prompt the LLM and add the cost to the totals

            Args:
                conversations: The conversations to answer in one batch
                regex: The pattern each response must match, if any

            Returns:
                One completion per conversation
            """
            completions = self.model.complete(
                conversations, self.temperature, self.max_tokens, regex
            )
            for completion in completions:
                usage['llm_calls'] += 1
                finish_reasons[str(completion.finish_reason)] += 1
                # Some backends report no tokens
                usage['prompt_tokens'] += completion.prompt_tokens or 0
                usage['output_tokens'] += completion.output_tokens or 0
            return completions

        # A resumed lattice already passed the halt check
        batches = (
            [pending]
            if previous
            else [
                [node for node in pending if len(node) <= 1],
                [node for node in pending if len(node) > 1],
            ]
        )
        for batch in batches:
            if not batch:
                continue
            if not previous and len(batch[0]) > 1:
                top_answers = {
                    rankings[concept][0].strip().lower()
                    for concept in concepts
                    if concept in rankings
                }
                if len(top_answers) == 1 and all(concept in rankings for concept in concepts):
                    halted_reason = (
                        f'all {len(concepts)} singletons produced the same top-1 answer: '
                        f"'{next(iter(top_answers))}'"
                    )
                    logger.info('query %s halted, %s', record.query_id, halted_reason)
                    break
            prompts = [
                ranking_prompt(concept_set, record.question, node, record.sources, record.answers)
                for node in batch
            ]
            completions = ask([self._conversation(prompt) for prompt in prompts])
            for node, prompt, completion in zip(batch, prompts, completions, strict=True):
                key = concept_key(node)
                ranking = parse_ranking(completion.text, record.answers)
                kept = completion
                attempts = 1
                if ranking is None and self.model.CONSTRAINED_DECODING:
                    # Constrained decoding comes before the correction prompt because some models
                    # answer the correction prompt with the same fixed ordering at every node
                    attempts += 1
                    kept = ask(
                        [self._conversation(prompt)], permutation_regex(len(record.answers))
                    )[0]
                    ranking = parse_ranking(kept.text, record.answers)
                    if ranking is not None:
                        constrained.append(key)
                while ranking is None and attempts < self.max_attempts:
                    attempts += 1
                    kept = ask([self._conversation(correction_prompt(prompt, kept.text))])[0]
                    ranking = parse_ranking(kept.text, record.answers)
                if ranking is None:
                    logger.warning('node %r did not parse after %d attempts', key, attempts)
                    continue
                rankings[key] = ranking
                node_tokens[key] = [kept.prompt_tokens, kept.output_tokens]

        if previous:
            halted = previous['halted']
            halted_reason = previous['halted_reason']
            for name in ('prompt_tokens', 'output_tokens', 'llm_calls'):
                usage[name] += previous['usage'][name]
            finish_reasons.update(previous['usage']['finish_reasons'])
            constrained = sorted(set(previous.get('constrained_nodes', [])) | set(constrained))
        else:
            halted = halted_reason is not None
        configuration: dict[str, Any] = {
            'question': record.question,
            'concept_set': concept_set,
            'concepts': list(concepts),
            'plausible_answers': list(record.answers),
            'sources': record.sources,
            'halted': halted,
            'halted_reason': halted_reason,
            'combos_total': len(nodes),
            'combos_run': len(rankings),
            'usage': {**usage, 'finish_reasons': dict(finish_reasons)},
            'constrained_nodes': sorted(constrained),
            'generation': {
                'model': self.model.model,
                'provider': self.model.PROVIDER,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'serving': self.model.serving,
            },
        }
        write_json(directory / 'answers.json', rankings)
        write_json(directory / 'config.json', configuration)
        write_json(directory / 'tokens.json', node_tokens, indent=None)
        return rankings

    @staticmethod
    def _conversation(prompt: str) -> list[dict[str, str]]:
        """Wrap a prompt with the system message

        Args:
            prompt: The user message

        Returns:
            The chat messages to send
        """
        return [{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user', 'content': prompt}]
