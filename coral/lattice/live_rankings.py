"""Rankings that ask the LLM about a node the first time a search reads it

The rankings get cached in a lattice folder for a rerun on the same question to read back
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import combinations
from typing import TYPE_CHECKING, Any

from coral.lattice.lattice import Lattice, concept_key
from coral.utilities.files import read_json, write_json

if TYPE_CHECKING:
    from pathlib import Path

    from coral.dataset.base_dataset import QueryRecord
    from coral.lattice.lattice_generator import LatticeGenerator
    from coral.llm.language_model import Completion, Message


class LiveRankings(Mapping[str, list[str]]):
    """Maps node keys to rankings and asks the LLM for each ranking not cached yet

    A key outside the query's concept subsets, or a node with no parseable reply, counts as missing
    """

    def __init__(
        self,
        generator: LatticeGenerator,
        record: QueryRecord,
        concept_set: str,
        directory: Path,
    ) -> None:
        """Index the concept subsets and read back the cache of an earlier run

        A cache written for a different question, concept list, or answer list gets replaced

        Args:
            generator: The lattice generator that ranks the answers with the LLM
            record: The question with its concepts and candidate answers
            concept_set: Either 'criteria' or 'pragmas'
            directory: The folder that holds the cached rankings
        """
        self.generator = generator
        self.record = record
        self.concept_set = concept_set
        self.directory = directory
        concepts = record.concepts(concept_set)
        self.nodes = {
            concept_key(subset): subset
            for size in range(len(concepts) + 1)
            for subset in combinations(concepts, size)
        }
        self.configuration: dict[str, Any] = {
            'question': record.question,
            'concept_set': concept_set,
            'concepts': list(concepts),
            'plausible_answers': list(record.answers),
            'sources': record.sources,
            'halted': False,
        }
        self.cache: dict[str, list[str]] = {}
        path = directory / 'config.json'
        if path.exists() and all(
            read_json(path).get(name) == value for name, value in self.configuration.items()
        ):
            self.cache = read_json(directory / 'answers.json')
        self.failed: set[str] = set()
        self.asked: list[str] = []
        self.usage = {'llm_calls': 0, 'prompt_tokens': 0, 'output_tokens': 0}

    def __getitem__(self, key: str) -> list[str]:
        """Return the ranking at a node and ask the LLM for it the first time

        Args:
            key: The node's lattice key

        Returns:
            The answers best first
        """
        if key not in self.nodes or key in self.failed:
            raise KeyError(key)
        if key not in self.cache:
            node = self.nodes[key]
            ranked = self.generator.rank_nodes(self.record, self.concept_set, [node], self._ask)
            self.asked.append(key)
            if ranked[0].ranking is None:
                self.failed.add(key)
                raise KeyError(key)
            self.cache[key] = ranked[0].ranking
            write_json(self.directory / 'answers.json', self.cache)
            write_json(self.directory / 'config.json', self.configuration)
        return self.cache[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the nodes with a cached ranking

        Returns:
            The cached node keys
        """
        return iter(self.cache)

    def __len__(self) -> int:
        """Return how many nodes have a cached ranking

        Returns:
            The number of cached rankings
        """
        return len(self.cache)

    def lattice(self) -> Lattice:
        """Return a lattice that reads its rankings through this mapping

        Returns:
            The live lattice of the question
        """
        return Lattice(
            query_id=self.directory.name,
            question=self.record.question,
            sources=self.record.sources,
            concept_set=self.concept_set,
            concepts=self.record.concepts(self.concept_set),
            answers=self.record.answers,
            rankings=self,
        )

    def _ask(
        self, conversations: list[list[Message]], regex: str | None = None
    ) -> list[Completion]:
        """Prompt the LLM and add the cost to the usage totals

        Args:
            conversations: The conversations to answer in one batch
            regex: The pattern each response must match, if any

        Returns:
            One completion per conversation
        """
        completions = self.generator.model.complete(
            conversations, self.generator.temperature, self.generator.max_tokens, regex
        )
        for completion in completions:
            self.usage['llm_calls'] += 1
            self.usage['prompt_tokens'] += completion.prompt_tokens or 0
            self.usage['output_tokens'] += completion.output_tokens or 0
        return completions
