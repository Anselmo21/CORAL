"""Reads a stored graph of thoughts and works out which concepts support each answer

An answer's support is the set of concepts with an edge to that answer
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from coral.utilities.files import read_json


@dataclass(frozen=True)
class GraphOfThoughts:
    """The graph an LLM drew from the question to sub-questions, concepts, and answers"""

    nodes: tuple[dict, ...]
    edges: tuple[dict, ...]

    @classmethod
    def from_file(cls, path: Path) -> GraphOfThoughts | None:
        """Read graph.json or return None if it never parsed

        Failed runs still write the file, with a null graph and the raw reply

        Args:
            path: The graph.json file

        Returns:
            The graph, or None if the file holds no parsed graph
        """
        graph = read_json(path).get('graph')
        if not graph:
            return None
        return cls(tuple(graph['nodes']), tuple(graph['edges']))

    def supports(
        self, concepts: Iterable[str], answers: Iterable[str]
    ) -> dict[str, frozenset[str]]:
        """Map each answer to the concepts with an edge to it

        Labels are matched loosely because models rename and renumber the concepts they were given

        Args:
            concepts: The query's concepts
            answers: The query's candidate answers

        Returns:
            Each answer with the concepts that support it, empty when no edge reaches it
        """
        concepts = set(concepts)
        answers = list(answers)

        def match(label: str, candidates: set[str]) -> str | None:
            """Return the candidate a label refers to or None if it is ambiguous

            Matching runs from exact to looser substring checks that each have to land on a single
            candidate

            Args:
                label: The node label the LLM wrote
                candidates: The exact names it may refer to

            Returns:
                The matching candidate, or None if no check finds exactly one
            """
            if label in candidates:
                return label
            stripped = re.sub(r'^\s*\d+[.)]\s*', '', label)
            if stripped in candidates:
                return stripped
            for hits in (
                [candidate for candidate in candidates if candidate.startswith(stripped)],
                [candidate for candidate in candidates if f'({stripped})' in candidate],
                [candidate for candidate in candidates if stripped in candidate],
            ):
                if len(hits) == 1:
                    return hits[0]
            hits = [
                candidate for candidate in candidates if stripped.casefold() in candidate.casefold()
            ]
            return hits[0] if len(hits) == 1 else None

        labels = {node['id']: node.get('label', node['id']) for node in self.nodes}
        concept_nodes = {node['id'] for node in self.nodes if str(node.get('layer')) == '3'}
        answer_nodes = {
            node['id']: match(labels[node['id']], set(answers))
            for node in self.nodes
            if str(node.get('layer')) == '4'
        }
        supports: dict[str, set[str]] = {answer: set() for answer in answers}
        for edge in self.edges:
            source, target = edge.get('source'), edge.get('target')
            if source in answer_nodes and target in concept_nodes:
                source, target = target, source
            if source not in concept_nodes or answer_nodes.get(target) is None:
                continue
            concept = match(labels[source], concepts)
            if concept is not None:
                supports[answer_nodes[target]].add(concept)
        return {answer: frozenset(support) for answer, support in supports.items()}
