"""Test cases stored as concept subsets and loaded with everything a search needs"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from coral.graph.graph_of_thoughts import GraphOfThoughts
from coral.lattice.lattice import read_lattices
from coral.lattice.search_space import Direction, SearchSpace
from coral.search.breadth_first_search import BreadthFirstSearch
from coral.utilities.files import read_json, write_json

if TYPE_CHECKING:
    from pathlib import Path

    from coral.lattice.lattice import Lattice
    from coral.search.search import Counterfactual, Supports
    from coral.utilities.files import DataLayout


"""Test cases on disk"""


@dataclass(frozen=True)
class TestCase:
    """A query's lattice restricted to a subset of its concepts

    The depth is how many steps the minimal counterfactual was from the initial node when the case
    was sampled
    """

    case_id: int
    query_id: str
    concepts: tuple[str, ...]
    depth: int


@dataclass(frozen=True)
class CasePool:
    """The test cases of one LLM, dataset, and concept set, with their search direction"""

    direction: Direction
    cases: tuple[TestCase, ...]

    @classmethod
    def read(cls, path: Path) -> CasePool:
        """Read a pool written by write

        The concept lists get converted back to tuples here because JSON has none

        Args:
            path: The case file

        Returns:
            The pool with its cases in file order
        """
        document = read_json(path)
        cases = tuple(
            TestCase(case['case_id'], case['query_id'], tuple(case['concepts']), case['depth'])
            for case in document['cases']
        )
        return cls(Direction(document['direction']), cases)

    def write(self, path: Path) -> None:
        """Write the pool as JSON

        Args:
            path: The case file to write
        """
        write_json(
            path,
            {
                'direction': self.direction.value,
                'cases': [
                    {
                        'case_id': case.case_id,
                        'query_id': case.query_id,
                        'concepts': list(case.concepts),
                        'depth': case.depth,
                    }
                    for case in self.cases
                ],
            },
        )


"""Test cases ready to search"""


@dataclass(frozen=True)
class EvaluationCase:
    """A test case loaded with its sublattice, search space, and minimal counterfactual

    The extra fields hold the answer supports from the graph of thoughts and the LLM calls BFS spent
    finding the minimal counterfactual
    """

    case: TestCase
    lattice: Lattice
    space: SearchSpace
    supports: Supports
    minimal: Counterfactual
    minimal_calls: int


def load_evaluation_cases(
    data: DataLayout,
    direction: Direction,
    llm: str,
    dataset: str,
    concept_set: str,
    with_graphs: bool = True,
) -> list[EvaluationCase]:
    """Load the test cases of one LLM, dataset, and concept set with sublattices and BFS results

    Callers other than LSM keep with_graphs on and get a ValueError for a query with no parsed graph

    Args:
        data: The data layout to read from
        direction: The search direction of the pool
        llm: The short name of the LLM
        dataset: The dataset name
        concept_set: Either 'criteria' or 'pragmas'
        with_graphs: Whether to read each query's graph and its answer supports

    Returns:
        One prepared case per entry in the pool
    """
    pool = CasePool.read(data.case_file(direction.value, llm, dataset, concept_set))
    lattices = read_lattices(data.lattice_directory(llm, dataset, concept_set))
    graph_directory = data.graph_directory(llm, dataset, concept_set)
    graphs: dict[str, GraphOfThoughts | None] = {}
    prepared = []
    for case in pool.cases:
        lattice = lattices[case.query_id]
        supports: Supports = {}
        if with_graphs:
            if case.query_id not in graphs:
                graphs[case.query_id] = GraphOfThoughts.from_file(
                    graph_directory / case.query_id / 'graph.json'
                )
            graph = graphs[case.query_id]
            if graph is None:
                raise ValueError(f'query {case.query_id} has no parsed graph of thoughts')
            # Supports use the query's full concept set instead of the case's subset because GERA
            # scores a node by the share of each answer's linked concepts the node keeps
            supports = graph.supports(lattice.concepts, lattice.answers)
        sublattice = lattice.restrict(case.concepts)
        space = SearchSpace(case.concepts, direction=pool.direction)
        minimal = BreadthFirstSearch().search(sublattice, space)
        if minimal.counterfactual is None:
            raise ValueError(f'case {case.case_id} has no counterfactual')
        prepared.append(
            EvaluationCase(case, sublattice, space, supports, minimal.counterfactual, minimal.calls)
        )
    return prepared
