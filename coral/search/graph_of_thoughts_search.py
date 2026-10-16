"""Graph of thoughts search (GoT) checks the nodes the graph proposes as counterfactuals"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coral.lattice.lattice import concept_key
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace
    from coral.search.search import Supports


class GraphOfThoughtsSearch(Search):
    """Verifies the counterfactuals the graph proposes

    Each rival answer proposes the anchor plus the concepts the graph links to that answer
    """

    STRATEGY = 'got'

    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Verify the proposals closest to the initial node

        One proposal gets checked per counterfactual wanted

        Args:
            lattice: The lattice to verify against
            space: The search space
            supports: Each answer's supporting concepts from the graph (required)

        Returns:
            The proposals that flipped the answer and the calls spent
        """
        if supports is None:
            raise ValueError(f'{self.STRATEGY} needs the supports of a graph of thoughts')
        initial_answer = self._initial_ranking(lattice, space)[0]
        initial_node = frozenset(space.initial_node)
        proposals: list[frozenset[str]] = []
        for answer, support in supports.items():
            proposal = (support & frozenset(space.concepts)) | frozenset(space.anchor)
            if answer == initial_answer or proposal == initial_node or proposal in proposals:
                continue
            proposals.append(proposal)
        proposals.sort(key=lambda proposal: (len(proposal ^ initial_node), concept_key(proposal)))

        oracle = LatticeOracle(lattice)
        counterfactuals = []
        for proposal in proposals[: self.count]:
            ranking = oracle.ask(concept_key(proposal))
            if ranking and ranking[0] != initial_answer:
                node = tuple(concept for concept in space.concepts if concept in proposal)
                counterfactuals.append(Counterfactual(node, ranking[0]))
        return SearchResult(tuple(counterfactuals), oracle.calls)
