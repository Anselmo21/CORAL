"""Linear surrogate model (LSM) search built on ContextCite

LSM fits a linear model on prompts with random concepts removed and checks the nodes the model
predicts will change the answer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from coral.lattice.ablation import distinct_ablations
from coral.lattice.lattice import concept_key
from coral.search.search import Counterfactual, LatticeOracle, Search, SearchResult
from coral.surrogate.concept_prompt import ConceptPrompt

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice
    from coral.lattice.search_space import SearchSpace
    from coral.search.search import Supports
    from coral.surrogate.context_citer import ContextCiter


class LinearSurrogateSearch(Search):
    """Evaluates the nodes a ContextCite surrogate predicts will flip the answer

    The LLM calls behind the random concept ablations the surrogate fits on count as the budget
    """

    STRATEGY = 'lsm'

    def __init__(
        self,
        citer: type[ContextCiter],
        model: Any,
        tokenizer: Any,
        budget: int = 16,
        batch_size: int = 1,
        count: int = 1,
    ) -> None:
        """Store the citer class, the loaded model, and the budget

        The caller loads the model once because every search call builds a new citer

        Args:
            citer: The ContextCiter class to build for each search
            model: The loaded LLM
            tokenizer: The LLM's tokenizer
            budget: The random ablations to fit the surrogate on
            batch_size: The ablations per forward pass with transformers
            count: The number of counterfactuals to look for
        """
        super().__init__(count)
        self.citer = citer
        self.model = model
        self.tokenizer = tokenizer
        self.budget = budget
        self.batch_size = batch_size
        self.usage: dict[str, int] = {}

    def search(
        self, lattice: Lattice, space: SearchSpace, supports: Supports | None = None
    ) -> SearchResult:
        """Fit the surrogate and evaluate the nodes in the order it scores them

        Nodes the surrogate scores above zero are pruned because it expects those to keep the answer

        Args:
            lattice: The lattice to verify against
            space: The search space
            supports: Ignored because LSM uses no graph

        Returns:
            The counterfactuals found, the calls and budget spent, and the pruned node count
        """
        del supports
        initial_answer = self._initial_ranking(lattice, space)[0]
        prompt = ConceptPrompt(lattice.question, space.concepts, lattice.sources, lattice.answers)
        citer = self.citer(
            self.model, self.tokenizer, prompt, self.budget, batch_size=self.batch_size
        )
        surrogate = citer.fit()
        budget_used = distinct_ablations(len(space.concepts), self.budget)
        self.usage = dict(citer.usage)
        positions = {concept: index for index, concept in enumerate(space.concepts)}
        oracle = LatticeOracle(lattice)
        counterfactuals = []
        pruned_nodes = 0
        for _size, nodes in space.layers():
            scored = []
            for node in nodes:
                mask = np.zeros(len(space.concepts), dtype=np.float32)
                mask[[positions[concept] for concept in node]] = 1.0
                scored.append(
                    (float(surrogate.predict(mask.reshape(1, -1))[0]), concept_key(node), node)
                )
            scored.sort(key=lambda entry: (entry[0], entry[1]))
            for score, key, node in scored:
                if score > 0:
                    pruned_nodes += 1
                    continue
                ranking = oracle.ask(key)
                if ranking is not None and ranking[0] != initial_answer:
                    counterfactuals.append(Counterfactual(node, ranking[0]))
                    if len(counterfactuals) == self.count:
                        return SearchResult(
                            tuple(counterfactuals), oracle.calls, budget_used, pruned_nodes
                        )
        return SearchResult(tuple(counterfactuals), oracle.calls, budget_used, pruned_nodes)
