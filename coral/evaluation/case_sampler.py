"""Samples test cases from the populated lattices"""

from __future__ import annotations

import logging
import random
from itertools import combinations
from typing import TYPE_CHECKING

from coral.evaluation.test_case import CasePool, TestCase
from coral.lattice.search_space import Direction, SearchSpace
from coral.search.breadth_first_search import BreadthFirstSearch

if TYPE_CHECKING:
    from coral.lattice.lattice import Lattice

logger = logging.getLogger(__name__)


class CaseSampler:
    """Samples sublattices with a minimal counterfactual three or more layers from the initial node

    Sublattices of three concepts only need two layers while full lattices need none
    """

    def __init__(
        self,
        sizes: range = range(3, 10),
        quota: int = 20,
        seed: int = 0,
        direction: Direction = Direction.TOP_DOWN,
    ) -> None:
        """Store the sizes to sample, how many cases to keep, and the seed

        The default sizes stop at nine because full lattices get their own quota in sample

        Args:
            sizes: The sublattice sizes to draw
            quota: The cases to keep per size and for the full lattices
            seed: The seed of the shuffles
            direction: Whether the cases search top-down or bottom-up
        """
        self.sizes = sizes
        self.quota = quota
        self.seed = seed
        self.direction = direction

    def sample(self, lattices: dict[str, Lattice]) -> CasePool:
        """Draw the test cases for one LLM, dataset, and concept set

        Lattices that halted or never finished cannot qualify while a size that runs short only logs
        a warning

        Args:
            lattices: The lattices to draw from, keyed by query id

        Returns:
            The pool of drawn cases, numbered in the order they were drawn
        """
        populated = {
            query_id: lattice for query_id, lattice in lattices.items() if lattice.is_populated
        }
        breadth_first_search = BreadthFirstSearch()

        def depth(lattice: Lattice, concepts: tuple[str, ...]) -> int | None:
            """Return how many steps the minimal counterfactual is from the initial node

            Args:
                lattice: The lattice to search
                concepts: The concept subset that bounds the search

            Returns:
                The step count, or None if the subset has no counterfactual
            """
            space = SearchSpace(concepts, direction=self.direction)
            counterfactual = breadth_first_search.search(lattice, space).counterfactual
            if counterfactual is None:
                return None
            return abs(counterfactual.size - len(space.initial_node))

        rng = random.Random(self.seed)
        drawn: list[tuple[str, tuple[str, ...], int]] = []
        for size in self.sizes:
            candidates = [
                (query_id, subset)
                for query_id, lattice in populated.items()
                for subset in combinations(sorted(lattice.concepts), size)
            ]
            rng.shuffle(candidates)
            minimum_depth = 2 if size == 3 else 3
            taken = 0
            for query_id, subset in candidates:
                if taken == self.quota:
                    break
                steps = depth(populated[query_id], subset)
                if steps is None or steps < minimum_depth:
                    continue
                drawn.append((query_id, subset, steps))
                taken += 1
            if taken < self.quota:
                logger.warning(
                    'size %d has only %d of %d qualifying subsets', size, taken, self.quota
                )

        full_lattices = []
        for query_id, lattice in populated.items():
            steps = depth(lattice, lattice.concepts)
            if steps is not None:
                full_lattices.append((query_id, tuple(sorted(lattice.concepts)), steps))
        rng.shuffle(full_lattices)
        if len(full_lattices) < self.quota:
            logger.warning('only %d full lattices have a counterfactual', len(full_lattices))
        drawn.extend(full_lattices[: self.quota])

        cases = tuple(
            TestCase(case_id, query_id, concepts, steps)
            for case_id, (query_id, concepts, steps) in enumerate(drawn)
        )
        return CasePool(self.direction, cases)
