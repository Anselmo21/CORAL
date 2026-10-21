"""Runs GERA with one entropy scorer or rank aggregator swapped in at a time"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from coral.evaluation.outcome import Outcome
from coral.evaluation.test_case import load_evaluation_cases
from coral.lattice.search_space import Direction
from coral.proxy.entropy import ENTROPY_SCORERS
from coral.search.graph_entropy_rank_aggregation import GraphEntropyRankAggregation

if TYPE_CHECKING:
    from coral.utilities.files import DataLayout

logger = logging.getLogger(__name__)

AGGREGATORS = ('borda', 'kemeny', 'mct', 'mc4', 'outranking')


class ConfigurationSweep:
    """Runs GERA once per entropy scorer and once per rank aggregator"""

    def __init__(
        self, data: DataLayout, direction: Direction = Direction.TOP_DOWN, budget: int = 16
    ) -> None:
        """Store where the test cases live and the budget every variant runs at

        Args:
            data: The data layout to read cases from
            direction: The search direction of the cases
            budget: The budget every variant runs at
        """
        self.data = data
        self.direction = direction
        self.budget = budget

    def run(self, llm: str, dataset: str, concept_set: str) -> list[dict[str, Any]]:
        """Evaluate every variant on the test cases of one LLM, dataset, and concept set

        Borda runs through the proxy's default aggregator while the other aggregators need PyFlagr

        Args:
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            One record per test case with each variant's outcome
        """
        # PyFlagr is an optional dependency and loads only when the sweep runs
        from coral.proxy.flagr_aggregator import FlagrAggregator

        searches = {
            'entropy': {
                name: GraphEntropyRankAggregation(self.budget, entropy_scorer=scorer)
                for name, scorer in ENTROPY_SCORERS.items()
            },
            'aggregator': {
                name: GraphEntropyRankAggregation(
                    self.budget, aggregator=None if name == 'borda' else FlagrAggregator(name)
                )
                for name in AGGREGATORS
            },
        }
        records = []
        evaluations = load_evaluation_cases(self.data, self.direction, llm, dataset, concept_set)
        for evaluation in evaluations:
            outcomes = {
                setting: {
                    name: Outcome.from_result(
                        evaluation.minimal,
                        search.search(evaluation.lattice, evaluation.space, evaluation.supports),
                    ).to_json()
                    for name, search in variants.items()
                }
                for setting, variants in searches.items()
            }
            records.append({'case_id': evaluation.case.case_id, 'outcomes': outcomes})
        logger.info('%s %s %s: %d cases', llm, dataset, concept_set, len(records))
        return records
