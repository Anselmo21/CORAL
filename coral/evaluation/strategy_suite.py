"""Runs every search strategy on the same test cases at every budget

Every strategy replays the same stored rankings while LSM results come from earlier runs
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from coral.evaluation.outcome import Outcome
from coral.evaluation.test_case import load_evaluation_cases
from coral.lattice.search_space import Direction
from coral.search.search_factory import SearchFactory
from coral.utilities.files import read_json

if TYPE_CHECKING:
    from coral.utilities.files import DataLayout

logger = logging.getLogger(__name__)

BUDGETS = (0, 2, 4, 8, 16, 32, 64, 128)


class StrategySuite:
    """Runs every strategy on the test cases at every budget"""

    def __init__(
        self,
        data: DataLayout,
        direction: Direction = Direction.TOP_DOWN,
        budgets: tuple[int, ...] = BUDGETS,
    ) -> None:
        """Store where the test cases live and the budgets to sweep

        Budget zero runs the proxy on the single-concept rankings alone

        Args:
            data: The data layout to read cases and LSM runs from
            direction: The search direction of the cases
            budgets: The budgets to run each budgeted strategy at
        """
        self.data = data
        self.direction = direction
        self.budgets = budgets

    def run(self, llm: str, dataset: str, concept_set: str) -> list[dict[str, Any]]:
        """Evaluate every strategy on the test cases of one LLM, dataset, and concept set

        SRA and GoT run once, the budgeted strategies once per budget, and LSM from saved runs that
        must cover every case

        Args:
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            One record per test case with the minimal counterfactual and every outcome
        """
        cases = load_evaluation_cases(self.data, self.direction, llm, dataset, concept_set)
        surrogate_runs = {}
        for budget in self.budgets:
            path = self.data.surrogate_file(self.direction.value, llm, dataset, concept_set, budget)
            if path.exists():
                surrogate_runs[budget] = {run['case_id']: run for run in read_json(path)}

        records = []
        for evaluation in cases:
            case = evaluation.case
            outcomes: dict[str, Any] = {}
            for strategy in ('sra', 'got'):
                result = SearchFactory.create(strategy).search(
                    evaluation.lattice, evaluation.space, evaluation.supports
                )
                outcomes[strategy] = Outcome.from_result(evaluation.minimal, result).to_json()
            for strategy in ('era', 'gra', 'gera'):
                outcomes[strategy] = {}
                for budget in self.budgets:
                    result = SearchFactory.create(strategy, budget).search(
                        evaluation.lattice, evaluation.space, evaluation.supports
                    )
                    outcome = Outcome.from_result(evaluation.minimal, result)
                    outcomes[strategy][str(budget)] = outcome.to_json()
            outcomes['lsm'] = {}
            for budget, runs in surrogate_runs.items():
                run = runs.get(case.case_id)
                if run is None or tuple(run['concepts']) != case.concepts:
                    # Never score LSM on fewer cases than the other strategies
                    raise ValueError(f'LSM at budget {budget} has no run for case {case.case_id}')
                outcome = Outcome.score(
                    evaluation.minimal, run['key'], run['size'], run['calls'], run['budget']
                )
                outcomes['lsm'][str(budget)] = outcome.to_json()
            records.append(
                {
                    'case_id': case.case_id,
                    'query_id': case.query_id,
                    'concepts': list(case.concepts),
                    'minimal': {
                        'key': evaluation.minimal.key,
                        'size': evaluation.minimal.size,
                        'calls': evaluation.minimal_calls,
                    },
                    'outcomes': outcomes,
                }
            )
        logger.info('%s %s %s: %d cases', llm, dataset, concept_set, len(records))
        return records
