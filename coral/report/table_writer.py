"""Writes the strategy and configuration tables as CSV files

The strategy table compares every strategy per LLM and dataset and the configuration table compares
GERA's entropy scorers and aggregators
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

from coral.evaluation.configuration_sweep import AGGREGATORS
from coral.proxy.entropy import ENTROPY_SCORERS
from coral.report.statistics import (
    accuracy,
    brute_force_cost,
    cost,
    distance,
    outcome,
    read_records,
)
from coral.utilities.catalog import CONCEPT_SETS, DATASETS, LANGUAGE_MODEL_NAMES
from coral.utilities.files import read_json

if TYPE_CHECKING:
    from pathlib import Path

    from coral.utilities.files import ResultLayout


def _rounded(value: float | None) -> str:
    """Round to one decimal or leave empty

    Args:
        value: The number to format, or None

    Returns:
        The formatted cell
    """
    return '' if value is None else f'{value:.1f}'


class TableWriter:
    """Writes the strategy and configuration tables"""

    def __init__(self, results: ResultLayout, budget: int = 16) -> None:
        """Store where the records live and which budget to report

        Args:
            results: The result layout to read records from and write tables to
            budget: The budget the strategy table reports
        """
        self.results = results
        self.budget = budget

    def write_strategy_table(self) -> Path:
        """Write accuracy, cost, and distance per LLM, dataset, and strategy

        BFS only fills in the cost and GoT only the accuracy

        Returns:
            The path of the CSV file
        """
        rows = []
        for llm in LANGUAGE_MODEL_NAMES:
            for dataset in DATASETS:
                records = read_records(self.results, llm, dataset, CONCEPT_SETS)
                rows.append([llm, dataset, 'bfs', '', _rounded(brute_force_cost(records)), ''])
                for strategy in ('sra', 'lsm', 'era', 'gra', 'gera'):
                    budget = None if strategy == 'sra' else self.budget
                    entries = [outcome(record, strategy, budget) for record in records]
                    rows.append(
                        [
                            llm,
                            dataset,
                            strategy,
                            _rounded(accuracy(entries)),
                            _rounded(cost(entries)),
                            _rounded(distance(entries)),
                        ]
                    )
                entries = [outcome(record, 'got') for record in records]
                rows.append([llm, dataset, 'got', _rounded(accuracy(entries)), '', ''])
        header = ['llm', 'dataset', 'strategy', 'accuracy', 'cost', 'distance']
        return self._write('strategies.csv', header, rows)

    def write_configuration_table(self) -> Path:
        """Write GERA's accuracy and cost per entropy scorer and aggregator

        Each row is the unweighted mean over every combination of LLM, dataset, and concept set

        Returns:
            The path of the CSV file
        """
        cells = [
            read_json(self.results.configuration_file(llm, dataset, concept_set))
            for llm in LANGUAGE_MODEL_NAMES
            for dataset in DATASETS
            for concept_set in CONCEPT_SETS
        ]
        rows = []
        for setting, variants in (('entropy', tuple(ENTROPY_SCORERS)), ('aggregator', AGGREGATORS)):
            for variant in variants:
                per_cell = [
                    [record['outcomes'][setting][variant] for record in records]
                    for records in cells
                ]
                rows.append(
                    [
                        setting,
                        variant,
                        _rounded(float(np.mean([accuracy(entries) for entries in per_cell]))),
                        _rounded(float(np.mean([cost(entries) for entries in per_cell]))),
                    ]
                )
        return self._write('configurations.csv', ['setting', 'variant', 'accuracy', 'cost'], rows)

    def _write(self, name: str, header: list[str], rows: list[list[str]]) -> Path:
        """Write one CSV table

        The table directory gets created if this is the first table

        Args:
            name: The file name inside the table directory
            header: The column names
            rows: The table's rows

        Returns:
            The path of the CSV file
        """
        path = self.results.table_directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path
