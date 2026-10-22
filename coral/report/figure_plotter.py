"""Plots the strategies' accuracy, cost, and distance from the evaluation records

One figure plots all three by lattice size for one LLM and the other plots accuracy and cost by
budget for every LLM
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from coral.evaluation.outcome import NOT_MINIMAL
from coral.evaluation.strategy_suite import BUDGETS
from coral.report.statistics import accuracy, brute_force_cost, cost, outcome, read_records
from coral.utilities.catalog import CONCEPT_SETS, DATASETS, LANGUAGE_MODEL_NAMES

if TYPE_CHECKING:
    from pathlib import Path

    from coral.report.statistics import Record
    from coral.utilities.files import ResultLayout

CONFIDENCE = 0.95

# Marker, colour, and label of each series
STYLES = {
    'bfs': ('o', 'black', 'BFS'),
    'sra': ('s', '#d62728', 'SRA'),
    'era': ('h', '#9467bd', 'ERA'),
    'gra': ('D', '#ff7f0e', 'GRA'),
    'gera': ('X', '#1f77b4', 'GERA'),
    'lsm': ('^', '#2ca02c', 'LSM'),
}

DATASET_TITLES = {'msmarco': 'MSMARCO', 'curated': 'Curated'}

# Font sizes and line weights of the per-size figure
SIZES_STYLE = {
    'font.size': 6.5,
    'axes.titlesize': 7.5,
    'axes.labelsize': 6.8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6.6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'axes.labelpad': 1.5,
}

# Font sizes of the budget figure
BUDGET_STYLE = {
    'font.size': 13,
    'axes.titlesize': 19,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 17,
    'legend.title_fontsize': 12,
    'figure.titlesize': 16,
}


def _bootstrap_mean(values: Sequence[float]) -> tuple[float, float, float] | None:
    """Return the mean and its 95% bootstrap interval or None

    Args:
        values: The samples to average

    Returns:
        The mean with the interval's low and high ends, or None when there are no values
    """
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    samples = np.random.default_rng(0).choice(array, size=(1000, array.size), replace=True)
    percentiles = [(1.0 - CONFIDENCE) / 2.0 * 100.0, (1.0 + CONFIDENCE) / 2.0 * 100.0]
    low, high = np.percentile(samples.mean(axis=1), percentiles)
    return mean, float(low), float(high)


class FigurePlotter:
    """Draws the per-size figure and the budget figure"""

    def __init__(self, results: ResultLayout, budget: int = 16) -> None:
        """Store where the records live and which budget to report

        The budget only applies to the per-size figure since the budget figure sweeps them all

        Args:
            results: The result layout to read records from and write figures to
            budget: The budget the per-size figure reports
        """
        self.results = results
        self.budget = budget

    def plot_lattice_sizes(self, llm: str = 'qwen3-32b') -> Path:
        """Plot accuracy, cost, and distance against lattice size for one LLM

        Each dataset gets its own column with both concept sets pooled

        Args:
            llm: The short name of the LLM

        Returns:
            The path of the saved figure
        """
        with rc_context(SIZES_STYLE):
            figure = Figure(figsize=(3.45, 3.9))
            FigureCanvasAgg(figure)
            axes = figure.subplots(3, 2)
            handles: dict[str, Any] = {}
            for column, dataset in enumerate(DATASETS):
                records = read_records(self.results, llm, dataset, CONCEPT_SETS)
                self._draw_column(axes[:, column], records, handles)
                axes[0, column].set_title(DATASET_TITLES[dataset], pad=3)
            for axis in axes.flat:
                for line in axis.get_lines():
                    line.set_markersize(2.6)
                    line.set_linewidth(1.0 if line.get_linestyle() == '--' else 0.85)
                axis.grid(True, alpha=0.3, linewidth=0.4)
            for axis in axes[:, 1]:
                axis.set_ylabel('')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_ylabel('Avg Cost (LLM Calls)')
            axes[2, 0].set_ylabel('Distance to Minimal\nCounterfactual')
            labels = [
                STYLES[strategy][2] for strategy in ('bfs', 'sra', 'era', 'gra', 'gera', 'lsm')
            ]
            figure.legend(
                [handles[label] for label in labels],
                labels,
                loc='upper center',
                ncol=len(labels),
                bbox_to_anchor=(0.53, 1.0),
                frameon=True,
                handlelength=1.6,
                handletextpad=0.4,
                columnspacing=0.9,
                borderpad=0.35,
                fancybox=False,
                edgecolor='0.7',
            )
            figure.subplots_adjust(
                left=0.135, right=0.985, top=0.9, bottom=0.08, wspace=0.2, hspace=0.6
            )
            path = self.results.figure_directory / 'lattice_sizes.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=300)
        return path

    def plot_budget_ablation(self) -> Path:
        """Plot accuracy and cost against the budget for every LLM

        LSM only appears at the nonzero budgets

        Returns:
            The path of the saved figure
        """
        with rc_context(BUDGET_STYLE):
            figure = Figure(figsize=(15, 6.2))
            FigureCanvasAgg(figure)
            axes = figure.subplots(2, 4)
            positions = list(range(len(BUDGETS)))
            surrogate_budgets = [budget for budget in BUDGETS if budget > 0]
            surrogate_positions = [BUDGETS.index(budget) for budget in surrogate_budgets]
            lines = {
                strategy: {
                    'marker': STYLES[strategy][0],
                    'color': STYLES[strategy][1],
                    'linestyle': '-',
                    'linewidth': 1.5,
                    'markersize': 6.5,
                }
                for strategy in ('era', 'gra', 'gera')
            }
            lines['gera'].update(linewidth=1.6)
            lines['gra'].update(markersize=5.5)
            lines['lsm'] = {
                'marker': '^',
                'color': STYLES['lsm'][1],
                'linestyle': ':',
                'linewidth': 1.7,
                'markersize': 6.5,
            }
            brute_force_line = {'linestyle': '--', 'color': 'black', 'linewidth': 1.7}

            def averaged(
                pooled: list[list[Record]], metric: Any, strategy: str, budget: int
            ) -> float:
                """Average a metric over an LLM's datasets

                Each dataset counts once however many records it has

                Args:
                    pooled: The records of each dataset
                    metric: The metric to compute on each dataset's outcomes
                    strategy: The strategy's abbreviation (sra, lsm, era, gra, gera, or got)
                    budget: The budget to read the outcomes at

                Returns:
                    The mean of the per-dataset values
                """
                return float(
                    np.mean(
                        [
                            metric([outcome(record, strategy, budget) for record in records])
                            for records in pooled
                        ]
                    )
                )

            for column, (llm, name) in enumerate(LANGUAGE_MODEL_NAMES.items()):
                pooled = [
                    read_records(self.results, llm, dataset, CONCEPT_SETS) for dataset in DATASETS
                ]
                series = {
                    strategy: (
                        [averaged(pooled, accuracy, strategy, budget) for budget in BUDGETS],
                        [averaged(pooled, cost, strategy, budget) for budget in BUDGETS],
                    )
                    for strategy in ('gera', 'gra', 'era')
                }
                surrogate_accuracy = [
                    averaged(pooled, accuracy, 'lsm', budget) for budget in surrogate_budgets
                ]
                surrogate_cost = [
                    averaged(pooled, cost, 'lsm', budget) for budget in surrogate_budgets
                ]
                brute_force = float(np.mean([brute_force_cost(records) for records in pooled]))

                accuracy_axis, cost_axis = axes[0][column], axes[1][column]
                accuracy_axis.plot(surrogate_positions, surrogate_accuracy, **lines['lsm'])
                for strategy in ('era', 'gra', 'gera'):
                    accuracy_axis.plot(
                        positions, series[strategy][0], label=strategy.upper(), **lines[strategy]
                    )
                accuracy_axis.set_title(name, pad=8)
                # The lower limit fits every line but the upper limit fits only ERA, GRA, and GERA
                low = min(min(min(series[arm][0]) for arm in series), min(surrogate_accuracy))
                high = max(max(series[arm][0]) for arm in series)
                margin = max(1.0, 0.03 * (high - low))
                accuracy_axis.set_ylim(low - margin, high + margin)
                accuracy_axis.yaxis.set_major_locator(MaxNLocator(5, integer=True))

                cost_axis.axhline(brute_force, label='BFS', **brute_force_line)
                cost_axis.plot(surrogate_positions, surrogate_cost, **lines['lsm'])
                for strategy in ('era', 'gra', 'gera'):
                    cost_axis.plot(positions, series[strategy][1], **lines[strategy])
                low = min(min(series[arm][1]) for arm in series)
                high = max(
                    brute_force,
                    *(max(series[arm][1]) for arm in series),
                    max(surrogate_cost, default=0),
                )
                margin = 0.05 * (high - low)
                cost_axis.set_ylim(low - margin, high + margin)
                cost_axis.yaxis.set_major_locator(MaxNLocator(5))
                for axis in (accuracy_axis, cost_axis):
                    axis.grid(True, alpha=0.3)
                    axis.set_xticks(positions)
                    axis.set_xticklabels([str(budget) for budget in BUDGETS])
                    axis.set_xlabel('Budget')
                if column == 0:
                    accuracy_axis.set_ylabel('Accuracy (%)')
                    cost_axis.set_ylabel('Avg Cost (LLM Calls)')

            handles = [
                Line2D([], [], label=STYLES[strategy][2], **lines[strategy])
                for strategy in ('gera', 'gra', 'era', 'lsm')
            ]
            handles.append(Line2D([], [], label='BFS', **brute_force_line))
            figure.legend(
                handles=handles,
                loc='upper center',
                ncol=len(handles),
                bbox_to_anchor=(0.5, 1.03),
                frameon=True,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            path = self.results.figure_directory / 'budget_ablation.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches='tight')
        return path

    def _draw_column(
        self, axes: Sequence[Any], records: list[Record], handles: dict[str, Any]
    ) -> None:
        """Draw the accuracy, cost, and distance panels of one dataset

        Handles are collected per label so the shared legend shows each strategy once

        Args:
            axes: The three panels of the column, top to bottom
            records: The dataset's evaluation records
            handles: The legend handles found so far, updated in place
        """
        by_size: dict[int, list[Record]] = defaultdict(list)
        for record in records:
            by_size[len(record['concepts'])].append(record)
        sizes = sorted(by_size)
        strategies = ('sra', 'era', 'gra', 'gera', 'lsm')

        def strategy_outcomes(strategy: str, size: int) -> list[dict[str, Any]]:
            """Return a strategy's outcomes on the cases of one size

            Args:
                strategy: The strategy's abbreviation (sra, lsm, era, gra, gera, or got)
                size: The number of concepts in the cases

            Returns:
                The matching outcome of every case of that size
            """
            budget = None if strategy == 'sra' else self.budget
            return [outcome(record, strategy, budget) for record in by_size[size]]

        for axis, metric, label in (
            (axes[0], accuracy, 'Accuracy (%)'),
            (axes[1], cost, 'Avg Cost (LLM Calls)'),
        ):
            if metric is cost:
                marker, colour, name = STYLES['bfs']
                (line,) = axis.plot(
                    sizes,
                    [brute_force_cost(by_size[size]) for size in sizes],
                    marker=marker,
                    linestyle='--',
                    color=colour,
                    linewidth=2.0,
                    markersize=7,
                    label=name,
                )
                handles.setdefault(name, line)
            for strategy in strategies:
                marker, colour, name = STYLES[strategy]
                (line,) = axis.plot(
                    sizes,
                    [metric(strategy_outcomes(strategy, size)) for size in sizes],
                    marker=marker,
                    linestyle='-',
                    color=colour,
                    linewidth=1.8,
                    markersize=7,
                    label=name,
                )
                handles.setdefault(name, line)
            axis.set_xlabel('Number of Concepts')
            axis.set_ylabel(label)
            axis.set_xticks(sizes)
            axis.grid(True, alpha=0.3)
            if metric is accuracy:
                axis.set_ylim(0, 108)

        distance_axis = axes[2]
        width = 0.84 / len(strategies)
        for index, strategy in enumerate(strategies):
            _marker, colour, name = STYLES[strategy]
            positions, means, below, above = [], [], [], []
            for size in sizes:
                distances = [
                    float(entry['distance'])
                    for entry in strategy_outcomes(strategy, size)
                    if entry['label'] == NOT_MINIMAL
                ]
                interval = _bootstrap_mean(distances)
                if interval is None:
                    continue
                mean, low, high = interval
                positions.append(size + (index - (len(strategies) - 1) / 2.0) * width)
                means.append(mean)
                below.append(mean - low)
                above.append(high - mean)
            if not positions:
                continue
            bars = distance_axis.bar(
                positions,
                means,
                width=width,
                color=colour,
                label=name,
                yerr=[below, above],
                capsize=0.8,
                edgecolor='black',
                linewidth=0.2,
                error_kw={'linewidth': 0.45},
            )
            handles.setdefault(name, bars[0])
        distance_axis.set_xlabel('Number of Concepts')
        distance_axis.set_ylabel('Distance to Minimal Counterfactual')
        distance_axis.set_xticks(sizes)
        distance_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        distance_axis.grid(True, alpha=0.3, axis='y')
