"""Writes the tables and figures from the records evaluate_strategies saved"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coral.evaluation.strategy_suite import BUDGETS
from coral.report.figure_plotter import FigurePlotter
from coral.report.table_writer import TableWriter
from coral.utilities.files import ResultLayout

logger = logging.getLogger(__name__)


def main() -> None:
    """Write the tables and figures

    The table of GERA's entropy scorers and aggregators is skipped until
    evaluate_strategies --configurations has run
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(description='Write the tables and figures')
    parser.add_argument(
        '--results', type=Path, default=Path('results'), help='results directory (default: results)'
    )
    parser.add_argument(
        '--direction',
        choices=('top-down', 'bottom-up'),
        default='top-down',
        help='search direction to report (default: top-down)',
    )
    parser.add_argument(
        '--budget',
        type=int,
        choices=BUDGETS,
        default=16,
        help='budget to report in the tables and the figure by lattice size (default: 16)',
    )
    parser.add_argument(
        '--llm', default='qwen3-32b', help='LLM of the figure by lattice size (default: qwen3-32b)'
    )
    arguments = parser.parse_args()

    results = ResultLayout(arguments.results / arguments.direction)
    tables = TableWriter(results, arguments.budget)
    paths = [tables.write_strategy_table()]
    if (results.root / 'configurations').exists():
        paths.append(tables.write_configuration_table())
    figures = FigurePlotter(results, arguments.budget)
    paths.append(figures.plot_lattice_sizes(arguments.llm))
    paths.append(figures.plot_budget_ablation())
    for path in paths:
        logger.info('wrote %s', path)


if __name__ == '__main__':
    main()
