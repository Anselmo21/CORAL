"""JSON helpers and the paths of everything under data and results"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

"""JSON input and output"""


def read_json(path: Path) -> Any:
    """Read a JSON file

    Args:
        path: The file to read

    Returns:
        The decoded document
    """
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, document: Any, indent: int | None = 2) -> None:
    """Write a JSON file and create its directory

    Args:
        path: The file to write
        document: The document to encode
        indent: The indent width, or None for one compact line
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=indent), encoding='utf-8')


"""Directory layout"""


@dataclass(frozen=True)
class DataLayout:
    """Paths of the datasets, lattices, graphs, test cases, and LSM runs"""

    root: Path = Path('data')

    def dataset_file(self, dataset: str) -> Path:
        """Return the path of a base dataset file

        Args:
            dataset: The dataset name

        Returns:
            The path of the JSON file
        """
        return self.root / 'datasets' / f'{dataset}.json'

    def lattice_directory(self, llm: str, dataset: str, concept_set: str) -> Path:
        """Return the lattice directory of one LLM, dataset, and concept set

        Each query has its own folder inside it, named by its query id

        Args:
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            The directory path
        """
        return self.root / 'lattices' / llm / dataset / concept_set

    def graph_directory(self, llm: str, dataset: str, concept_set: str) -> Path:
        """Return the graph directory of one LLM, dataset, and concept set

        Graphs mirror the lattice layout with one folder per query

        Args:
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            The directory path
        """
        return self.root / 'graphs' / llm / dataset / concept_set

    def case_file(self, direction: str, llm: str, dataset: str, concept_set: str) -> Path:
        """Return the test case file of one LLM, dataset, and concept set

        The direction comes first in the path because cases are sampled per direction

        Args:
            direction: The folder name of the search direction
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            The path of the JSON file
        """
        return self.root / 'cases' / direction / llm / dataset / f'{concept_set}.json'

    def surrogate_file(
        self, direction: str, llm: str, dataset: str, concept_set: str, budget: int
    ) -> Path:
        """Return the LSM run file of one LLM, dataset, and concept set at a budget

        Args:
            direction: The folder name of the search direction
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'
            budget: The number of ablations

        Returns:
            The path of the JSON file
        """
        return (
            self.root
            / 'surrogate'
            / direction
            / llm
            / dataset
            / concept_set
            / f'budget_{budget}.json'
        )


@dataclass(frozen=True)
class ResultLayout:
    """Paths of the evaluation records, tables, and figures

    The root normally ends in the search direction to keep top-down and bottom-up results apart
    """

    root: Path = Path('results') / 'top-down'

    def evaluation_file(self, llm: str, dataset: str, concept_set: str) -> Path:
        """Return the evaluation records of one LLM, dataset, and concept set

        Args:
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            The path of the JSON file
        """
        return self.root / 'evaluation' / llm / dataset / f'{concept_set}.json'

    def configuration_file(self, llm: str, dataset: str, concept_set: str) -> Path:
        """Return the ConfigurationSweep records of one LLM, dataset, and concept set

        Args:
            llm: The short name of the LLM
            dataset: The dataset name
            concept_set: Either 'criteria' or 'pragmas'

        Returns:
            The path of the JSON file
        """
        return self.root / 'configurations' / llm / dataset / f'{concept_set}.json'

    @property
    def table_directory(self) -> Path:
        """Return the table directory

        Returns:
            The directory path
        """
        return self.root / 'tables'

    @property
    def figure_directory(self) -> Path:
        """Return the figure directory

        Returns:
            The directory path
        """
        return self.root / 'figures'
