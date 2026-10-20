"""Scores a search result against the minimal counterfactual found by breadth-first search"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coral.search.search import Counterfactual, SearchResult

MINIMAL = 'minimal'
NOT_MINIMAL = 'not_minimal'
NOT_FOUND = 'not_found'


@dataclass(frozen=True)
class Outcome:
    """How close a strategy got to the minimal counterfactual and what it cost

    The key, size, and distance stay None when the strategy found nothing
    """

    label: str
    key: str | None
    size: int | None
    distance: int | None
    calls: int
    budget_used: int

    @classmethod
    def score(
        cls,
        minimal: Counterfactual,
        key: str | None,
        size: int | None,
        calls: int,
        budget_used: int,
    ) -> Outcome:
        """Score a found counterfactual against the minimal one

        A counterfactual counts as minimal when it has as many concepts as the one BFS finds

        Args:
            minimal: The counterfactual BFS found for this case
            key: The lattice key of the found node, or None if nothing was found
            size: The concept count of the found node, or None if nothing was found
            calls: The LLM calls the traversal spent
            budget_used: The budget calls spent before the traversal

        Returns:
            The outcome with its label and its distance from the minimal size
        """
        if key is None or size is None:
            return cls(NOT_FOUND, None, None, None, calls, budget_used)
        distance = abs(size - minimal.size)
        return cls(
            MINIMAL if distance == 0 else NOT_MINIMAL, key, size, distance, calls, budget_used
        )

    @classmethod
    def from_result(cls, minimal: Counterfactual, result: SearchResult) -> Outcome:
        """Score the first counterfactual in a search result

        Args:
            minimal: The counterfactual BFS found for this case
            result: The result of the search being scored

        Returns:
            The scored outcome
        """
        found = result.counterfactual
        return cls.score(
            minimal,
            found.key if found else None,
            found.size if found else None,
            result.calls,
            result.budget_used,
        )

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> Outcome:
        """Read an outcome written by to_json

        Args:
            document: The decoded JSON object

        Returns:
            The outcome it describes
        """
        return cls(**document)

    @property
    def cost(self) -> int:
        """Return the traversal calls plus the budget

        Returns:
            The total LLM calls
        """
        return self.calls + self.budget_used

    def to_json(self) -> dict[str, Any]:
        """Return the outcome as a plain dictionary

        Returns:
            The outcome's fields by name
        """
        return asdict(self)
