"""The rank aggregation interface and the score helpers aggregators share"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

"""Score vectors"""


def borda_vector(ranking: Sequence[str], answers: Sequence[str]) -> dict[str, float]:
    """Score each answer by the number of answers minus its rank

    Answers missing from the ranking stay at zero, below every ranked one

    Args:
        ranking: The answers best first
        answers: The answers to score

    Returns:
        The score of each answer
    """
    vector = dict.fromkeys(answers, 0.0)
    for rank, answer in enumerate(ranking):
        if answer in vector:
            vector[answer] = float(len(answers) - rank)
    return vector


def normalize_max(scores: Mapping[str, float]) -> dict[str, float]:
    """Divide the scores by their largest magnitude

    All-zero scores come back unchanged

    Args:
        scores: The score of each answer

    Returns:
        The scaled scores, with the largest magnitude at one
    """
    largest = max((abs(score) for score in scores.values()), default=0.0)
    if largest == 0:
        return dict(scores)
    return {answer: score / largest for answer, score in scores.items()}


def rank_by_score(scores: Mapping[str, float]) -> list[str]:
    """Rank answers by score with ties broken by text

    Args:
        scores: The score of each answer

    Returns:
        The answers best first
    """
    return sorted(scores, key=lambda answer: (-scores[answer], answer))


"""Aggregator interface"""


class Aggregator(ABC):
    """Combines the answer scores of several smaller nodes into a consensus ranking"""

    NAME: str

    @abstractmethod
    def aggregate(
        self, voters: Mapping[str, Mapping[str, float]], answers: Sequence[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Return the consensus ranking and scores

        Voters are nodes below the one being estimated, each with its own score per answer

        Args:
            voters: Each voter's score for each answer
            answers: The answers to rank

        Returns:
            The answers best first and the score of each one
        """
