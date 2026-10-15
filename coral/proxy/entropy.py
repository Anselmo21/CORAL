"""Entropy scores that measure how unsure an estimated ranking is

A high score means several answers sit close to first place
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import partial

import numpy as np

EntropyScorer = Callable[[Mapping[str, float]], float]


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Turn the scores into probabilities with a softmax

    Args:
        scores: The score of each answer

    Returns:
        The probability of each answer
    """
    values = np.array(list(scores.values()), dtype=float)
    exponentials = np.exp(values - values.max())
    return dict(zip(scores.keys(), (exponentials / exponentials.sum()).tolist(), strict=True))


def shannon_entropy(scores: Mapping[str, float]) -> float:
    """Return the Shannon entropy of the positive scores normalized by their sum

    The entropy is zero when no score is above zero

    Args:
        scores: The score of each answer

    Returns:
        The entropy in bits
    """
    values = [value for value in scores.values() if value > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum(value / total * math.log2(value / total) for value in values)


def softmax_entropy(scores: Mapping[str, float]) -> float:
    """Return the Shannon entropy of the softmax of the scores

    Softmax entropy is the default because the softmax separates a tie for first place from a tie
    for second

    Args:
        scores: The score of each answer

    Returns:
        The entropy in bits
    """
    return shannon_entropy(softmax(scores))


def weighted_entropy(scores: Mapping[str, float]) -> float:
    """Return the softmax entropy with each term weighted by one over its rank

    Args:
        scores: The score of each answer

    Returns:
        The weighted entropy in bits
    """
    probabilities = np.sort(np.array(list(softmax(scores).values())))[::-1]
    weights = 1.0 / np.arange(1, len(probabilities) + 1)
    return -float(np.sum(weights * probabilities * np.log2(probabilities + 1e-12)))


def renyi_entropy(scores: Mapping[str, float], alpha: float) -> float:
    """Return the Rényi entropy of the softmax of the scores

    An alpha below one stresses the bottom of the ranking and an alpha above one stresses the top

    Args:
        scores: The score of each answer
        alpha: The order of the entropy (any value except one)

    Returns:
        The entropy in bits
    """
    probabilities = np.array(list(softmax(scores).values()))
    return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(probabilities**alpha) + 1e-12))


ENTROPY_SCORERS: dict[str, EntropyScorer] = {
    'softmax': softmax_entropy,
    'shannon': shannon_entropy,
    'weighted': weighted_entropy,
    'renyi_0.5': partial(renyi_entropy, alpha=0.5),
    'renyi_2': partial(renyi_entropy, alpha=2.0),
}
