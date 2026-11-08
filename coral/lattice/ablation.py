"""Draws the random concept ablations that LSM fits its surrogate on"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

KEEP_PROBABILITY = 0.5


def create_mask(count: int, keep_probability: float, seed: int) -> NDArray[np.bool_]:
    """Draw which concepts one ablation keeps

    The seed is the ablation's index

    Args:
        count: The number of concepts
        keep_probability: The chance each concept is kept
        seed: The seed of this mask

    Returns:
        The boolean mask over the concepts
    """
    return np.random.RandomState(seed).choice(
        [False, True], size=count, p=[1 - keep_probability, keep_probability]
    )


def distinct_ablations(
    count: int, ablations: int, keep_probability: float = KEEP_PROBABILITY
) -> int:
    """Return how many different masks the ablations draw

    Args:
        count: The number of concepts
        ablations: The number of ablations
        keep_probability: The chance each concept is kept

    Returns:
        The number of distinct masks
    """
    return len({tuple(create_mask(count, keep_probability, seed)) for seed in range(ablations)})
