"""Contrast scores read off a query's Graph of Thoughts.

The graph says which concepts back each candidate answer. A node's contrast is
how much more of the baseline's backing it strips than its best rival's, so a
positive contrast is the model's own explanation calling for a flip there.
"""
from __future__ import annotations


def graph_contrast(subset, supports, answers, baseline):
    """max over rivals of cov(rival) - cov(baseline), 0.0 when there are no rivals."""
    present = set(subset)
    cov = lambda a: (len(present & supports.get(a, set()))
                     / len(supports[a]) if supports.get(a) else 0.0)
    base = cov(baseline)
    return max((cov(a) - base for a in answers if a != baseline), default=0.0)
