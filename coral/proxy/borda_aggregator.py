"""Implements the Borda count the proxy uses by default to combine rankings

The point totals set the combined order after each input ranking gives an answer more points the
higher the answer sits
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from coral.proxy.aggregator import Aggregator, rank_by_score


class BordaAggregator(Aggregator):
    """Sums the voters' Borda scores"""

    NAME = 'borda'

    def aggregate(
        self, voters: Mapping[str, Mapping[str, float]], answers: Sequence[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Return the answers ranked by summed score and the sums

        A voter with no score for an answer adds nothing to it

        Args:
            voters: Each voter's score for each answer
            answers: The answers to rank

        Returns:
            The answers best first and the summed score of each one
        """
        totals = dict.fromkeys(answers, 0.0)
        for vector in voters.values():
            for answer in answers:
                totals[answer] += vector.get(answer, 0.0)
        return rank_by_score(totals), totals
