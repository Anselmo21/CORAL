"""Kemeny, MCT, MC4, and Outranking rank aggregation through the PyFlagr library"""

from __future__ import annotations

import csv
import ctypes
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence

from pyflagr import Kemeny, Majoritarian, MarkovChains

from coral.proxy.aggregator import Aggregator, normalize_max, rank_by_score

METHODS = {
    'kemeny': Kemeny.KemenyOptimal,
    'mct': lambda: MarkovChains.MCT(10),
    'mc4': lambda: MarkovChains.MC4(10),
    'outranking': Majoritarian.OutrankingApproach,
}

# Kemeny raises a ValueError above this many answers because FLAGR's Kemeny tries every ordering
KEMENY_ANSWER_LIMIT = 8


class FlagrAggregator(Aggregator):
    """Runs a PyFlagr method with short aliases in place of the answers

    The aliases keep commas and long fields out of FLAGR's CSV parser
    """

    def __init__(self, method: str) -> None:
        """Create the PyFlagr method by name

        An unknown name raises a ValueError that lists the ones available

        Args:
            method: One of 'kemeny', 'mct', 'mc4', or 'outranking'
        """
        if method not in METHODS:
            raise ValueError(f'unknown PyFlagr method {method!r}, choose from {sorted(METHODS)}')
        self.NAME = method
        self.method = METHODS[method]()

    def aggregate(
        self, voters: Mapping[str, Mapping[str, float]], answers: Sequence[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Aggregate the voters' rankings with the PyFlagr method

        Scores fall back to rank-based ones when FLAGR's own are missing, flat, or out of order

        Args:
            voters: Each voter's score for each answer
            answers: The answers to rank

        Returns:
            The answers best first and their scores scaled to a largest magnitude of one
        """
        if self.NAME == 'kemeny' and len(answers) > KEMENY_ANSWER_LIMIT:
            raise ValueError(f'kemeny would enumerate {len(answers)}! orderings')
        alias = {answer: f'a{index}' for index, answer in enumerate(answers)}
        inverse = {name: answer for answer, name in alias.items()}

        file_descriptor, input_path = tempfile.mkstemp(suffix='.csv', prefix='flagr_in_')
        os.close(file_descriptor)
        output_directory = tempfile.mkdtemp(prefix='flagr_out_')
        try:
            with open(input_path, 'w', newline='', encoding='ascii') as input_file:
                writer = csv.writer(input_file)
                for voter_index, vector in enumerate(voters.values()):
                    ranking = rank_by_score({answer: vector.get(answer, 0.0) for answer in answers})
                    for rank, answer in enumerate(ranking):
                        writer.writerow(
                            [
                                'q',
                                f'v{voter_index}',
                                alias[answer],
                                rank + 1,
                                len(answers) - 1 - rank,
                                'coral',
                            ]
                        )
            # Only redirecting the file descriptor silences FLAGR because it prints from C
            ctypes.CDLL(None).fflush(None)
            null_descriptor = os.open(os.devnull, os.O_WRONLY)
            saved_descriptor = os.dup(1)
            os.dup2(null_descriptor, 1)
            try:
                output = self.method.aggregate(input_file=input_path, out_dir=output_directory)
            finally:
                ctypes.CDLL(None).fflush(None)
                os.dup2(saved_descriptor, 1)
                os.close(saved_descriptor)
                os.close(null_descriptor)
            frame = output[0].sort_values('Rank', kind='stable')
            ordered = [inverse[item] for item in frame['ItemID'] if item in inverse]
            raw_scores = {
                inverse[item]: score
                for item, score in zip(frame['ItemID'], frame['Score'], strict=True)
                if item in inverse
            }
        finally:
            os.unlink(input_path)
            shutil.rmtree(output_directory, ignore_errors=True)

        missing = sorted(answer for answer in answers if answer not in raw_scores)
        ordered += missing
        raw = [raw_scores[answer] for answer in ordered if answer in raw_scores]
        descending = all(raw[index] >= raw[index + 1] for index in range(len(raw) - 1))
        if not missing and descending and len(set(raw)) > 1:
            scores = {answer: float(raw_scores[answer]) for answer in ordered}
        else:
            scores = {answer: float(len(answers) - rank) for rank, answer in enumerate(ordered)}
        return ordered, normalize_max(scores)
