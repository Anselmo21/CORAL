"""pyflagr rank aggregation with sanitized inputs.

FLAGR's CSV parser breaks on long fields and splits on every comma. Answers
and voters are aliased to short ids like a0 and v0 before each call and
mapped back afterwards.
"""
from __future__ import annotations

import csv
import ctypes
import os
import shutil
import tempfile
from contextlib import contextmanager

import pyflagr.Kemeny as Kemeny
import pyflagr.Linear as Linear
import pyflagr.Majoritarian as Majoritarian
import pyflagr.MarkovChains as MarkovChains
import pyflagr.RRA as RRA
import pyflagr.Weighted as Weighted

KEMENY_MAX_ANSWERS = 8  # FLAGR's Kemeny brute-forces all n! orderings

_FACTORIES = {
    "borda":         Linear.BordaCount,
    "combsum":       Linear.CombSUM,
    "combmnz":       Linear.CombMNZ,
    "condorcet":     Majoritarian.CondorcetWinners,
    "copeland":      Majoritarian.CopelandWinners,
    "outranking":    Majoritarian.OutrankingApproach,
    "kemeny":        Kemeny.KemenyOptimal,
    "rra":           RRA.RRA,
    # mc1 and mc3 are omitted because FLAGR 1.0.21 builds their chains wrong
    "mc2":           lambda: MarkovChains.MC2(10),
    "mc4":           lambda: MarkovChains.MC4(10),
    "mct":           lambda: MarkovChains.MCT(10),
    "prefrel":       Weighted.PreferenceRelationsGraph,
    "agglomerative": Weighted.Agglomerative,
    "dibra":         Weighted.DIBRA,
}

_method_cache: dict[str, object] = {}


@contextmanager
def _silence_stdout():
    """Suppress FLAGR's C-level progress lines via fd redirect."""
    def flush_c():
        try:
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        flush_c()
        os.dup2(devnull_fd, 1)
        yield
    finally:
        flush_c()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)


def flagr_aggregate(voter_rankings, answers, method="borda", verbose=False):
    """Run one pyflagr aggregation and return (ranking, scores).

    Voter labels and answers may be long or contain commas. They never reach
    FLAGR verbatim. Returns ([], {}) when there is nothing to aggregate.
    """
    if not voter_rankings or not answers:
        return [], {}

    # resolve the method name to a cached aggregator instance
    if isinstance(method, str):
        key = method.lower()
        if key not in _FACTORIES:
            raise ValueError(f"unknown flagr method {method!r}; choose from {sorted(_FACTORIES)}")
        if key not in _method_cache:
            _method_cache[key] = _FACTORIES[key]()
        agg = _method_cache[key]
    else:
        agg = method
    if type(agg).__name__ == "KemenyOptimal" and len(answers) > KEMENY_MAX_ANSWERS:
        raise ValueError(
            f"kemeny is O(n!) in FLAGR; refusing n={len(answers)} answers "
            f"(limit {KEMENY_MAX_ANSWERS})")

    n = len(answers)
    alias = {a: f"a{i}" for i, a in enumerate(answers)}
    inverse = {v: k for k, v in alias.items()}

    fd, in_path = tempfile.mkstemp(suffix=".csv", prefix="flagr_in_")
    os.close(fd)
    out_dir = tempfile.mkdtemp(prefix="flagr_out_")
    try:
        # write the aliased votes as one plain csv row each
        with open(in_path, "w", newline="", encoding="ascii") as f:
            w = csv.writer(f)
            for vi, ranking in enumerate(voter_rankings.values()):
                for rank, ans in enumerate(ranking):
                    if ans in alias:
                        w.writerow(["q", f"v{vi}", alias[ans], rank + 1, n - 1 - rank, "coral"])
        if verbose:
            out = agg.aggregate(input_file=in_path, out_dir=out_dir)
        else:
            with _silence_stdout():
                out = agg.aggregate(input_file=in_path, out_dir=out_dir)
        if out is None:
            return [], {}
        df = out[0]
        if df.empty:
            return [], {}
        df = df.sort_values("Rank", kind="stable")
        ordered = [inverse[i] for i in df["ItemID"] if i in inverse]
        raw_scores = {inverse[i]: s for i, s in zip(df["ItemID"], df["Score"]) if i in inverse}
    finally:
        os.unlink(in_path)
        shutil.rmtree(out_dir, ignore_errors=True)

    missing = sorted(a for a in answers if a not in raw_scores)
    ordered += missing
    # keep raw scores only when they decrease along the full ranking so a
    # re-sort reproduces the order. otherwise score each answer n minus rank
    raw = [raw_scores[a] for a in ordered if a in raw_scores]
    monotone = all(raw[i] >= raw[i + 1] for i in range(len(raw) - 1))
    if not missing and monotone and len(set(raw)) > 1:
        scores = {a: float(raw_scores[a]) for a in ordered}
    else:
        scores = {a: float(n - r) for r, a in enumerate(ordered)}
    return ordered, scores


def make_sra_aggregator(method="borda"):
    """Build an aggregator usable as sra_search(aggregator=...).

    sra_search re-sorts the returned scores with a lexicographic tie break.
    """
    def _agg(voter_rankings, answers):
        _, scores = flagr_aggregate(voter_rankings, answers, method=method)
        return scores

    return _agg


def make_era_aggregator(method="borda"):
    """Build an aggregator usable as era_search(aggregator_fn=...).

    ERA voters are score vectors. Each one is collapsed to a ranking before
    aggregation and the result is max normalized like the native aggregator.
    Rank-only methods such as kemeny return positional scores so their
    entropy reflects ordering rather than margins.
    """
    def _agg(voter_scores, answers):
        if not voter_scores:
            return [], {}
        voter_rankings = {
            v: sorted(answers, key=lambda a: (-sv.get(a, 0.0), a))
            for v, sv in voter_scores.items()
        }
        ranking, scores = flagr_aggregate(voter_rankings, answers, method=method)
        max_abs = max((abs(s) for s in scores.values()), default=0.0)
        if max_abs:
            scores = {a: s / max_abs for a, s in scores.items()}
        return ranking, scores

    return _agg
