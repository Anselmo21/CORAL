"""Tests for the pluggable aggregator API on ERA and MRA.

Run from repo root:
    conda run -n coral python3 -m pytest tests/test_rank_aggregators.py -q
"""
from __future__ import annotations

import math

import pytest

from search.rank import (
    _borda_scores,
    _combmnz_scores,
    _combsum_scores,
    _SCORE_AGGREGATORS,
    aggregator_default,
    era_search,
    make_aggregator,
    mra_search,
)


# ─── Score-vector combiner unit tests ──────────────────────────────────────

def test_combsum_scores_sums_entrywise():
    voters = {
        "v1": {"X": 3.0, "Y": 1.0, "Z": 0.0},
        "v2": {"X": 2.0, "Y": 2.0, "Z": 1.0},
    }
    assert _combsum_scores(voters, ["X", "Y", "Z"]) == {"X": 5.0, "Y": 3.0, "Z": 1.0}


def test_combsum_missing_answer_defaults_to_zero():
    voters = {"v1": {"X": 3.0}, "v2": {"Y": 2.0}}
    assert _combsum_scores(voters, ["X", "Y", "Z"]) == {"X": 3.0, "Y": 2.0, "Z": 0.0}


def test_combmnz_multiplies_by_positive_voter_count():
    voters = {
        "v1": {"X": 3.0, "Y": 0.0, "Z": 0.0},
        "v2": {"X": 2.0, "Y": 4.0, "Z": 0.0},
        "v3": {"X": 1.0, "Y": 1.0, "Z": 2.0},
    }
    # X: sum=6, count=3 -> 18;  Y: sum=5, count=2 -> 10;  Z: sum=2, count=1 -> 2
    assert _combmnz_scores(voters, ["X", "Y", "Z"]) == {"X": 18.0, "Y": 10.0, "Z": 2.0}


def test_borda_scores_equals_combsum():
    voters = {"v1": {"X": 3.0, "Y": 1.0}, "v2": {"X": 2.0, "Y": 4.0}}
    answers = ["X", "Y"]
    assert _borda_scores(voters, answers) == _combsum_scores(voters, answers)


def test_score_aggregator_registry_keys():
    assert set(_SCORE_AGGREGATORS) == {"combsum", "combmnz", "borda"}


# ─── make_aggregator factory tests ─────────────────────────────────────────

def test_make_aggregator_default_max_normalizes_per_voter_and_post_sum():
    """Default = ERA behavior. Per-voter max-norm then sum then max-norm."""
    agg = make_aggregator()
    voters = {
        "v1": {"X": 3.0, "Y": 1.5, "Z": 0.0},  # per-voter norm -> {1.0, 0.5, 0.0}
        "v2": {"X": 2.0, "Y": 2.0, "Z": 1.0},  # per-voter norm -> {1.0, 1.0, 0.5}
    }
    # sum: {2.0, 1.5, 0.5}; post-max-norm: {1.0, 0.75, 0.25}
    ranking, scores = agg(voters, ["X", "Y", "Z"])
    assert ranking == ["X", "Y", "Z"]
    assert math.isclose(scores["X"], 1.0)
    assert math.isclose(scores["Y"], 0.75)
    assert math.isclose(scores["Z"], 0.25)


def test_make_aggregator_empty_voters_returns_empty():
    assert make_aggregator()({}, ["X", "Y"]) == ([], {})


def test_make_aggregator_none_normalization_matches_raw_sum():
    """MRA default: no per-voter norm, no post-sum norm."""
    agg = make_aggregator(method="combsum", voter_norm_fn=None, norm_fn=None)
    voters = {"v1": {"X": 3.0, "Y": 1.0}, "v2": {"X": 2.0, "Y": 4.0}}
    ranking, scores = agg(voters, ["X", "Y"])
    assert scores == {"X": 5.0, "Y": 5.0}
    # tie broken alphabetically by (-score, key)
    assert ranking == ["X", "Y"]


def test_make_aggregator_combmnz_differs_from_combsum():
    voters = {
        "v1": {"X": 5.0, "Y": 0.0},
        "v2": {"X": 5.0, "Y": 0.0},
        "v3": {"X": 5.0, "Y": 0.0},
    }
    csum = make_aggregator(method="combsum", voter_norm_fn=None, norm_fn=None)
    cmnz = make_aggregator(method="combmnz", voter_norm_fn=None, norm_fn=None)
    _, s_csum = csum(voters, ["X", "Y"])
    _, s_cmnz = cmnz(voters, ["X", "Y"])
    assert s_csum == {"X": 15.0, "Y": 0.0}
    assert s_cmnz == {"X": 45.0, "Y": 0.0}  # 15 * 3


def test_make_aggregator_borda_equals_combsum():
    voters = {"v1": {"X": 3.0, "Y": 1.0}, "v2": {"X": 2.0, "Y": 4.0}}
    borda = make_aggregator(method="borda", voter_norm_fn=None, norm_fn=None)
    csum  = make_aggregator(method="combsum", voter_norm_fn=None, norm_fn=None)
    assert borda(voters, ["X", "Y"]) == csum(voters, ["X", "Y"])


def test_make_aggregator_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown aggregator"):
        make_aggregator(method="bogus")


def test_make_aggregator_custom_callable_combiner():
    """Custom combiner: entry-wise min instead of sum."""
    def _min_combiner(voter_scores, answers):
        return {a: min(sv.get(a, 0.0) for sv in voter_scores.values()) for a in answers}

    agg = make_aggregator(method=_min_combiner, voter_norm_fn=None, norm_fn=None)
    voters = {"v1": {"X": 3.0, "Y": 1.0}, "v2": {"X": 2.0, "Y": 4.0}}
    ranking, scores = agg(voters, ["X", "Y"])
    assert scores == {"X": 2.0, "Y": 1.0}
    assert ranking == ["X", "Y"]


def test_aggregator_default_output_stable():
    """Regression: fixed input → fixed output. Locks the default ERA pipeline."""
    voters = {
        "v1": {"X": 3.0, "Y": 1.5, "Z": 0.0},
        "v2": {"X": 2.0, "Y": 2.0, "Z": 1.0},
    }
    ranking, scores = aggregator_default(voters, ["X", "Y", "Z"])
    assert ranking == ["X", "Y", "Z"]
    assert math.isclose(scores["X"], 1.0)
    assert math.isclose(scores["Y"], 0.75)
    assert math.isclose(scores["Z"], 0.25)


# ─── Crafted lattice used by ERA / MRA integration tests ───────────────────

def _crafted_lattice():
    """3-criterion lattice where a counterfactual sits at pair 'a, b'.

    Baseline top-1 = 'X'. Only 'a, b' flips to 'Y'.
    """
    return {
        "criteria": ["a", "b", "c"],
        "rankings_by_combo": {
            "":         ["X", "Y", "Z"],
            "a":        ["X", "Z", "Y"],
            "b":        ["X", "Y", "Z"],
            "c":        ["X", "Y", "Z"],
            "a, b":     ["Y", "X", "Z"],   # CF here
            "a, c":     ["X", "Y", "Z"],
            "b, c":     ["X", "Y", "Z"],
            "a, b, c":  ["X", "Y", "Z"],   # baseline
        },
    }


_ERA_KEYS = {"found", "key", "size", "ranking", "calls", "budget_used", "pruned_nodes"}


# ─── era_search integration tests ──────────────────────────────────────────

def test_era_default_finds_cf():
    lat = _crafted_lattice()
    res = era_search(lat["criteria"], lat["rankings_by_combo"], budget=8)
    assert set(res) == _ERA_KEYS
    assert res["found"] is True
    assert res["key"] == "a, b"


def test_era_string_combsum_equals_default():
    """Passing 'combsum' string must reproduce the None-default result exactly."""
    lat = _crafted_lattice()
    default = era_search(lat["criteria"], lat["rankings_by_combo"], budget=8)
    string  = era_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="combsum",
    )
    assert default == string


def test_era_string_borda_equals_combsum():
    lat = _crafted_lattice()
    csum = era_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="combsum",
    )
    bord = era_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="borda",
    )
    assert csum == bord


def test_era_combmnz_returns_well_formed_result():
    lat = _crafted_lattice()
    res = era_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="combmnz",
    )
    assert set(res) == _ERA_KEYS
    # Any CF found must be a real one (baseline flip); non-found is also acceptable.
    if res["found"]:
        assert lat["rankings_by_combo"][res["key"]][0] != "X"


def test_era_invalid_aggregator_string_raises():
    lat = _crafted_lattice()
    with pytest.raises(ValueError, match="unknown aggregator"):
        era_search(
            lat["criteria"], lat["rankings_by_combo"], budget=4,
            aggregator_fn="not_a_real_method",
        )


def test_era_wrapped_callable_still_accepted():
    """Back-compat: passing a fully-wrapped aggregator callable still works."""
    lat = _crafted_lattice()
    default = era_search(lat["criteria"], lat["rankings_by_combo"], budget=8)
    custom  = era_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8,
        aggregator_fn=make_aggregator(method="combsum"),
    )
    assert default == custom


# ─── mra_search integration tests ──────────────────────────────────────────

def test_mra_default_returns_well_formed_result():
    lat = _crafted_lattice()
    res = mra_search(lat["criteria"], lat["rankings_by_combo"], budget=8)
    assert set(res) == _ERA_KEYS
    assert res["found"] is True
    assert res["key"] == "a, b"


def test_mra_string_combsum_equals_default():
    lat = _crafted_lattice()
    default = mra_search(lat["criteria"], lat["rankings_by_combo"], budget=8)
    string  = mra_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="combsum",
    )
    assert default == string


def test_mra_string_borda_equals_combsum():
    lat = _crafted_lattice()
    csum = mra_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="combsum",
    )
    bord = mra_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="borda",
    )
    assert csum == bord


def test_mra_combmnz_returns_well_formed_result():
    lat = _crafted_lattice()
    res = mra_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8, aggregator_fn="combmnz",
    )
    assert set(res) == _ERA_KEYS
    if res["found"]:
        assert lat["rankings_by_combo"][res["key"]][0] != "X"


def test_mra_invalid_aggregator_string_raises():
    lat = _crafted_lattice()
    with pytest.raises(ValueError, match="unknown aggregator"):
        mra_search(
            lat["criteria"], lat["rankings_by_combo"], budget=4,
            aggregator_fn="not_real",
        )


def test_mra_wrapped_callable_still_accepted():
    lat = _crafted_lattice()
    default = mra_search(lat["criteria"], lat["rankings_by_combo"], budget=8)
    custom  = mra_search(
        lat["criteria"], lat["rankings_by_combo"], budget=8,
        aggregator_fn=make_aggregator(
            method="combsum", voter_norm_fn=None, norm_fn=None,
        ),
    )
    assert default == custom
