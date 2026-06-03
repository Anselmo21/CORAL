"""Mock-based unit tests for search.contextcite.contextcite_search_entropy.

Mocks ContextCiter + the Lasso solver so we exercise only the entropy/pruning/
search-loop logic without needing a GPU or HuggingFace model.

Run:
    conda run -n coral python test_contextcite_entropy.py
"""

import numpy as np

import search.contextcite as cc_mod
from search.contextcite import contextcite_search_entropy
from search.lattice import combo_key


class FakeSolver:
    def __init__(self, logit_by_mask: dict[tuple[int, ...], float]):
        self.logit_by_mask = logit_by_mask

    def predict(self, masks: np.ndarray) -> np.ndarray:
        # masks: shape (n_samples, n_criteria) of 0/1 floats
        out = []
        for row in masks:
            key = tuple(int(v) for v in row)
            out.append(self.logit_by_mask.get(key, 0.0))
        return np.array(out, dtype=np.float32)


class FakeContextCiter:
    """Stand-in for context_cite.ContextCiter. Configured per-test via class state."""

    next_solver: FakeSolver | None = None
    next_attr: list[float] | None = None
    next_response: str = "<mock-response>"

    def __init__(self, model, tokenizer, **kwargs):
        self._kwargs = kwargs

    def get_attributions(self, as_dataframe: bool = False, verbose: bool = False):
        assert FakeContextCiter.next_attr is not None, "configure next_attr"
        return list(FakeContextCiter.next_attr)

    @property
    def last_fitted_solver(self) -> FakeSolver:
        assert FakeContextCiter.next_solver is not None, "configure next_solver"
        return FakeContextCiter.next_solver

    @property
    def response(self) -> str:
        return FakeContextCiter.next_response


class FakePartitioner:
    def __init__(self, question, concepts, sources_block, answers):
        self.question = question
        self.concepts = concepts
        self.context = sources_block
        self.answers = answers


cc_mod.ContextCiter = FakeContextCiter
cc_mod.ConceptCriteriaPartitioner = FakePartitioner


def _mask(criteria: list[str], subset: list[str]) -> tuple[int, ...]:
    idx = {c: i for i, c in enumerate(criteria)}
    bits = [0] * len(criteria)
    for c in subset:
        bits[idx[c]] = 1
    return tuple(bits)


def setup_fake(criteria: list[str], logits: dict[str, float], attr: list[float] | None = None) -> None:
    """Configure the fake solver: subset_key (e.g. 'a, b') -> logit-prob."""
    logit_by_mask: dict[tuple[int, ...], float] = {}
    # enumerate all 2^n subsets so any predict() call resolves
    n = len(criteria)
    for bits in range(2 ** n):
        subset = [criteria[i] for i in range(n) if (bits >> i) & 1]
        key = combo_key(subset)
        logit_by_mask[_mask(criteria, subset)] = logits.get(key, 0.0)
    FakeContextCiter.next_solver = FakeSolver(logit_by_mask)
    FakeContextCiter.next_attr = attr if attr is not None else [0.0] * n


# ─────────────────────────────── Test cases ────────────────────────────────


def test_finds_cf_when_present() -> None:
    criteria = ["a", "b", "c"]
    # Baseline answer at the full set is "X". One CF exists at {a, b} returning "Y".
    rbc = {
        "a, b, c": ["X", "Y", "Z"],
        "a, b":    ["Y", "X", "Z"],   # CF
        "a, c":    ["X", "Y", "Z"],
        "b, c":    ["X", "Y", "Z"],
        "a":       ["X", "Y", "Z"],
        "b":       ["X", "Y", "Z"],
        "c":       ["X", "Y", "Z"],
    }
    # Current pruning keeps LOW-entropy nodes (surrogate-confident) and drops
    # HIGH-entropy ones. To find the CF at {a,b}, the surrogate must be confident
    # there (logit far from 0 → p near 0 or 1 → low entropy). Other layer-2 nodes
    # are placed at logit=0 (peak entropy) so they get pruned.
    logits = {
        "a, b": -5.0,        # confident: p(baseline)≈0 → low entropy → SURVIVES
        "a, c":  0.0,        # peak entropy → pruned
        "b, c":  0.0,
        "a":     5.0, "b": 5.0, "c": 5.0,
    }
    setup_fake(criteria, logits)

    result = contextcite_search_entropy(
        criteria=criteria,
        rankings_by_combo=rbc,
        query="q",
        sources="src",
        answers=["X", "Y", "Z"],
        model=None,
        tokenizer=None,
        num_ablations=4,
        start_node=criteria,
        direction="down",
    )
    assert result["found"] is True, result
    assert result["key"] == "a, b", result
    assert result["ranking"][0] == "Y"
    assert result["calls"] >= 1
    print(f"  PASS test_finds_cf_when_present: key={result['key']}, calls={result['calls']}, pruned={result['pruned_nodes']}")


def test_returns_not_found_when_no_cf() -> None:
    criteria = ["a", "b", "c"]
    # No subset has a different top-1. Search should exhaust without finding.
    rbc = {combo_key(s): ["X", "Y", "Z"] for s in [
        ["a", "b", "c"], ["a", "b"], ["a", "c"], ["b", "c"], ["a"], ["b"], ["c"], [],
    ]}
    setup_fake(criteria, logits={})

    result = contextcite_search_entropy(
        criteria=criteria,
        rankings_by_combo=rbc,
        query="q", sources="src", answers=["X", "Y", "Z"],
        model=None, tokenizer=None, num_ablations=4,
        start_node=criteria, direction="down",
    )
    assert result["found"] is False, result
    assert result["key"] is None
    # All visited nodes should be counted in calls + pruned_nodes
    total_seen = result["calls"] + result["pruned_nodes"]
    # Lattice has 7 strict subsets going down from full set
    assert total_seen <= 7, (total_seen, result)
    print(f"  PASS test_returns_not_found_when_no_cf: calls={result['calls']}, pruned={result['pruned_nodes']}")


def test_pruning_skips_high_entropy_nodes() -> None:
    """When the max-jump cutoff is clean, the high-entropy nodes are pruned."""
    criteria = ["a", "b", "c"]
    # CF at {a, c}, but also make {a, b} look like a CF with the highest entropy
    # (logit=0 → p=0.5 → max entropy). The cutoff should put {a, b} in the
    # 'high entropy' bucket, removing it from the search list.
    rbc = {
        "a, b, c": ["X", "Y", "Z"],
        "a, b":    ["Y", "X", "Z"],   # is a CF — but should get pruned
        "a, c":    ["Y", "X", "Z"],   # also a CF; should be found
        "b, c":    ["X", "Y", "Z"],
        "a":       ["X", "Y", "Z"], "b": ["X", "Y", "Z"], "c": ["X", "Y", "Z"],
    }
    logits = {
        "a, b":  0.0,   # entropy ≈ 1.0 (peak)
        "a, c":  5.0,   # entropy ≈ 0.04
        "b, c":  5.0,
        "a": 5.0, "b": 5.0, "c": 5.0,
    }
    setup_fake(criteria, logits)

    result = contextcite_search_entropy(
        criteria=criteria, rankings_by_combo=rbc,
        query="q", sources="src", answers=["X", "Y", "Z"],
        model=None, tokenizer=None, num_ablations=4,
        start_node=criteria, direction="down",
    )
    assert result["found"] is True
    # Should find a, c (the unpruned CF), not a, b (pruned)
    assert result["key"] == "a, c", f"expected a, c (pruning skipped a, b), got {result['key']}"
    assert result["pruned_nodes"] >= 1, result
    print(f"  PASS test_pruning_skips_high_entropy_nodes: key={result['key']}, pruned={result['pruned_nodes']}")


def test_upward_direction() -> None:
    """Bottom-up search from {a} should find a CF at a superset."""
    criteria = ["a", "b", "c"]
    rbc = {
        "a":       ["X", "Y", "Z"],   # baseline (start node)
        "a, b":    ["X", "Y", "Z"],
        "a, c":    ["Y", "X", "Z"],   # CF
        "a, b, c": ["X", "Y", "Z"],
    }
    # CF at {a,c}: surrogate confident (low entropy) → survives pruning.
    # Sibling {a,b} at logit=0 (peak entropy) → pruned.
    setup_fake(criteria, logits={"a, b": 0.0, "a, c": -5.0, "a, b, c": 5.0})

    result = contextcite_search_entropy(
        criteria=criteria, rankings_by_combo=rbc,
        query="q", sources="src", answers=["X", "Y", "Z"],
        model=None, tokenizer=None, num_ablations=4,
        start_node=["a"], direction="up",
    )
    assert result["found"] is True
    assert result["key"] == "a, c", result
    print(f"  PASS test_upward_direction: key={result['key']}")


if __name__ == "__main__":
    print("Running contextcite_search_entropy mock tests…")
    test_finds_cf_when_present()
    test_returns_not_found_when_no_cf()
    test_pruning_skips_high_entropy_nodes()
    test_upward_direction()
    print("All tests passed.")
