"""Shared sub-lattice case sampling for criteria and pragma datasets.

For each effective size k in SIZES, enumerates every (qid, k-subset) across
the supplied lattices, shuffles deterministically, and accepts the first
TARGET_PER_SIZE candidates whose BFS top-down search finds a counterfactual
that requires removing at least MIN_REMOVED items from the start_node (i.e.
the CF layer is at most k - MIN_REMOVED). Then appends every full k=10
lattice as a size-10 case (one per qid).

This is the single source of truth shared by all four datasets so the
criteria and pragma pipelines stay symmetric.
"""
from __future__ import annotations

import random
from itertools import combinations

from search.bfs import bfs_search

SIZES: list[int] = list(range(3, 10))   # 3..9 sub-lattice sampling
TARGET_PER_SIZE: int = 20
SEED: int = 42
MIN_REMOVED: int = 2                    # BFS CF must require removing >= MIN_REMOVED items
FULL_SIZE: int = 10                     # k for the appended full-lattice cases


def _qualifies(
    rbc: dict[str, list[str]],
    criteria: list[str],
    start_node: list[str],
    min_removed: int = MIN_REMOVED,
) -> bool:
    res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
    if not res["found"]:
        return False
    return len(start_node) - res["size"] >= min_removed


def sample_cases(
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
    seed: int = SEED,
    sizes: list[int] = SIZES,
    target_per_size: int = TARGET_PER_SIZE,
    full_size: int = FULL_SIZE,
    min_removed: int = MIN_REMOVED,
) -> list[dict]:
    """Sample sub-lattice cases per the unified recipe (see module docstring)."""
    rng = random.Random(seed)
    cases: list[dict] = []
    for size in sizes:
        candidates: list[dict] = []
        for qid, (criteria, _) in lattices.items():
            for combo in combinations(criteria, size):
                candidates.append({"qid": qid, "size": size, "start_node": list(combo)})
        rng.shuffle(candidates)
        accepted: list[dict] = []
        for cand in candidates:
            criteria, rbc = lattices[cand["qid"]]
            if _qualifies(rbc, criteria, cand["start_node"], min_removed=min_removed):
                accepted.append(cand)
                if len(accepted) >= target_per_size:
                    break
        print(f"  size={size}: pool={len(candidates):>6}  accepted={len(accepted):>3}/{target_per_size}")
        cases.extend(accepted)

    for qid, (criteria, _) in sorted(lattices.items()):
        cases.append({"qid": qid, "size": full_size, "start_node": criteria})
    print(f"  size={full_size}: added {len(lattices)} full lattices")

    return cases
