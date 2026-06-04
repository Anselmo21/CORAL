"""Run ERA with weighted_entropy at budget=64 on test top-down cases."""

import json
import os

from era_budget_ablation import (
    LATTICES_PATH,
    TEST_CASES_PATH,
    effective_size,
    lattice_size,
    load_manifest_cases,
)
from experiment import score_proxy
from search.bfs import bfs_search
from search.heuristics import weighted_entropy
from search.rank import era_search


OUT_PATH = "data/era_ablation_study/test/top_down_weighted_64/era_weighted_64_results.json"


def main() -> None:
    cases = load_manifest_cases(LATTICES_PATH, TEST_CASES_PATH, manifest_directions=["top-down"])
    print(f"Loaded {len(cases)} test top-down cases")

    results: list[dict] = []
    for case in cases:
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        eff = effective_size(case)
        budget = 64

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        era_res = era_search(
            criteria, rbc,
            budget=budget,
            entropy_fn=weighted_entropy,
            start_node=start_node,
            direction=direction,
        )
        scored = score_proxy(bfs_res, era_res, rbc, start_node)
        scored["calls"] = era_res["calls"]
        scored["budget_used"] = era_res.get("budget_used", 0)
        scored["budget_allocated"] = budget

        results.append({
            "query_id": case["query_id"],
            "effective_size": eff,
            "lattice_size": lattice_size(eff),
            "direction": direction,
            "budget_mode": "absolute",
            "entropy": "weighted",
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "budget_64": scored,
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
