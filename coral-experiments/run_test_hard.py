"""Run BFS, SRA, ERA(10%), ERA(64) on data/test_hard.jsonl.

Saves a single JSON file with all four methods aligned by case index.

Run:
    conda run -n coral python run_test_hard.py
"""

import json
import os

from era_budget_ablation import (
    LATTICES_PATH,
    budget_from_pct,
    effective_size,
    lattice_size,
    load_manifest_cases,
)
from experiment import score_proxy
from search.bfs import bfs_search
from search.rank import era_search, sra_search


CASES_PATH = "data/test_hard.jsonl"
OUT_PATH = "data/era_ablation_study/test_hard/test_hard_results.json"


def main() -> None:
    cases = load_manifest_cases(LATTICES_PATH, CASES_PATH, manifest_directions=["top-down"])
    print(f"Loaded {len(cases)} test_hard top-down cases")

    results: list[dict] = []
    for i, case in enumerate(cases):
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        eff = effective_size(case)
        budget_pct10 = budget_from_pct(eff, 10)
        budget_64 = 64

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        sra_res = sra_search(criteria, rbc, start_node=start_node, direction=direction)
        era10_res = era_search(criteria, rbc,
                                budget=budget_pct10,
                                start_node=start_node, direction=direction)
        era64_res = era_search(criteria, rbc,
                                budget=budget_64,
                                start_node=start_node, direction=direction)

        sra_scored = score_proxy(bfs_res, sra_res, rbc, start_node)
        sra_scored["calls"] = sra_res["calls"]

        era10_scored = score_proxy(bfs_res, era10_res, rbc, start_node)
        era10_scored["calls"] = era10_res["calls"]
        era10_scored["budget_used"] = era10_res.get("budget_used", 0)
        era10_scored["budget_allocated"] = budget_pct10
        era10_scored["budget_pct"] = 10

        era64_scored = score_proxy(bfs_res, era64_res, rbc, start_node)
        era64_scored["calls"] = era64_res["calls"]
        era64_scored["budget_used"] = era64_res.get("budget_used", 0)
        era64_scored["budget_allocated"] = budget_64

        results.append({
            "query_id": case["query_id"],
            "effective_size": eff,
            "lattice_size": lattice_size(eff),
            "direction": direction,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "sra": sra_scored,
            "pct_10": era10_scored,
            "budget_64": era64_scored,
        })

        if (i + 1) % 25 == 0:
            print(f"  done {i+1}/{len(cases)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
