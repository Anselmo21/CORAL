"""Run ERA(budget=32) on test_hard cases and merge into existing results file."""

import json

from era_budget_ablation import LATTICES_PATH, load_manifest_cases
from experiment import score_proxy
from search.bfs import bfs_search
from search.rank import era_search


CASES_PATH = "data/test_hard.jsonl"
RESULTS_PATH = "data/era_ablation_study/test_hard/test_hard_results.json"


def main() -> None:
    cases = load_manifest_cases(LATTICES_PATH, CASES_PATH, manifest_directions=["top-down"])
    print(f"Loaded {len(cases)} test_hard top-down cases")

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    assert len(results) == len(cases), (len(results), len(cases))

    for i, case in enumerate(cases):
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        era_res = era_search(criteria, rbc,
                              budget=32,
                              start_node=start_node, direction=direction)

        scored = score_proxy(bfs_res, era_res, rbc, start_node)
        scored["calls"] = era_res["calls"]
        scored["budget_used"] = era_res.get("budget_used", 0)
        scored["budget_allocated"] = 32

        results[i]["budget_32"] = scored

        if (i + 1) % 25 == 0:
            print(f"  done {i+1}/{len(cases)}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Merged budget_32 into {RESULTS_PATH}")


if __name__ == "__main__":
    main()
