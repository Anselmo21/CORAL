"""BFS / SRA / ERA(budget=16) evaluation on data/msmarco_llama_size10.json.

All 16 queries are size-10 top-down cases starting at the full criteria node.
Writes data/msmarco_eval/results.json.
CLI: conda run -n coral python3 run_msmarco_eval.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, ".")
from experiment import score_proxy
from search.bfs import bfs_search
from search.rank import era_search, sra_search

DATASET_PATH = "data/msmarco_llama_size10.json"
OUT_PATH = "data/msmarco_eval/results.json"
ERA_BUDGET = 16


def run(dataset_path: str = DATASET_PATH, out_path: str = OUT_PATH) -> list[dict]:
    with open(dataset_path) as f:
        data = json.load(f)

    results: list[dict] = []
    for qid, rec in data["lattices"].items():
        criteria = rec["criteria"]
        rankings = rec["rankings"]
        start_node = criteria  # full top-down

        bfs_res = bfs_search(criteria, rankings, start_node=start_node, direction="down")
        sra_res = sra_search(criteria, rankings, start_node=start_node, direction="down")
        era_res = era_search(
            criteria, rankings, budget=ERA_BUDGET,
            start_node=start_node, direction="down",
        )

        sra_scored = score_proxy(bfs_res, sra_res, rankings, start_node)
        sra_scored["calls"] = sra_res["calls"]

        era_scored = score_proxy(bfs_res, era_res, rankings, start_node)
        era_scored["calls"] = era_res["calls"]
        era_scored["budget_used"] = era_res.get("budget_used", 0)

        results.append({
            "query_id": rec["query_id"],
            "bfs": {"outcome": "correct", "calls": bfs_res["calls"], "key": bfs_res["key"], "size": bfs_res["size"]},
            "sra": sra_scored,
            "era": era_scored,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"era_budget": ERA_BUDGET, "results": results}, f, indent=2)
    print(f"Wrote {len(results)} results to {out_path}")
    return results


if __name__ == "__main__":
    run()
