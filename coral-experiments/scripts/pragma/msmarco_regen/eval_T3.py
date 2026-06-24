"""Rerun ERA on the 16-lattice case set with softmax T=3 instead of T=1.

Uses the existing cases.json from data/16_lattices_cf_layer2to8/. BFS and SRA
are temperature-independent so we don't re-run them; we just copy them from
results.json. Only the era_* columns get recomputed.

Output: data/16_lattices_cf_layer2to8/results_T3.json
"""
from __future__ import annotations

import json
import os
from functools import partial

from scripts.pragma.msmarco_regen.run_subsample import (
    ERA_BUDGETS_FIXED,
    ERA_BUDGETS_PCT,
    _load_lattices,
    score_proxy,
)
from scripts.pragma.msmarco_regen.run_16_lattices import _load_qids
from search.heuristics import softmax_entropy
from search.rank import era_search

T_VALUE: float = 3.0
CASES_PATH    = "data/16_lattices_cf_layer2to8/cases.json"
T1_RESULTS    = "data/16_lattices_cf_layer2to8/results.json"
OUT_PATH      = "data/16_lattices_cf_layer2to8/results_T3.json"


def main() -> None:
    with open(CASES_PATH) as f:
        cases = json.load(f)
    with open(T1_RESULTS) as f:
        t1 = json.load(f)
    t1_by_case = {(r["qid"], r["size"], tuple(sorted(r["start_node"]))): r
                  for r in t1}

    qids = _load_qids()
    lattices = _load_lattices(qids)

    entropy_fn = partial(softmax_entropy, T=T_VALUE)
    print(f"Re-running ERA with softmax T={T_VALUE} on {len(cases)} cases ...")

    results: list[dict] = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        criteria, rbc = lattices[qid]
        start_node = case["start_node"]
        size = case["size"]
        key = (qid, size, tuple(sorted(start_node)))

        # Reuse BFS + SRA from T=1 results
        prior = t1_by_case[key]
        row: dict = {
            "case_id":    i,
            "qid":        qid,
            "size":       size,
            "start_node": start_node,
            "bfs":        prior["bfs"],
            "sra":        prior["sra"],
            "era":        {},
        }
        bfs_for_proxy = {"key":   prior["bfs"]["key"],
                         "size":  prior["bfs"]["layer"],
                         "found": prior["bfs"]["found"]}

        for b in ERA_BUDGETS_FIXED:
            res = era_search(criteria, rbc, budget=b, entropy_fn=entropy_fn,
                             start_node=start_node, direction="down")
            row["era"][f"era_fixed_{b}"] = score_proxy(bfs_for_proxy, res, rbc, start_node)

        for p in ERA_BUDGETS_PCT:
            budget = max(1, int(2**size * p / 100))
            res = era_search(criteria, rbc, budget=budget, entropy_fn=entropy_fn,
                             start_node=start_node, direction="down")
            row["era"][f"era_pct_{p}"] = score_proxy(bfs_for_proxy, res, rbc, start_node)

        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>3}/{len(cases)}] done", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {OUT_PATH}")


if __name__ == "__main__":
    main()
