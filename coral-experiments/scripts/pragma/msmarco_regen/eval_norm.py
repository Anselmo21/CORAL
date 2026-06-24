"""Rerun ERA on the 16-lattice case set with scale-normalized softmax_entropy.

Same as eval_T3.py but at T=1 (default) — the change is the new max-normalization
inside softmax_entropy itself, so this measures the effect of normalization
independently of any temperature change.

Output: data/16_lattices_cf_layer2to8/results_norm.json
"""
from __future__ import annotations

import json
import os

from scripts.pragma.msmarco_regen.run_subsample import (
    ERA_BUDGETS_FIXED,
    ERA_BUDGETS_PCT,
    _load_lattices,
    score_proxy,
)
from scripts.pragma.msmarco_regen.run_16_lattices import _load_qids
from search.rank import era_search

CASES_PATH = "data/16_lattices_cf_layer2to8/cases.json"
T1_RESULTS = "data/16_lattices_cf_layer2to8/results.json"
OUT_PATH   = "data/16_lattices_cf_layer2to8/results_norm.json"


def main() -> None:
    with open(CASES_PATH) as f:
        cases = json.load(f)
    with open(T1_RESULTS) as f:
        t1 = json.load(f)
    t1_by_case = {(r["qid"], r["size"], tuple(sorted(r["start_node"]))): r
                  for r in t1}

    qids = _load_qids()
    lattices = _load_lattices(qids)

    print(f"Re-running ERA with normalized softmax (T=1) on {len(cases)} cases ...")

    results: list[dict] = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        criteria, rbc = lattices[qid]
        start_node = case["start_node"]
        size = case["size"]
        key = (qid, size, tuple(sorted(start_node)))

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
            res = era_search(criteria, rbc, budget=b,
                             start_node=start_node, direction="down")
            row["era"][f"era_fixed_{b}"] = score_proxy(bfs_for_proxy, res, rbc, start_node)

        for p in ERA_BUDGETS_PCT:
            budget = max(1, int(2**size * p / 100))
            res = era_search(criteria, rbc, budget=budget,
                             start_node=start_node, direction="down")
            row["era"][f"era_pct_{p}"] = score_proxy(bfs_for_proxy, res, rbc, start_node)

        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>3}/{len(cases)}] done", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {OUT_PATH}")

    # Quick aggregate compare
    print("\nHeadline accuracy (T1 raw vs T1 normalized):")
    for label in ["era_pct_10", "era_fixed_16"]:
        n = len(results)
        t1_corr = sum(1 for r in t1 if r["era"].get(label, {}).get("outcome") == "correct")
        nm_corr = sum(1 for r in results if r["era"].get(label, {}).get("outcome") == "correct")
        print(f"  {label:<15} raw={t1_corr:>3}/{n} ({100*t1_corr/n:.1f}%)   "
              f"norm={nm_corr:>3}/{n} ({100*nm_corr/n:.1f}%)   "
              f"Δ = {100*(nm_corr-t1_corr)/n:+.1f}pp")


if __name__ == "__main__":
    main()
