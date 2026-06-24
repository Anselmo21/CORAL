"""Compare ERA accuracy + cost: existing results.json (cached, pre-normalization)
vs a fresh ERA run with the new max-normalized softmax_entropy.

For each of the four main datasets, re-runs ERA only at pct_10 and fixed_16
on the same cases, reusing BFS / SRA from results.json.

Usage:
    conda run -n coral python3 -m scripts.shared.eval_norm_compare
"""
from __future__ import annotations

import json
from importlib import import_module

import numpy as np

DATASETS = [
    ("msmarco_subsample", "scripts.msmarco.run_subsample"),
    ("sonnet_subsample",  "scripts.sonnet.run_subsample"),
    ("msmarco_pragma",    "scripts.pragma.msmarco.run_subsample"),
    ("sonnet_pragma",     "scripts.pragma.sonnet.run_subsample"),
]


def acc(rows, label):
    cells = [r["era"].get(label) for r in rows]
    cells = [c for c in cells if c is not None]
    if not cells:
        return float("nan"), 0
    correct = sum(1 for c in cells if c["outcome"] == "correct")
    return 100 * correct / len(cells), len(cells)


def cost(rows, label):
    cells = [r["era"].get(label) for r in rows]
    cells = [c for c in cells if c is not None]
    if not cells:
        return float("nan")
    return float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))


def rerun_era(cases, results, mod):
    from search.rank import era_search

    _load_lattices = getattr(mod, "_load_lattices", None) or mod._load_pragma_lattices
    score_proxy    = mod.score_proxy

    prior_by_case = {
        (r["qid"], r["size"], tuple(sorted(r["start_node"]))): r
        for r in results
    }

    qids = sorted({c["qid"] for c in cases})
    lattices = _load_lattices(qids)

    new_rows = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        size = case["size"]
        start = case["start_node"]
        key = (qid, size, tuple(sorted(start)))
        prior = prior_by_case.get(key)
        if prior is None:
            continue
        criteria, rbc = lattices[qid]

        bfs_for_proxy = {"key":   prior["bfs"]["key"],
                         "size":  prior["bfs"]["layer"],
                         "found": prior["bfs"]["found"]}

        row = {"size": size, "era": {}}

        b = 16
        res = era_search(criteria, rbc, budget=b, start_node=start, direction="down")
        row["era"][f"era_fixed_{b}"] = score_proxy(bfs_for_proxy, res, rbc, start)

        budget = max(1, int(2**size * 10 / 100))
        res = era_search(criteria, rbc, budget=budget, start_node=start, direction="down")
        row["era"]["era_pct_10"] = score_proxy(bfs_for_proxy, res, rbc, start)

        new_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1:>4}/{len(cases)}] done", flush=True)
    return new_rows


def main():
    print(f"\n{'='*88}")
    print(f"  ERA accuracy + cost: pre-normalization (cached) vs post-normalization (re-run)")
    print(f"{'='*88}\n")

    for name, modpath in DATASETS:
        print(f"\n[{name}]")
        cases_path = f"data/{name}/cases.json"
        results_path = f"data/{name}/results.json"

        with open(cases_path) as f:
            cases = json.load(f)
        with open(results_path) as f:
            results = json.load(f)
        if isinstance(results, dict) and "results" in results:
            results = results["results"]

        mod = import_module(modpath)

        print(f"  Re-running ERA(pct_10, b=16) on {len(cases)} cases with normalized softmax ...")
        new_rows = rerun_era(cases, results, mod)
        print(f"  Done. {len(new_rows)} rows.")

        print(f"\n  Accuracy:")
        print(f"    {'config':<14} | {'raw':>10} | {'norm':>10} | {'Δ':>6}")
        for label in ["era_pct_10", "era_fixed_16"]:
            a_raw, n_raw = acc(results, label)
            a_norm, n_norm = acc(new_rows, label)
            print(f"    {label:<14} | {a_raw:>9.1f}% | {a_norm:>9.1f}% | {a_norm-a_raw:>+5.1f}")

        print(f"\n  Avg cost (calls + budget_used):")
        print(f"    {'config':<14} | {'raw':>10} | {'norm':>10} | {'Δ':>6}")
        for label in ["era_pct_10", "era_fixed_16"]:
            c_raw  = cost(results, label)
            c_norm = cost(new_rows, label)
            print(f"    {label:<14} | {c_raw:>10.2f} | {c_norm:>10.2f} | {c_norm-c_raw:>+6.2f}")


if __name__ == "__main__":
    main()
