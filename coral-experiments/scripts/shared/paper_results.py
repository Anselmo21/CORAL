"""Paper-results pipeline (Phase A — no GPU).

For each of the 4 main datasets, refresh ERA(b=16) + ERA(pct_10) under the
current normalized softmax_entropy, copy BFS + SRA from existing results,
compute hop distance from BFS's CF for every method, and emit:

    data/<dataset>/paper_results.json   (per-case rows: bfs/sra/era_b16/era_pct10)
    stdout                             (per-dataset table: accuracy, cost, hop)

Datasets:
    msmarco_criteria           = MSMARCO criteria
    sonnet_criteria            = Sonnet criteria
    msmarco_pragma    = MSMARCO pragma (regen)
    sonnet_pragma               = Sonnet pragma

Usage:
    conda run -n coral python3 -m scripts.shared.paper_results
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from importlib import import_module

import numpy as np

from search.lattice import hop_distance
from search.rank import era_search

DATASETS = [
    # (data dir, module path, label, propagate_scores)
    ("msmarco_criteria",        "scripts.criteria.msmarco.run_subsample",          "MSMARCO criteria",       False),
    ("sonnet_criteria",         "scripts.criteria.sonnet.run_subsample",           "Sonnet criteria",        False),
    ("msmarco_pragma", "scripts.pragma.msmarco.run_subsample", "MSMARCO pragma (regen)", True),
    ("sonnet_pragma",            "scripts.pragma.sonnet.run_subsample",    "Sonnet pragma",          True),
]


def _load_results(path: str) -> list[dict]:
    with open(path) as f:
        r = json.load(f)
    if isinstance(r, dict) and "results" in r:
        return r["results"]
    return r


def _load_lattices(mod):
    for name in ("_load_lattices", "_load_pragma_lattices", "_load_dataset"):
        fn = getattr(mod, name, None)
        if fn:
            return fn
    raise RuntimeError(f"No loader on {mod.__name__}")


def rerun_dataset(name: str, modpath: str, propagate_scores: bool = False) -> list[dict]:
    cases_path   = f"data/{name}/cases.json"
    results_path = f"data/{name}/results.json"

    with open(cases_path) as f:
        cases = json.load(f)
    results = _load_results(results_path)
    prior_by_case = {
        (r["qid"], r["size"], tuple(sorted(r["start_node"]))): r
        for r in results
    }

    mod = import_module(modpath)
    loader = _load_lattices(mod)
    score_proxy = mod.score_proxy

    qids = sorted({c["qid"] for c in cases})
    lattices = loader(qids)

    new_rows: list[dict] = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        size = case["size"]
        start = case["start_node"]
        prior = prior_by_case.get((qid, size, tuple(sorted(start))))
        if prior is None:
            continue
        criteria, rbc = lattices[qid]

        bfs_for_proxy = {"key":   prior["bfs"]["key"],
                         "size":  prior["bfs"]["layer"],
                         "found": prior["bfs"]["found"]}

        row = {
            "case_id":   prior.get("case_id", i),
            "qid":       qid,
            "size":      size,
            "start_node": start,
            "bfs":       prior["bfs"],
            "sra":       prior["sra"],
        }

        # ERA b=16
        r16 = era_search(criteria, rbc, budget=16, start_node=start, direction="down",
                         propagate_scores=propagate_scores)
        row["era_b16"] = score_proxy(bfs_for_proxy, r16, rbc, start)

        # ERA pct_10
        budget = max(1, int(2**size * 10 / 100))
        rp = era_search(criteria, rbc, budget=budget, start_node=start, direction="down",
                        propagate_scores=propagate_scores)
        row["era_pct10"] = score_proxy(bfs_for_proxy, rp, rbc, start)

        # Hop distance from BFS's CF to each method's CF
        bfs_key = prior["bfs"].get("key")
        for col in ("sra", "era_b16", "era_pct10"):
            cell = row[col]
            method_key = cell.get("proxy_key") or cell.get("key") if isinstance(cell, dict) else None
            row[col]["hop_to_bfs"] = hop_distance(method_key, bfs_key)
        row["bfs"]["hop_to_bfs"] = 0   # by definition

        new_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1:>4}/{len(cases)}] done", flush=True)
    return new_rows


def summarize(rows: list[dict]) -> str:
    """Build a per-dataset table:  method | accuracy | avg cost | avg hop."""
    methods = [
        ("BFS (oracle)",  "bfs",       True),
        ("SRA",           "sra",       False),
        ("ERA (b=16)",    "era_b16",   False),
        ("ERA (pct_10)",  "era_pct10", False),
    ]
    lines = []
    lines.append(f"  {'method':<14} | {'N':>4} | {'acc':>7} | {'cost':>8} | {'hop μ':>7} | {'hop med':>7}")
    lines.append("  " + "-" * 60)
    for label, col, is_bfs in methods:
        cells = [r[col] for r in rows if col in r]
        n = len(cells)
        if is_bfs:
            acc = 100.0   # BFS is the oracle
        else:
            acc = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / n
        if is_bfs:
            cost = float(np.mean([c["calls"] for c in cells]))
        else:
            cost = float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
        hops = [c.get("hop_to_bfs") for c in cells]
        hops = [h for h in hops if h is not None]
        hop_mean = float(np.mean(hops)) if hops else float("nan")
        hop_med  = float(np.median(hops)) if hops else float("nan")
        lines.append(f"  {label:<14} | {n:>4} | {acc:>6.1f}% | {cost:>8.2f} | {hop_mean:>7.2f} | {hop_med:>7.1f}")
    return "\n".join(lines)


def main():
    for name, modpath, nice, propagate in DATASETS:
        flag = "  [propagate_scores=ON]" if propagate else ""
        print(f"\n{'='*70}\n  {nice}   (data/{name}/){flag}\n{'='*70}")
        rows = rerun_dataset(name, modpath, propagate_scores=propagate)
        out_path = f"data/{name}/paper_results.json"
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n  → {out_path}  ({len(rows)} rows)\n")
        print(summarize(rows))


if __name__ == "__main__":
    main()
