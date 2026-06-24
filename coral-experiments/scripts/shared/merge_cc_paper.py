"""Merge CC (b=16, pct_10) results into paper_results.json for each dataset.

After CC has been run on a GPU machine, copy the cc_results_*.json files back to
data/<dataset>/ and then run this script to update paper_results.json with
cc_b16 / cc_pct10 columns, recompute the per-dataset table, and print the final
6-method summary.

If a dataset is missing one or both CC files, the merge silently omits that
column for that dataset (table prints "—").

Usage:
    conda run -n coral python3 -m scripts.shared.merge_cc_paper
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from search.lattice import hop_distance

DATASETS = [
    ("msmarco_subsample",        "MSMARCO criteria"),
    ("sonnet_subsample",         "Sonnet criteria"),
    ("16_lattices_cf_layer2to8", "MSMARCO pragma (regen)"),
    ("sonnet_pragma",            "Sonnet pragma"),
]


def _load_paper(name: str) -> list[dict]:
    with open(f"data/{name}/paper_results.json") as f:
        return json.load(f)


def _load_cc(name: str, config_label: str) -> dict[int, dict] | None:
    path = f"data/{name}/cc_results_{config_label}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    return {r["case_id"]: r for r in rows}


def merge_dataset(name: str) -> list[dict]:
    rows = _load_paper(name)
    cc_b16   = _load_cc(name, "budget_16")
    cc_pct10 = _load_cc(name, "pct_10")

    for r in rows:
        bfs_key = r["bfs"].get("key")
        cid = r.get("case_id")
        for col, lookup in [("cc_b16", cc_b16), ("cc_pct10", cc_pct10)]:
            if lookup is None:
                continue
            cc = lookup.get(cid)
            if cc is None:
                continue
            entry = {
                "outcome":     cc.get("outcome"),
                "calls":       cc.get("calls", 0),
                "budget_used": cc.get("budget", 0),
                "proxy_key":   cc.get("proxy_key"),
                "proxy_layer": cc.get("proxy_layer"),
            }
            entry["hop_to_bfs"] = hop_distance(entry["proxy_key"], bfs_key)
            r[col] = entry

    with open(f"data/{name}/paper_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def summarize(rows: list[dict]) -> str:
    methods = [
        ("BFS (oracle)",  "bfs",       True),
        ("SRA",           "sra",       False),
        ("ERA (b=16)",    "era_b16",   False),
        ("ERA (pct_10)",  "era_pct10", False),
        ("CC (b=16)",     "cc_b16",    False),
        ("CC (pct_10)",   "cc_pct10",  False),
    ]
    lines = []
    lines.append(f"  {'method':<14} | {'N':>4} | {'acc':>7} | {'cost':>8} | {'hop μ':>7} | {'hop med':>7}")
    lines.append("  " + "-" * 60)
    for label, col, is_bfs in methods:
        cells = [r[col] for r in rows if col in r and r[col] is not None]
        n = len(cells)
        if n == 0:
            lines.append(f"  {label:<14} |    — |       — |        — |       — |       —")
            continue
        if is_bfs:
            acc = 100.0
            cost = float(np.mean([c["calls"] for c in cells]))
        else:
            acc = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / n
            cost = float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
        hops = [c.get("hop_to_bfs") for c in cells]
        hops = [h for h in hops if h is not None]
        hop_mean = float(np.mean(hops)) if hops else float("nan")
        hop_med  = float(np.median(hops)) if hops else float("nan")
        lines.append(f"  {label:<14} | {n:>4} | {acc:>6.1f}% | {cost:>8.2f} | {hop_mean:>7.2f} | {hop_med:>7.1f}")
    return "\n".join(lines)


def main():
    for name, nice in DATASETS:
        print(f"\n{'='*70}\n  {nice}   (data/{name}/)\n{'='*70}")
        rows = merge_dataset(name)
        print(summarize(rows))


if __name__ == "__main__":
    main()
