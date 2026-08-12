"""Run SRA with each pyflagr rank-aggregation algorithm on the 4 Llama-3.1-8B datasets.

Uses the subsampled cases in logged_results/<dataset>/cases.json as the start nodes.
For each (dataset, aggregator) pair, reports accuracy (matches BFS layer) and mean cost.

Usage:
    conda run -n coral python3 -m scripts.shared.run_sra_pyflagr_llama
"""
from __future__ import annotations

import json
import os

import numpy as np
import pyflagr.Linear as Linear

from search.bfs import bfs_search
from search.lattice import combo_key, restrict_rbc
from search.rank_pyflagr import sra_search

DATASETS = [
    ("msmarco_criteria", "data/llama31-8b/msmarco-criteria-lattices.json", "MSMARCO criteria"),
    ("msmarco_pragma",   "data/llama31-8b/msmarco-pragmas-lattices.json",  "MSMARCO pragma"),
    ("curated_criteria",  "data/llama31-8b/curated-criteria-lattices.json",  "Curated criteria"),
    ("curated_pragma",    "data/llama31-8b/curated-pragmas-lattices.json",   "Curated pragma"),
]

AGGREGATORS = [
    ("CombSUM",          Linear.CombSUM),
    ("CombMNZ",          Linear.CombMNZ),
    ("BordaCount",       Linear.BordaCount),
    ("SimpleBordaCount", Linear.SimpleBordaCount),
]

OUT_PATH = "logged_results/sra_pyflagr_llama.json"


def load_lattices(path: str) -> dict[str, dict]:
    with open(path) as f:
        d = json.load(f)
    return {qid: q for qid, q in d.items() if qid != "_meta" and not q.get("halted")}


def score(proxy: dict, rbc: dict, start_node: list[str], bfs_layer: int) -> tuple[bool, int]:
    key = proxy.get("key")
    if not proxy.get("found") or key is None:
        return False, proxy.get("calls", 0)
    baseline = (rbc.get(combo_key(start_node)) or [""])[0]
    ranking = rbc.get(key, [])
    verified = bool(ranking) and ranking[0] != baseline
    correct = verified and proxy.get("size") == bfs_layer
    return correct, proxy.get("calls", 0)


def run_dataset(name: str, path: str, label: str) -> dict:
    lattices = load_lattices(path)
    with open(f"logged_results/{name}/cases.json") as f:
        cases = json.load(f)

    # Pre-compute BFS on each case's sub-lattice
    prepared = []
    for case in cases:
        qid = str(case["qid"])
        if qid not in lattices:
            continue
        full_criteria = list(lattices[qid]["criteria"])
        full_rbc = lattices[qid]["raw_answers"]
        start_node = case["start_node"]
        sub_rbc = restrict_rbc(full_rbc, start_node, full_criteria)
        bfs = bfs_search(start_node, sub_rbc, start_node=start_node, direction="down")
        if not bfs["found"]:
            continue
        prepared.append((start_node, sub_rbc, bfs))

    n = len(prepared)
    print(f"\n{'='*60}\n  {label}  ({n} valid cases)\n{'='*60}")
    bfs_cost = float(np.mean([b["calls"] for _, _, b in prepared]))
    print(f"  BFS mean cost: {bfs_cost:.2f}")

    results = {"n_cases": n, "bfs_cost": bfs_cost, "aggregators": {}}
    for agg_name, agg_cls in AGGREGATORS:
        nc, costs = 0, []
        for start_node, sub_rbc, bfs in prepared:
            r = sra_search(start_node, sub_rbc, aggregator=agg_cls(),
                           start_node=start_node, direction="down")
            ok, cost = score(r, sub_rbc, start_node, bfs["size"])
            nc += ok
            costs.append(cost)
        acc = 100 * nc / n
        cost = float(np.mean(costs))
        results["aggregators"][agg_name] = {"acc": acc, "cost": cost, "n_correct": nc}
        print(f"  SRA[{agg_name:<17s}] acc={acc:5.1f}%  cost={cost:5.2f}")
    return results


def main() -> None:
    all_results = {}
    for name, path, label in DATASETS:
        all_results[name] = {"label": label, **run_dataset(name, path, label)}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")

    print("\n" + "="*70)
    print(f"{'Dataset':<20s} " + " ".join(f"{a:>18s}" for a, _ in AGGREGATORS))
    print("="*70)
    for name, _, _ in DATASETS:
        cells = []
        for a, _ in AGGREGATORS:
            r = all_results[name]["aggregators"][a]
            cells.append(f"{r['acc']:5.1f}% / {r['cost']:4.1f}")
        print(f"{name:<20s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
