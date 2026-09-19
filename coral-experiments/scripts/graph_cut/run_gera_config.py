"""GERA under each entropy scorer and each rank aggregation method.

One setting is swapped at a time into graph_cut_search on the regenerated
pools, at the operating budget, with everything else left at its default. The
softmax and Borda rows rerun that default, so they have to land on the suite's
GERA cell; a mismatch means the pools or the search moved since the suite ran.

--tie-tol and --full-support are the suite's flags of the same name and have
to match the run the suite was made with, or the default rows will not land on
its GERA cell.

Writes results/gera_config.json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from scripts.criteria.msmarco.run_subsample import score_proxy
from scripts.graph_cut.run_gera_suite import DATASETS, agg, load_cases
from search.flagr import make_era_aggregator
from search.graph_cut import graph_cut_search
from search.heuristics import max_jump_cutoff, renyi_entropy, shannon_entropy, weighted_entropy

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

BUDGET = 16
MODELS = ("llama31-8b-regen", "llama31-70b-regen", "qwen3-32b-regen", "qwen35-122b-regen")

# None keeps graph_cut_search's own default
ENTROPY = {
    "softmax": None,
    "shannon": shannon_entropy,
    "weighted": weighted_entropy,
    "renyi_0.5": partial(renyi_entropy, alpha=0.5),
    "renyi_2": partial(renyi_entropy, alpha=2.0),
}
AGGREGATOR = {"borda": None, "kemeny": "kemeny", "mct": "mct", "mc4": "mc4",
              "outranking": "outranking"}


def _cells(rows, budget, **kw):
    out = []
    for r in rows:
        res = graph_cut_search(r["criteria"], r["rbc"], r["supports"],
                               budget=budget, start_node=r["start"], **kw)
        out.append(score_proxy(r["bfs"], res, r["rbc"], r["start"]))
    return out


def run_cell(model, ds, budget, tie_tol=0.0, full_support=False):
    rows = load_cases(model, ds, full_support=full_support)
    cutoff_fn = partial(max_jump_cutoff, tie_tol=tie_tol) if tie_tol else None
    out = {"entropy": {}, "aggregator": {}}
    for name, fn in ENTROPY.items():
        out["entropy"][name] = agg(_cells(rows, budget, entropy_fn=fn, cutoff_fn=cutoff_fn))
    for name, method in AGGREGATOR.items():
        fn = make_era_aggregator(method) if method else None
        out["aggregator"][name] = agg(_cells(rows, budget, aggregator_fn=fn, cutoff_fn=cutoff_fn))
    return model, ds, out


def check_defaults(cells, budget):
    """The default rows against the suite's GERA cell, where the suite has one."""
    suite_path = RESULTS / "gera_suite.json"
    if not suite_path.exists():
        return
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    for model, by_ds in cells.items():
        for ds, c in by_ds.items():
            ref = suite.get(model, {}).get(ds, {}).get("sweep", {}).get(str(budget), {}).get("gera")
            for got in (c["entropy"]["softmax"], c["aggregator"]["borda"]):
                if ref and got != ref:
                    raise ValueError(f"{model}/{ds}: default row {got} != suite {ref}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tie-tol", type=float, default=0.0)
    ap.add_argument("--full-support", action="store_true")
    args = ap.parse_args()

    cells = {m: {} for m in MODELS}
    jobs = [(m, ds) for m in MODELS for ds in DATASETS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        n = len(jobs)
        for model, ds, out in pool.map(run_cell, *zip(*jobs), [args.budget] * n,
                                       [args.tie_tol] * n, [args.full_support] * n):
            cells[model][ds] = out
            g = out["entropy"]["softmax"]
            print(f"{model}/{ds}: gera {g['accuracy_pct']}%@{g['mean_calls']}")
    check_defaults(cells, args.budget)

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "gera_config.json"
    dest.write_text(json.dumps({"budget": args.budget, "tie_tol": args.tie_tol,
                                "full_support": args.full_support, "cells": cells}, indent=2),
                    encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
