"""Sub-lattice BFS / SRA / ERA sweep on a selected set of 5 MSMARCO pragma lattices.

QIDs: 51718, 51894, 441101, 51957, 52040

Samples 20 sub-lattices per effective size k in 3..9 that have a BFS-found CF
requiring removal of ≥2 pragmas, plus the 5 full k=10 lattices.
Total: 20*7 + 5 = 145 cases.

Also exports CC-ready cases to data/msmarco_pragma_selected/cc_cases.json
(sub-lattice-restricted rankings_by_combo + sources) for GPU evaluation.

Outputs:
  data/msmarco_pragma_selected/cases.json
  data/msmarco_pragma_selected/results.json
  data/msmarco_pragma_selected/cc_cases.json
  plots/msmarco_pragma_selected/accuracy_by_size.png
  plots/msmarco_pragma_selected/cost_by_size.png

CLI:
  conda run -n coral python3 run_msmarco_pragma_selected_eval.py
  conda run -n coral python3 run_msmarco_pragma_selected_eval.py --resample
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style
from run_msmarco_pragma_subsample import score_proxy
from search.bfs import bfs_search
from search.heuristics import softmax_entropy
from search.lattice import combo_key
from search.rank import era_search, sra_search

DATASET_PATH = "data/msmarco-pragma-final8-with-lattices.json"
OUT_DIR      = "data/msmarco_pragma_selected"
PLOT_DIR     = "plots/msmarco_pragma_selected"
CASES_PATH   = f"{OUT_DIR}/cases.json"
RESULTS_PATH = f"{OUT_DIR}/results.json"
CC_CASES_PATH = f"{OUT_DIR}/cc_cases.json"

TARGET_QIDS: list[str] = ["51718", "51894", "441101", "51957", "52040"]

SIZES: list[int]         = list(range(3, 10))
TARGET_PER_SIZE: int     = 20
MIN_PRAGMAS_REMOVED: int = 2
ERA_BUDGET_FIXED: int    = 32
SEED: int                = 42

METHOD_STYLES: dict[str, dict] = {
    "bfs": {"color": "black",    "marker": "o", "line": ":",  "lw": 2.0, "ms": 7},
    "sra": {"color": "#d62728",  "marker": "s", "line": "--", "lw": 1.7, "ms": 6},
    "era": {"color": "#1f77b4",  "marker": "D", "line": "-",  "lw": 1.7, "ms": 6},
    "cc":  {"color": "#ff7f0e",  "marker": "^", "line": "-",  "lw": 1.7, "ms": 6},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_lattices() -> dict[str, tuple[list[str], dict[str, list[str]], dict]]:
    """Return {qid: (criteria, rbc, rec)} for the target QIDs."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    out: dict[str, tuple[list[str], dict[str, list[str]], dict]] = {}
    for rec in data:
        qid = str(rec["query_id"])
        if qid not in TARGET_QIDS:
            continue
        criteria = rec["lattice"]["config"]["criteria"]
        rbc      = rec["lattice"]["answers"]
        if combo_key(criteria) not in rbc:
            print(f"  Skipping qid={qid}: start_key missing")
            continue
        out[qid] = (criteria, rbc, rec)
    return out


# ---------------------------------------------------------------------------
# Case sampling
# ---------------------------------------------------------------------------

def _qualifies(rbc: dict[str, list[str]], criteria: list[str], start_node: list[str]) -> bool:
    res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
    if not res["found"]:
        return False
    return (len(start_node) - res["size"]) >= MIN_PRAGMAS_REMOVED


def sample_cases(
    lattices: dict[str, tuple[list[str], dict[str, list[str]], dict]],
) -> list[dict]:
    from itertools import combinations

    rng   = random.Random(SEED)
    cases: list[dict] = []

    for size in SIZES:
        candidates: list[dict] = []
        for qid, (criteria, _, _) in lattices.items():
            for combo in combinations(criteria, size):
                candidates.append({"qid": qid, "size": size, "start_node": list(combo)})
        rng.shuffle(candidates)
        accepted: list[dict] = []
        for cand in candidates:
            criteria, rbc, _ = lattices[cand["qid"]]
            if _qualifies(rbc, criteria, cand["start_node"]):
                accepted.append(cand)
                if len(accepted) >= TARGET_PER_SIZE:
                    break
        print(f"  size={size}: pool={len(candidates):>5}  accepted={len(accepted):>3}/{TARGET_PER_SIZE}")
        cases.extend(accepted)

    for qid in TARGET_QIDS:
        if qid not in lattices:
            continue
        criteria, _, _ = lattices[qid]
        cases.append({"qid": qid, "size": 10, "start_node": criteria})
    print(f"  size=10: added {len(TARGET_QIDS)} full lattices")

    return cases


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]], dict]],
) -> list[dict]:
    results: list[dict] = []
    skipped = 0
    for i, case in enumerate(cases):
        qid        = str(case["qid"])
        start_node = case["start_node"]
        size       = case["size"]
        criteria, rbc, _ = lattices[qid]

        pct10_budget = max(1, int(2**size * 10 // 100))

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        sra_res      = sra_search(criteria, rbc, start_node=start_node, direction="down")
        era_res      = era_search(criteria, rbc, budget=ERA_BUDGET_FIXED,
                                  entropy_fn=softmax_entropy,
                                  start_node=start_node, direction="down")
        era_pct10_res = era_search(criteria, rbc, budget=pct10_budget,
                                   entropy_fn=softmax_entropy,
                                   start_node=start_node, direction="down")

        results.append({
            "case_id":    i,
            "qid":        qid,
            "size":       size,
            "start_node": start_node,
            "bfs": {
                "calls": bfs_res["calls"],
                "key":   bfs_res["key"],
                "layer": bfs_res["size"],
            },
            "sra": score_proxy(bfs_res, sra_res, rbc, start_node),
            "era": {
                **score_proxy(bfs_res, era_res, rbc, start_node),
                "calls":       era_res["calls"],
                "budget_used": era_res.get("budget_used", 0),
            },
            "era_pct10": {
                **score_proxy(bfs_res, era_pct10_res, rbc, start_node),
                "calls":       era_pct10_res["calls"],
                "budget_used": era_pct10_res.get("budget_used", 0),
            },
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)} done", flush=True)

    if skipped:
        print(f"Skipped {skipped} cases (no BFS CF)")
    return results


# ---------------------------------------------------------------------------
# CC export
# ---------------------------------------------------------------------------

def export_cc_cases(
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]], dict]],
) -> None:
    cc_cases: list[dict] = []
    for cid, case in enumerate(cases):
        qid        = str(case["qid"])
        start_node = case["start_node"]
        criteria, rbc, rec = lattices[qid]
        cfg = rec["lattice"]["config"]

        pragma_set = set(start_node)
        sub_rbc = {
            key: ranking for key, ranking in rbc.items()
            if all(p in pragma_set for p in (key.split(", ") if key else []))
        }

        cc_cases.append({
            "case_id":           cid,
            "query_id":          qid,
            "query":             cfg["question"],
            "sources":           cfg["sources"],
            "plausible_answers": cfg["plausible_answers"],
            "criteria":          start_node,
            "rankings_by_combo": sub_rbc,
            "start_node":        start_node,
            "effective_size":    case["size"],
        })

    with open(CC_CASES_PATH, "w") as f:
        json.dump(cc_cases, f, indent=2)
    sizes = defaultdict(int)
    for c in cc_cases:
        sizes[c["effective_size"]] += 1
    print(f"Exported {len(cc_cases)} CC cases → {CC_CASES_PATH}")
    print("By size:", dict(sorted(sizes.items())))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    keys = ["sra", "era", "era_pct10"]
    print(f"\n{'='*80}")
    print(f"  Selected pragma lattices — {len(results)} cases")
    print(f"{'='*80}")
    header = f"  {'k':>3} | {'N':>3} | {'BFS calls':>10} |"
    for k in keys:
        header += f" {k:>14} |"
    print(header)

    for size in sorted(by_size):
        bucket = by_size[size]
        n      = len(bucket)
        bfs_c  = np.mean([r["bfs"]["calls"] for r in bucket])
        line   = f"  {size:>3} | {n:>3} | {bfs_c:>10.1f} |"
        for k in keys:
            correct = sum(1 for r in bucket if r[k]["outcome"] == "correct")
            line   += f" {correct/n:>4.0%} ({correct:>2}/{n:<2}) |"
        print(line)

    total   = len(results)
    bfs_all = np.mean([r["bfs"]["calls"] for r in results])
    totals  = f"  {'All':>3} | {total:>3} | {bfs_all:>10.1f} |"
    for k in keys:
        correct = sum(1 for r in results if r[k]["outcome"] == "correct")
        totals += f" {correct/total:>4.0%} ({correct:>2}/{total:<2}) |"
    print(totals)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _acc(bucket: list[dict], key: str) -> float:
    return 100.0 * sum(1 for r in bucket if r[key]["outcome"] == "correct") / len(bucket)


def _cost(bucket: list[dict], key: str) -> float:
    return float(np.mean([r[key]["calls"] + r[key].get("budget_used", 0) for r in bucket]))


def plot_results(
    results: list[dict],
    cc_results: list[dict] | None = None,
) -> None:
    apply_plot_style()
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    cc_by_size: dict[int, list[dict]] = defaultdict(list)
    if cc_results:
        for r in cc_results:
            cc_by_size[r["effective_size"]].append(r)

    sizes = sorted(by_size)

    all_methods = [
        ("BFS",                    "bfs"),
        ("SRA",                    "sra"),
        (f"ERA b={ERA_BUDGET_FIXED}", "era"),
    ]

    for ylabel_kind, fname in [("accuracy", "accuracy_by_size"), ("cost", "cost_by_size")]:
        method_keys = (
            [m for m in all_methods if m[1] != "bfs"]
            if ylabel_kind == "accuracy"
            else all_methods
        )
        # fig, ax = plt.subplots(figsize=(5, 3))
        fig, ax = plt.subplots(figsize=(7, 4))

        for label, key in method_keys:
            st = METHOD_STYLES[key]
            if ylabel_kind == "accuracy":
                y = ([100.0] * len(sizes) if key == "bfs"
                     else [_acc(by_size[s], key) for s in sizes])
            else:
                y = ([float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
                     if key == "bfs" else [_cost(by_size[s], key) for s in sizes])
            ax.plot(sizes, y, f"{st['marker']}{st['line']}", color=st["color"],
                    linewidth=st["lw"], markersize=st["ms"], label=label)

        if cc_results:
            st = METHOD_STYLES["cc"]
            if ylabel_kind == "accuracy":
                cc_y = [
                    100.0 * sum(1 for r in cc_by_size[s] if r["outcome"] == "correct") / len(cc_by_size[s])
                    if cc_by_size[s] else float("nan") for s in sizes
                ]
            else:
                cc_y = [
                    float(np.mean([r["calls"] + r["budget"] for r in cc_by_size[s]]))
                    if cc_by_size[s] else float("nan") for s in sizes
                ]
            ax.plot(sizes, cc_y, f"{st['marker']}{st['line']}", color=st["color"],
                    linewidth=st["lw"], markersize=st["ms"], label="LSM b=10%")

        ax.set_xlabel("Number of Available Pragmas")
        if ylabel_kind == "accuracy":
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 108)
        else:
            ax.set_ylabel("Average LLM Calls")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", ncol=4, fontsize=9)
        fig.tight_layout()
        out_path = f"{PLOT_DIR}/{fname}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample",  action="store_true", help="Re-roll sampled cases")
    parser.add_argument("--plot-only", action="store_true", help="Skip eval; replot only")
    args = parser.parse_args()

    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    cc_results: list[dict] | None = None
    cc_out = f"{OUT_DIR}/cc_results_pct10.json"
    if os.path.exists(cc_out):
        with open(cc_out) as f:
            cc_results = json.load(f)
        print(f"Loaded {len(cc_results)} CC results from {cc_out}")

    if args.plot_only:
        with open(RESULTS_PATH) as f:
            results: list[dict] = json.load(f)
        print(f"Loaded {len(results)} results")
        plot_results(results, cc_results)
        return

    lattices = _load_lattices()
    print(f"Loaded {len(lattices)} lattices: {sorted(lattices)}")

    if args.resample or not os.path.exists(CASES_PATH):
        print("\nSampling cases ...")
        cases = sample_cases(lattices)
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Saved {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} existing cases")

    export_cc_cases(cases, lattices)

    print(f"\nRunning BFS / SRA / ERA(b={ERA_BUDGET_FIXED}) / ERA(pct_10) on {len(cases)} cases ...")
    results = run_eval(cases, lattices)

    print_summary(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")

    plot_results(results, cc_results)


if __name__ == "__main__":
    main()
