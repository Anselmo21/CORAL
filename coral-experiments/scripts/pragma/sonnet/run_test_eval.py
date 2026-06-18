"""BFS / SRA / ERA(b=16, softmax) / ERA(b=16, shannon) on pragma sub-lattice test set.

Reads data/sonnet_pragma/cases_test.json (148 test cases).
Outputs:
  data/sonnet_pragma/test_results.json
  plots/sonnet_pragma/test_{accuracy_by_size,cost_by_size}.png

CLI: conda run -n coral python3 run_sonnet_pragma_test_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from search.lattice import hop_distance
from scripts.pragma.sonnet.run_subsample import (
    DATASET_PATH,
    SPLIT_PATH,
    _load_pragma_lattices,
    _load_split,
    score_proxy,
)
from search.bfs import bfs_search
from search.heuristics import shannon_entropy, softmax_entropy
from search.rank import era_search, sra_search

CASES_PATH   = "data/sonnet_pragma/cases_test.json"
OUT_PATH     = "data/sonnet_pragma/test_results.json"
PLOT_DIR     = "plots/sonnet_pragma"

BUDGET: int = 16
CC_RESULTS_PATH = "data/sonnet_pragma/cc_results_test_b16.json"

METHODS = [
    ("BFS",            None),
    ("SRA",            None),
    ("ERA (softmax)",  softmax_entropy),
    ("ERA (shannon)",  shannon_entropy),
]

COLORS  = ["black", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
MARKERS = ["o",     "s",       "D",       "D",       "^"]
LINES   = [":",     "--",      "-",       "-",       "-"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
) -> list[dict]:
    results: list[dict] = []
    skipped = 0
    for i, case in enumerate(cases):
        qid        = str(case["qid"])
        start_node = case["start_node"]
        size       = case["size"]
        criteria, rbc = lattices[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        sra_res = sra_search(criteria, rbc, start_node=start_node, direction="down")

        era_softmax = era_search(
            criteria, rbc, budget=BUDGET,
            entropy_fn=softmax_entropy,
            start_node=start_node, direction="down",
        )
        era_shannon = era_search(
            criteria, rbc, budget=BUDGET,
            entropy_fn=shannon_entropy,
            start_node=start_node, direction="down",
        )

        results.append({
            "case_id":   i,
            "qid":       qid,
            "size":      size,
            "start_node": start_node,
            "bfs": {
                "calls": bfs_res["calls"],
                "key":   bfs_res["key"],
                "layer": bfs_res["size"],
            },
            "sra":          score_proxy(bfs_res, sra_res,     rbc, start_node),
            "era_softmax":  {**score_proxy(bfs_res, era_softmax, rbc, start_node),
                             "calls": era_softmax["calls"],
                             "budget_used": era_softmax.get("budget_used", 0)},
            "era_shannon":  {**score_proxy(bfs_res, era_shannon, rbc, start_node),
                             "calls": era_shannon["calls"],
                             "budget_used": era_shannon.get("budget_used", 0)},
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cases)} cases done", flush=True)

    if skipped:
        print(f"Skipped {skipped} cases (no BFS CF)")
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    keys = ["sra", "era_softmax", "era_shannon"]
    col_w = 18

    print(f"\n{'='*80}")
    print(f"  Pragma test set — BFS / SRA / ERA(b={BUDGET}) — {len(results)} cases")
    print(f"{'='*80}")
    header = f"  {'Size':>4} | {'N':>3} |"
    for k in keys:
        header += f" {k:>{col_w-2}} |"
    print(header)
    sep = f"  {'-'*4}-+-{'-'*3}-+" + f"-{'-'*(col_w-2)}-+" * len(keys)
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        line = f"  {size:>4} | {n:>3} |"
        for k in keys:
            correct = sum(1 for r in bucket if r[k]["outcome"] == "correct")
            line += f" {correct/n:>4.0%} ({correct:>2}/{n:<2}) |"
        print(line)

    print(sep)
    total = len(results)
    totals = f"  {'All':>4} | {total:>3} |"
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
    out_dir: str,
    cc_results: list[dict] | None = None,
) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    cc_by_size: dict[int, list[dict]] = defaultdict(list)
    if cc_results:
        for r in cc_results:
            cc_by_size[r["effective_size"]].append(r)

    sizes = sorted(by_size)

    method_keys = [
        ("BFS",            "bfs"),
        ("SRA",            "sra"),
        (f"ERA b={BUDGET}", "era_softmax"),
    ]

    for ylabel_kind, fname in [("accuracy", "test_accuracy_by_size"), ("cost", "test_cost_by_size")]:
        fig, ax = plt.subplots(figsize=(6, 3.5))

        for (label, key), color, marker, line in zip(method_keys, COLORS, MARKERS, LINES):
            if ylabel_kind == "accuracy" and key == "bfs":
                continue
            if ylabel_kind == "accuracy":
                y = [_acc(by_size[s], key) for s in sizes]
            else:
                if key == "bfs":
                    y = [float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
                else:
                    y = [_cost(by_size[s], key) for s in sizes]

            lw = 2.0 if key == "bfs" else 1.7
            ms = 7   if key == "bfs" else 6
            ax.plot(sizes, y, f"{marker}{line}", color=color, linewidth=lw,
                    markersize=ms, label=label)

        if cc_results:
            cc_color, cc_marker, cc_line = COLORS[4], MARKERS[4], LINES[4]
            if ylabel_kind == "accuracy":
                cc_y = [
                    100.0 * sum(1 for r in cc_by_size[s] if r["outcome"] == "correct") / len(cc_by_size[s])
                    if cc_by_size[s] else float("nan")
                    for s in sizes
                ]
            else:
                cc_y = [
                    float(np.mean([r["calls"] + r["budget"] for r in cc_by_size[s]]))
                    if cc_by_size[s] else float("nan")
                    for s in sizes
                ]
            ax.plot(sizes, cc_y, f"{cc_marker}{cc_line}", color=cc_color,
                    linewidth=1.7, markersize=6, label=f"LSM b={BUDGET}")

        ax.set_xlabel("Number of Available Pragmas")
        if ylabel_kind == "accuracy":
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 108)
        else:
            ax.set_ylabel("Average LLM Calls")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_path = f"{out_dir}/{fname}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    # CF distance plot
    def _avg_hop(bucket: list[dict], key: str) -> float:
        dists = []
        for r in bucket:
            bfs_key = r["bfs"].get("key")
            m = r.get(key)
            if not bfs_key or m is None:
                continue
            proxy_key = m.get("proxy_key")
            if proxy_key is None:
                continue
            d = hop_distance(bfs_key, proxy_key)
            if d is not None:
                dists.append(d)
        return float(np.mean(dists)) if dists else float("nan")

    hop_methods = [
        ("SRA",                    "sra",         "#d62728", "s--"),
        (f"ERA b={BUDGET}",        "era_softmax", "#1f77b4", "D-"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, key, color, style in hop_methods:
        y = [_avg_hop(by_size[s], key) for s in sizes]
        ax.plot(sizes, y, style, label=label, color=color, linewidth=2, markersize=8)

    if cc_results:
        cc_id_map = {r["case_id"]: r for r in cc_results}
        def _avg_hop_cc(bucket: list[dict]) -> float:
            dists = []
            for r in bucket:
                bfs_key = r["bfs"].get("key")
                cc = cc_id_map.get(r["case_id"])
                if not bfs_key or cc is None:
                    continue
                proxy_key = cc.get("proxy_key")
                if proxy_key is None:
                    continue
                d = hop_distance(bfs_key, proxy_key)
                if d is not None:
                    dists.append(d)
            return float(np.mean(dists)) if dists else float("nan")
        y_cc = [_avg_hop_cc(by_size[s]) for s in sizes]
        ax.plot(sizes, y_cc, f"{MARKERS[4]}{LINES[4]}", color=COLORS[4],
                linewidth=2, markersize=8, label=f"LSM b={BUDGET}")

    ax.set_xlabel("Number of Available Pragmas")
    ax.set_ylabel("Avg hop distance from minimal CF")
    ax.set_xticks(sizes)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    out_path = f"{out_dir}/test_cf_distance_by_size.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip eval; just re-plot from existing results JSON")
    args = parser.parse_args()

    apply_plot_style()

    cc_results: list[dict] | None = None
    if os.path.exists(CC_RESULTS_PATH):
        with open(CC_RESULTS_PATH) as f:
            cc_results = json.load(f)
        print(f"Loaded {len(cc_results)} CC results from {CC_RESULTS_PATH}")

    if args.plot_only:
        with open(OUT_PATH) as f:
            results: list[dict] = json.load(f)
        print(f"Loaded {len(results)} results from {OUT_PATH}")
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_results(results, PLOT_DIR, cc_results)
        return

    with open(CASES_PATH) as f:
        cases: list[dict] = json.load(f)
    print(f"Loaded {len(cases)} test cases from {CASES_PATH}")

    _, test_qids = _load_split()
    lattices = _load_pragma_lattices(DATASET_PATH, set(test_qids))
    print(f"Loaded {len(lattices)} test lattices")

    print(f"\nRunning BFS / SRA / ERA(b={BUDGET}, softmax) / ERA(b={BUDGET}, shannon) ...")
    results = run_eval(cases, lattices)

    print_summary(results)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {OUT_PATH}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_results(results, PLOT_DIR, cc_results)


if __name__ == "__main__":
    main()
