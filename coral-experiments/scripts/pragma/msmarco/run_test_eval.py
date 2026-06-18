"""BFS / SRA / ERA(b=32, softmax) on the MSMARCO pragma sub-lattice test set.

Reads  data/msmarco_pragma/cases_test.json (144 test cases).
Writes data/msmarco_pragma/test_results.json
       plots/msmarco_pragma/test_{accuracy_by_size,cost_by_size}.png

CC results merged from data/msmarco_pragma/cc_results_test.json if present.

CLI: conda run -n coral python3 run_msmarco_pragma_test_eval.py
     conda run -n coral python3 run_msmarco_pragma_test_eval.py --plot-only
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
from scripts.pragma.msmarco.run_subsample import _load_lattices, _load_split, score_proxy
from search.bfs import bfs_search
from search.heuristics import softmax_entropy
from search.lattice import hop_distance
from search.rank import era_search, sra_search

CASES_PATH      = "data/msmarco_pragma/cases_test.json"
OUT_PATH        = "data/msmarco_pragma/test_results.json"
PLOT_DIR        = "plots/msmarco_pragma"
CC_RESULTS_PATH = "data/msmarco_pragma/cc_results_test_pct10.json"

BUDGET: int = 32

COLORS  = ["black",  "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
MARKERS = ["o",      "s",       "D",       "D",       "^"]
LINES   = [":",      "--",      "-",       "-",       "-"]


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

        pct10_budget = max(1, int(2**size * 10 // 100))

        sra_res      = sra_search(criteria, rbc, start_node=start_node, direction="down")
        era_res      = era_search(
            criteria, rbc, budget=BUDGET,
            entropy_fn=softmax_entropy,
            start_node=start_node, direction="down",
        )
        era_pct10_res = era_search(
            criteria, rbc, budget=pct10_budget,
            entropy_fn=softmax_entropy,
            start_node=start_node, direction="down",
        )

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

    keys  = ["sra", "era"]
    col_w = 18

    print(f"\n{'='*70}")
    print(f"  MSMARCO pragma test set — BFS / SRA / ERA(b={BUDGET}, softmax) — {len(results)} cases")
    print(f"{'='*70}")
    header = f"  {'Size':>4} | {'N':>3} | {'BFS calls':>10} |"
    for k in keys:
        header += f" {k:>{col_w-2}} |"
    print(header)
    sep = f"  {'-'*4}-+-{'-'*3}-+-{'-'*10}-+" + f"-{'-'*(col_w-2)}-+" * len(keys)
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        bfs_avg = np.mean([r["bfs"]["calls"] for r in bucket])
        line = f"  {size:>4} | {n:>3} | {bfs_avg:>10.1f} |"
        for k in keys:
            correct = sum(1 for r in bucket if r[k]["outcome"] == "correct")
            line += f" {correct/n:>4.0%} ({correct:>2}/{n:<2}) |"
        print(line)

    print(sep)
    total = len(results)
    bfs_overall = np.mean([r["bfs"]["calls"] for r in results])
    totals = f"  {'All':>4} | {total:>3} | {bfs_overall:>10.1f} |"
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
        ("BFS",             "bfs"),
        ("SRA",             "sra"),
        (f"ERA b={BUDGET}", "era"),
        ("ERA pct_10",      "era_pct10"),
    ]

    for ylabel_kind, fname in [("accuracy", "test_accuracy_by_size"), ("cost", "test_cost_by_size")]:
        fig, ax = plt.subplots(figsize=(6, 3.5))

        for (label, key), color, marker, line in zip(method_keys, COLORS, MARKERS, LINES):
            if ylabel_kind == "accuracy":
                y = [100.0] * len(sizes) if key == "bfs" else [_acc(by_size[s], key) for s in sizes]
            else:
                y = ([float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
                     if key == "bfs" else [_cost(by_size[s], key) for s in sizes])
            lw = 2.0 if key == "bfs" else 1.7
            ax.plot(sizes, y, f"{marker}{line}", color=color, linewidth=lw,
                    markersize=7 if key == "bfs" else 6, label=label)

        if cc_results:
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
            ax.plot(sizes, cc_y, f"{MARKERS[4]}{LINES[4]}", color=COLORS[4],
                    linewidth=1.7, markersize=6, label="CC pct_10")

        ax.set_xlabel("Number of Available Pragmas")
        if ylabel_kind == "accuracy":
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 108)
        else:
            ax.set_ylabel("Average LLM Calls")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", ncol=5, fontsize=9)
        fig.tight_layout()
        out_path = f"{out_dir}/{fname}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    # CF distance plot
    def _avg_hop_inline(bucket: list[dict], key: str) -> float:
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
        ("SRA",             "sra", "#d62728", "s--"),
        (f"ERA b={BUDGET}", "era", "#1f77b4", "D-"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, key, color, style in hop_methods:
        y = [_avg_hop_inline(by_size[s], key) for s in sizes]
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
                linewidth=2, markersize=8, label="CC pct_10")

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
                        help="Skip eval; re-plot from existing results JSON")
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
    lattices = _load_lattices(set(test_qids))
    print(f"Loaded {len(lattices)} test lattices")

    print(f"\nRunning BFS / SRA / ERA(b={BUDGET}, softmax) on {len(cases)} cases ...")
    results = run_eval(cases, lattices)

    print_summary(results)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {OUT_PATH}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_results(results, PLOT_DIR, cc_results)


if __name__ == "__main__":
    main()
