"""Compare BFS / SRA / ERA(budget=64) / ERA(budget=10%) / CC(budget=64) on test top-down cases.

Reads existing ERA + CC results (both embed BFS) and runs SRA inline on the
same 200 test top-down cases (deterministic case ordering ensures alignment).
Writes SRA results to disk, then plots accuracy and call-cost by effective size.

Run:
    conda run -n coral python plot_test_top_down.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from era_budget_ablation import LATTICES_PATH, TEST_CASES_PATH, effective_size, load_manifest_cases
from experiment import score_proxy
from plot_style import apply_plot_style
from search.bfs import bfs_search
from search.rank import sra_search


ERA64_PATH = "data/era_ablation_study/test/top_down_budget_64/era_budget_ablation_results.json"
ERA10_PATH = "data/era_ablation_study/test/top_down_budget_10pct/era_budget_pct_ablation_results.json"
CC64_PATH = "data/era_ablation_study/test/cc_results_test_topdown.json"
SRA_OUT = "data/era_ablation_study/test/top_down_sra/sra_results.json"
PLOT_DIR = "data/era_ablation_study/test/comparison"


def run_sra(cases: list[dict]) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        sra_res = sra_search(criteria, rbc, start_node=start_node, direction=direction)
        scored = score_proxy(bfs_res, sra_res, rbc, start_node)
        scored["calls"] = sra_res["calls"]

        results.append({
            "query_id": case["query_id"],
            "effective_size": effective_size(case),
            "direction": direction,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "sra": scored,
        })
    return results


def _load_or_run_sra(cases: list[dict]) -> list[dict]:
    if os.path.exists(SRA_OUT):
        with open(SRA_OUT) as f:
            data = json.load(f)
        if len(data) == len(cases):
            print(f"Loaded {len(data)} SRA rows from {SRA_OUT}")
            return data
        print(f"SRA cache row count mismatch ({len(data)} vs {len(cases)}); recomputing")
    print(f"Running SRA on {len(cases)} cases...")
    results = run_sra(cases)
    os.makedirs(os.path.dirname(SRA_OUT), exist_ok=True)
    with open(SRA_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved SRA results to {SRA_OUT}")
    return results


def main() -> None:
    apply_plot_style()

    with open(ERA64_PATH) as f:
        era64_rows = json.load(f)
    with open(ERA10_PATH) as f:
        era10_rows = json.load(f)
    with open(CC64_PATH) as f:
        cc64_rows = json.load(f)
    print(f"Loaded ERA(64): {len(era64_rows)} rows")
    print(f"Loaded ERA(10%): {len(era10_rows)} rows")
    print(f"Loaded CC(64):  {len(cc64_rows)} rows")

    cases = load_manifest_cases(LATTICES_PATH, TEST_CASES_PATH, manifest_directions=["top-down"])
    sra_rows = _load_or_run_sra(cases)
    assert len(sra_rows) == len(era64_rows) == len(era10_rows) == len(cc64_rows), \
        (len(sra_rows), len(era64_rows), len(era10_rows), len(cc64_rows))

    # Index by effective_size
    sizes = sorted({r["effective_size"] for r in era64_rows})

    def by_size(rows: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            out[r["effective_size"]].append(r)
        return out

    era64_by = by_size(era64_rows)
    era10_by = by_size(era10_rows)
    sra_by = by_size(sra_rows)
    cc64_by = by_size(cc64_rows)

    # Accuracy curves
    def acc(rows: list[dict], result_key: str) -> float:
        if not rows:
            return float("nan")
        return sum(1 for r in rows if r[result_key]["outcome"] == "correct") / len(rows)

    bfs_acc = [1.0] * len(sizes)
    sra_acc = [acc(sra_by[s], "sra") for s in sizes]
    era64_acc = [acc(era64_by[s], "budget_64") for s in sizes]
    era10_acc = [acc(era10_by[s], "pct_10") for s in sizes]
    cc64_acc = [
        sum(1 for r in cc64_by[s] if r["outcome"] == "correct") / len(cc64_by[s])
        for s in sizes
    ]

    # Calls curves (means)
    def mean_calls(rows: list[dict], call_fn) -> float:
        vals = [call_fn(r) for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    bfs_calls = [mean_calls(era64_by[s], lambda r: r["bfs_calls"]) for s in sizes]
    sra_calls = [mean_calls(sra_by[s], lambda r: r["sra"]["calls"]) for s in sizes]
    era64_calls = [
        mean_calls(era64_by[s], lambda r: r["budget_64"]["calls"] + r["budget_64"]["budget_used"])
        for s in sizes
    ]
    era10_calls = [
        mean_calls(era10_by[s], lambda r: r["pct_10"]["calls"] + r["pct_10"]["budget_used"])
        for s in sizes
    ]
    # CC total cost: search calls + num_ablations spent fitting the surrogate.
    cc64_calls = [
        mean_calls(cc64_by[s], lambda r: r["calls"] + r["budget"])
        for s in sizes
    ]

    os.makedirs(PLOT_DIR, exist_ok=True)
    notitle_dir = os.path.join(PLOT_DIR, "notitle")
    os.makedirs(notitle_dir, exist_ok=True)

    def _save_dual(fig, name: str) -> None:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150)
        fig.suptitle("")
        fig.savefig(os.path.join(notitle_dir, name), dpi=150)
        print(f"Saved {path}")

    # ─── Accuracy plot ────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sizes, bfs_acc, "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax1.plot(sizes, sra_acc, "s-", label="SRA",            color="#d62728", linewidth=2, markersize=7)
    ax1.plot(sizes, era10_acc, "D-", label="ERA (10% budget)", color="#1f77b4", linewidth=2, markersize=7)
    ax1.plot(sizes, era64_acc, "P-", label="ERA (budget=64)",  color="#9467bd", linewidth=2, markersize=7)
    ax1.plot(sizes, cc64_acc, "X-", label="CC (budget=64)",    color="#ff7f0e", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Toggleable Criteria (k)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Top-Down Test")
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=True)
    ax1.grid(True, alpha=0.3)
    fig1.suptitle("Accuracy of Search Methods (test, top-down)", fontweight="bold")
    fig1.tight_layout()
    fig1.subplots_adjust(bottom=0.22)
    _save_dual(fig1, "accuracy_by_size.png")
    plt.close(fig1)

    # ─── Calls plot ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(sizes, bfs_calls, "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax2.plot(sizes, sra_calls, "s-", label="SRA",            color="#d62728", linewidth=2, markersize=7)
    ax2.plot(sizes, era10_calls, "D-", label="ERA (10% budget)", color="#1f77b4", linewidth=2, markersize=7)
    ax2.plot(sizes, era64_calls, "P-", label="ERA (budget=64)",  color="#9467bd", linewidth=2, markersize=7)
    ax2.plot(sizes, cc64_calls, "X-", label="CC (budget=64)",    color="#ff7f0e", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Toggleable Criteria (k)")
    ax2.set_ylabel("Average Total Calls")
    ax2.set_title("Top-Down Test")
    ax2.set_xticks(sizes)
    ax2.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=True)
    ax2.grid(True, alpha=0.3)
    fig2.suptitle("LLM Call Cost to Find First Counterfactual (test, top-down)", fontweight="bold")
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.22)
    _save_dual(fig2, "calls_by_size.png")
    plt.close(fig2)

    # ─── Summary table ───────────────────────────────────────────────────
    print()
    print(
        f"{'k':>3} | {'N':>3} | "
        f"{'BFS acc':>8} | {'SRA acc':>8} | {'ERA10 acc':>10} | {'ERA64 acc':>10} | {'CC64 acc':>9} || "
        f"{'BFS calls':>10} | {'SRA calls':>10} | {'ERA10 calls':>12} | {'ERA64 calls':>12} | {'CC64 calls':>11}"
    )
    print("-" * 165)
    for s in sizes:
        n = len(era64_by[s])
        cc_correct = sum(1 for r in cc64_by[s] if r["outcome"] == "correct")
        cc_acc_s = cc_correct / len(cc64_by[s])
        print(
            f"{s:>3} | {n:>3} | "
            f"{1.0:>8.1%} | "
            f"{acc(sra_by[s], 'sra'):>8.1%} | "
            f"{acc(era10_by[s], 'pct_10'):>10.1%} | "
            f"{acc(era64_by[s], 'budget_64'):>10.1%} | "
            f"{cc_acc_s:>9.1%} || "
            f"{mean_calls(era64_by[s], lambda r: r['bfs_calls']):>10.1f} | "
            f"{mean_calls(sra_by[s], lambda r: r['sra']['calls']):>10.1f} | "
            f"{mean_calls(era10_by[s], lambda r: r['pct_10']['calls'] + r['pct_10']['budget_used']):>12.1f} | "
            f"{mean_calls(era64_by[s], lambda r: r['budget_64']['calls'] + r['budget_64']['budget_used']):>12.1f} | "
            f"{mean_calls(cc64_by[s], lambda r: r['calls'] + r['budget']):>11.1f}"
        )


if __name__ == "__main__":
    main()
