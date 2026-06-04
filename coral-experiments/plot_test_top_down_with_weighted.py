"""Comparison plot: BFS / SRA / ERA-softmax-10% / ERA-softmax-64 / ERA-weighted-10%
/ ERA-weighted-64 / CC-64 on test top-down cases.

Run:
    conda run -n coral python plot_test_top_down_with_weighted.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style


ERA64_PATH         = "data/era_ablation_study/test/top_down_budget_64/era_budget_ablation_results.json"
ERA10_PATH         = "data/era_ablation_study/test/top_down_budget_10pct/era_budget_pct_ablation_results.json"
ERA_WEIGHTED_PATH  = "data/era_ablation_study/test/top_down_weighted_10pct/era_weighted_10pct_results.json"
ERA_WEIGHTED64_PATH = "data/era_ablation_study/test/top_down_weighted_64/era_weighted_64_results.json"
CC64_PATH          = "data/era_ablation_study/test/cc_results_test_topdown.json"
SRA_PATH           = "data/era_ablation_study/test/top_down_sra/sra_results.json"
PLOT_DIR           = "data/era_ablation_study/test/comparison/weighted"


def main() -> None:
    apply_plot_style()

    with open(ERA64_PATH) as f:
        era64 = json.load(f)
    with open(ERA10_PATH) as f:
        era10 = json.load(f)
    with open(ERA_WEIGHTED_PATH) as f:
        erawt = json.load(f)
    with open(ERA_WEIGHTED64_PATH) as f:
        erawt64 = json.load(f)
    with open(CC64_PATH) as f:
        cc64 = json.load(f)
    with open(SRA_PATH) as f:
        sra = json.load(f)

    n = len(era64)
    assert len(era10) == len(erawt) == len(erawt64) == len(cc64) == len(sra) == n, \
        (len(era64), len(era10), len(erawt), len(erawt64), len(cc64), len(sra))

    sizes = sorted({r["effective_size"] for r in era64})

    def by_size(rows: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            out[r["effective_size"]].append(r)
        return out

    era64_by   = by_size(era64)
    era10_by   = by_size(era10)
    erawt_by   = by_size(erawt)
    erawt64_by = by_size(erawt64)
    cc64_by    = by_size(cc64)
    sra_by     = by_size(sra)

    def acc_era(rows: list[dict], result_key: str) -> float:
        if not rows:
            return float("nan")
        return sum(1 for r in rows if r[result_key]["outcome"] == "correct") / len(rows)

    def acc_cc(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return sum(1 for r in rows if r["outcome"] == "correct") / len(rows)

    def mean_calls_era(rows: list[dict], result_key: str) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r[result_key]["calls"] + r[result_key]["budget_used"] for r in rows]))

    def mean_calls_cc(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["calls"] + r["budget"] for r in rows]))

    def mean_bfs(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["bfs_calls"] for r in rows]))

    def mean_sra(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["sra"]["calls"] for r in rows]))

    # ─── Curves ─────────────────────────────────────────────────────────
    bfs_acc       = [1.0] * len(sizes)
    sra_acc       = [acc_era(sra_by[s], "sra") for s in sizes]
    era10_acc     = [acc_era(era10_by[s], "pct_10") for s in sizes]
    erawt_acc     = [acc_era(erawt_by[s], "pct_10") for s in sizes]
    era64_acc     = [acc_era(era64_by[s], "budget_64") for s in sizes]
    erawt64_acc   = [acc_era(erawt64_by[s], "budget_64") for s in sizes]
    cc64_acc      = [acc_cc(cc64_by[s]) for s in sizes]

    bfs_calls     = [mean_bfs(era64_by[s]) for s in sizes]
    sra_calls     = [mean_sra(sra_by[s]) for s in sizes]
    era10_calls   = [mean_calls_era(era10_by[s], "pct_10") for s in sizes]
    erawt_calls   = [mean_calls_era(erawt_by[s], "pct_10") for s in sizes]
    era64_calls   = [mean_calls_era(era64_by[s], "budget_64") for s in sizes]
    erawt64_calls = [mean_calls_era(erawt64_by[s], "budget_64") for s in sizes]
    cc64_calls    = [mean_calls_cc(cc64_by[s]) for s in sizes]

    os.makedirs(PLOT_DIR, exist_ok=True)
    notitle_dir = os.path.join(PLOT_DIR, "notitle")
    os.makedirs(notitle_dir, exist_ok=True)

    def _save_dual(fig, name: str) -> None:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150)
        fig.suptitle("")
        fig.savefig(os.path.join(notitle_dir, name), dpi=150)
        print(f"Saved {path}")

    colors = {
        "BFS":     "black",
        "SRA":     "#d62728",
        "ERA10":   "#1f77b4",
        "ERAwt":   "#17becf",
        "ERA64":   "#9467bd",
        "ERAwt64": "#8c564b",
        "CC64":    "#ff7f0e",
    }

    # ─── Accuracy plot ─────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sizes, bfs_acc,     "o:", label="BFS (baseline)",            color=colors["BFS"],     linewidth=2, markersize=7)
    ax1.plot(sizes, sra_acc,     "s-", label="SRA",                       color=colors["SRA"],     linewidth=2, markersize=7)
    ax1.plot(sizes, era10_acc,   "D-", label="ERA softmax (10%)",         color=colors["ERA10"],   linewidth=2, markersize=7)
    ax1.plot(sizes, erawt_acc,   "D-", label="ERA weighted (10%)",        color=colors["ERAwt"],   linewidth=2, markersize=7)
    ax1.plot(sizes, era64_acc,   "P-", label="ERA softmax (budget=64)",   color=colors["ERA64"],   linewidth=2, markersize=7)
    ax1.plot(sizes, erawt64_acc, "P-", label="ERA weighted (budget=64)",  color=colors["ERAwt64"], linewidth=2, markersize=7)
    ax1.plot(sizes, cc64_acc,    "X-", label="CC (budget=64)",            color=colors["CC64"],    linewidth=2, markersize=7)
    ax1.set_xlabel("Number of Toggleable Criteria (k)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Top-Down Test (N=200, 25 per k)")
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(ncol=2, loc="lower left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig1.suptitle("Accuracy of Search Methods (test, top-down)", fontweight="bold")
    fig1.tight_layout()
    _save_dual(fig1, "accuracy_by_size.png")
    plt.close(fig1)

    # ─── Calls plot ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(sizes, bfs_calls,     "o:", label="BFS (baseline)",            color=colors["BFS"],     linewidth=2, markersize=7)
    ax2.plot(sizes, sra_calls,     "s-", label="SRA",                       color=colors["SRA"],     linewidth=2, markersize=7)
    ax2.plot(sizes, era10_calls,   "D-", label="ERA softmax (10%)",         color=colors["ERA10"],   linewidth=2, markersize=7)
    ax2.plot(sizes, erawt_calls,   "D-", label="ERA weighted (10%)",        color=colors["ERAwt"],   linewidth=2, markersize=7)
    ax2.plot(sizes, era64_calls,   "P-", label="ERA softmax (budget=64)",   color=colors["ERA64"],   linewidth=2, markersize=7)
    ax2.plot(sizes, erawt64_calls, "P-", label="ERA weighted (budget=64)",  color=colors["ERAwt64"], linewidth=2, markersize=7)
    ax2.plot(sizes, cc64_calls,    "X-", label="CC (budget=64)",            color=colors["CC64"],    linewidth=2, markersize=7)
    ax2.set_xlabel("Number of Toggleable Criteria (k)")
    ax2.set_ylabel("Average Total Calls")
    ax2.set_title("Top-Down Test (N=200, 25 per k)")
    ax2.set_xticks(sizes)
    ax2.legend(ncol=2, loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.suptitle("LLM Call Cost to Find First Counterfactual (test, top-down)", fontweight="bold")
    fig2.tight_layout()
    _save_dual(fig2, "calls_by_size.png")
    plt.close(fig2)

    # ─── Summary table ─────────────────────────────────────────────────
    print()
    print(
        f"{'k':>3} | {'N':>3} | {'SRA':>10} | {'ERA10':>10} | {'ERAwt10':>11} | "
        f"{'ERA64':>10} | {'ERAwt64':>11} | {'CC64':>10} | {'BFS':>10}"
    )
    print("-" * 145)
    for s in sizes:
        n_s = len(era64_by[s])
        print(
            f"{s:>3} | {n_s:>3} | "
            f"{acc_era(sra_by[s], 'sra'):>4.0%} {mean_sra(sra_by[s]):>5.1f} | "
            f"{acc_era(era10_by[s], 'pct_10'):>4.0%} {mean_calls_era(era10_by[s], 'pct_10'):>5.1f} | "
            f"{acc_era(erawt_by[s], 'pct_10'):>4.0%} {mean_calls_era(erawt_by[s], 'pct_10'):>6.1f} | "
            f"{acc_era(era64_by[s], 'budget_64'):>4.0%} {mean_calls_era(era64_by[s], 'budget_64'):>5.1f} | "
            f"{acc_era(erawt64_by[s], 'budget_64'):>4.0%} {mean_calls_era(erawt64_by[s], 'budget_64'):>6.1f} | "
            f"{acc_cc(cc64_by[s]):>4.0%} {mean_calls_cc(cc64_by[s]):>5.1f} | "
            f"{mean_bfs(era64_by[s]):>10.1f}"
        )


if __name__ == "__main__":
    main()
