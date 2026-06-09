"""Depth-restricted comparison plot (test, top-down, CF depth >= 3).

Mirrors plot_test_top_down.py but filters to cases where the BFS-found CF is
at least 3 layers below the start node — i.e. the regime where pruning should
matter. Reuses the same on-disk SRA/ERA/CC result files and uses ERA-64's
`bfs_layer` as canonical depth across all methods.

Run:
    conda run -n coral python plot_test_top_down_deep.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style


ERA64_PATH = "data/era_ablation_study/test/top_down_budget_64/era_budget_ablation_results.json"
ERA10_PATH = "data/era_ablation_study/test/top_down_budget_10pct/era_budget_pct_ablation_results.json"
CC64_PATH = "data/era_ablation_study/test/cc_results_test_topdown.json"
SRA_PATH = "data/era_ablation_study/test/top_down_sra/sra_results.json"
PLOT_DIR = "plots/era_ablation_study/test/comparison/deep"

DEPTH_THRESHOLD = 3


def main() -> None:
    apply_plot_style()

    with open(ERA64_PATH) as f:
        era64 = json.load(f)
    with open(ERA10_PATH) as f:
        era10 = json.load(f)
    with open(SRA_PATH) as f:
        sra = json.load(f)
    with open(CC64_PATH) as f:
        cc64 = json.load(f)

    n = len(era64)
    assert len(era10) == len(sra) == len(cc64) == n, \
        (len(era64), len(era10), len(sra), len(cc64))

    # Canonical depth from ERA-64's BFS results (locally computed, post-revert).
    def depth(r: dict) -> int:
        return r["effective_size"] - r["bfs_layer"]

    deep_idx = [i for i, r in enumerate(era64) if depth(r) >= DEPTH_THRESHOLD]
    print(f"Total cases: {n}; depth >= {DEPTH_THRESHOLD}: {len(deep_idx)}")
    if not deep_idx:
        print("No cases match — nothing to plot.")
        return

    era64_d = [era64[i] for i in deep_idx]
    era10_d = [era10[i] for i in deep_idx]
    sra_d   = [sra[i]   for i in deep_idx]
    cc64_d  = [cc64[i]  for i in deep_idx]

    # ─── Group by effective_size (k) ──────────────────────────────────────
    sizes = sorted({r["effective_size"] for r in era64_d})
    def by_size(rows: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            out[r["effective_size"]].append(r)
        return out
    era64_by = by_size(era64_d)
    era10_by = by_size(era10_d)
    sra_by   = by_size(sra_d)
    cc64_by  = by_size(cc64_d)

    def acc_era(rows: list[dict], result_key: str) -> float:
        return sum(1 for r in rows if r[result_key]["outcome"] == "correct") / len(rows) if rows else float("nan")

    def acc_cc(rows: list[dict]) -> float:
        return sum(1 for r in rows if r["outcome"] == "correct") / len(rows) if rows else float("nan")

    def mean_calls_era(rows: list[dict], result_key: str) -> float:
        if not rows:
            return float("nan")
        vals = [r[result_key]["calls"] + r[result_key]["budget_used"] for r in rows]
        return float(np.mean(vals))

    def mean_calls_cc(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        vals = [r["calls"] + r["budget"] for r in rows]  # CC: search + budget (num_ablations)
        return float(np.mean(vals))

    def mean_bfs(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["bfs_calls"] for r in rows]))

    def mean_sra_calls(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["sra"]["calls"] for r in rows]))

    bfs_acc   = [1.0] * len(sizes)
    sra_acc   = [acc_era(sra_by[s], "sra") for s in sizes]
    era10_acc = [acc_era(era10_by[s], "pct_10") for s in sizes]
    era64_acc = [acc_era(era64_by[s], "budget_64") for s in sizes]
    cc64_acc  = [acc_cc(cc64_by[s]) for s in sizes]

    bfs_calls   = [mean_bfs(era64_by[s]) for s in sizes]
    sra_calls   = [mean_sra_calls(sra_by[s]) for s in sizes]
    era10_calls = [mean_calls_era(era10_by[s], "pct_10") for s in sizes]
    era64_calls = [mean_calls_era(era64_by[s], "budget_64") for s in sizes]
    cc64_calls  = [mean_calls_cc(cc64_by[s]) for s in sizes]

    os.makedirs(PLOT_DIR, exist_ok=True)

    def _save_dual(fig, name: str) -> None:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    colors = {
        "BFS":   "#2ca02c",
        "SRA":   "#d62728",
        "ERA10": "#1f77b4",
        "ERA64": "#9467bd",
        "CC64":  "#ff7f0e",
    }

    # ─── Accuracy by k ────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sizes, bfs_acc,   "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax1.plot(sizes, sra_acc,   "s-", label="SRA",            color=colors["SRA"], linewidth=2, markersize=7)
    ax1.plot(sizes, era10_acc, "D-", label="ERA (10% budget)", color=colors["ERA10"], linewidth=2, markersize=7)
    ax1.plot(sizes, era64_acc, "P-", label="ERA (budget=64)",  color=colors["ERA64"], linewidth=2, markersize=7)
    ax1.plot(sizes, cc64_acc,  "X-", label="CC (budget=64)",   color=colors["CC64"], linewidth=2, markersize=7)
    ax1.set_xlabel("Number of Toggleable Criteria (k)")
    ax1.set_ylabel("Accuracy")
    n_str = ", ".join(f"k={s}: {len(era64_by[s])}" for s in sizes)
    ax1.set_title(f"Top-Down Test (CF depth ≥ {DEPTH_THRESHOLD}, N={len(deep_idx)})")
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(ncol=3, loc="lower left")
    ax1.grid(True, alpha=0.3)
    fig1.suptitle(f"Accuracy at Deep Counterfactuals (depth ≥ {DEPTH_THRESHOLD})", fontweight="bold")
    fig1.tight_layout()
    _save_dual(fig1, "accuracy_by_size_deep.png")
    plt.close(fig1)

    # ─── Calls by k ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(sizes, bfs_calls,   "o:", label="BFS (baseline)",   color="black", linewidth=2, markersize=7)
    ax2.plot(sizes, sra_calls,   "s-", label="SRA",              color=colors["SRA"], linewidth=2, markersize=7)
    ax2.plot(sizes, era10_calls, "D-", label="ERA (10% budget)", color=colors["ERA10"], linewidth=2, markersize=7)
    ax2.plot(sizes, era64_calls, "P-", label="ERA (budget=64)",  color=colors["ERA64"], linewidth=2, markersize=7)
    ax2.plot(sizes, cc64_calls,  "X-", label="CC (budget=64)",   color=colors["CC64"], linewidth=2, markersize=7)
    ax2.set_xlabel("Number of Toggleable Criteria (k)")
    ax2.set_ylabel("Average Total Calls")
    ax2.set_title(f"Top-Down Test (CF depth ≥ {DEPTH_THRESHOLD}, N={len(deep_idx)})")
    ax2.set_xticks(sizes)
    ax2.legend(ncol=3, loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.suptitle(f"LLM Call Cost at Deep Counterfactuals (depth ≥ {DEPTH_THRESHOLD})", fontweight="bold")
    fig2.tight_layout()
    _save_dual(fig2, "calls_by_size_deep.png")
    plt.close(fig2)

    # ─── Also: bin by depth (k pooled) ───────────────────────────────────
    by_depth: dict[int, list[int]] = defaultdict(list)
    for i in deep_idx:
        by_depth[depth(era64[i])].append(i)
    depths = sorted(by_depth)

    d_sra_acc, d_era10_acc, d_era64_acc, d_cc64_acc = [], [], [], []
    d_bfs_calls, d_sra_calls, d_era10_calls, d_era64_calls, d_cc64_calls = [], [], [], [], []
    for d in depths:
        ids = by_depth[d]
        d_sra_acc.append(acc_era([sra[i] for i in ids], "sra"))
        d_era10_acc.append(acc_era([era10[i] for i in ids], "pct_10"))
        d_era64_acc.append(acc_era([era64[i] for i in ids], "budget_64"))
        d_cc64_acc.append(acc_cc([cc64[i] for i in ids]))
        d_bfs_calls.append(mean_bfs([era64[i] for i in ids]))
        d_sra_calls.append(mean_sra_calls([sra[i] for i in ids]))
        d_era10_calls.append(mean_calls_era([era10[i] for i in ids], "pct_10"))
        d_era64_calls.append(mean_calls_era([era64[i] for i in ids], "budget_64"))
        d_cc64_calls.append(mean_calls_cc([cc64[i] for i in ids]))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(depths, [1.0] * len(depths), "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax3.plot(depths, d_sra_acc,   "s-", label="SRA",            color=colors["SRA"], linewidth=2, markersize=7)
    ax3.plot(depths, d_era10_acc, "D-", label="ERA (10% budget)", color=colors["ERA10"], linewidth=2, markersize=7)
    ax3.plot(depths, d_era64_acc, "P-", label="ERA (budget=64)",  color=colors["ERA64"], linewidth=2, markersize=7)
    ax3.plot(depths, d_cc64_acc,  "X-", label="CC (budget=64)",   color=colors["CC64"], linewidth=2, markersize=7)
    ax3.set_xlabel("CF Depth (criteria removed to reach CF)")
    ax3.set_ylabel("Accuracy")
    ax3.set_xticks(depths)
    ax3.set_ylim(0, 1.05)
    ax3.legend(ncol=3, loc="lower left")
    ax3.grid(True, alpha=0.3)
    n_labels = " · ".join(f"d={d}: n={len(by_depth[d])}" for d in depths)
    ax3.set_title(n_labels)
    fig3.suptitle("Accuracy by CF Depth (deep cases pooled across k)", fontweight="bold")
    fig3.tight_layout()
    _save_dual(fig3, "accuracy_by_depth.png")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(depths, d_bfs_calls,   "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax4.plot(depths, d_sra_calls,   "s-", label="SRA",            color=colors["SRA"], linewidth=2, markersize=7)
    ax4.plot(depths, d_era10_calls, "D-", label="ERA (10% budget)", color=colors["ERA10"], linewidth=2, markersize=7)
    ax4.plot(depths, d_era64_calls, "P-", label="ERA (budget=64)",  color=colors["ERA64"], linewidth=2, markersize=7)
    ax4.plot(depths, d_cc64_calls,  "X-", label="CC (budget=64)",   color=colors["CC64"], linewidth=2, markersize=7)
    ax4.set_xlabel("CF Depth (criteria removed to reach CF)")
    ax4.set_ylabel("Average Total Calls")
    ax4.set_xticks(depths)
    ax4.legend(ncol=3, loc="upper left")
    ax4.grid(True, alpha=0.3)
    ax4.set_title(n_labels)
    fig4.suptitle("Call Cost by CF Depth (deep cases pooled across k)", fontweight="bold")
    fig4.tight_layout()
    _save_dual(fig4, "calls_by_depth.png")
    plt.close(fig4)

    # ─── Summary table ───────────────────────────────────────────────────
    print()
    print(
        f"{'k':>3} | {'N':>3} | {'SRA':>6} {'calls':>8} | "
        f"{'ERA10':>6} {'calls':>8} | {'ERA64':>6} {'calls':>8} | "
        f"{'CC64':>6} {'calls':>8} | {'BFS calls':>10}"
    )
    print("-" * 110)
    for s in sizes:
        n_s = len(era64_by[s])
        print(
            f"{s:>3} | {n_s:>3} | "
            f"{acc_era(sra_by[s],'sra'):>5.0%} {mean_sra_calls(sra_by[s]):>8.1f} | "
            f"{acc_era(era10_by[s],'pct_10'):>5.0%} {mean_calls_era(era10_by[s],'pct_10'):>8.1f} | "
            f"{acc_era(era64_by[s],'budget_64'):>5.0%} {mean_calls_era(era64_by[s],'budget_64'):>8.1f} | "
            f"{acc_cc(cc64_by[s]):>5.0%} {mean_calls_cc(cc64_by[s]):>8.1f} | "
            f"{mean_bfs(era64_by[s]):>10.1f}"
        )


if __name__ == "__main__":
    main()
