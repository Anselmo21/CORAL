"""Plot BFS / SRA / ERA(10%) / ERA(64) on test_hard.jsonl.

Reads data/era_ablation_study/test_hard/test_hard_results.json.

Run:
    conda run -n coral python plot_test_hard.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style


RESULTS_PATH = "data/era_ablation_study/test_hard/test_hard_results.json"
PLOT_DIR = "data/era_ablation_study/test_hard/plots"


def main() -> None:
    apply_plot_style()

    with open(RESULTS_PATH) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} test_hard rows")

    sizes = sorted({r["effective_size"] for r in rows})

    def by_size(rows: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            out[r["effective_size"]].append(r)
        return out

    bucket = by_size(rows)

    def acc(rows: list[dict], key: str) -> float:
        if not rows:
            return float("nan")
        return sum(1 for r in rows if r[key]["outcome"] == "correct") / len(rows)

    def mean_calls_proxy(rows: list[dict], key: str) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r[key]["calls"] + r[key].get("budget_used", 0) for r in rows]))

    def mean_bfs(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["bfs_calls"] for r in rows]))

    has_era32 = all("budget_32" in r for r in rows)

    bfs_acc   = [1.0] * len(sizes)
    sra_acc   = [acc(bucket[s], "sra") for s in sizes]
    era10_acc = [acc(bucket[s], "pct_10") for s in sizes]
    era32_acc = [acc(bucket[s], "budget_32") for s in sizes] if has_era32 else None
    era64_acc = [acc(bucket[s], "budget_64") for s in sizes]

    bfs_calls   = [mean_bfs(bucket[s]) for s in sizes]
    sra_calls   = [mean_calls_proxy(bucket[s], "sra") for s in sizes]
    era10_calls = [mean_calls_proxy(bucket[s], "pct_10") for s in sizes]
    era32_calls = [mean_calls_proxy(bucket[s], "budget_32") for s in sizes] if has_era32 else None
    era64_calls = [mean_calls_proxy(bucket[s], "budget_64") for s in sizes]

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
        "BFS":   "black",
        "SRA":   "#d62728",
        "ERA10": "#1f77b4",
        "ERA32": "#2ca02c",
        "ERA64": "#9467bd",
    }

    # ─── Accuracy ─────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sizes, bfs_acc,   "o:", label="BFS (baseline)", color=colors["BFS"],   linewidth=2, markersize=7)
    ax1.plot(sizes, sra_acc,   "s-", label="SRA",            color=colors["SRA"],   linewidth=2, markersize=7)
    ax1.plot(sizes, era10_acc, "D-", label="ERA (10% budget)", color=colors["ERA10"], linewidth=2, markersize=7)
    if has_era32:
        ax1.plot(sizes, era32_acc, "v-", label="ERA (budget=32)",  color=colors["ERA32"], linewidth=2, markersize=7)
    ax1.plot(sizes, era64_acc, "P-", label="ERA (budget=64)",  color=colors["ERA64"], linewidth=2, markersize=7)
    ax1.set_xlabel("Number of Toggleable Criteria (k)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"test_hard (top-down, N={len(rows)})")
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(ncol=3, loc="lower left")
    ax1.grid(True, alpha=0.3)
    fig1.suptitle("Accuracy of Search Methods (test_hard)", fontweight="bold")
    fig1.tight_layout()
    _save_dual(fig1, "accuracy_by_size.png")
    plt.close(fig1)

    # ─── Calls ────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(sizes, bfs_calls,   "o:", label="BFS (baseline)", color=colors["BFS"],   linewidth=2, markersize=7)
    ax2.plot(sizes, sra_calls,   "s-", label="SRA",            color=colors["SRA"],   linewidth=2, markersize=7)
    ax2.plot(sizes, era10_calls, "D-", label="ERA (10% budget)", color=colors["ERA10"], linewidth=2, markersize=7)
    if has_era32:
        ax2.plot(sizes, era32_calls, "v-", label="ERA (budget=32)",  color=colors["ERA32"], linewidth=2, markersize=7)
    ax2.plot(sizes, era64_calls, "P-", label="ERA (budget=64)",  color=colors["ERA64"], linewidth=2, markersize=7)
    ax2.set_xlabel("Number of Toggleable Criteria (k)")
    ax2.set_ylabel("Average Total Calls")
    ax2.set_title(f"test_hard (top-down, N={len(rows)})")
    ax2.set_xticks(sizes)
    ax2.legend(ncol=3, loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.suptitle("LLM Call Cost to Find First Counterfactual (test_hard)", fontweight="bold")
    fig2.tight_layout()
    _save_dual(fig2, "calls_by_size.png")
    plt.close(fig2)

    # ─── Summary table ─────────────────────────────────────────────────
    print()
    if has_era32:
        print(f"{'k':>3} | {'N':>3} | {'SRA':>10} | {'ERA10':>10} | {'ERA32':>10} | {'ERA64':>10} | {'BFS calls':>10}")
        print("-" * 95)
        for s in sizes:
            n_s = len(bucket[s])
            print(
                f"{s:>3} | {n_s:>3} | "
                f"{acc(bucket[s], 'sra'):>4.0%} {mean_calls_proxy(bucket[s], 'sra'):>5.1f} | "
                f"{acc(bucket[s], 'pct_10'):>4.0%} {mean_calls_proxy(bucket[s], 'pct_10'):>5.1f} | "
                f"{acc(bucket[s], 'budget_32'):>4.0%} {mean_calls_proxy(bucket[s], 'budget_32'):>5.1f} | "
                f"{acc(bucket[s], 'budget_64'):>4.0%} {mean_calls_proxy(bucket[s], 'budget_64'):>5.1f} | "
                f"{mean_bfs(bucket[s]):>10.1f}"
            )
    else:
        print(f"{'k':>3} | {'N':>3} | {'SRA':>10} | {'ERA10':>10} | {'ERA64':>10} | {'BFS calls':>10}")
        print("-" * 80)
        for s in sizes:
            n_s = len(bucket[s])
            print(
                f"{s:>3} | {n_s:>3} | "
                f"{acc(bucket[s], 'sra'):>4.0%} {mean_calls_proxy(bucket[s], 'sra'):>5.1f} | "
                f"{acc(bucket[s], 'pct_10'):>4.0%} {mean_calls_proxy(bucket[s], 'pct_10'):>5.1f} | "
                f"{acc(bucket[s], 'budget_64'):>4.0%} {mean_calls_proxy(bucket[s], 'budget_64'):>5.1f} | "
                f"{mean_bfs(bucket[s]):>10.1f}"
            )

    # Pooled
    print()
    n_all = len(rows)
    print(f"Pooled (N={n_all}):")
    print(f"  SRA   acc: {acc(rows, 'sra'):.1%}   mean calls: {mean_calls_proxy(rows, 'sra'):.1f}")
    print(f"  ERA10 acc: {acc(rows, 'pct_10'):.1%}   mean calls: {mean_calls_proxy(rows, 'pct_10'):.1f}")
    if has_era32:
        print(f"  ERA32 acc: {acc(rows, 'budget_32'):.1%}   mean calls: {mean_calls_proxy(rows, 'budget_32'):.1f}")
    print(f"  ERA64 acc: {acc(rows, 'budget_64'):.1%}   mean calls: {mean_calls_proxy(rows, 'budget_64'):.1f}")
    print(f"  BFS  mean calls: {mean_bfs(rows):.1f}")


if __name__ == "__main__":
    main()
