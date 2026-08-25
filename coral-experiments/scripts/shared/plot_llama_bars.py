"""Bar chart summary of accuracy, cost, and hop distance across 4 llama31 datasets.

Aggregates across all lattice sizes; hop distance uses incorrect cases only.
Error bars show 95% CI (Wilson for accuracy, SE-based for cost/hop).

Output: plots/llama31/combined_bars.png

Usage:
    conda run -n coral python3 -m scripts.shared.plot_llama_bars
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

DATASETS = [
    ("msmarco_criteria", "MSMARCO\ncriteria"),
    ("msmarco_pragma",   "MSMARCO\npragma"),
    ("curated_criteria",  "Curated\ncriteria"),
    ("curated_pragma",    "Curated\npragma"),
]

METHODS = [
    ("bfs",            "BFS",        "black",   True),
    ("sra",            "SRA",        "#d62728", False),
    ("cc_b16",         "LSM (b=16)", "#2ca02c", False),
    ("cc_pct10",       "LSM (10%)",  "#98df8a", False),
    ("mra_softmax_b4", "ERA (b=4)",  "#9467bd", False),
    ("mra_softmax_b8", "ERA (b=8)",  "#c5b0d5", False),
]


def _load(dataset: str) -> list[dict]:
    with open(f"logged_results/llama31-8b/{dataset}/paper_results.json") as f:
        return json.load(f)


def _wilson95(successes: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), 0.0, 0.0
    p = successes / n
    z = 1.96
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p * 100, (centre - half) * 100, (centre + half) * 100


def _mean_ci95(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0.0
    m = float(np.mean(vals))
    if n == 1:
        return m, m, m
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    return m, m - 1.96 * se, m + 1.96 * se


def _stats(rows: list[dict], col: str, is_bfs: bool):
    acc_hits, acc_n = 0, 0
    cost_vals, hop_vals = [], []

    for r in rows:
        if is_bfs:
            cell = r.get("bfs")
            if not cell:
                continue
            acc_hits += 1
            acc_n += 1
            cost_vals.append(cell["calls"])
        else:
            cell = r.get(col)
            if not cell:
                continue
            acc_n += 1
            correct = cell.get("outcome") == "correct"
            if correct:
                acc_hits += 1
            cost_vals.append(cell["calls"] + cell.get("budget_used", 0))
            if not correct:
                pl = cell.get("proxy_layer")
                bl = r["bfs"]["layer"]
                if pl is not None:
                    hop_vals.append(abs(pl - bl))

    acc_m, acc_lo, acc_hi   = _wilson95(acc_hits, acc_n)
    cost_m, cost_lo, cost_hi = _mean_ci95(cost_vals)
    hop_m,  hop_lo,  hop_hi  = _mean_ci95(hop_vals)
    return (acc_m, acc_lo, acc_hi), (cost_m, cost_lo, cost_hi), (hop_m, hop_lo, hop_hi)


def main() -> None:
    apply_plot_style()

    x = np.arange(len(METHODS))
    width = 0.6
    metric_rows = [
        ("acc",  "Accuracy (%)"),
        ("cost", "Avg Cost (LLM Calls)"),
        ("hop",  "Avg Hop Distance\n(incorrect cases only)"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle("Method comparison — Llama-3.1-8B datasets (all sizes pooled)",
                 fontsize=13, fontweight="bold")

    for col_idx, (ds, ds_label) in enumerate(DATASETS):
        rows = _load(ds)
        col_stats = {}
        for col, _, _, is_bfs in METHODS:
            col_stats[col] = _stats(rows, col, is_bfs)

        for row_idx, (metric, ylabel) in enumerate(metric_rows):
            ax = axes[row_idx, col_idx]
            ki = {"acc": 0, "cost": 1, "hop": 2}[metric]

            means, lo_errs, hi_errs, colors = [], [], [], []
            for col, _, color, _ in METHODS:
                m, lo, hi = col_stats[col][ki]
                means.append(m)
                lo_errs.append(max(0.0, m - lo) if not np.isnan(m) else 0.0)
                hi_errs.append(max(0.0, hi - m) if not np.isnan(m) else 0.0)
                colors.append(color)

            means_arr = np.array(means, dtype=float)
            valid = ~np.isnan(means_arr)
            ax.bar(x[valid], means_arr[valid], width,
                   color=[colors[i] for i in range(len(colors)) if valid[i]],
                   alpha=0.85, zorder=3)
            lo_arr = np.array(lo_errs)[valid]
            hi_arr = np.array(hi_errs)[valid]
            ax.errorbar(x[valid], means_arr[valid], yerr=[lo_arr, hi_arr],
                        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

            if row_idx == 0:
                ax.set_title(ds_label, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels([lab for _, lab, _, _ in METHODS],
                               fontsize=7, rotation=35, ha="right")
            ax.grid(True, alpha=0.3, axis="y", zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if metric == "acc":
                ax.set_ylim(0, 108)

    fig.tight_layout()
    out = "plots/llama31/combined_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
