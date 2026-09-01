"""Side-by-side ERA vs MRA entropy ablation comparison (budget=8, llama31).

Loads:
  logged_results/llama31-8b/{ds}/era_ablation/entropy/results_b8.json
  logged_results/llama31-8b/{ds}/mra_entropy_ablation/results_b8.json

Output:
  plots/llama31/era_mra_entropy_comparison_b8.png

ERA lines: solid; MRA lines: dashed. Same color per entropy variant.

Usage:
    conda run -n coral python3 -m scripts.shared.plot_era_mra_entropy_comparison
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

DATASETS = [
    ("msmarco_criteria", "MSMARCO criteria", "Criteria"),
    ("msmarco_pragma",   "MSMARCO pragma",   "Pragmas"),
    ("curated_criteria",  "Curated criteria",  "Criteria"),
    ("curated_pragma",    "Curated pragma",    "Pragmas"),
]

VARIANTS = ["softmax", "shannon", "weighted", "renyi_0.5", "renyi_2"]
COLORS   = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
MARKERS  = ["s", "o", "^", "D", "v"]


def _load(path: str) -> list[dict]:
    d = json.load(open(path))
    return d["results"] if isinstance(d, dict) else d


def _by_size(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return dict(sorted(out.items()))


def _acc(bucket: list[dict], variant: str) -> float:
    return 100.0 * sum(1 for r in bucket if r[variant]["outcome"] == "correct") / len(bucket)


def _cost(bucket: list[dict], variant: str) -> float:
    return float(np.mean([r[variant]["calls"] + r[variant].get("budget_used", 0) for r in bucket]))


def _hop(bucket: list[dict], variant: str) -> float | None:
    vals = [
        abs(r[variant]["proxy_layer"] - r["bfs_layer"])
        for r in bucket
        if r[variant].get("outcome") != "correct" and r[variant].get("proxy_layer") is not None
    ]
    return float(np.mean(vals)) if vals else None


def _series(by_size, variant, metric_fn):
    xs, ys = [], []
    for s, bucket in by_size.items():
        v = metric_fn(bucket, variant)
        if v is not None:
            xs.append(s); ys.append(v)
    return xs, ys


def main() -> None:
    apply_plot_style()

    metrics = [
        ("Accuracy (%)",          _acc),
        ("Avg Cost (LLM Calls)",  _cost),
        ("Avg Hop Distance\n(incorrect only)", _hop),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    fig.suptitle("ERA vs MRA entropy ablation — budget=8 (Llama-3.1-8B)\n"
                 "solid = ERA, dashed = MRA",
                 fontsize=12, fontweight="bold")

    for col_idx, (ds, nice, x_label) in enumerate(DATASETS):
        era_path = f"logged_results/llama31-8b/{ds}/era_ablation/entropy/results_b8.json"
        mra_path = f"logged_results/llama31-8b/{ds}/mra_entropy_ablation/results_b8.json"

        era_rows = _load(era_path)
        mra_rows = _load(mra_path)
        era_by   = _by_size(era_rows)
        mra_by   = _by_size(mra_rows)
        sizes    = sorted(set(era_by) | set(mra_by))

        for row_idx, (ylabel, metric_fn) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for vi, variant in enumerate(VARIANTS):
                color  = COLORS[vi]
                marker = MARKERS[vi]

                xs_e, ys_e = _series(era_by, variant, metric_fn)
                if xs_e:
                    ax.plot(xs_e, ys_e, marker=marker, linestyle="-",
                            color=color, linewidth=2, markersize=6)

                xs_m, ys_m = _series(mra_by, variant, metric_fn)
                if xs_m:
                    ax.plot(xs_m, ys_m, marker=marker, linestyle="--",
                            color=color, linewidth=1.5, markersize=6, alpha=0.8)

            if metric_fn is _cost:
                bfs_cost = [float(np.mean([r["bfs_calls"] for r in era_by[s]])) for s in sizes if s in era_by]
                xs_bfs   = [s for s in sizes if s in era_by]
                ax.plot(xs_bfs, bfs_cost, marker="o", linestyle=":", color="black",
                        linewidth=1.5, markersize=5, label="BFS" if col_idx == 0 and row_idx == 1 else "")

            if row_idx == 0:
                ax.set_title(nice, fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_idx == 2:
                ax.set_xlabel(f"Number of Available {x_label}", fontsize=8)
            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric_fn is _acc:
                ax.set_ylim(0, 108)

    # Legend: variant colors + ERA/MRA linestyles
    variant_handles = [
        plt.Line2D([0], [0], marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=6, label=v)
        for i, v in enumerate(VARIANTS)
    ]
    style_handles = [
        plt.Line2D([0], [0], color="gray", linewidth=2, linestyle="-",  label="ERA (solid)"),
        plt.Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="MRA (dashed)"),
        plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle=":", label="BFS cost"),
    ]
    fig.legend(handles=variant_handles + style_handles,
               loc="upper center", ncol=len(VARIANTS) + 3,
               bbox_to_anchor=(0.5, 1.01), fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = "plots/llama31/era_mra_entropy_comparison_b8.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
