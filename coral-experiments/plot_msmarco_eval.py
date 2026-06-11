"""Plot BFS / SRA / ERA(budget=16) comparison for data/msmarco_llama_size10.json.

Reads data/msmarco_eval/results.json.
Writes plots/msmarco_eval/{accuracy,cost,outcomes}.png.
CLI: conda run -n coral python3 plot_msmarco_eval.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

RESULTS_PATH = "data/msmarco_eval/results.json"
OUT_DIR = "plots/msmarco_eval"

METHODS = [
    ("BFS",      "#333333"),
    ("SRA",      "#d62728"),
    ("ERA b=16", "#1f77b4"),
]


def _load() -> tuple[list[dict], int]:
    with open(RESULTS_PATH) as f:
        d = json.load(f)
    return d["results"], d["era_budget"]


def plot_accuracy(results: list[dict], out_dir: str) -> None:
    n = len(results)
    accs = [
        100.0,  # BFS always correct
        100.0 * sum(1 for r in results if r["sra"]["outcome"] == "correct") / n,
        100.0 * sum(1 for r in results if r["era"]["outcome"] == "correct") / n,
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, accs, color=[c for _, c in METHODS], width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in METHODS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Accuracy — MS MARCO size-10 (n={n})", fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cost(results: list[dict], out_dir: str) -> None:
    def mean_cost(key: str) -> float:
        vals = [r[key]["calls"] + r[key].get("budget_used", 0) for r in results]
        return float(np.mean(vals))

    costs = [
        float(np.mean([r["bfs"]["calls"] for r in results])),
        float(np.mean([r["sra"]["calls"] for r in results])),
        mean_cost("era"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, costs, color=[c for _, c in METHODS], width=0.5)
    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in METHODS])
    ax.set_ylabel("Mean LLM calls (+ budget)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Cost — MS MARCO size-10 (n={len(results)})", fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/cost.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_outcomes(results: list[dict], out_dir: str) -> None:
    outcome_order = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#aec7e8"]

    proxy_methods = [("SRA", "sra"), ("ERA b=16", "era")]
    counts: dict[str, dict[str, int]] = {}
    for label, key in proxy_methods:
        c: dict[str, int] = defaultdict(int)
        for r in results:
            c[r[key]["outcome"]] += 1
        counts[label] = dict(c)

    x = np.arange(len(proxy_methods))
    bottoms = np.zeros(len(proxy_methods))

    fig, ax = plt.subplots(figsize=(6, 4))
    for outcome, color in zip(outcome_order, outcome_colors):
        vals = [counts[label].get(outcome, 0) for label, _ in proxy_methods]
        ax.bar(x, vals, bottom=bottoms, color=color, label=outcome, width=0.5)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in proxy_methods])
    ax.set_ylabel("Cases")
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Outcome breakdown — MS MARCO size-10 (n={len(results)})", fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/outcomes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    results, era_budget = _load()
    plot_accuracy(results, OUT_DIR)
    plot_cost(results, OUT_DIR)
    plot_outcomes(results, OUT_DIR)


if __name__ == "__main__":
    main()
