"""Plots for the gemini-dataset eval — BFS vs SRA vs ERA (budget=16).

Reads data/gemini_eval/gemini_eval_results.json (produced by run_gemini_eval.py).
Writes PNGs under plots/gemini_eval/:
  * accuracy_and_cost.png   - headline comparison (accuracy bars + avg cost bars w/ 95% CI)
  * cost_per_query.png      - per-query scatter of ERA b=16 cost vs BFS cost (log-log)
  * outcomes.png            - stacked outcome breakdown for SRA and ERA b=16
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

RESULTS_PATH = "data/gemini_eval/gemini_eval_results.json"
OUT_DIR = "plots/gemini_eval"

# Visual conventions match the rest of the project (see plot_test_top_down.py).
COLORS = {"BFS": "black", "SRA": "#d62728", "ERA": "#1f77b4"}
ERA_LABEL_KEY = "era_fixed_16"
ERA_LABEL = "ERA (budget=16)"


def _load_results() -> list[dict]:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data["results"]


def _method_cost(r: dict, method: str) -> float | None:
    if method == "bfs":
        return r.get("bfs", {}).get("calls")
    if method == "sra":
        m = r.get("sra")
        return None if m is None else m.get("calls", 0) + m.get("budget_used", 0)
    if method == ERA_LABEL_KEY:
        m = r.get("era", {}).get(ERA_LABEL_KEY)
        return None if m is None else m.get("calls", 0) + m.get("budget_used", 0)
    raise ValueError(method)


def _accuracy(r_list: list[dict], method: str) -> tuple[int, int]:
    if method == "bfs":
        # BFS is the ground truth; trivially "correct" whenever it found a CF.
        n = sum(1 for r in r_list if "bfs" in r)
        c = sum(1 for r in r_list if r.get("bfs", {}).get("found"))
        return c, n
    if method == "sra":
        sras = [r["sra"] for r in r_list if "sra" in r]
        return sum(1 for m in sras if m["outcome"] == "correct"), len(sras)
    eras = [r["era"][ERA_LABEL_KEY] for r in r_list if r.get("era", {}).get(ERA_LABEL_KEY)]
    return sum(1 for m in eras if m["outcome"] == "correct"), len(eras)


def _mean_ci(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    ci = float(1.96 * np.std(arr, ddof=1) / np.sqrt(n))
    return mean, ci


def plot_accuracy_and_cost(results: list[dict], out_dir: str) -> None:
    methods = ["bfs", "sra", ERA_LABEL_KEY]
    labels = ["BFS", "SRA", ERA_LABEL]
    colors = [COLORS["BFS"], COLORS["SRA"], COLORS["ERA"]]

    accs = []
    for m in methods:
        c, n = _accuracy(results, m)
        accs.append(100.0 * c / n if n else 0.0)

    means, cis = [], []
    for m in methods:
        vals = [v for r in results if (v := _method_cost(r, m)) is not None]
        mean, ci = _mean_ci(vals)
        means.append(mean)
        cis.append(ci)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))

    ax = axes[0]
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy (% queries where method's CF matches BFS layer)")
    ax.grid(True, axis="y", alpha=0.3)
    for rect, a in zip(bars, accs):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5,
                f"{a:.0f}%", ha="center", va="bottom", fontweight="bold")

    ax = axes[1]
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.8,
                  yerr=cis, capsize=6, error_kw={"elinewidth": 1.2, "capthick": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg cost (calls + budget_used)")
    ax.set_title("Average cost per query (95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    for rect, mean in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(means) * 0.02,
                f"{mean:.1f}", ha="center", va="bottom", fontweight="bold")

    fig.suptitle(f"Gemini dataset (n={len(results)} queries, 10 criteria, top-down)",
                 fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/accuracy_and_cost.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cost_per_query(results: list[dict], out_dir: str) -> None:
    pairs: list[tuple[float, float, float]] = []  # (bfs, era, sra)
    for r in results:
        b = _method_cost(r, "bfs")
        e = _method_cost(r, ERA_LABEL_KEY)
        s = _method_cost(r, "sra")
        if b is not None and e is not None:
            pairs.append((b, e, s if s is not None else float("nan")))
    if not pairs:
        return
    bfs_x = np.array([p[0] for p in pairs], dtype=float)
    era_y = np.array([p[1] for p in pairs], dtype=float)
    sra_y = np.array([p[2] for p in pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    lo = max(1.0, min(bfs_x.min(), era_y.min(), np.nanmin(sra_y)))
    hi = max(bfs_x.max(), era_y.max(), np.nanmax(sra_y)) * 1.2
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="y = x")
    ax.scatter(bfs_x, era_y, color=COLORS["ERA"], s=40, alpha=0.75,
               edgecolor="black", linewidth=0.5, label=ERA_LABEL)
    ax.scatter(bfs_x, sra_y, color=COLORS["SRA"], s=40, alpha=0.75, marker="s",
               edgecolor="black", linewidth=0.5, label="SRA")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BFS cost (calls)")
    ax.set_ylabel("Proxy cost (calls + budget_used)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("BFS cost (calls)")
    ax.set_ylabel("Proxy cost (calls + budget_used)")
    ax.set_title("Per-query cost: proxy vs BFS (log-log)\nPoints below y=x mean the proxy is cheaper than BFS")
    fig.tight_layout()
    path = f"{out_dir}/cost_per_query.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_outcomes(results: list[dict], out_dir: str) -> None:
    outcomes = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = {
        "correct": "#2ca02c",
        "wrong_layer": "#ff7f0e",
        "false_positive": "#d62728",
        "not_found": "#aec7e8",
    }

    counts: dict[str, Counter] = {}
    counts["SRA"] = Counter(r["sra"]["outcome"] for r in results if "sra" in r)
    counts[ERA_LABEL] = Counter(
        r["era"][ERA_LABEL_KEY]["outcome"] for r in results if r.get("era", {}).get(ERA_LABEL_KEY)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(counts.keys())
    x = np.arange(len(labels))
    width = 0.55
    bottoms = np.zeros(len(labels))
    for o in outcomes:
        vals = np.array([counts[lbl].get(o, 0) for lbl in labels], dtype=float)
        ax.bar(x, vals, width, bottom=bottoms, color=outcome_colors[o], label=o,
               edgecolor="black", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(i, b + v / 2, f"{int(v)}", ha="center", va="center",
                        fontweight="bold", color="white" if o != "not_found" else "black")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={sum(counts[lbl].values())})" for lbl in labels])
    ax.set_ylabel("Number of queries")
    ax.legend(loc="lower center", ncol=len(outcomes), bbox_to_anchor=(0.5, -0.32))
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Outcome breakdown", fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/outcomes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    results = _load_results()
    print(f"Loaded {len(results)} results from {RESULTS_PATH}")
    plot_accuracy_and_cost(results, OUT_DIR)
    plot_cost_per_query(results, OUT_DIR)
    plot_outcomes(results, OUT_DIR)


if __name__ == "__main__":
    main()
