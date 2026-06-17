"""Dev + test combined plot for MSMARCO pragma sub-lattice sweep.

Methods: BFS / SRA / ERA b=32 / ERA pct_10 / CC pct_10
2×2 grid: rows = accuracy | cost, cols = dev | test

Reads:
  data/msmarco_pragma/results.json                        (dev BFS/SRA/ERA)
  data/msmarco_pragma/cc_ablation_subsample_results.json  (dev CC, filter pct_10)
  data/msmarco_pragma/test_results.json                   (test BFS/SRA/ERA/ERA pct_10)
  data/msmarco_pragma/cc_results_test_pct10.json          (test CC pct_10)

Writes:
  plots/msmarco_pragma/dev_test_combined.png

CLI: conda run -n coral python3 plot_msmarco_pragma_dev_test_combined.py
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

DEV_ERA_PATH  = "data/msmarco_pragma/results.json"
DEV_CC_PATH   = "data/msmarco_pragma/cc_ablation_subsample_results.json"
TEST_ERA_PATH = "data/msmarco_pragma/test_results.json"
TEST_CC_PATH  = "data/msmarco_pragma/cc_results_test_pct10.json"
OUT_PATH      = "plots/msmarco_pragma/dev_test_combined.png"

COLORS  = {"BFS": "black", "SRA": "#d62728", "ERA b=32": "#1f77b4",
           "ERA pct_10": "#2ca02c", "CC pct_10": "#ff7f0e"}
MARKERS = {"BFS": "o",  "SRA": "s", "ERA b=32": "D", "ERA pct_10": "D", "CC pct_10": "^"}
LINES   = {"BFS": ":", "SRA": "--", "ERA b=32": "-", "ERA pct_10": "-", "CC pct_10": "-"}


def _acc(rows: list[dict], key: str) -> float:
    return 100.0 * sum(1 for r in rows if r[key]["outcome"] == "correct") / len(rows)


def _cost(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key]["calls"] + r[key].get("budget_used", 0) for r in rows]))


def _acc_cc(rows: list[dict]) -> float:
    return 100.0 * sum(1 for r in rows if r["outcome"] == "correct") / len(rows)


def _cost_cc(rows: list[dict]) -> float:
    return float(np.mean([r["calls"] + r["budget"] for r in rows]))


def _plot_panel(
    ax: matplotlib.axes.Axes,
    sizes: list[int],
    by_size: dict[int, list[dict]],
    cc_by_size: dict[int, list[dict]],
    ylabel_kind: str,
    era_key: str,
    era_pct_key: str,
    title: str,
) -> None:
    methods = [
        ("BFS",        lambda s: (100.0 if ylabel_kind == "accuracy"
                                  else float(np.mean([r["bfs"]["calls"] for r in by_size[s]])))),
        ("SRA",        lambda s: (_acc(by_size[s], "sra") if ylabel_kind == "accuracy"
                                  else _cost(by_size[s], "sra"))),
        ("ERA b=32",   lambda s: (_acc(by_size[s], era_key) if ylabel_kind == "accuracy"
                                  else _cost(by_size[s], era_key))),
        ("ERA pct_10", lambda s: (_acc(by_size[s], era_pct_key) if ylabel_kind == "accuracy"
                                  else _cost(by_size[s], era_pct_key))),
    ]

    for name, fn in methods:
        y = [fn(s) for s in sizes]
        lw = 2.0 if name == "BFS" else 1.7
        ms = 7   if name == "BFS" else 6
        ax.plot(sizes, y, f"{MARKERS[name]}{LINES[name]}",
                color=COLORS[name], linewidth=lw, markersize=ms, label=name)

    cc_y = [
        (_acc_cc(cc_by_size[s]) if ylabel_kind == "accuracy" else _cost_cc(cc_by_size[s]))
        if cc_by_size[s] else float("nan")
        for s in sizes
    ]
    ax.plot(sizes, cc_y, f"{MARKERS['CC pct_10']}{LINES['CC pct_10']}",
            color=COLORS["CC pct_10"], linewidth=1.7, markersize=6, label="CC pct_10")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Available Pragmas")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls (fewer is better)")
        ax.invert_yaxis()
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)


def main() -> None:
    apply_plot_style()

    with open(DEV_ERA_PATH) as f:
        dev_era = json.load(f)
    with open(DEV_CC_PATH) as f:
        dev_cc_all = json.load(f)
    with open(TEST_ERA_PATH) as f:
        test_era = json.load(f)
    with open(TEST_CC_PATH) as f:
        test_cc = json.load(f)

    # Dev: remap era nested keys to flat for uniform access
    for r in dev_era:
        r["era_b32"]   = r["era"]["era_fixed_32"]
        r["era_pct10"] = r["era"]["era_pct_10"]

    dev_by_size: dict[int, list[dict]] = defaultdict(list)
    for r in dev_era:
        dev_by_size[r["size"]].append(r)

    dev_cc_by_size: dict[int, list[dict]] = defaultdict(list)
    for r in dev_cc_all:
        if r["budget_label"] == "pct_10":
            dev_cc_by_size[r["effective_size"]].append(r)

    test_by_size: dict[int, list[dict]] = defaultdict(list)
    for r in test_era:
        test_by_size[r["size"]].append(r)

    test_cc_by_size: dict[int, list[dict]] = defaultdict(list)
    for r in test_cc:
        test_cc_by_size[r["effective_size"]].append(r)

    dev_sizes  = sorted(dev_by_size)
    test_sizes = sorted(test_by_size)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    _plot_panel(axes[0, 0], dev_sizes,  dev_by_size,  dev_cc_by_size,
                "accuracy", "era_b32", "era_pct10", "Dev — Accuracy")
    _plot_panel(axes[0, 1], test_sizes, test_by_size, test_cc_by_size,
                "accuracy", "era",     "era_pct10",  "Test — Accuracy")
    _plot_panel(axes[1, 0], dev_sizes,  dev_by_size,  dev_cc_by_size,
                "cost",     "era_b32", "era_pct10", "Dev — Cost")
    _plot_panel(axes[1, 1], test_sizes, test_by_size, test_cc_by_size,
                "cost",     "era",     "era_pct10",  "Test — Cost")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("MSMARCO pragma sub-lattice — dev vs test", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
