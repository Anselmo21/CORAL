"""Plot CC budget ablation vs BFS and ERA(b=16) for sonnet dev sub-lattices.

Reads:
  data/sonnet_subsample/cc_ablation_results.json   (CC sweep, from GPU)
  data/sonnet_subsample/results.json               (BFS/ERA, local)

Writes plots/cc_sonnet_dev_ablation/:
  combined_fixed.png   — fixed budgets: accuracy | avg LLM calls
  combined_pct.png     — pct budgets:   accuracy | avg LLM calls
  accuracy_heatmap.png — rows = CC budget configs, cols = sizes
  cost_heatmap.png     — same for cost

CLI: conda run -n coral python3 plot_cc_sonnet_dev_ablation.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
import matplotlib.axes
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from scripts.sonnet.run_cc_test_ablation import BUDGETS_FIXED, BUDGETS_PCT

CC_RESULTS_PATH  = "data/sonnet_subsample/cc_ablation_results.json"
ERA_RESULTS_PATH = "data/sonnet_subsample/results.json"
OUT_DIR          = "plots/cc_sonnet_dev_ablation"

REF_COLORS = {"BFS": "black", "ERA b=16": "#1f77b4"}
_COLORS    = plt.colormaps["tab10"].colors
_MARKERS   = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]


def _load() -> tuple[list[dict], list[dict]]:
    with open(CC_RESULTS_PATH) as f:
        cc = json.load(f)
    with open(ERA_RESULTS_PATH) as f:
        era_d = json.load(f)
    return cc, era_d["results"]


def _by_size_cc(cc: list[dict]) -> dict[int, dict[str, list[dict]]]:
    out: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in cc:
        out[r["effective_size"]][r["budget_label"]].append(r)
    return out


def _by_size_era(era: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in era:
        out[r["size"]].append(r)
    return out


def _acc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if r["outcome"] == "correct") / len(rows)


def _cost(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r["calls"] + r["budget"] for r in rows]))


def _budget_panel(
    ax: matplotlib.axes.Axes,
    sizes: list[int],
    cc_by_size: dict[int, dict[str, list[dict]]],
    era_by_size: dict[int, list[dict]],
    labels: list[str],
    ylabel_kind: str,
) -> None:
    for i, label in enumerate(labels):
        y = [
            _acc(cc_by_size.get(s, {}).get(label, [])) if ylabel_kind == "accuracy"
            else _cost(cc_by_size.get(s, {}).get(label, []))
            for s in sizes
        ]
        legend_label = (label.replace("fixed_", "").replace("pct_", "")
                        + ("%" if "pct_" in label else ""))
        ax.plot(sizes, y, f"{_MARKERS[i % len(_MARKERS)]}-",
                color=_COLORS[i % len(_COLORS)],
                linewidth=1.7, markersize=6, label=legend_label)

    if ylabel_kind == "cost":
        bfs_y = [float(np.mean([r["bfs"]["calls"] for r in era_by_size[s]])) for s in sizes]
        ax.plot(sizes, bfs_y, "o:", color=REF_COLORS["BFS"], linewidth=2, markersize=7,
                label="BFS (baseline)")

    ax.set_xlabel("Number of Available Criteria")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_combined(cc: list[dict], era: list[dict], out_dir: str) -> None:
    cc_by_size  = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    fixed_labels = [f"fixed_{b}" for b in BUDGETS_FIXED]
    pct_labels   = [f"pct_{p}"   for p in BUDGETS_PCT]

    for labels, name in [(fixed_labels, "fixed"), (pct_labels, "pct")]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        _budget_panel(axes[0], sizes, cc_by_size, era_by_size, labels, "accuracy")
        _budget_panel(axes[1], sizes, cc_by_size, era_by_size, labels, "cost")
        fig.tight_layout()
        out_path = f"{out_dir}/combined_{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_heatmap(cc: list[dict], era: list[dict], kind: str, out_path: str) -> None:
    cc_by_size  = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    row_labels = [f"{b}"  for b in BUDGETS_FIXED] + [f"{p}%" for p in BUDGETS_PCT]
    raw_labels = [f"fixed_{b}" for b in BUDGETS_FIXED] + [f"pct_{p}" for p in BUDGETS_PCT]

    matrix = np.full((len(raw_labels), len(sizes)), float("nan"))
    for i, label in enumerate(raw_labels):
        for j, s in enumerate(sizes):
            rows = cc_by_size.get(s, {}).get(label, [])
            matrix[i, j] = _acc(rows) if kind == "accuracy" else _cost(rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap_name = "RdYlGn" if kind == "accuracy" else "YlOrRd"
    vmin, vmax = (0.0, 100.0) if kind == "accuracy" else (None, None)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Number of Available Criteria")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.0f}" if kind == "accuracy" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if (kind == "accuracy" and v >= 50) or
                                     (kind == "cost" and v < 0.6 * np.nanmax(matrix))
                                  else "white")

    fig.colorbar(im, ax=ax).set_label("Accuracy (%)" if kind == "accuracy" else "Average LLM Calls")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    cc, era = _load()
    print(f"Loaded {len(cc)} CC rows, {len(era)} ERA rows")
    plot_combined(cc, era, OUT_DIR)
    plot_heatmap(cc, era, "accuracy", f"{OUT_DIR}/accuracy_heatmap.png")
    plot_heatmap(cc, era, "cost",     f"{OUT_DIR}/cost_heatmap.png")


if __name__ == "__main__":
    main()
