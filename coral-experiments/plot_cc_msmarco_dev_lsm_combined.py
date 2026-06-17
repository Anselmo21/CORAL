"""LSM (CC) budget ablation combined plots for MSMARCO dev sub-lattices.

Reads:
  data/msmarco_subsample/cc_ablation_results.json
  data/msmarco_subsample/results.json   (BFS reference)

Writes:
  plots/msmarco_subsample/lsm_combined_fixed.png
  plots/msmarco_subsample/lsm_combined_pct.png

CLI: conda run -n coral python3 plot_cc_msmarco_dev_lsm_combined.py
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

CC_RESULTS_PATH  = "data/msmarco_subsample/cc_ablation_results.json"
ERA_RESULTS_PATH = "data/msmarco_subsample/results.json"
OUT_DIR          = "plots/msmarco_subsample"

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT:   list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

_COLORS  = plt.colormaps["tab10"].colors
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]


def _load() -> tuple[list[dict], list[dict]]:
    with open(CC_RESULTS_PATH) as f:
        cc: list[dict] = json.load(f)
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
    ax,
    sizes: list[int],
    cc_by_size: dict[int, dict[str, list[dict]]],
    era_by_size: dict[int, list[dict]],
    labels: list[str],
    legend_names: list[str],
    ylabel_kind: str,
) -> None:
    for i, (label, name) in enumerate(zip(labels, legend_names)):
        y = [
            _acc(cc_by_size.get(s, {}).get(label, [])) if ylabel_kind == "accuracy"
            else _cost(cc_by_size.get(s, {}).get(label, []))
            for s in sizes
        ]
        ax.plot(sizes, y, f"{_MARKERS[i % len(_MARKERS)]}-",
                color=_COLORS[i % len(_COLORS)],
                linewidth=1.7, markersize=6, label=name)

    if ylabel_kind == "cost":
        bfs_y = [float(np.mean([r["bfs"]["calls"] for r in era_by_size[s]])) for s in sizes]
        ax.plot(sizes, bfs_y, "o:", color="black", linewidth=2, markersize=7,
                label="BFS (baseline)")

    ax.set_xlabel("Number of Available Criteria")
    ax.set_ylabel("Accuracy (%)" if ylabel_kind == "accuracy" else "Average LLM Calls")
    if ylabel_kind == "accuracy":
        ax.set_ylim(0, 108)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_combined(cc: list[dict], era: list[dict], out_dir: str) -> None:
    cc_by_size  = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    specs = [
        ([f"fixed_{b}" for b in BUDGETS_FIXED], [str(b) for b in BUDGETS_FIXED], "fixed"),
        ([f"pct_{p}"   for p in BUDGETS_PCT],   [f"{p}%" for p in BUDGETS_PCT],  "pct"),
    ]

    for labels, names, name in specs:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        _budget_panel(axes[0], sizes, cc_by_size, era_by_size, labels, names, "accuracy")
        _budget_panel(axes[1], sizes, cc_by_size, era_by_size, labels, names, "cost")
        fig.tight_layout()
        out_path = f"{out_dir}/lsm_combined_{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    cc, era = _load()
    print(f"Loaded {len(cc)} CC rows, {len(era)} ERA/BFS rows")
    plot_combined(cc, era, OUT_DIR)


if __name__ == "__main__":
    main()
