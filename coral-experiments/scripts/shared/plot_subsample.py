"""Unified per-dataset subsample plotter.

Reads `logged_results/<dataset>/results.json` for one or all of the four paper datasets
and writes the following PNGs to `plots/<dataset>/`:

    era_combined_fixed.png   acc + cost side-by-side, fixed ERA budgets
    era_combined_pct.png     acc + cost side-by-side, pct ERA budgets

CLI:
    python -m scripts.shared.plot_subsample --dataset msmarco_criteria
    python -m scripts.shared.plot_subsample --all
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

DATASETS = ["msmarco_criteria", "curated_criteria", "msmarco_pragma", "curated_pragma"]

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT:   list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

REF_COLORS = {"BFS": "black", "SRA": "#d62728"}
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]
_COLORS  = plt.colormaps["tab10"].colors


def _load_results(dataset: str) -> list[dict]:
    """Load results.json for a dataset; handle both dict-wrapped (criteria) and flat-list (pragma) forms."""
    with open(f"logged_results/{dataset}/results.json") as f:
        d = json.load(f)
    if isinstance(d, dict) and "results" in d:
        return d["results"]
    return d


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _acc(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m["outcome"] == "correct") / len(ms)


def _cost(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return float(np.mean([m["calls"] + m.get("budget_used", 0) for m in ms]))


def _budget_panel(ax, sizes, by_size, labels_and_pickers, ylabel_kind: str,
                  show_sra: bool = True, show_bfs: bool = True) -> None:
    for i, (label, picker) in enumerate(labels_and_pickers):
        y = [_acc(by_size[s], picker) if ylabel_kind == "accuracy"
             else _cost(by_size[s], picker)
             for s in sizes]
        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(sizes, y, f"{marker}-", color=color, linewidth=1.7, markersize=6, label=label)

    if show_bfs:
        if ylabel_kind == "accuracy":
            bfs_y = [100.0 for _ in sizes]
        else:
            bfs_y = [float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
        ax.plot(sizes, bfs_y, "o:", color=REF_COLORS["BFS"], linewidth=2, markersize=7,
                label="BFS (baseline)")

    if show_sra:
        if ylabel_kind == "accuracy":
            sra_y = [_acc(by_size[s], lambda r: r.get("sra")) for s in sizes]
        else:
            sra_y = [_cost(by_size[s], lambda r: r.get("sra")) for s in sizes]
        ax.plot(sizes, sra_y, "s--", color=REF_COLORS["SRA"], linewidth=2, markersize=7,
                label="SRA")

    ax.set_xlabel("Number of Available Criteria")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def _make_pickers() -> tuple[list[tuple[str, callable]], list[tuple[str, callable]]]:
    fixed_specs = [(f"{b}",
                    (lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab)))
                   for b in ERA_BUDGETS_FIXED]
    pct_specs = [(f"{p}%",
                  (lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab)))
                 for p in ERA_BUDGETS_PCT]
    return fixed_specs, pct_specs


def plot_combined(results: list[dict], out_dir: str) -> None:
    by_size = _by_size(results)
    sizes = sorted(by_size)
    fixed_specs, pct_specs = _make_pickers()
    for specs, name in [(fixed_specs, "fixed"), (pct_specs, "pct")]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        _budget_panel(axes[0], sizes, by_size, specs, "accuracy", show_sra=False, show_bfs=False)
        _budget_panel(axes[1], sizes, by_size, specs, "cost",     show_sra=False, show_bfs=True)
        fig.tight_layout()
        out_path = f"{out_dir}/era_combined_{name}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"  saved {out_path}")


def plot_dataset(dataset: str) -> None:
    out_dir = f"plots/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    results = _load_results(dataset)
    print(f"[{dataset}] {len(results)} cases")
    plot_combined(results, out_dir)


def main() -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=DATASETS,
                   help="One of the four paper datasets.")
    g.add_argument("--all", action="store_true",
                   help="Plot all four datasets.")
    args = parser.parse_args()

    targets = DATASETS if args.all else [args.dataset]
    for d in targets:
        plot_dataset(d)


if __name__ == "__main__":
    main()
