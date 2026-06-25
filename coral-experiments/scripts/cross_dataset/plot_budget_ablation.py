"""ERA budget ablation plots for all four main datasets.

Sweeps already exist in each dataset's results.json (produced by the
per-dataset run_subsample.py). This script renders accuracy and cost vs
effective lattice size for the fixed-budget grid `[4,8,16,32,64,128,256]`
and the percent-of-lattice grid `[10,20,...,100]%`, plus BFS oracle and
SRA reference curves.

Outputs (per dataset):
  final_plots/<dataset>/budget_ablation/era_fixed_accuracy.png
  final_plots/<dataset>/budget_ablation/era_fixed_cost.png
  final_plots/<dataset>/budget_ablation/era_pct_accuracy.png
  final_plots/<dataset>/budget_ablation/era_pct_cost.png
  final_plots/<dataset>/budget_ablation/era_combined_fixed.png
  final_plots/<dataset>/budget_ablation/era_combined_pct.png
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

DATASETS: list[tuple[str, str, str]] = [
    ("msmarco_criteria", "MSMARCO criteria", "data/msmarco_criteria/results.json"),
    ("sonnet_criteria",  "Sonnet criteria",  "data/sonnet_criteria/results.json"),
    ("msmarco_pragma",   "MSMARCO pragma",   "data/msmarco_pragma/results.json"),
    ("sonnet_pragma",    "Sonnet pragma",    "data/sonnet_pragma/results.json"),
]

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

REF_COLORS = {"BFS": "black", "SRA": "#d62728"}
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]
_COLORS = plt.colormaps["tab10"].colors


def _load_results(path: str) -> list[dict]:
    with open(path) as f:
        blob = json.load(f)
    return blob if isinstance(blob, list) else blob.get("results", [])


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _acc(bucket: list[dict], picker: Callable[[dict], dict | None]) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m["outcome"] == "correct") / len(ms)


def _cost(bucket: list[dict], picker: Callable[[dict], dict | None]) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return float(np.mean([m["calls"] + m.get("budget_used", 0) for m in ms]))


def _budget_panel(
    ax,
    sizes: list[int],
    by_size: dict[int, list[dict]],
    labels_and_pickers: list[tuple[str, Callable[[dict], dict | None]]],
    ylabel_kind: str,
    title: str,
    xlabel: str,
    show_sra: bool = True,
    show_bfs: bool = True,
) -> None:
    for i, (label, picker) in enumerate(labels_and_pickers):
        y = [_acc(by_size[s], picker) if ylabel_kind == "accuracy"
             else _cost(by_size[s], picker)
             for s in sizes]
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(sizes, y, f"{marker}-", color=color, linewidth=1.7,
                markersize=6, label=label)

    if show_bfs:
        if ylabel_kind == "accuracy":
            bfs_y = [100.0 for _ in sizes]
        else:
            bfs_y = [float(np.mean([r["bfs"]["calls"] for r in by_size[s]]))
                     for s in sizes]
        ax.plot(sizes, bfs_y, "o:", color=REF_COLORS["BFS"], linewidth=2,
                markersize=7, label="BFS (baseline)")

    if show_sra:
        if ylabel_kind == "accuracy":
            sra_y = [_acc(by_size[s], lambda r: r.get("sra")) for s in sizes]
        else:
            sra_y = [_cost(by_size[s], lambda r: r.get("sra")) for s in sizes]
        ax.plot(sizes, sra_y, "s--", color=REF_COLORS["SRA"], linewidth=2,
                markersize=7, label="SRA")

    ax.set_xlabel(xlabel)
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)


def _fixed_specs() -> list[tuple[str, Callable[[dict], dict | None]]]:
    return [(f"{b}",
             (lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab)))
            for b in ERA_BUDGETS_FIXED]


def _pct_specs() -> list[tuple[str, Callable[[dict], dict | None]]]:
    return [(f"{p}%",
             (lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab)))
            for p in ERA_BUDGETS_PCT]


def _save_single(
    by_size: dict[int, list[dict]],
    sizes: list[int],
    specs: list[tuple[str, Callable[[dict], dict | None]]],
    ylabel_kind: str,
    title: str,
    xlabel: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    show_bfs = ylabel_kind != "accuracy"
    _budget_panel(ax, sizes, by_size, specs, ylabel_kind, title,
                  xlabel, show_sra=False, show_bfs=show_bfs)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_combined(
    by_size: dict[int, list[dict]],
    sizes: list[int],
    specs: list[tuple[str, Callable[[dict], dict | None]]],
    title_prefix: str,
    xlabel: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    _budget_panel(axes[0], sizes, by_size, specs, "accuracy",
                  f"{title_prefix} — accuracy", xlabel,
                  show_sra=False, show_bfs=False)
    _budget_panel(axes[1], sizes, by_size, specs, "cost",
                  f"{title_prefix} — cost", xlabel, show_sra=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _verify_budgets(results: list[dict], slug: str) -> None:
    """Sanity check: confirm the expected budget labels exist in the data."""
    seen_fixed: set[int] = set()
    seen_pct: set[int] = set()
    for r in results:
        era = r.get("era", {})
        for k in era:
            if k.startswith("era_fixed_"):
                seen_fixed.add(int(k.split("_")[2]))
            elif k.startswith("era_pct_"):
                seen_pct.add(int(k.split("_")[2]))
    missing_fixed = [b for b in ERA_BUDGETS_FIXED if b not in seen_fixed]
    missing_pct = [p for p in ERA_BUDGETS_PCT if p not in seen_pct]
    if missing_fixed or missing_pct:
        print(f"  WARNING [{slug}] missing fixed={missing_fixed} pct={missing_pct}")


def main() -> None:
    apply_plot_style()
    for slug, title, path in DATASETS:
        results = _load_results(path)
        _verify_budgets(results, slug)
        by_size = _by_size(results)
        sizes = sorted(by_size)
        xlabel = ("Number of Available Pragmas" if "pragma" in slug
                  else "Number of Available Criteria")

        out_dir = f"final_plots/{slug}/budget_ablation"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{slug}] n={len(results)} sizes={sizes}")

        _save_single(by_size, sizes, _fixed_specs(), "accuracy",
                     f"{title}: ERA fixed-budget accuracy", xlabel,
                     f"{out_dir}/era_fixed_accuracy.png")
        _save_single(by_size, sizes, _fixed_specs(), "cost",
                     f"{title}: ERA fixed-budget cost", xlabel,
                     f"{out_dir}/era_fixed_cost.png")
        _save_single(by_size, sizes, _pct_specs(), "accuracy",
                     f"{title}: ERA percent-budget accuracy", xlabel,
                     f"{out_dir}/era_pct_accuracy.png")
        _save_single(by_size, sizes, _pct_specs(), "cost",
                     f"{title}: ERA percent-budget cost", xlabel,
                     f"{out_dir}/era_pct_cost.png")
        _save_combined(by_size, sizes, _fixed_specs(),
                       f"{title}: ERA fixed budgets", xlabel,
                       f"{out_dir}/era_combined_fixed.png")
        _save_combined(by_size, sizes, _pct_specs(),
                       f"{title}: ERA percent budgets", xlabel,
                       f"{out_dir}/era_combined_pct.png")


if __name__ == "__main__":
    main()
