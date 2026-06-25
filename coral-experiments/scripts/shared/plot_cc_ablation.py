"""Unified CC budget-ablation plotter (replaces the three per-dataset variants).

Reads:
  data/<dataset>/cc_ablation_results.json   (produced by run_cc_ablation.py)
  data/<dataset>/results.json                (BFS/SRA/ERA references)

Writes plots/<dataset>/:
  cc_fixed_budgets.png    fixed CC budgets vs refs (accuracy | cost)
  cc_pct_budgets.png      pct CC budgets vs refs
  cc_accuracy_heatmap.png accuracy heatmap, rows = CC configs, cols = sizes
  cc_cost_heatmap.png     cost heatmap

NOTE: as of the paper handoff, none of the four datasets actually have a
`cc_ablation_results.json` on disk. This script is here as the canonical
plotter for when that sweep is run — see `run_cc_ablation.py` in each
per-dataset directory.

CLI:
    python -m scripts.shared.plot_cc_ablation --dataset msmarco_criteria
    python -m scripts.shared.plot_cc_ablation --all
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

DATASETS = ["msmarco_criteria", "sonnet_criteria", "msmarco_pragma", "sonnet_pragma"]

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT:   list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ERA_PCT_REF:   int       = 10  # ERA reference budget for overlay

REF_COLORS = {"BFS": "black", "SRA": "#d62728", f"ERA pct_{ERA_PCT_REF}": "#1f77b4"}
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]
_COLORS  = plt.colormaps["tab10"].colors


def _load(dataset: str) -> tuple[list[dict], list[dict]]:
    cc_path = f"data/{dataset}/cc_ablation_results.json"
    era_path = f"data/{dataset}/results.json"
    with open(cc_path) as f:
        cc_rows = json.load(f)
    with open(era_path) as f:
        era_d = json.load(f)
    era_rows = era_d["results"] if isinstance(era_d, dict) and "results" in era_d else era_d
    return cc_rows, era_rows


def _by_size_cc(rows: list[dict], label: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("budget_label") == label:
            size_key = r.get("effective_size") or r.get("size")
            if size_key is not None:
                out[size_key].append(r)
    return out


def _by_size_era(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return out


def _acc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if r.get("outcome") == "correct") / len(rows)


def _cost(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r["calls"] + r.get("budget", 0) for r in rows]))


def _era_acc(bucket: list[dict], label: str) -> float:
    ms = [r["era"].get(label) for r in bucket if r.get("era")]
    ms = [m for m in ms if m is not None]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m["outcome"] == "correct") / len(ms)


def _era_cost(bucket: list[dict], label: str) -> float:
    ms = [r["era"].get(label) for r in bucket if r.get("era")]
    ms = [m for m in ms if m is not None]
    if not ms:
        return float("nan")
    return float(np.mean([m["calls"] + m.get("budget_used", 0) for m in ms]))


def _budget_panel(ax, sizes, cc_rows, era_rows, budgets, label_fmt: str,
                  kind: str) -> None:
    era_by_size = _by_size_era(era_rows)
    for i, b in enumerate(budgets):
        lab = label_fmt.format(b=b)
        cc_by_size = _by_size_cc(cc_rows, lab)
        y = [_acc(cc_by_size.get(s, [])) if kind == "accuracy"
             else _cost(cc_by_size.get(s, [])) for s in sizes]
        ax.plot(sizes, y, f"{_MARKERS[i % len(_MARKERS)]}-",
                color=_COLORS[i % len(_COLORS)], linewidth=1.7, markersize=6,
                label=f"CC {lab}")

    # Reference lines
    if kind == "accuracy":
        ax.plot(sizes, [100.0]*len(sizes), "o:", color=REF_COLORS["BFS"],
                linewidth=2, markersize=7, label="BFS")
        sra_y = [_era_acc(era_by_size.get(s, []), "sra_acc") for s in sizes]
        # SRA isn't under era; use sra outcome directly
        sra_y = [100.0 * sum(1 for r in era_by_size.get(s, []) if r["sra"]["outcome"] == "correct") / max(1, len(era_by_size.get(s, []))) for s in sizes]
        ax.plot(sizes, sra_y, "s--", color=REF_COLORS["SRA"], linewidth=2, markersize=7, label="SRA")
        ax.plot(sizes, [_era_acc(era_by_size.get(s, []), f"era_pct_{ERA_PCT_REF}") for s in sizes],
                "^-", color=REF_COLORS[f"ERA pct_{ERA_PCT_REF}"], linewidth=2, markersize=7,
                label=f"ERA pct_{ERA_PCT_REF}")
    else:
        ax.plot(sizes, [float(np.mean([r["bfs"]["calls"] for r in era_by_size.get(s, [])])) for s in sizes],
                "o:", color=REF_COLORS["BFS"], linewidth=2, markersize=7, label="BFS")
        ax.plot(sizes, [float(np.mean([r["sra"]["calls"] + r["sra"].get("budget_used", 0)
                                       for r in era_by_size.get(s, [])])) for s in sizes],
                "s--", color=REF_COLORS["SRA"], linewidth=2, markersize=7, label="SRA")
        ax.plot(sizes, [_era_cost(era_by_size.get(s, []), f"era_pct_{ERA_PCT_REF}") for s in sizes],
                "^-", color=REF_COLORS[f"ERA pct_{ERA_PCT_REF}"], linewidth=2, markersize=7,
                label=f"ERA pct_{ERA_PCT_REF}")

    ax.set_xlabel("Number of Available Criteria")
    if kind == "accuracy":
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_pair(dataset: str, kind_panels: list[str], out_path: str) -> None:
    cc_rows, era_rows = _load(dataset)
    era_by_size = _by_size_era(era_rows)
    sizes = sorted(era_by_size)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _budget_panel(axes[0], sizes, cc_rows, era_rows, BUDGETS_FIXED, "fixed_{b}", kind_panels[0])
    _budget_panel(axes[1], sizes, cc_rows, era_rows, BUDGETS_FIXED, "fixed_{b}", kind_panels[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_dataset(dataset: str) -> None:
    cc_path = f"data/{dataset}/cc_ablation_results.json"
    if not os.path.exists(cc_path):
        print(f"[{dataset}] SKIP — {cc_path} doesn't exist (no CC ablation has been run)")
        return
    out_dir = f"plots/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{dataset}]")
    cc_rows, era_rows = _load(dataset)
    era_by_size = _by_size_era(era_rows)
    sizes = sorted(era_by_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _budget_panel(axes[0], sizes, cc_rows, era_rows, BUDGETS_FIXED, "fixed_{b}", "accuracy")
    _budget_panel(axes[1], sizes, cc_rows, era_rows, BUDGETS_FIXED, "fixed_{b}", "cost")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cc_fixed_budgets.png", dpi=150); plt.close(fig)
    print(f"  saved {out_dir}/cc_fixed_budgets.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _budget_panel(axes[0], sizes, cc_rows, era_rows, BUDGETS_PCT, "pct_{b}", "accuracy")
    _budget_panel(axes[1], sizes, cc_rows, era_rows, BUDGETS_PCT, "pct_{b}", "cost")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cc_pct_budgets.png", dpi=150); plt.close(fig)
    print(f"  saved {out_dir}/cc_pct_budgets.png")


def main() -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=DATASETS)
    g.add_argument("--all", action="store_true")
    args = parser.parse_args()
    for d in (DATASETS if args.all else [args.dataset]):
        plot_dataset(d)


if __name__ == "__main__":
    main()
