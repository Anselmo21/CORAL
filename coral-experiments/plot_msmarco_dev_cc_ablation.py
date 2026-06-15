"""Plot CC budget ablation on the MSMARCO dev set (k=10, 8 queries).

Reads data/msmarco_dev_cc_ablation_results.json.
Computes BFS / SRA / ERA(pct=10) reference lines inline from data/msmarco_dataset.json.

Writes plots/msmarco_dev_cc_ablation/{fixed_budgets,pct_budgets}.png
  Each PNG: left panel = accuracy vs budget, right panel = avg total cost vs budget.
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
from search.bfs import bfs_search
from search.rank import era_search, sra_search
from search.lattice import combo_key
from run_msmarco_subsample import _load_lattices, _load_split, score_proxy

CC_PATH = "data/msmarco_dev_cc_ablation_results.json"
DATA_PATH = "data/msmarco_dataset.json"
OUT_DIR = "plots/msmarco_dev_cc_ablation"

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90]
K: int = 10
ERA_PCT: int = 10

REF_COLORS = {"BFS": "black", "SRA": "#d62728", f"ERA {ERA_PCT}%": "#1f77b4"}


def _pct_budget(pct: int) -> int:
    return max(1, int(2**K * pct / 100))


def _load_cc() -> list[dict]:
    with open(CC_PATH) as f:
        return json.load(f)


def _compute_refs() -> dict[str, dict]:
    """Run BFS / SRA / ERA(pct=10) on the 8 dev queries. Returns {qid: {method: result}}."""
    dev_qids, _ = _load_split()
    lattices = _load_lattices(dev_qids)
    budget = _pct_budget(ERA_PCT)
    refs: dict[str, dict] = {}
    for qid, (criteria, rbc) in lattices.items():
        bfs = bfs_search(criteria, rbc, start_node=criteria, direction="down")
        sra = sra_search(criteria, rbc, start_node=criteria, direction="down")
        era = era_search(criteria, rbc, budget=budget, start_node=criteria, direction="down")
        bfs_for_proxy = {"key": bfs["key"], "size": bfs.get("size", len(bfs["key"].split(", ")) if bfs["key"] else 0)}
        refs[qid] = {
            "bfs_calls": bfs["calls"],
            "sra": score_proxy(bfs_for_proxy, sra, rbc, criteria),
            "era": score_proxy(bfs_for_proxy, era, rbc, criteria),
        }
    return refs


def _ref_hlines(ax, refs: dict[str, dict], ylabel_kind: str) -> None:
    n = len(refs)
    if ylabel_kind == "accuracy":
        bfs_val = 100.0
        sra_val = 100.0 * sum(1 for r in refs.values() if r["sra"]["outcome"] == "correct") / n
        era_val = 100.0 * sum(1 for r in refs.values() if r["era"]["outcome"] == "correct") / n
    else:
        bfs_val = float(np.mean([r["bfs_calls"] for r in refs.values()]))
        sra_val = float(np.mean([r["sra"]["calls"] + r["sra"].get("budget_used", 0) for r in refs.values()]))
        era_val = float(np.mean([r["era"]["calls"] + r["era"].get("budget_used", 0) for r in refs.values()]))

    ax.axhline(bfs_val, color=REF_COLORS["BFS"], linewidth=1.5, linestyle=":", label="BFS")
    ax.axhline(sra_val, color=REF_COLORS["SRA"], linewidth=1.5, linestyle="--", label="SRA")
    ax.axhline(era_val, color=REF_COLORS[f"ERA {ERA_PCT}%"], linewidth=1.5, linestyle="-.", label=f"ERA {ERA_PCT}%")


def _plot_panel(ax, cc_by_label: dict[str, list[dict]], labels_budgets: list[tuple[str, int]],
                refs: dict[str, dict], ylabel_kind: str, cmap_name: str) -> None:
    cmap = plt.colormaps[cmap_name]
    n = len(labels_budgets)
    xs = [b for _, b in labels_budgets]

    y = []
    for label, _ in labels_budgets:
        rows = cc_by_label.get(label, [])
        if not rows:
            y.append(float("nan"))
        elif ylabel_kind == "accuracy":
            y.append(100.0 * sum(1 for r in rows if r["outcome"] == "correct") / len(rows))
        else:
            y.append(float(np.mean([r["calls"] + r["budget"] for r in rows])))

    ax.plot(xs, y, "D-", color=cmap(0.6), linewidth=2, markersize=7, label="LSM (CC)")
    _ref_hlines(ax, refs, ylabel_kind)

    ax.set_xlabel("Budget (# ablations)")
    ax.set_ylabel("Accuracy (%)" if ylabel_kind == "accuracy" else "Avg total LLM calls")
    if ylabel_kind == "accuracy":
        ax.set_ylim(0, 108)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_budget_type(cc: list[dict], refs: dict[str, dict], budget_type: str, out_path: str) -> None:
    cc_by_label: dict[str, list[dict]] = defaultdict(list)
    for r in cc:
        cc_by_label[r["budget_label"]].append(r)

    if budget_type == "fixed":
        specs = [(f"fixed_{b}", b) for b in BUDGETS_FIXED]
        cmap = "Oranges"
        title = f"Fixed budgets — MSMARCO dev (k={K}, n=8)"
    else:
        specs = [(f"pct_{p}", _pct_budget(p)) for p in BUDGETS_PCT]
        cmap = "Greens"
        title = f"Percent-of-lattice budgets — MSMARCO dev (k={K}, n=8)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_panel(axes[0], cc_by_label, specs, refs, "accuracy", cmap)
    _plot_panel(axes[1], cc_by_label, specs, refs, "cost", cmap)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    cc = _load_cc()
    print(f"Loaded {len(cc)} CC rows from {CC_PATH}")

    print("Computing BFS/SRA/ERA reference lines...")
    refs = _compute_refs()
    print(f"  Done ({len(refs)} dev queries).")

    plot_budget_type(cc, refs, "fixed", f"{OUT_DIR}/fixed_budgets.png")
    plot_budget_type(cc, refs, "pct",   f"{OUT_DIR}/pct_budgets.png")


if __name__ == "__main__":
    main()
