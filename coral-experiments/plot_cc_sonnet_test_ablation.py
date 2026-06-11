"""Plot CC budget ablation vs BFS / SRA / ERA(b=16) for sonnet test sub-lattices.

Reads:
  data/sonnet_test_subsample/cc_ablation_results.json  (CC sweep, from GPU)
  data/sonnet_test_subsample/results.json              (BFS/SRA/ERA, local)

Writes plots/cc_sonnet_test_ablation/:
  accuracy_by_size.png   — two panels: fixed / pct budgets, with BFS+SRA+ERA refs
  cost_by_size.png       — same layout for avg total cost
  accuracy_heatmap.png   — rows = CC budget configs, cols = sizes
  cost_heatmap.png       — same for cost
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
from run_cc_sonnet_test_ablation import BUDGETS_FIXED, BUDGETS_PCT, _pct_budget

CC_ABLATION_RESULTS_PATH = "data/sonnet_test_subsample/cc_ablation_results.json"
ERA_RESULTS_PATH = "data/sonnet_test_subsample/results.json"
OUT_DIR = "plots/cc_sonnet_test_ablation"

import argparse as _argparse
_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--cc-results", default=None)
_p.add_argument("--era-results", default=None)
_p.add_argument("--out-dir", default=None)
_p.add_argument("--no-era", action="store_true")
_p.add_argument("--no-sra", action="store_true")
_args, _ = _p.parse_known_args()
if _args.cc_results:
    CC_ABLATION_RESULTS_PATH = _args.cc_results
if _args.era_results:
    ERA_RESULTS_PATH = _args.era_results
if _args.out_dir:
    OUT_DIR = _args.out_dir
SHOW_ERA = not _args.no_era
SHOW_SRA = not _args.no_sra

ERA_BUDGET = 16
REF_COLORS = {"BFS": "black", "SRA": "#d62728", "ERA b=16": "#1f77b4"}


def _load() -> tuple[list[dict], list[dict]]:
    with open(CC_ABLATION_RESULTS_PATH) as f:
        cc = json.load(f)
    with open(ERA_RESULTS_PATH) as f:
        era_d = json.load(f)
    return cc, era_d["results"]


def _by_size_cc(cc: list[dict]) -> dict[int, dict[str, list[dict]]]:
    """Returns {size: {budget_label: [rows]}}."""
    out: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in cc:
        out[r["effective_size"]][r["budget_label"]].append(r)
    return out


def _by_size_era(era: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in era:
        out[r["size"]].append(r)
    return out


def _acc_cc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if r["outcome"] == "correct") / len(rows)


def _cost_cc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r["calls"] + r["budget"] for r in rows]))


def _get_era_result(r: dict) -> dict:
    """Return the ERA-b=16 result regardless of whether era is a single result or a budget dict."""
    era = r.get("era", {})
    if "outcome" in era:
        return era  # test schema: single result
    return era.get("era_fixed_16", {})  # dev schema: keyed by budget


def _ref_lines(ax, sizes, era_by_size, ylabel_kind: str) -> None:
    bfs_y = ([100.0] * len(sizes) if ylabel_kind == "accuracy"
             else [float(np.mean([r["bfs"]["calls"] for r in era_by_size[s]])) for s in sizes])
    sra_y = ([100.0 * sum(1 for r in era_by_size[s] if r.get("sra", {}).get("outcome") == "correct") / len(era_by_size[s])
              for s in sizes] if ylabel_kind == "accuracy"
             else [float(np.mean([r["sra"]["calls"] + r["sra"].get("budget_used", 0)
                                  for r in era_by_size[s] if r.get("sra")]))
                   for s in sizes])
    era_y = ([100.0 * sum(1 for r in era_by_size[s] if _get_era_result(r).get("outcome") == "correct") / len(era_by_size[s])
              for s in sizes] if ylabel_kind == "accuracy"
             else [float(np.mean([_get_era_result(r)["calls"] + _get_era_result(r).get("budget_used", 0)
                                  for r in era_by_size[s] if _get_era_result(r)]))
                   for s in sizes])
    ax.plot(sizes, bfs_y,  "o:", color=REF_COLORS["BFS"],     linewidth=2, markersize=7, label="BFS")
    if SHOW_SRA:
        ax.plot(sizes, sra_y,  "s--", color=REF_COLORS["SRA"],    linewidth=2, markersize=7, label="SRA")
    if SHOW_ERA:
        ax.plot(sizes, era_y,  "D-",  color=REF_COLORS["ERA b=16"], linewidth=2, markersize=7, label="ERA b=16")


def _budget_panel(ax, sizes, cc_by_size, era_by_size, labels_budgets,
                  ylabel_kind: str, title: str, cmap_name: str) -> None:
    cmap = plt.colormaps[cmap_name]
    n = len(labels_budgets)
    for i, (label, _) in enumerate(labels_budgets):
        y = []
        for s in sizes:
            rows = cc_by_size.get(s, {}).get(label, [])
            y.append(_acc_cc(rows) if ylabel_kind == "accuracy" else _cost_cc(rows))
        color = cmap(0.15 + 0.7 * i / max(1, n - 1))
        budget_val = _
        legend_label = f"budget={budget_val}%" if label.startswith("pct_") else f"budget={budget_val}"
        ax.plot(sizes, y, "D-", color=color, linewidth=1.7, markersize=6, label=legend_label)

    _ref_lines(ax, sizes, era_by_size, ylabel_kind)
    ax.set_xlabel("Effective lattice size (k)")
    ax.set_ylabel("Accuracy (%)" if ylabel_kind == "accuracy" else "Average LLM Calls")
    if ylabel_kind == "accuracy":
        ax.set_ylim(0, 108)
    ax.set_xticks(sizes)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_budget_type(cc: list[dict], era: list[dict], budget_type: str, out_path: str) -> None:
    """One file per budget type: [accuracy | cost] side by side."""
    cc_by_size = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    if budget_type == "fixed":
        specs = [(f"fixed_{b}", b) for b in BUDGETS_FIXED]
        cmap = "Oranges"
    else:
        specs = [(f"pct_{p}", p) for p in BUDGETS_PCT]
        cmap = "Greens"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    _budget_panel(axes[0], sizes, cc_by_size, era_by_size, specs, "accuracy", "", cmap)
    _budget_panel(axes[1], sizes, cc_by_size, era_by_size, specs, "cost",     "", cmap)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _heatmap_grid(cc: list[dict], era: list[dict], kind: str) -> tuple[list[str], list[int], np.ndarray]:
    cc_by_size = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    row_specs: list[tuple[str, callable]] = []
    for b in BUDGETS_FIXED:
        row_specs.append((f"budget={b}", f"fixed_{b}"))
    for p in BUDGETS_PCT:
        row_specs.append((f"budget={p}%", f"pct_{p}"))

    matrix = np.zeros((len(row_specs), len(sizes)), dtype=float)
    for i, (_, label) in enumerate(row_specs):
        for j, s in enumerate(sizes):
            rows = cc_by_size.get(s, {}).get(label, [])
            matrix[i, j] = _acc_cc(rows) if kind == "accuracy" else _cost_cc(rows)

    return [r[0] for r in row_specs], sizes, matrix


def plot_heatmap(cc: list[dict], era: list[dict], kind: str, out_path: str) -> None:
    labels, sizes, matrix = _heatmap_grid(cc, era, kind)
    era_by_size = _by_size_era(era)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_name = "RdYlGn" if kind == "accuracy" else "YlOrRd"
    vmin, vmax = (0.0, 100.0) if kind == "accuracy" else (None, None)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Effective lattice size (k)")

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

    if kind == "cost":
        bfs_row = ", ".join(f"k={s}: {np.mean([r['bfs']['calls'] for r in era_by_size[s]]):.1f}"
                            for s in sizes)
        ax.set_title(f"Avg total cost (calls + budget)\nBFS baseline — {bfs_row}", fontsize=10)
        cbar_label = "Avg cost"
    else:
        ax.set_title("Accuracy (%) — BFS is 100% at every k", fontsize=10)
        cbar_label = "Accuracy (%)"

    fig.colorbar(im, ax=ax).set_label(cbar_label)
    fig.suptitle(f"CC budget ablation — sonnet test sub-lattices", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    cc, era = _load()
    print(f"Loaded {len(cc)} CC ablation rows, {len(era)} ERA/BFS/SRA rows.")
    plot_budget_type(cc, era, "fixed", f"{OUT_DIR}/fixed_budgets.png")
    plot_budget_type(cc, era, "pct",   f"{OUT_DIR}/pct_budgets.png")
    plot_heatmap(cc, era, "accuracy", f"{OUT_DIR}/accuracy_heatmap.png")
    plot_heatmap(cc, era, "cost",     f"{OUT_DIR}/cost_heatmap.png")


if __name__ == "__main__":
    main()
