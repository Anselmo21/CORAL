"""Plot CC budget ablation vs BFS / SRA / ERA(pct_10) — MSMARCO full set.

Reads:
  data/msmarco_criteria/cc_ablation_results.json  (CC sweep over sizes 3-10)
  data/msmarco_criteria/results.json              (BFS/SRA/ERA references)

Writes plots/msmarco_criteria/:
  cc_fixed_budgets.png    — fixed CC budgets vs refs (accuracy | cost)
  cc_pct_budgets.png      — pct CC budgets vs refs
  cc_accuracy_heatmap.png — accuracy heatmap, rows = CC configs, cols = sizes
  cc_cost_heatmap.png     — cost heatmap
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

CC_RESULTS_PATH  = "data/msmarco_criteria/cc_ablation_results.json"
ERA_RESULTS_PATH = "data/msmarco_criteria/results.json"
OUT_DIR          = "plots/msmarco_criteria"

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT:   list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ERA_PCT:       int       = 10  # ERA reference budget (pct_10)

REF_COLORS = {"BFS": "black", "SRA": "#d62728", f"ERA pct_{ERA_PCT}": "#1f77b4"}
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]
_COLORS  = plt.colormaps["tab10"].colors


def _load() -> tuple[list[dict], list[dict]]:
    with open(CC_RESULTS_PATH) as f:
        cc: list[dict] = json.load(f)
    with open(ERA_RESULTS_PATH) as f:
        era_data: dict = json.load(f)
    return cc, era_data["results"]


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


def _acc_cc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if r["outcome"] == "correct") / len(rows)


def _cost_cc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r["calls"] + r["budget"] for r in rows]))


def _era_result(r: dict) -> dict:
    return r.get("era", {}).get(f"era_pct_{ERA_PCT}", {})


def _ref_lines(ax, sizes: list[int], era_by_size: dict, ylabel_kind: str) -> None:
    bfs_y = ([100.0] * len(sizes) if ylabel_kind == "accuracy"
             else [float(np.mean([r["bfs"]["calls"] for r in era_by_size[s]])) for s in sizes])
    sra_y = ([100.0 * sum(1 for r in era_by_size[s] if r.get("sra", {}).get("outcome") == "correct") / len(era_by_size[s])
              for s in sizes] if ylabel_kind == "accuracy"
             else [float(np.mean([r["sra"]["calls"] + r["sra"].get("budget_used", 0)
                                  for r in era_by_size[s] if r.get("sra")]))
                   for s in sizes])
    era_y = ([100.0 * sum(1 for r in era_by_size[s] if _era_result(r).get("outcome") == "correct") / len(era_by_size[s])
              for s in sizes] if ylabel_kind == "accuracy"
             else [float(np.mean([_era_result(r)["calls"] + _era_result(r).get("budget_used", 0)
                                  for r in era_by_size[s] if _era_result(r)]))
                   for s in sizes])
    ax.plot(sizes, bfs_y, "o:", color=REF_COLORS["BFS"],                  linewidth=2, markersize=7, label="BFS")
    ax.plot(sizes, sra_y, "s--", color=REF_COLORS["SRA"],                 linewidth=2, markersize=7, label="SRA")
    ax.plot(sizes, era_y, "D-",  color=REF_COLORS[f"ERA pct_{ERA_PCT}"],  linewidth=2, markersize=7, label=f"ERA pct_{ERA_PCT}")


def _budget_panel(ax, sizes, cc_by_size, era_by_size, specs: list[tuple[str, int]],
                  ylabel_kind: str) -> None:
    for i, (label, budget_val) in enumerate(specs):
        y = [_acc_cc(cc_by_size.get(s, {}).get(label, [])) if ylabel_kind == "accuracy"
             else _cost_cc(cc_by_size.get(s, {}).get(label, []))
             for s in sizes]
        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        legend_label = f"{budget_val}%" if label.startswith("pct_") else str(budget_val)
        ax.plot(sizes, y, f"{marker}-", color=color, linewidth=1.7, markersize=6, label=legend_label)

    _ref_lines(ax, sizes, era_by_size, ylabel_kind)
    ax.set_xlabel("Number of Available Criteria")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_budget_type(cc: list[dict], era: list[dict], budget_type: str, out_path: str) -> None:
    cc_by_size  = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    if budget_type == "fixed":
        specs = [(f"fixed_{b}", b) for b in BUDGETS_FIXED]
        title = "Fixed CC budgets — MSMARCO sub-lattices"
    else:
        specs = [(f"pct_{p}", p) for p in BUDGETS_PCT]
        title = "Percent-of-lattice CC budgets — MSMARCO sub-lattices"

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.suptitle(title, fontweight="bold")
    _budget_panel(axes[0], sizes, cc_by_size, era_by_size, specs, "accuracy")
    _budget_panel(axes[1], sizes, cc_by_size, era_by_size, specs, "cost")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap(cc: list[dict], era: list[dict], kind: str, out_path: str) -> None:
    cc_by_size  = _by_size_cc(cc)
    era_by_size = _by_size_era(era)
    sizes = sorted(set(cc_by_size) | set(era_by_size))

    row_specs: list[tuple[str, str]] = (
        [(f"b={b}", f"fixed_{b}") for b in BUDGETS_FIXED]
        + [(f"{p}%", f"pct_{p}") for p in BUDGETS_PCT]
    )

    matrix = np.zeros((len(row_specs), len(sizes)), dtype=float)
    metric_fn = _acc_cc if kind == "accuracy" else _cost_cc
    for i, (_, label) in enumerate(row_specs):
        for j, s in enumerate(sizes):
            matrix[i, j] = metric_fn(cc_by_size.get(s, {}).get(label, []))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_name = "RdYlGn" if kind == "accuracy" else "YlOrRd"
    vmin, vmax = (0.0, 100.0) if kind == "accuracy" else (None, None)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(row_specs)))
    ax.set_yticklabels([r[0] for r in row_specs], fontsize=9)
    ax.set_xlabel("Number of Available Criteria")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.0f}" if kind == "accuracy" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if (kind == "accuracy" and v >= 50) or
                                     (kind == "cost" and v < 0.6 * float(np.nanmax(matrix)))
                                  else "white")

    cbar_label = "Accuracy (%)" if kind == "accuracy" else "Avg cost (calls + budget)"
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    title = ("CC accuracy (%) — BFS is 100% at every k" if kind == "accuracy"
             else "CC avg total cost (calls + budget)")
    ax.set_title(title, fontsize=10)
    fig.suptitle("CC budget ablation — MSMARCO sub-lattices", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    cc, era = _load()
    print(f"Loaded {len(cc)} CC rows, {len(era)} ERA/BFS/SRA rows")

    plot_budget_type(cc, era, "fixed", f"{OUT_DIR}/cc_fixed_budgets.png")
    plot_budget_type(cc, era, "pct",   f"{OUT_DIR}/cc_pct_budgets.png")
    plot_heatmap(cc, era, "accuracy",  f"{OUT_DIR}/cc_accuracy_heatmap.png")
    plot_heatmap(cc, era, "cost",      f"{OUT_DIR}/cc_cost_heatmap.png")


if __name__ == "__main__":
    main()
