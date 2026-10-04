"""The pruning ablation over budget: GERA, GRA and ERA, one column per model.

Two rows (accuracy, cost) by four models. Each panel macro-averages the four
datasets, so no dataset outweighs another. All three arms are swept over the
same budgets - dropping the entropy cut leaves GRA, dropping the graph leaves
ERA. LSM (dotted) rides both rows over its own surrogate budgets, so the
accuracy band stretches to hold it; BFS (dashed) joins the cost row. SRA
has no budget at all, so it stays out of this figure - the strips figure
and Table I carry it. Colours match the Figure 4 strips series for series.
Reads results/gera_suite.json.

Writes results/paper_figs/gera/budget_ablation.png.

Usage:
    python3 -m scripts.shared.plot_gera_budget
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.shared.plot_style import apply_plot_style

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "results" / "gera_suite.json"
OUT = REPO / "results" / "paper_figs" / "gera" / "budget_ablation.png"

BUDGETS = (0, 2, 4, 8, 16, 32, 64, 128)
MODELS = [
    ("llama31-8b-regen", "Llama3.1-8B"),
    ("llama31-70b-regen", "Llama3.1-70B"),
    ("qwen3-32b-regen", "Qwen3-32B"),
    ("qwen35-122b-regen", "Qwen3.5-122B"),
]
DATASETS = ("msmarco_criteria", "msmarco_pragma", "curated_criteria", "curated_pragma")

GERA_STYLE = dict(linestyle="-", marker="X", color="#1f77b4", linewidth=1.6, markersize=6.5)
GRA_STYLE = dict(linestyle="-", marker="D", color="#ff7f0e", linewidth=1.5, markersize=5.5)
ERA_STYLE = dict(linestyle="-", marker="h", color="#9467bd", linewidth=1.5, markersize=6.5)
LSM_STYLE = dict(linestyle=":", marker="^", color="#2ca02c", linewidth=1.7, markersize=6.5)
BFS_STYLE = dict(linestyle="--", color="black", linewidth=1.7)


def macro(suite, model, pick):
    """Mean of a per-dataset statistic over the model's four datasets."""
    return float(np.mean([pick(suite[model][ds]) for ds in DATASETS]))


def main() -> None:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    apply_plot_style()
    # the figure prints at \textwidth, so type is sized for that reduction
    plt.rcParams.update({"axes.titlesize": 19, "axes.labelsize": 18,
                         "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 17})
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    xs = list(range(len(BUDGETS)))

    for col, (model, nice) in enumerate(MODELS):
        ax_acc, ax_cost = axes[0][col], axes[1][col]
        series = {}
        for m in ("gera", "gra", "era"):
            series[m] = (
                [macro(suite, model, lambda s, b=b, m=m: s["sweep"][str(b)][m]["accuracy_pct"])
                 for b in BUDGETS],
                [macro(suite, model, lambda s, b=b, m=m: s["sweep"][str(b)][m]["mean_calls"])
                 for b in BUDGETS])
        bfs_cost = macro(suite, model, lambda s: s["bfs_calls"])
        lsm_bs = sorted(int(b) for b in suite[model][DATASETS[0]]["lsm"])
        lx = [BUDGETS.index(b) for b in lsm_bs]
        lsm_acc = [macro(suite, model, lambda s, b=b: s["lsm"][str(b)]["accuracy_pct"])
                   for b in lsm_bs]
        lsm_cost = [macro(suite, model, lambda s, b=b: s["lsm"][str(b)]["mean_calls"])
                    for b in lsm_bs]

        ax_acc.plot(lx, lsm_acc, **LSM_STYLE)
        for m, style in (("era", ERA_STYLE), ("gra", GRA_STYLE), ("gera", GERA_STYLE)):
            ax_acc.plot(xs, series[m][0], label=m.upper(), **style)
        ax_acc.set_title(nice, pad=8)
        # span from LSM's floor to the arms' ceiling, the way the cost row scales
        lo = min(min(min(series[m][0]) for m in ("gera", "gra", "era")), min(lsm_acc))
        hi = max(max(series[m][0]) for m in ("gera", "gra", "era"))
        pad = max(1.0, 0.03 * (hi - lo))
        ax_acc.set_ylim(lo - pad, hi + pad)
        ax_acc.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        ax_acc.set_xticks(xs)
        ax_acc.set_xticklabels([str(b) for b in BUDGETS])
        ax_acc.set_xlabel("Budget")
        ax_acc.grid(True, alpha=0.3)

        ax_cost.axhline(bfs_cost, label="BFS", **BFS_STYLE)
        ax_cost.plot(lx, lsm_cost, **LSM_STYLE)
        for m, style in (("era", ERA_STYLE), ("gra", GRA_STYLE), ("gera", GERA_STYLE)):
            ax_cost.plot(xs, series[m][1], **style)
        # start at the data floor, same as the accuracy row
        clo = min(min(series[m][1]) for m in ("gera", "gra", "era"))
        chi = max(bfs_cost, *(max(series[m][1]) for m in ("gera", "gra", "era")),
                  max(lsm_cost, default=0))
        pad = 0.05 * (chi - clo)
        ax_cost.set_ylim(clo - pad, chi + pad)
        ax_cost.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax_cost.grid(True, alpha=0.3)
        ax_cost.set_xticks(xs)
        ax_cost.set_xticklabels([str(b) for b in BUDGETS])
        ax_cost.set_xlabel("Budget")
        if col == 0:
            ax_acc.set_ylabel("Accuracy (%)")
            ax_cost.set_ylabel("Avg Cost (LLM Calls)")

    handles = [Line2D([], [], label="GERA", **GERA_STYLE),
               Line2D([], [], label="GRA", **GRA_STYLE),
               Line2D([], [], label="ERA", **ERA_STYLE),
               Line2D([], [], label="LSM", **LSM_STYLE),
               Line2D([], [], label="BFS", **BFS_STYLE)]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
