"""Hasse diagrams for all 16 lattices in data/msmarco_llama_size10.json.

Produces a single 4×4 grid PNG. Each subplot shows the 1024-node lattice
layered top→bottom by subset size, nodes colored by top-1 answer, BFS CF
marked with a red star, and ERA(budget=16) accuracy annotated in the title.

Output: plots/msmarco_hasse/all_lattices.png
CLI: conda run -n coral python3 plot_msmarco_hasse.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from search.bfs import bfs_search
from search.lattice import combo_key

DATA_PATH = "data/msmarco_llama_size10.json"
ERA_RESULTS_PATH = "data/msmarco_eval/results.json"
OUT_DIR = "plots/msmarco_hasse"
OUT_PATH = "plots/msmarco_hasse/all_lattices.png"

OUTCOME_COLOR = {
    "correct":        "#2ca02c",
    "wrong_layer":    "#ff7f0e",
    "false_positive": "#d62728",
    "not_found":      "#aaaaaa",
}
OUTCOME_LABEL = {
    "correct":        "ERA ✓",
    "wrong_layer":    "ERA wrong layer",
    "false_positive": "ERA false positive",
    "not_found":      "ERA not found",
}


def _subset_to_bits(subset: list[str], idx: dict[str, int]) -> int:
    bits = 0
    for c in subset:
        bits |= 1 << idx[c]
    return bits


def draw_lattice(
    ax: plt.Axes,
    qid: str,
    rec: dict,
    era_result: dict | None,
) -> None:
    criteria = sorted(rec["criteria"])
    rankings: dict[str, list[str]] = rec["rankings"]
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}

    bfs = bfs_search(criteria, rankings, start_node=list(criteria), direction="down")

    layers: dict[int, list[list[str]]] = defaultdict(list)
    for k in range(n + 1):
        for combo in combinations(criteria, k):
            layers[k].append(list(combo))

    counts: dict[str, int] = defaultdict(int)
    for ranking in rankings.values():
        if ranking:
            counts[ranking[0]] += 1
    answers_sorted = sorted(counts, key=lambda a: -counts[a])
    palette = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    color_for = {ans: palette[i % len(palette)] for i, ans in enumerate(answers_sorted)}

    positions: dict[str, tuple[float, float]] = {}
    for k in range(n + 1):
        layer_nodes = sorted(layers[k], key=lambda s: _subset_to_bits(s, idx))
        m = len(layer_nodes)
        xs = np.linspace(-1.0, 1.0, m) if m > 1 else np.array([0.0])
        for x, subset in zip(xs, layer_nodes):
            positions[combo_key(subset)] = (float(x), -float(k))

    # Covering edges
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for k in range(n):
        for subset in layers[k]:
            s_key = combo_key(subset)
            if s_key not in positions:
                continue
            x1, y1 = positions[s_key]
            present = set(subset)
            for c in criteria:
                if c in present:
                    continue
                p_key = combo_key(sorted(subset + [c]))
                if p_key not in positions:
                    continue
                x2, y2 = positions[p_key]
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]
    ax.plot(edge_x, edge_y, color="lightgray", linewidth=0.1, alpha=0.2, zorder=1)

    # Nodes
    xs_list, ys_list, colors = [], [], []
    for key, (x, y) in positions.items():
        ranking = rankings.get(key) or []
        if not ranking:
            continue
        xs_list.append(x)
        ys_list.append(y)
        colors.append(color_for[ranking[0]])
    ax.scatter(xs_list, ys_list, c=colors, s=4, zorder=2, edgecolors="none")

    # BFS CF star
    if bfs["found"] and bfs["key"] in positions:
        cfx, cfy = positions[bfs["key"]]
        ax.scatter([cfx], [cfy], s=120, marker="*", c="red",
                   edgecolors="black", linewidth=0.8, zorder=3)

    # ERA CF marker (blue diamond) if different from BFS
    if era_result:
        era_key = era_result.get("proxy_key")
        if era_key and era_key in positions and era_key != (bfs["key"] if bfs["found"] else None):
            ex, ey = positions[era_key]
            ax.scatter([ex], [ey], s=80, marker="D", c="#1f77b4",
                       edgecolors="black", linewidth=0.6, zorder=3)

    ax.set_xlim(-1.12, 1.05)
    ax.set_ylim(-n - 0.5, 0.5)
    ax.axis("off")

    # Title: qid + BFS CF layer + ERA outcome
    era_outcome = era_result["outcome"] if era_result else "n/a"
    era_color = OUTCOME_COLOR.get(era_outcome, "gray")
    era_tag = OUTCOME_LABEL.get(era_outcome, era_outcome)
    bfs_tag = f"BFS L{bfs['size']}" if bfs["found"] else "no BFS CF"
    ax.set_title(f"qid {qid}  |  {bfs_tag}  |  {era_tag}",
                 fontsize=7, color=era_color, fontweight="bold", pad=3)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    lattices = data["lattices"]

    with open(ERA_RESULTS_PATH) as f:
        era_data = json.load(f)
    era_by_qid: dict[str, dict] = {
        str(r["query_id"]): r["era"] for r in era_data["results"]
    }

    qids = sorted(lattices.keys())
    n_cols = 4
    n_rows = (len(qids) + n_cols - 1) // n_cols  # 4 rows for 16 lattices

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 6))
    axes_flat = axes.flatten()

    for i, qid in enumerate(qids):
        draw_lattice(axes_flat[i], qid, lattices[qid], era_by_qid.get(qid))

    # Hide unused subplots
    for j in range(len(qids), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend
    era_budget = era_data["era_budget"]
    legend_handles = [
        mpatches.Patch(color=c, label=OUTCOME_LABEL[o])
        for o, c in OUTCOME_COLOR.items()
    ]
    legend_handles += [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=10, label="BFS CF"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=7, label=f"ERA CF (b={era_budget}, if different)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"MS MARCO size-10 lattices — Llama-3.1-8B  |  ERA budget={era_budget}  |  n={len(qids)}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
