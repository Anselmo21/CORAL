"""Hasse diagrams for the size-10 lattices in
data/baseline_lattice_runs_skeleton_llama31_8b_local.json.

Each lattice has 2^10 = 1024 nodes. The diagram is layered top→bottom by subset
size; nodes are colored by their top-1 answer; the BFS-found counterfactual
(top-down from the full set) is highlighted with a red star. Covering edges are
drawn faintly so they don't dominate.
"""

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

DATA_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"
OUT_DIR = "plots/hasse_size10"


def _subset_to_bits(subset: list[str], idx: dict[str, int], n: int) -> int:
    """Encode a subset as a binary integer using a fixed criterion-index map."""
    bits = 0
    for c in subset:
        bits |= 1 << idx[c]
    return bits


def plot_lattice(run: dict) -> str:
    criteria = sorted(run["criteria_used"])
    rbc: dict[str, list[str]] = run["rankings_by_combo"]
    qid = run["query_id"]
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}

    bfs = bfs_search(criteria, rbc, start_node=list(criteria), direction="down")

    layers: dict[int, list[list[str]]] = defaultdict(list)
    for k in range(n + 1):
        for combo in combinations(criteria, k):
            layers[k].append(list(combo))

    # Distinct top-1 answers; order by frequency so the legend reads naturally.
    counts: dict[str, int] = defaultdict(int)
    for ranking in rbc.values():
        if ranking:
            counts[ranking[0]] += 1
    answers_sorted = sorted(counts, key=lambda a: -counts[a])
    palette = plt.cm.tab10.colors + plt.cm.Set2.colors
    color_for = {ans: palette[i % len(palette)] for i, ans in enumerate(answers_sorted)}

    # Layout: within each layer sort by binary encoding so adjacent nodes in the
    # plot are adjacent in the boolean lattice — this surfaces color-region
    # structure without the eye-melting random shuffle.
    positions: dict[str, tuple[float, float]] = {}
    for k in range(n + 1):
        layer_nodes = layers[k]
        layer_nodes.sort(key=lambda s: _subset_to_bits(s, idx, n))
        m = len(layer_nodes)
        xs = np.linspace(-1.0, 1.0, m) if m > 1 else np.array([0.0])
        for x, subset in zip(xs, layer_nodes):
            positions[combo_key(subset)] = (float(x), -float(k))

    fig, ax = plt.subplots(figsize=(18, 11))

    # Faint covering edges (child → parent, parent = child ∪ {c}).
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
    ax.plot(edge_x, edge_y, color="lightgray", linewidth=0.15, alpha=0.25, zorder=1)

    # Nodes colored by top-1.
    xs: list[float] = []
    ys: list[float] = []
    colors: list = []
    for key, (x, y) in positions.items():
        ranking = rbc.get(key) or []
        if not ranking:
            continue
        xs.append(x)
        ys.append(y)
        colors.append(color_for[ranking[0]])
    ax.scatter(xs, ys, c=colors, s=10, zorder=2, edgecolors="none")

    # BFS CF marker.
    if bfs["found"] and bfs["key"] in positions:
        cfx, cfy = positions[bfs["key"]]
        ax.scatter([cfx], [cfy], s=260, marker="*", c="red",
                   edgecolors="black", linewidth=1.5, zorder=3)

    # Layer labels on the left.
    for k in range(n + 1):
        ax.text(-1.08, -k, f"L{k}  ({len(layers[k])})",
                ha="right", va="center", fontsize=9, color="dimgray")

    n_distinct = len(answers_sorted)
    title = (f"qid={qid}  —  n={n} criteria, {len(rbc)} nodes, "
             f"{n_distinct} distinct top-1 answer{'s' if n_distinct != 1 else ''}")
    if bfs["found"]:
        title += f"  •  BFS CF at layer {bfs['size']}"
    else:
        title += "  •  no CF found by BFS"
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1.18, 1.05)
    ax.set_ylim(-n - 0.5, 0.6)
    ax.axis("off")

    legend_handles: list = [
        mpatches.Patch(color=color_for[a], label=f"{a}  ({counts[a]})")
        for a in answers_sorted
    ]
    if bfs["found"]:
        legend_handles.append(Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="red",
            markeredgecolor="black", markersize=14,
            label=f"BFS CF (L{bfs['size']})",
        ))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              title="Top-1 answer (node count)", title_fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"qid_{qid}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DATA_PATH) as f:
        data = json.load(f)
    size10 = [r for r in data["runs"] if len(r["criteria_used"]) == 10]
    print(f"Plotting {len(size10)} size-10 lattices to {OUT_DIR}/")
    for run in size10:
        path = plot_lattice(run)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
