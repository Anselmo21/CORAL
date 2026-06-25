"""Hasse diagram for sonnet_pragma qid=6 (k=10).

Same layout as scripts/cross_dataset/plot_size10_hasse.py — 1024 nodes layered
top→bottom by subset size, colored by top-1 answer, BFS CF marked with a red
star. Source: data/sonnet-pragma-dataset.json (via _load_pragma_lattices).

Output: reports/era_pragma_pruning/figures/qid6_hasse.png
"""
from __future__ import annotations

import os
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.pragma.sonnet.run_subsample import _load_pragma_lattices
from search.bfs import bfs_search
from search.lattice import combo_key

OUT_DIR = "reports/era_pragma_pruning/figures"
QID     = "6"


def _subset_to_bits(subset: list[str], idx: dict[str, int], n: int) -> int:
    bits = 0
    for c in subset:
        bits |= 1 << idx[c]
    return bits


def _short(label: str, n_chars: int = 60) -> str:
    """Trim the long Sonnet-pragma answer text to fit the legend."""
    return label if len(label) <= n_chars else label[:n_chars - 1] + "…"


def plot_lattice(criteria: list[str], rbc: dict[str, list[str]], qid: str, out_path: str) -> None:
    criteria = sorted(criteria)
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}

    bfs = bfs_search(criteria, rbc, start_node=list(criteria), direction="down")

    layers: dict[int, list[list[str]]] = defaultdict(list)
    for k in range(n + 1):
        for combo in combinations(criteria, k):
            layers[k].append(list(combo))

    counts: dict[str, int] = defaultdict(int)
    for ranking in rbc.values():
        if ranking:
            counts[ranking[0]] += 1
    answers_sorted = sorted(counts, key=lambda a: -counts[a])
    palette = plt.cm.tab10.colors + plt.cm.Set2.colors
    color_for = {ans: palette[i % len(palette)] for i, ans in enumerate(answers_sorted)}

    positions: dict[str, tuple[float, float]] = {}
    for k in range(n + 1):
        layer_nodes = layers[k]
        layer_nodes.sort(key=lambda s: _subset_to_bits(s, idx, n))
        m = len(layer_nodes)
        xs = np.linspace(-1.0, 1.0, m) if m > 1 else np.array([0.0])
        for x, subset in zip(xs, layer_nodes):
            positions[combo_key(subset)] = (float(x), -float(k))

    fig, ax = plt.subplots(figsize=(18, 11))

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

    xs: list[float] = []
    ys: list[float] = []
    colors: list = []
    for key, (x, y) in positions.items():
        ranking = rbc.get(key) or []
        if not ranking:
            continue
        xs.append(x); ys.append(y)
        colors.append(color_for[ranking[0]])
    ax.scatter(xs, ys, c=colors, s=10, zorder=2, edgecolors="none")

    if bfs["found"] and bfs["key"] in positions:
        cfx, cfy = positions[bfs["key"]]
        ax.scatter([cfx], [cfy], s=300, marker="*", c="red",
                   edgecolors="black", linewidth=1.5, zorder=3)

    for k in range(n + 1):
        ax.text(-1.08, -k, f"L{k}  ({len(layers[k])})",
                ha="right", va="center", fontsize=9, color="dimgray")

    title = (f"Sonnet pragma qid={qid} — n={n} pragmas, {len(rbc)} nodes, "
             f"{len(answers_sorted)} distinct top-1 answer(s)")
    if bfs["found"]:
        title += f"  •  BFS CF at L{bfs['size']}  ({bfs['calls']} calls)"
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1.18, 1.05)
    ax.set_ylim(-n - 0.5, 0.6)
    ax.axis("off")

    legend_handles: list = [
        mpatches.Patch(color=color_for[a], label=f"{_short(a)}  ({counts[a]})")
        for a in answers_sorted
    ]
    if bfs["found"]:
        legend_handles.append(Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="red",
            markeredgecolor="black", markersize=14,
            label=f"BFS CF (L{bfs['size']})",
        ))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              title="Top-1 answer (node count)", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lats = _load_pragma_lattices([QID])
    criteria, rbc = lats[QID]
    out = f"{OUT_DIR}/qid6_hasse.png"
    plot_lattice(criteria, rbc, QID, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
