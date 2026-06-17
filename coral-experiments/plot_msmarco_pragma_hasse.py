"""Hasse diagrams for the MSMARCO pragma lattices (k=10).

Individual PNGs:  plots/msmarco_pragma/hasse/qid_<qid>.png
Dev grid PNG:     plots/msmarco_pragma/hasse/dev_grid.png  (2×2)

CLI:
  conda run -n coral python3 plot_msmarco_pragma_hasse.py          # test individuals
  conda run -n coral python3 plot_msmarco_pragma_hasse.py --dev    # dev 2×2 grid
  conda run -n coral python3 plot_msmarco_pragma_hasse.py --all    # both
"""
from __future__ import annotations

import argparse
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

DATASET_PATH = "data/msmarco-pragma-final8-with-lattices.json"
SPLIT_PATH   = "data/msmarco_pragma/split.json"
OUT_DIR      = "plots/msmarco_pragma/hasse"

PALETTE = plt.cm.tab10.colors + plt.cm.Set2.colors


def _subset_to_bits(subset: list[str], idx: dict[str, int]) -> int:
    bits = 0
    for c in subset:
        bits |= 1 << idx[c]
    return bits


def _build_layout(
    criteria: list[str],
    rbc: dict[str, list[str]],
) -> tuple[dict, dict, dict, dict, dict]:
    n   = len(criteria)
    idx = {c: i for i, c in enumerate(sorted(criteria))}

    layers: dict[int, list[list[str]]] = defaultdict(list)
    for k in range(n + 1):
        for combo in combinations(criteria, k):
            layers[k].append(list(combo))

    counts: dict[str, int] = defaultdict(int)
    for ranking in rbc.values():
        if ranking:
            counts[ranking[0]] += 1
    answers_sorted = sorted(counts, key=lambda a: -counts[a])
    color_for = {ans: PALETTE[i % len(PALETTE)] for i, ans in enumerate(answers_sorted)}

    positions: dict[str, tuple[float, float]] = {}
    for k in range(n + 1):
        layer_nodes = sorted(layers[k], key=lambda s: _subset_to_bits(s, idx))
        m  = len(layer_nodes)
        xs = np.linspace(-1.0, 1.0, m) if m > 1 else np.array([0.0])
        for x, subset in zip(xs, layer_nodes):
            positions[combo_key(subset)] = (float(x), -float(k))

    return layers, counts, answers_sorted, color_for, positions


def _render_ax(
    ax: plt.Axes,
    qid: str,
    criteria: list[str],
    rbc: dict[str, list[str]],
    fontsize: int = 9,
) -> None:
    n   = len(criteria)
    bfs = bfs_search(criteria, rbc, start_node=list(criteria), direction="down")
    layers, counts, answers_sorted, color_for, positions = _build_layout(criteria, rbc)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for k in range(n):
        for subset in layers[k]:
            s_key = combo_key(subset)
            if s_key not in positions:
                continue
            x1, y1 = positions[s_key]
            for c in criteria:
                if c in set(subset):
                    continue
                p_key = combo_key(sorted(subset + [c]))
                if p_key not in positions:
                    continue
                x2, y2 = positions[p_key]
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]
    ax.plot(edge_x, edge_y, color="lightgray", linewidth=0.15, alpha=0.25, zorder=1)

    xs_nodes, ys_nodes, colors = [], [], []
    for key, (x, y) in positions.items():
        ranking = rbc.get(key) or []
        if not ranking:
            continue
        xs_nodes.append(x)
        ys_nodes.append(y)
        colors.append(color_for[ranking[0]])
    ax.scatter(xs_nodes, ys_nodes, c=colors, s=6, zorder=2, edgecolors="none")

    if bfs["found"] and bfs["key"] in positions:
        cfx, cfy = positions[bfs["key"]]
        ax.scatter([cfx], [cfy], s=200, marker="*", c="red",
                   edgecolors="black", linewidth=1.2, zorder=3)

    for k in range(n + 1):
        ax.text(-1.08, -k, f"L{k}", ha="right", va="center",
                fontsize=fontsize - 2, color="dimgray")

    title = f"qid={qid}"
    if bfs["found"]:
        title += f"  •  BFS CF @ L{bfs['size']}  ({bfs['calls']} calls)"
    ax.set_title(title, fontsize=fontsize, fontweight="bold")
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
            markeredgecolor="black", markersize=12,
            label=f"BFS CF (L{bfs['size']})",
        ))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=fontsize - 2,
              title="Top-1 answer", title_fontsize=fontsize - 1)


def plot_individual(qid: str, criteria: list[str], rbc: dict[str, list[str]]) -> str:
    fig, ax = plt.subplots(figsize=(18, 11))
    _render_ax(ax, qid, criteria, rbc, fontsize=10)

    bfs = bfs_search(criteria, rbc, start_node=list(criteria), direction="down")
    cf_str = bfs["key"] if bfs["found"] else "none"
    ax.set_title(
        f"qid={qid}  —  {len(criteria)} pragmas, {len(rbc)} nodes\n"
        + (f"BFS CF at L{bfs['size']}: {cf_str}" if bfs["found"] else "no CF found"),
        fontsize=10,
    )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"qid_{qid}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_grid(
    entries: list[tuple[str, list[str], dict[str, list[str]]]],
    out_path: str,
    title: str,
    ncols: int = 2,
) -> None:
    nrows = (len(entries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18 * ncols, 12 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, (qid, criteria, rbc) in zip(axes_flat, entries):
        _render_ax(ax, qid, criteria, rbc, fontsize=10)

    for ax in axes_flat[len(entries):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev",  action="store_true", help="Plot dev grid")
    parser.add_argument("--test", action="store_true", help="Plot test individuals")
    parser.add_argument("--all",  action="store_true", help="Plot everything")
    args = parser.parse_args()

    do_dev  = args.dev  or args.all
    do_test = args.test or args.all or (not args.dev and not args.all)

    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    by_qid = {str(r["query_id"]): r for r in dataset}
    os.makedirs(OUT_DIR, exist_ok=True)

    if do_test:
        for qid in split["test"]:
            rec = by_qid.get(qid)
            if rec is None:
                continue
            criteria = rec["lattice"]["config"]["criteria"]
            rbc      = rec["lattice"]["answers"]
            print(f"Plotting test qid={qid} ...", end=" ", flush=True)
            path = plot_individual(qid, criteria, rbc)
            print(f"saved {path}")

    if do_dev:
        entries = []
        for qid in split["dev"]:
            rec = by_qid.get(qid)
            if rec is None:
                continue
            entries.append((qid, rec["lattice"]["config"]["criteria"],
                            rec["lattice"]["answers"]))
        print(f"Plotting dev grid ({len(entries)} lattices) ...")
        plot_grid(entries, os.path.join(OUT_DIR, "dev_grid.png"),
                  "MSMARCO pragma — dev lattices (k=10)")


if __name__ == "__main__":
    main()
