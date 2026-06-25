"""4×4 grid Hasse diagrams for all 16 lattices in each of the two datasets.

Outputs:
  plots/hasse_msmarco_grid.png  — 16 MSMARCO lattices (data/sources/msmarco_criteria.json)
  plots/hasse_sonnet_grid.png   — 16 Sonnet lattices  (data/sources/sonnet_criteria.json)

Each cell: nodes colored by top-1 answer, BFS CF marked with a red star.
Covering edges are omitted at this scale (1024 nodes per cell, 16 cells).

CLI: conda run -n coral python3 plot_dataset_hasse_grids.py
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

from search.bfs import bfs_search
from search.lattice import combo_key

OUT_DIR = "plots"
MSMARCO_PATH = "data/sources/msmarco_criteria.json"
SONNET_PATH = "data/sources/sonnet_criteria.json"

GRID_ROWS, GRID_COLS = 4, 4
CELL_W, CELL_H = 4.5, 6.5  # inches per cell
PALETTE = plt.cm.tab10.colors + plt.cm.Set2.colors


# ── Sonnet key normalisation (criteria names can contain ", ") ─────────────

def _parse_combo_key(raw_key: str, sorted_criteria: list[str]) -> list[str]:
    if not raw_key:
        return []
    result: list[str] = []
    remaining = raw_key
    for c in sorted_criteria:
        if remaining == c:
            result.append(c)
            remaining = ""
            break
        if remaining.startswith(c + ", "):
            result.append(c)
            remaining = remaining[len(c) + 2:]
    return result if not remaining else []


def _sonnet_rbc(q: dict) -> tuple[list[str], dict[str, list[str]]] | None:
    if q.get("halted"):
        return None
    raw = q.get("raw_answers") or {}
    criteria = list(q["criteria"])
    sorted_crit = sorted(criteria)
    rbc: dict[str, list[str]] = {}
    for k, v in raw.items():
        parts = _parse_combo_key(k, sorted_crit)
        rbc[combo_key(parts)] = v
    if combo_key(criteria) not in rbc:
        return None
    return criteria, rbc


# ── Per-cell rendering ────────────────────────────────────────────────────

def _render_cell(
    ax: plt.Axes,
    qid: str,
    criteria: list[str],
    rbc: dict[str, list[str]],
) -> None:
    n = len(criteria)
    sorted_crit = sorted(criteria)
    idx = {c: i for i, c in enumerate(sorted_crit)}

    bfs = bfs_search(criteria, rbc, start_node=list(criteria), direction="down")

    # Layer → node lists (sorted by binary encoding for structured layout)
    layers: dict[int, list[list[str]]] = defaultdict(list)
    for k in range(n + 1):
        layer = sorted(combinations(sorted_crit, k),
                       key=lambda s: sum(1 << idx[c] for c in s))
        layers[k] = [list(combo) for combo in layer]

    # Answer colours
    counts: dict[str, int] = defaultdict(int)
    for ranking in rbc.values():
        if ranking:
            counts[ranking[0]] += 1
    answers_sorted = sorted(counts, key=lambda a: -counts[a])
    color_for = {ans: PALETTE[i % len(PALETTE)] for i, ans in enumerate(answers_sorted)}

    # Node positions
    positions: dict[str, tuple[float, float]] = {}
    for k in range(n + 1):
        nodes = layers[k]
        m = len(nodes)
        xs = np.linspace(0.0, 1.0, m) if m > 1 else np.array([0.5])
        for x, subset in zip(xs, nodes):
            positions[combo_key(subset)] = (float(x), float(n - k))

    # Scatter nodes
    px, py, pc = [], [], []
    for key, (x, y) in positions.items():
        ranking = rbc.get(key)
        if not ranking:
            continue
        px.append(x)
        py.append(y)
        pc.append(color_for[ranking[0]])
    ax.scatter(px, py, c=pc, s=3, zorder=2, edgecolors="none", rasterized=True)

    # BFS CF star
    if bfs["found"] and bfs["key"] in positions:
        cfx, cfy = positions[bfs["key"]]
        ax.scatter([cfx], [cfy], s=120, marker="*", c="red",
                   edgecolors="black", linewidth=0.8, zorder=3)

    # Subtle layer lines
    for k in range(n + 1):
        ax.axhline(float(n - k), color="lightgray", linewidth=0.3, zorder=0)

    # Colour legend (compact, right side)
    handles = [mpatches.Patch(color=color_for[a], label=a[:22])
               for a in answers_sorted]
    if bfs["found"]:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="*", color="w",
                               markerfacecolor="red", markeredgecolor="black",
                               markersize=7, label=f"CF L{bfs['size']}"))
    ax.legend(handles=handles, fontsize=4.5, loc="lower right",
              framealpha=0.7, handlelength=1, borderpad=0.4)

    cf_note = f"  CF→L{bfs['size']}" if bfs["found"] else "  no CF"
    ax.set_title(f"qid {qid}{cf_note}", fontsize=7, pad=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, n + 0.5)
    ax.axis("off")


# ── Grid figure ───────────────────────────────────────────────────────────

def plot_grid(
    records: list[tuple[str, list[str], dict[str, list[str]]]],
    title: str,
    out_path: str,
) -> None:
    n_cells = len(records)
    rows = GRID_ROWS
    cols = GRID_COLS
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * CELL_W, rows * CELL_H))
    axes_flat = axes.flatten()

    for i, (qid, criteria, rbc) in enumerate(records):
        print(f"  [{i+1}/{n_cells}] qid={qid} ...", flush=True)
        _render_cell(axes_flat[i], qid, criteria, rbc)

    # Hide unused cells
    for j in range(n_cells, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=14, y=1.002)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Loaders ───────────────────────────────────────────────────────────────

def load_msmarco() -> list[tuple[str, list[str], dict[str, list[str]]]]:
    with open(MSMARCO_PATH) as f:
        data = json.load(f)
    records = []
    for qid, rec in data["lattices"].items():
        records.append((qid, rec["criteria"], rec["rankings"]))
    records.sort(key=lambda r: r[0])
    return records


def load_sonnet() -> list[tuple[str, list[str], dict[str, list[str]]]]:
    with open(SONNET_PATH) as f:
        data = json.load(f)
    records = []
    for qid, rec in data.items():
        if qid == "_meta":
            continue
        result = _sonnet_rbc(rec)
        if result is None:
            print(f"  Skipping qid={qid} (halted or missing start key)")
            continue
        criteria, rbc = result
        records.append((qid, criteria, rbc))
    records.sort(key=lambda r: r[0])
    return records


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading MSMARCO lattices ...")
    msmarco = load_msmarco()
    print(f"  {len(msmarco)} lattices")
    plot_grid(msmarco, "MSMARCO — 16 lattices (Llama-3.1-8B, 10 criteria)",
              f"{OUT_DIR}/hasse_msmarco_grid.png")

    print("\nLoading Sonnet lattices ...")
    sonnet = load_sonnet()
    print(f"  {len(sonnet)} lattices")
    plot_grid(sonnet, "Sonnet — 16 lattices (Claude Sonnet, 10 criteria)",
              f"{OUT_DIR}/hasse_sonnet_grid.png")


if __name__ == "__main__":
    main()
