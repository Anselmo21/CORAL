"""Visualize the size-10 Hasse lattices from data/gemini_datasets.json.

Each query's lattice has 2**10 = 1024 nodes — too many to draw with edges.
Instead we show the *counterfactual structure* of each lattice:

  * lattice_cf_heatmap.png — 59 rows (one per query) × 11 columns (one per layer).
    Cell color = fraction of nodes at that layer whose top-1 answer differs
    from the full-set baseline (i.e. fraction of "CF nodes"). BFS's minimal-CF
    layer for each query is marked with a black box.

  * lattice_hasse_grid.png — a small-multiples grid (one mini-panel per query)
    where each panel is a stacked horizontal-bar Hasse: layer 10 at the top
    (full set, never a CF), layer 0 at the bottom (empty set). Bar width per
    layer = C(10, layer); green segment = non-CF nodes, red segment = CF nodes.
    Title per panel = qid + BFS-found minimal-CF layer.

Cost-of-search context: the lower the BFS-CF layer, the harder the search
(more removals required to flip the answer).
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style
from search.lattice import combo_key

DATA_PATH = "data/gemini_datasets.json"
EVAL_PATH = "data/gemini_eval/gemini_eval_results.json"
OUT_DIR = "plots/gemini_lattices"

N_LAYERS = 11  # 0..10 inclusive (since size-10 lattice)
LAYER_SIZES = [math.comb(10, k) for k in range(N_LAYERS)]


def _load_lattices() -> dict[str, dict[str, list[str]]]:
    """Returns {qid: rankings_by_combo} for non-halted queries."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    out: dict[str, dict[str, list[str]]] = {}
    for qid, q in data.items():
        if qid == "_meta" or q.get("halted"):
            continue
        rbc: dict[str, list[str]] = {}
        for k, v in q["raw_answers"].items():
            parts = [p.strip() for p in k.split(",") if p.strip()]
            rbc[combo_key(parts)] = v
        if combo_key(q["criteria"]) in rbc:
            out[qid] = rbc
    return out


def _load_bfs_layers() -> dict[str, int]:
    """Returns {qid: bfs_layer} from the size-10 eval results."""
    try:
        with open(EVAL_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    return {r["query_id"]: r["bfs"]["layer"] for r in data["results"] if r["bfs"]["found"]}


def _load_bfs_calls() -> dict[str, int]:
    """Returns {qid: bfs_calls} from the size-10 eval results."""
    try:
        with open(EVAL_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    return {r["query_id"]: r["bfs"]["calls"] for r in data["results"] if r["bfs"]["found"]}


def _cf_counts_per_layer(rbc: dict[str, list[str]]) -> tuple[list[int], list[int]]:
    """Returns (cf_counts, total_counts) indexed by layer 0..10."""
    baseline_top1 = rbc[combo_key(_full_criteria_from_keys(rbc))][0]
    cf = [0] * N_LAYERS
    tot = [0] * N_LAYERS
    for key, ranking in rbc.items():
        layer = 0 if key == "" else key.count(",") + 1
        if layer >= N_LAYERS:
            continue
        tot[layer] += 1
        if ranking and ranking[0] != baseline_top1:
            cf[layer] += 1
    return cf, tot


def _full_criteria_from_keys(rbc: dict[str, list[str]]) -> list[str]:
    """Recover the full criteria list from the longest key in rbc."""
    longest = max(rbc.keys(), key=lambda k: len(k.split(",")) if k else 0)
    return [p.strip() for p in longest.split(",") if p.strip()]


def plot_cf_heatmap(out_path: str) -> None:
    lattices = _load_lattices()
    bfs_layers = _load_bfs_layers()
    qids = sorted(lattices.keys(), key=int)

    matrix = np.full((len(qids), N_LAYERS), np.nan)
    for i, qid in enumerate(qids):
        cf, tot = _cf_counts_per_layer(lattices[qid])
        for layer in range(N_LAYERS):
            matrix[i, layer] = (cf[layer] / tot[layer]) if tot[layer] else np.nan

    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    # Mark BFS minimal-CF layer per row.
    for i, qid in enumerate(qids):
        bfs_layer = bfs_layers.get(qid)
        if bfs_layer is not None and 0 <= bfs_layer < N_LAYERS:
            ax.add_patch(patches.Rectangle((bfs_layer - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="black", linewidth=1.8))

    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"{k}\n({LAYER_SIZES[k]})" for k in range(N_LAYERS)])
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels([f"qid={q}" for q in qids], fontsize=8)
    ax.set_xlabel("Lattice layer (criteria kept) — node count in parens")
    ax.set_ylabel("Query")
    ax.set_title(
        "CF fraction per layer — gemini size-10 lattices (n=59 queries)\n"
        "Black box = BFS minimal-CF layer (the shallowest layer with any CF reachable from layer 10)",
        fontsize=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("CF fraction at layer  (red = many nodes flipped, green = baseline still wins)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_hasse_grid(out_path: str, ncols: int = 6) -> None:
    """Stacked-bar Hasse per query in a small-multiples grid."""
    lattices = _load_lattices()
    bfs_layers = _load_bfs_layers()
    bfs_calls = _load_bfs_calls()
    qids = sorted(lattices.keys(), key=int)

    nrows = math.ceil(len(qids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 1.7 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    layers = list(range(N_LAYERS))
    cf_color = "#d62728"
    non_color = "#2ca02c"

    for idx, qid in enumerate(qids):
        ax = axes[idx // ncols, idx % ncols]
        cf, tot = _cf_counts_per_layer(lattices[qid])
        widths_cf = np.array(cf, dtype=float)
        widths_non = np.array(tot, dtype=float) - widths_cf
        norm = max(LAYER_SIZES) or 1
        # Center each bar around x=0 so wide layers look wider.
        ax.barh(layers, widths_non / norm, left=-(widths_non + widths_cf) / (2 * norm),
                color=non_color, edgecolor="none", height=0.9)
        ax.barh(layers, widths_cf / norm,
                left=(-(widths_non + widths_cf) / (2 * norm)) + widths_non / norm,
                color=cf_color, edgecolor="none", height=0.9)

        bfs_layer = bfs_layers.get(qid)
        bfs_call_n = bfs_calls.get(qid)
        title = f"qid={qid}"
        if bfs_layer is not None:
            title += f"  L={bfs_layer}, {bfs_call_n} calls" if bfs_call_n is not None else f"  L={bfs_layer}"
            ax.axhline(bfs_layer, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([str(k) if k % 2 == 0 else "" for k in range(N_LAYERS)],
                           fontsize=7)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)

    # Hide unused panels.
    for j in range(len(qids), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    # Single legend
    fig.legend(
        handles=[
            patches.Patch(color=non_color, label="baseline still top-1 (non-CF)"),
            patches.Patch(color=cf_color, label="answer flipped (CF)"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=11,
    )
    fig.suptitle(
        "Size-10 Hasse lattices — each panel = one query, y-axis = layer (10=top, 0=bottom)\n"
        "Bar width ∝ # nodes in layer; dashed line = BFS minimal-CF layer; "
        "title L = BFS-CF layer, calls = BFS ground-truth lookups",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_cf_heatmap(f"{OUT_DIR}/lattice_cf_heatmap.png")
    plot_hasse_grid(f"{OUT_DIR}/lattice_hasse_grid.png")


if __name__ == "__main__":
    main()
