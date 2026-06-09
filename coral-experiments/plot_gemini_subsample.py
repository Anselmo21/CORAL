"""Plot BFS / SRA / ERA(budget grid) by effective lattice size.

Reads data/gemini_subsample/gemini_subsample_results.json (produced by
run_gemini_subsample.py). Writes PNGs under plots/gemini_subsample/:
  * accuracy_by_size.png     — accuracy across sizes for every ERA config (fixed + pct panels)
  * cost_by_size.png         — avg total cost across sizes for every ERA config
  * accuracy_heatmap.png     — accuracy heatmap, rows = ERA budgets, cols = sizes
  * cost_heatmap.png         — avg cost heatmap, rows = ERA budgets, cols = sizes
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

RESULTS_PATH = "data/gemini_subsample/gemini_subsample_results.json"
OUT_DIR = "plots/gemini_subsample"

REF_COLORS = {"BFS": "black", "SRA": "#d62728"}


def _load() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _acc(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m["outcome"] == "correct") / len(ms)


def _cost(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return float(np.mean([m["calls"] + m.get("budget_used", 0) for m in ms]))


def _ref_lines(ax, sizes, by_size, ylabel_kind: str) -> None:
    if ylabel_kind == "accuracy":
        bfs = [100.0 for _ in sizes]
        sra = [_acc(by_size[s], lambda r: r.get("sra")) for s in sizes]
    else:
        bfs = [float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
        sra = [_cost(by_size[s], lambda r: r.get("sra")) for s in sizes]
    ax.plot(sizes, bfs, "o:", color=REF_COLORS["BFS"], linewidth=2, markersize=7,
            label="BFS (baseline)")
    ax.plot(sizes, sra, "s--", color=REF_COLORS["SRA"], linewidth=2, markersize=7,
            label="SRA")


def _budget_panel(ax, sizes, by_size, labels_and_pickers, ylabel_kind: str, title: str,
                  cmap_name: str) -> None:
    cmap = plt.colormaps[cmap_name]
    for i, (label, picker) in enumerate(labels_and_pickers):
        if ylabel_kind == "accuracy":
            y = [_acc(by_size[s], picker) for s in sizes]
        else:
            y = [_cost(by_size[s], picker) for s in sizes]
        color = cmap(0.15 + 0.7 * i / max(1, len(labels_and_pickers) - 1))
        ax.plot(sizes, y, "D-", color=color, linewidth=1.7, markersize=6, label=label)

    _ref_lines(ax, sizes, by_size, ylabel_kind)

    ax.set_xlabel("Effective lattice size (k)")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Avg cost (calls + budget_used)")
    ax.set_xticks(sizes)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, ncol=2)


def plot_pair(d: dict, ylabel_kind: str, out_path: str) -> None:
    results = d["results"]
    by_size = _by_size(results)
    sizes = sorted(by_size)

    fixed_budgets = d["era_budgets_fixed"]
    pct_budgets = d["era_budgets_pct"]

    def picker_fixed(b):
        label = f"era_fixed_{b}"
        return lambda r: r.get("era", {}).get(label)

    def picker_pct(p):
        label = f"era_pct_{p}"
        return lambda r: r.get("era", {}).get(label)

    fixed_specs = [(f"ERA b={b}", picker_fixed(b)) for b in fixed_budgets]
    pct_specs = [(f"ERA {p}%", picker_pct(p)) for p in pct_budgets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _budget_panel(axes[0], sizes, by_size, fixed_specs, ylabel_kind,
                  f"Fixed ERA budgets — {ylabel_kind}", "Blues")
    _budget_panel(axes[1], sizes, by_size, pct_specs, ylabel_kind,
                  f"Percent-of-lattice ERA budgets — {ylabel_kind}", "Purples")

    n_per_size = {s: len(by_size[s]) for s in sizes}
    fig.suptitle(f"BFS vs SRA vs ERA(budget grid) — gemini sub-lattices, "
                 f"n={n_per_size[sizes[0]]} per size",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _heatmap_grid(d: dict, kind: str) -> tuple[list[str], list[int], np.ndarray]:
    """Returns (row_labels, sizes, value_matrix) for ERA configs × sizes.

    Rows are ordered: SRA reference, then 7 fixed budgets ascending, then 5 percent budgets ascending.
    """
    results = d["results"]
    by_size = _by_size(results)
    sizes = sorted(by_size)
    fixed_budgets = d["era_budgets_fixed"]
    pct_budgets = d["era_budgets_pct"]

    row_specs: list[tuple[str, callable]] = [("SRA", lambda r: r.get("sra"))]
    for b in fixed_budgets:
        row_specs.append((f"ERA b={b}",
                          (lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab))))
    for p in pct_budgets:
        row_specs.append((f"ERA {p}%",
                          (lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab))))

    matrix = np.zeros((len(row_specs), len(sizes)), dtype=float)
    for i, (_, picker) in enumerate(row_specs):
        for j, s in enumerate(sizes):
            if kind == "accuracy":
                matrix[i, j] = _acc(by_size[s], picker)
            else:
                matrix[i, j] = _cost(by_size[s], picker)
    labels = [r[0] for r in row_specs]
    return labels, sizes, matrix


def plot_heatmap(d: dict, kind: str, out_path: str) -> None:
    labels, sizes, matrix = _heatmap_grid(d, kind)
    # Also collect BFS reference cost row for the cost heatmap.
    bfs_costs = [float(np.mean([r["bfs"]["calls"] for r in _by_size(d["results"])[s]]))
                 for s in sizes]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap_name = "RdYlGn" if kind == "accuracy" else "YlOrRd"
    vmin, vmax = (0.0, 100.0) if kind == "accuracy" else (None, None)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effective lattice size (k)")

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = f"{v:.0f}" if kind == "accuracy" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="black" if (kind == "accuracy" and v >= 50) or
                                     (kind == "cost" and v < 0.6 * matrix.max())
                                  else "white")

    # BFS reference along the bottom for cost heatmap
    if kind == "cost":
        bfs_row = ", ".join(f"k={s}: {c:.1f}" for s, c in zip(sizes, bfs_costs))
        ax.set_title(f"Avg total cost (calls + budget_used)\nBFS baseline — {bfs_row}",
                     fontsize=12)
        cbar_label = "Avg cost"
    else:
        ax.set_title("Accuracy (%) — BFS baseline is trivially 100 at every k", fontsize=12)
        cbar_label = "Accuracy (%)"

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.suptitle(f"BFS vs SRA vs ERA(budget grid) — gemini sub-lattices, "
                 f"n={d['target_per_size']} per size",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    d = _load()
    print(f"Loaded {d['n_cases']} results.")
    plot_pair(d, "accuracy", f"{OUT_DIR}/accuracy_by_size.png")
    plot_pair(d, "cost", f"{OUT_DIR}/cost_by_size.png")
    plot_heatmap(d, "accuracy", f"{OUT_DIR}/accuracy_heatmap.png")
    plot_heatmap(d, "cost", f"{OUT_DIR}/cost_heatmap.png")


if __name__ == "__main__":
    main()
