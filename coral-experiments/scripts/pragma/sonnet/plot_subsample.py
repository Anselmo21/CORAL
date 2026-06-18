"""Plot BFS / SRA / ERA(budget grid) by effective lattice size — sonnet pragma sub-lattices.

Reads data/sonnet_pragma/results.json (produced by run_sonnet_pragma_subsample.py).
Writes PNGs under plots/sonnet_pragma/:
  accuracy_by_size.png   — two-panel line plot (fixed / pct budgets)
  cost_by_size.png       — two-panel line plot (fixed / pct budgets)
  accuracy_heatmap.png   — accuracy heatmap, rows = configs, cols = sizes
  cost_heatmap.png       — avg cost heatmap, rows = configs, cols = sizes

CLI: conda run -n coral python3 plot_sonnet_pragma_subsample.py
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

RESULTS_PATH = "data/sonnet_pragma/results.json"
OUT_DIR = "plots/sonnet_pragma"

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT:  list[int]  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

REF_COLORS = {"BFS": "black", "SRA": "#d62728"}
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]
_COLORS  = plt.colormaps["tab10"].colors


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


def _budget_panel(
    ax: matplotlib.axes.Axes,
    sizes: list[int],
    by_size: dict[int, list[dict]],
    specs: list[tuple[str, callable]],
    ylabel_kind: str,
    show_sra: bool = True,
    show_bfs: bool = True,
) -> None:
    for i, (label, picker) in enumerate(specs):
        y = [
            _acc(by_size[s], picker) if ylabel_kind == "accuracy"
            else _cost(by_size[s], picker)
            for s in sizes
        ]
        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(sizes, y, f"{marker}-", color=color, linewidth=1.7, markersize=6, label=label)

    if show_bfs:
        if ylabel_kind == "accuracy":
            bfs_y = [100.0] * len(sizes)
        else:
            bfs_y = [
                float(np.mean([r["bfs"]["calls"] for r in by_size[s]]))
                for s in sizes
            ]
        ax.plot(sizes, bfs_y, "o:", color=REF_COLORS["BFS"], linewidth=2,
                markersize=7, label="BFS (baseline)")

    if show_sra:
        if ylabel_kind == "accuracy":
            sra_y = [_acc(by_size[s], lambda r: r.get("sra")) for s in sizes]
        else:
            sra_y = [_cost(by_size[s], lambda r: r.get("sra")) for s in sizes]
        ax.plot(sizes, sra_y, "s--", color=REF_COLORS["SRA"], linewidth=2,
                markersize=7, label="SRA")

    ax.set_xlabel("Number of Available Pragmas")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def _fixed_specs() -> list[tuple[str, callable]]:
    return [
        (f"{b}", lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab))
        for b in ERA_BUDGETS_FIXED
    ]


def _pct_specs() -> list[tuple[str, callable]]:
    return [
        (f"{p}%", lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab))
        for p in ERA_BUDGETS_PCT
    ]


def plot_pair(
    results: list[dict],
    ylabel_kind: str,
    out_path: str,
) -> None:
    by_size = _by_size(results)
    sizes   = sorted(by_size)
    show_sra = ylabel_kind != "accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _budget_panel(axes[0], sizes, by_size, _fixed_specs(), ylabel_kind,
                  show_sra=show_sra)
    axes[0].set_title("Fixed ERA budgets")
    _budget_panel(axes[1], sizes, by_size, _pct_specs(), ylabel_kind,
                  show_sra=show_sra)
    axes[1].set_title("Percent-of-lattice ERA budgets")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_heatmap(results: list[dict], kind: str, out_path: str) -> None:
    by_size = _by_size(results)
    sizes   = sorted(by_size)

    row_specs: list[tuple[str, callable]] = [
        ("SRA", lambda r: r.get("sra"))
    ]
    for b in ERA_BUDGETS_FIXED:
        row_specs.append((f"{b}", lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab)))
    for p in ERA_BUDGETS_PCT:
        row_specs.append((f"{p}%", lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab)))

    matrix = np.zeros((len(row_specs), len(sizes)), dtype=float)
    for i, (_, picker) in enumerate(row_specs):
        for j, s in enumerate(sizes):
            matrix[i, j] = (
                _acc(by_size[s], picker) if kind == "accuracy"
                else _cost(by_size[s], picker)
            )

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_name = "RdYlGn" if kind == "accuracy" else "YlOrRd"
    vmin, vmax = (0.0, 100.0) if kind == "accuracy" else (None, None)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(row_specs)))
    ax.set_yticklabels([r[0] for r in row_specs])
    ax.set_xlabel("Number of Available Pragmas")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = f"{v:.0f}" if kind == "accuracy" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if (kind == "accuracy" and v >= 50) or
                                     (kind == "cost" and v < 0.6 * matrix.max())
                                  else "white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)" if kind == "accuracy" else "Avg cost")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined(results: list[dict], out_dir: str) -> None:
    """Two-panel (accuracy | cost) plots for fixed and pct budgets."""
    by_size = _by_size(results)
    sizes   = sorted(by_size)

    for specs, name in [(_fixed_specs(), "fixed"), (_pct_specs(), "pct")]:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

        # Left: accuracy only (no SRA, no BFS)
        _budget_panel(axes[0], sizes, by_size, specs, "accuracy",
                      show_sra=False, show_bfs=False)

        # Right: cost only (BFS reference, no SRA)
        _budget_panel(axes[1], sizes, by_size, specs, "cost",
                      show_sra=False, show_bfs=True)

        fig.tight_layout()
        out_path = f"{out_dir}/era_combined_{name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RESULTS_PATH) as f:
        results: list[dict] = json.load(f)
    print(f"Loaded {len(results)} cases from {RESULTS_PATH}")

    plot_pair(results, "accuracy", f"{OUT_DIR}/accuracy_by_size.png")
    plot_pair(results, "cost",     f"{OUT_DIR}/cost_by_size.png")
    plot_heatmap(results, "accuracy", f"{OUT_DIR}/accuracy_heatmap.png")
    plot_heatmap(results, "cost",     f"{OUT_DIR}/cost_heatmap.png")
    plot_combined(results, OUT_DIR)


if __name__ == "__main__":
    main()
