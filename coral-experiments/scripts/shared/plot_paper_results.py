"""Plot the paper results from paper_results.json across 4 datasets.

For each dataset, writes a 3-panel figure (accuracy, cost, hop distance) over
the effective lattice size (k=3..10), one line per method. CC methods are
omitted if their column is missing from paper_results.json (so this can run
before CC has merged in).

Outputs:
    final_plots/<dataset>/{accuracy,cost,hop}_by_size.png
    final_plots/<dataset>/combined.png        (3 panels in one figure)
    final_plots/headline_acc_vs_cost.png      (aggregate per dataset, all methods)

Usage:
    conda run -n coral python3 -m scripts.shared.plot_paper_results
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

DATASETS = [
    ("msmarco_subsample",        "msmarco_criteria", "MSMARCO criteria"),
    ("sonnet_subsample",         "sonnet_criteria",  "Sonnet criteria"),
    ("16_lattices_cf_layer2to8", "msmarco_pragma",   "MSMARCO pragma"),
    ("sonnet_pragma",            "sonnet_pragma",    "Sonnet pragma"),
]

METHODS = [
    ("bfs",       "BFS",          "o",  "--", "black",   2.0, True),
    ("sra",       "SRA",          "s",  "-",  "#d62728", 1.6, False),
    ("era_b16",   "ERA (b=16)",   "v",  "-",  "#1f77b4", 2.0, False),
    ("era_pct10", "ERA (b=10%)",  "D",  "-",  "#7fbfff", 2.0, False),
    ("cc_b16",    "LSM (b=16)",   "^",  "-",  "#2ca02c", 2.0, False),
    ("cc_pct10",  "LSM (b=10%)",  "P",  "-",  "#98df8a", 2.0, False),
]

OUT_BASE = "final_plots"


def _load(dataset: str) -> list[dict]:
    with open(f"data/{dataset}/paper_results.json") as f:
        return json.load(f)


def _has_method(rows: list[dict], col: str) -> bool:
    return any(col in r and r[col] is not None for r in rows)


def _by_size(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return dict(sorted(out.items()))


def _metric(cells: list[dict], kind: str, is_bfs: bool) -> float | None:
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    if kind == "acc":
        if is_bfs:
            return 100.0
        return 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)
    if kind == "cost":
        if is_bfs:
            return float(np.mean([c["calls"] for c in cells]))
        return float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
    if kind == "hop":
        hops = [c.get("hop_to_bfs") for c in cells]
        hops = [h for h in hops if h is not None]
        return float(np.mean(hops)) if hops else None
    raise ValueError(kind)


def _series(rows_by_size: dict[int, list[dict]], col: str, kind: str, is_bfs: bool):
    xs, ys = [], []
    for size, bucket in rows_by_size.items():
        cells = [r.get(col) for r in bucket]
        v = _metric(cells, kind, is_bfs)
        if v is not None:
            xs.append(size); ys.append(v)
    return xs, ys


def plot_dataset(name: str, out_name: str, nice: str) -> None:
    rows = _load(name)
    rows_by = _by_size(rows)
    available = [(col, lab, mk, ls, c, lw, is_bfs) for col, lab, mk, ls, c, lw, is_bfs in METHODS
                 if _has_method(rows, col)]

    xlabel = ("Number of Available Pragmas" if "pragma" in out_name
              else "Number of Available Criteria")

    out_dir = f"{OUT_BASE}/{out_name}"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    titles = [
        ("Accuracy (%)",       "acc",  axes[0]),
        ("Avg Cost (LLM Calls)", "cost", axes[1]),
        ("Avg. Hop Distance to BFS CF", "hop", axes[2]),
    ]
    handles_by_label: dict[str, object] = {}
    for ylabel, kind, ax in titles:
        for col, lab, mk, ls, color, lw, is_bfs in available:
            if is_bfs and kind in ("acc", "hop"):
                continue
            xs, ys = _series(rows_by, col, kind, is_bfs)
            if xs:
                line, = ax.plot(xs, ys, marker=mk, linestyle=ls, color=color, linewidth=lw,
                                markersize=7, label=lab)
                handles_by_label.setdefault(lab, line)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rows_by.keys()))
        ax.grid(True, alpha=0.3)
        if kind == "acc":
            ax.set_ylim(0, 108)
    ordered_labels = [lab for _col, lab, *_ in METHODS if lab in handles_by_label]
    fig.legend([handles_by_label[lab] for lab in ordered_labels], ordered_labels,
               loc="upper center", ncol=len(ordered_labels),
               bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = f"{out_dir}/combined.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")

    for kind, ylabel in [("acc", "Accuracy (%)"),
                         ("cost", "Avg Cost (LLM Calls)"),
                         ("hop",  "Avg. Hop Distance to BFS CF")]:
        fig, ax = plt.subplots(figsize=(7.5, 3))
        for col, lab, mk, ls, color, lw, is_bfs in available:
            if is_bfs and kind in ("acc", "hop"):
                continue
            xs, ys = _series(rows_by, col, kind, is_bfs)
            if xs:
                ax.plot(xs, ys, marker=mk, linestyle=ls, color=color, linewidth=lw,
                        markersize=7, label=lab)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rows_by.keys()))
        ax.grid(True, alpha=0.3)
        if kind == "acc":
            ax.set_ylim(0, 108)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        fig.tight_layout()
        out = f"{out_dir}/{kind}_by_size.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_headline() -> None:
    """One scatter per method: x = avg cost across all cases, y = accuracy.
    One marker per (dataset, method)."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    dataset_markers = {"msmarco_subsample": "o", "sonnet_subsample": "s",
                       "16_lattices_cf_layer2to8": "^", "sonnet_pragma": "D"}

    for ds, _out, nice in DATASETS:
        rows = _load(ds)
        for col, lab, _mk, _ls, color, _lw, is_bfs in METHODS:
            if not _has_method(rows, col):
                continue
            cells = [r.get(col) for r in rows]
            cells = [c for c in cells if c is not None]
            if not cells:
                continue
            if is_bfs:
                acc = 100.0
                cost = float(np.mean([c["calls"] for c in cells]))
            else:
                acc = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)
                cost = float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
            ax.scatter(cost, acc, marker=dataset_markers[ds], color=color, s=90,
                       edgecolors="black", linewidths=0.5, zorder=3)

    for col, lab, _mk, _ls, color, _lw, is_bfs in METHODS:
        ax.scatter([], [], marker="o", color=color, s=80, label=lab)
    for ds, _out, nice in DATASETS:
        ax.scatter([], [], marker=dataset_markers[ds], color="gray",
                   edgecolors="black", s=80, label=nice)

    ax.set_xlabel("Avg cost (LLM calls + budget)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT_BASE, exist_ok=True)
    out = f"{OUT_BASE}/headline_acc_vs_cost.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n→ {out}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_BASE, exist_ok=True)
    for name, out_name, nice in DATASETS:
        print(f"\n[{nice}]")
        plot_dataset(name, out_name, nice)
    plot_headline()


if __name__ == "__main__":
    main()
