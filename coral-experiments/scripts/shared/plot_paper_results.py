"""Plot the paper results from paper_results.json across 4 datasets.

For each dataset, writes a 3-panel figure (accuracy, cost, hop distance) over
the effective lattice size (k=3..10), one line per method. CC methods are
omitted if their column is missing from paper_results.json (so this can run
before CC has merged in).

Outputs:
    plots/<dataset>/combined.png        (3 panels in one figure)

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
    ("msmarco_criteria",        "msmarco_criteria", "MSMARCO criteria"),
    ("curated_criteria",         "curated_criteria",  "Curated criteria"),
    ("msmarco_pragma", "msmarco_pragma",   "MSMARCO pragma"),
    ("curated_pragma",            "curated_pragma",    "Curated pragma"),
]

METHODS = [
    ("bfs",     "BFS",        "o",  "--", "black",   2.0, True),
    ("sra",     "SRA",        "s",  "-",  "#d62728", 1.6, False),
    ("cc_b4",   "LSM (b=4)",  "^",  "-",  "#2ca02c", 2.0, False),
    ("cc_b8",   "LSM (b=8)",  "P",  "-",  "#98df8a", 2.0, False),
    ("era_b4",  "ERA (b=4)",  "X",  "-",  "#9467bd", 2.0, False),
    ("era_b8",  "ERA (b=8)",  "D",  "-",  "#c5b0d5", 2.0, False),
]

OUT_BASE = "plots"


def _load(dataset: str) -> list[dict]:
    with open(f"logged_results/{dataset}/paper_results.json") as f:
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


def _hop_stats(rows_by_size: dict[int, list[dict]], col: str):
    """Per-size mean and std of hop_to_bfs for the bar panel."""
    xs, means, stds = [], [], []
    for size, bucket in rows_by_size.items():
        hops = [r[col].get("hop_to_bfs") for r in bucket if r.get(col)]
        hops = [h for h in hops if h is not None]
        if hops:
            xs.append(size)
            means.append(float(np.mean(hops)))
            stds.append(float(np.std(hops)))
    return xs, means, stds


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
        if kind == "hop":
            bar_methods = [m for m in available if not m[6]]
            width = 0.8 / len(bar_methods)
            for j, (col, lab, mk, ls, color, lw, is_bfs) in enumerate(bar_methods):
                xs, means, stds = _hop_stats(rows_by, col)
                if xs:
                    pos = [x + (j - (len(bar_methods) - 1) / 2) * width for x in xs]
                    yerr = [[min(m, s) for m, s in zip(means, stds)], stds]
                    bars = ax.bar(pos, means, width=width, yerr=yerr, color=color,
                                  error_kw={"elinewidth": 0.8, "capsize": 2}, label=lab)
                    handles_by_label.setdefault(lab, bars)
        else:
            for col, lab, mk, ls, color, lw, is_bfs in available:
                if is_bfs and kind == "acc":
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


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_BASE, exist_ok=True)
    for name, out_name, nice in DATASETS:
        print(f"\n[{nice}]")
        plot_dataset(name, out_name, nice)


if __name__ == "__main__":
    main()
