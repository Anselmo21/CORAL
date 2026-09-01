"""Plot the SIGNED layer-size gap (proxy_layer - bfs_layer) per dataset.

Mirrors the hop-distance panel from plot_paper_results.py, but preserves sign:
positive means proxy stopped too shallow (larger CF than BFS's minimal),
negative means proxy went too deep (smaller layer than BFS).

Outputs:
    temp_plots/<dataset>/signed_gap.png

Usage:
    conda run -n coral python3 -m scripts.shared.plot_signed_gap
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
    ("msmarco_criteria",  "MSMARCO criteria"),
    ("curated_criteria",  "Curated criteria"),
    ("msmarco_pragma",    "MSMARCO pragma"),
    ("curated_pragma",    "Curated pragma"),
]

METHODS = [
    ("sra",    "SRA",       "#d62728"),
    ("cc_b4",  "LSM (b=4)", "#2ca02c"),
    ("cc_b8",  "LSM (b=8)", "#98df8a"),
    ("era_b4", "ERA (b=4)", "#9467bd"),
    ("era_b8", "ERA (b=8)", "#c5b0d5"),
]

OUT_BASE = "temp_plots"


def _load(dataset: str) -> list[dict]:
    with open(f"logged_results/llama31-8b/{dataset}/paper_results.json") as f:
        return json.load(f)


def _has_method(rows: list[dict], col: str) -> bool:
    return any(col in r and r[col] is not None for r in rows)


def _by_size(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return dict(sorted(out.items()))


def _signed_diffs(bucket: list[dict], col: str) -> list[int]:
    diffs = []
    for r in bucket:
        cell = r.get(col)
        bfs  = r.get("bfs")
        if not cell or not bfs:
            continue
        p = cell.get("proxy_layer")
        b = bfs.get("layer")
        if p is None or b is None:
            continue
        diffs.append(p - b)
    return diffs


def _stats(rows_by_size: dict[int, list[dict]], col: str):
    xs, means, stds = [], [], []
    for size, bucket in rows_by_size.items():
        d = _signed_diffs(bucket, col)
        if d:
            xs.append(size)
            means.append(float(np.mean(d)))
            stds.append(float(np.std(d)))
    return xs, means, stds


def plot_dataset(name: str, nice: str) -> None:
    rows = _load(name)
    rows_by = _by_size(rows)
    available = [(col, lab, c) for col, lab, c in METHODS if _has_method(rows, col)]

    xlabel = ("Number of Available Pragmas" if "pragma" in name
              else "Number of Available Criteria")

    out_dir = f"{OUT_BASE}/{name}"
    os.makedirs(out_dir, exist_ok=True)

    # 2 panels: (a) mean signed gap vs size (grouped bars),
    #           (b) overall signed-gap distribution histogram per method.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    ax_bar, ax_hist = axes

    # --- panel (a): grouped bars of mean signed gap by size ---
    width = 0.8 / max(1, len(available))
    handles_by_label: dict[str, object] = {}
    for j, (col, lab, color) in enumerate(available):
        xs, means, stds = _stats(rows_by, col)
        if not xs:
            continue
        pos = [x + (j - (len(available) - 1) / 2) * width for x in xs]
        bars = ax_bar.bar(pos, means, width=width, yerr=stds, color=color,
                          error_kw={"elinewidth": 0.8, "capsize": 2}, label=lab)
        handles_by_label.setdefault(lab, bars)
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel(xlabel)
    ax_bar.set_ylabel("Mean signed layer gap\n(proxy − BFS)")
    ax_bar.set_xticks(sorted(rows_by.keys()))
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_title("Mean per size (± std)")

    # --- panel (b): histogram of all signed diffs across cases ---
    all_diffs = {lab: [] for _, lab, _ in available}
    for size, bucket in rows_by.items():
        for col, lab, _ in available:
            all_diffs[lab].extend(_signed_diffs(bucket, col))
    lo = min((min(v) for v in all_diffs.values() if v), default=0)
    hi = max((max(v) for v in all_diffs.values() if v), default=0)
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    for col, lab, color in available:
        v = all_diffs[lab]
        if v:
            ax_hist.hist(v, bins=bins, alpha=0.5, color=color, label=lab,
                         edgecolor="black", linewidth=0.4)
    ax_hist.axvline(0, color="black", linewidth=0.8)
    ax_hist.set_xlabel("Signed layer gap (proxy − BFS)")
    ax_hist.set_ylabel("Cases")
    ax_hist.grid(True, alpha=0.3, axis="y")
    ax_hist.set_title("Distribution across all cases")

    ordered = [lab for _c, lab, _col in METHODS if lab in handles_by_label]
    fig.legend([handles_by_label[lab] for lab in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=True)
    fig.suptitle(f"{nice} — signed layer gap", y=1.06, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = f"{out_dir}/signed_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_BASE, exist_ok=True)
    for name, nice in DATASETS:
        print(f"\n[{nice}]")
        plot_dataset(name, nice)


if __name__ == "__main__":
    main()
