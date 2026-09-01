"""Bar chart of accuracy, cost, and hop distance per lattice size across 4 llama31 datasets.

No confidence intervals — just per-size means. One bar per method per size.

Output: artifacts/llama31_bars_by_size.png

Usage:
    conda run -n coral python3 -m scripts.shared.plot_llama_bars_by_size
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

DATASETS = [
    ("msmarco_criteria", "MSMARCO\ncriteria"),
    ("msmarco_pragma",   "MSMARCO\npragma"),
    ("curated_criteria",  "Curated\ncriteria"),
    ("curated_pragma",    "Curated\npragma"),
]

METHODS = [
    ("bfs",            "BFS",        "black",   True),
    ("sra",            "SRA",        "#d62728", False),
    ("cc_b16",         "LSM (b=16)", "#2ca02c", False),
    ("cc_pct10",       "LSM (10%)",  "#98df8a", False),
    ("mra_softmax_b4", "ERA (b=4)",  "#9467bd", False),
    ("mra_softmax_b8", "ERA (b=8)",  "#c5b0d5", False),
]

METRICS = [
    ("acc",  "Accuracy (%)"),
    ("cost", "Avg Cost (LLM Calls)"),
    ("hop",  "Avg Hop Distance\n(incorrect cases only)"),
]


def _load(dataset: str) -> dict[int, list[dict]]:
    rows = json.load(open(f"logged_results/llama31-8b/{dataset}/paper_results.json"))
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_size[r["size"]].append(r)
    return dict(sorted(by_size.items()))


def _wilson95(successes: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), 0.0, 0.0
    p = successes / n
    z = 1.96
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p * 100, (centre - half) * 100, (centre + half) * 100


def _mean_ci95(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0.0
    m = float(np.mean(vals))
    if n == 1:
        return m, m, m
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    return m, m - 1.96 * se, m + 1.96 * se


def _cell_stats(rows: list[dict], col: str, is_bfs: bool, metric: str) -> tuple[float, float, float]:
    vals, hits, n = [], 0, 0
    for r in rows:
        if is_bfs:
            cell = r.get("bfs")
            if not cell:
                continue
            if metric == "acc":
                hits += 1; n += 1
            elif metric == "cost":
                vals.append(cell["calls"])
        else:
            cell = r.get(col)
            if not cell:
                continue
            correct = cell.get("outcome") == "correct"
            if metric == "acc":
                hits += correct; n += 1
            elif metric == "cost":
                vals.append(cell["calls"] + cell.get("budget_used", 0))
            elif metric == "hop" and not correct:
                pl = cell.get("proxy_layer")
                bl = r["bfs"]["layer"]
                if pl is not None:
                    vals.append(abs(pl - bl))
    if metric == "acc":
        return _wilson95(hits, n)
    return _mean_ci95(vals)


def main() -> None:
    apply_plot_style()

    n_methods = len(METHODS)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    fig.suptitle("Method comparison by lattice size — Llama-3.1-8B",
                 fontsize=13, fontweight="bold")

    for col_idx, (ds, ds_label) in enumerate(DATASETS):
        by_size = _load(ds)
        sizes = sorted(by_size.keys())
        x = np.arange(len(sizes))
        width = 0.8 / n_methods

        for row_idx, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]

            for m_idx, (col, lab, color, is_bfs) in enumerate(METHODS):
                stats = [_cell_stats(by_size[s], col, is_bfs, metric) for s in sizes]
                means  = np.array([s[0] for s in stats])
                lo_err = np.array([max(0.0, s[0] - s[1]) if not np.isnan(s[0]) else 0.0 for s in stats])
                hi_err = np.array([max(0.0, s[2] - s[0]) if not np.isnan(s[0]) else 0.0 for s in stats])
                offset = (m_idx - n_methods / 2 + 0.5) * width
                xpos = x + offset
                valid = ~np.isnan(means)
                ax.bar(xpos[valid], means[valid], width, color=color, alpha=0.85,
                       label=lab, zorder=3)
                ax.errorbar(xpos[valid], means[valid],
                            yerr=[lo_err[valid], hi_err[valid]],
                            fmt="none", color="black", capsize=2,
                            linewidth=0.8, zorder=4)

            if row_idx == 0:
                ax.set_title(ds_label, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(sizes, fontsize=8)
            ax.set_xlabel("Lattice size" if row_idx == 2 else "")
            ax.grid(True, alpha=0.3, axis="y", zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if metric == "acc":
                ax.set_ylim(0, 108)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for _, _, c, _ in METHODS]
    labels  = [lab for _, lab, _, _ in METHODS]
    fig.legend(handles, labels, loc="upper center", ncol=n_methods,
               bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = "artifacts/llama31_bars_by_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
