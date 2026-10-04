"""Plot BFS and SRA accuracy + cost vs lattice size across 4 Qwen3 datasets.

Writes plots/qwen3_subsample.png (2 rows × 4 cols: accuracy top, cost bottom).

Usage:
    conda run -n coral python3 -m scripts.shared.plot_qwen3_subsample
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
    ("qwen3_msmarco_criteria", "MSMARCO\ncriteria"),
    ("qwen3_msmarco_pragma",   "MSMARCO\npragma"),
    ("qwen3_curated_criteria",  "Curated\ncriteria"),
    ("qwen3_curated_pragma",    "Curated\npragma"),
]

COLOR_BFS = "#4e79a7"
COLOR_SRA = "#e15759"


def _load(dataset: str) -> list[dict]:
    with open(f"logged_results/qwen3-32b-prelim/{dataset.removeprefix('qwen3_')}/results.json") as f:
        d = json.load(f)
    return d["results"] if isinstance(d, dict) and "results" in d else d


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _sra_acc(bucket: list[dict]) -> float:
    ms = [r["sra"] for r in bucket if r.get("sra")]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m["outcome"] == "correct") / len(ms)


def _cost(bucket: list[dict], key: str) -> float:
    if key == "bfs":
        vals = [r["bfs"]["calls"] for r in bucket if r.get("bfs", {}).get("found")]
    else:
        vals = [r["sra"]["calls"] + r["sra"].get("budget_used", 0)
                for r in bucket if r.get("sra")]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    apply_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("BFS vs SRA — Qwen3-32B subsample (20 lattices per size)",
                 fontsize=13, fontweight="bold")

    for col, (ds, label) in enumerate(DATASETS):
        results = _load(ds)
        by_size = _by_size(results)
        sizes = sorted(s for s in by_size if s != 10)

        ax_acc  = axes[0, col]
        ax_cost = axes[1, col]

        sra_acc  = [_sra_acc(by_size[s])       for s in sizes]
        bfs_cost = [_cost(by_size[s], "bfs")   for s in sizes]
        sra_cost = [_cost(by_size[s], "sra")   for s in sizes]

        # accuracy
        ax_acc.plot(sizes, [100.0] * len(sizes), "o:", color=COLOR_BFS,
                    linewidth=2, markersize=6, label="BFS")
        ax_acc.plot(sizes, sra_acc, "s--", color=COLOR_SRA,
                    linewidth=2, markersize=6, label="SRA")
        ax_acc.set_ylim(0, 108)
        ax_acc.set_title(label, fontsize=11, fontweight="bold")
        ax_acc.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax_acc.set_xticks(sizes)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)
        if col == 0:
            ax_acc.legend(fontsize=9, loc="lower right")

        # cost
        ax_cost.plot(sizes, bfs_cost, "o:", color=COLOR_BFS,
                     linewidth=2, markersize=6, label="BFS")
        ax_cost.plot(sizes, sra_cost, "s--", color=COLOR_SRA,
                     linewidth=2, markersize=6, label="SRA")
        ax_cost.set_xlabel("Lattice size")
        ax_cost.set_ylabel("Avg LLM calls" if col == 0 else "")
        ax_cost.set_xticks(sizes)
        ax_cost.grid(True, alpha=0.3)
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out = "plots/qwen3_subsample.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
