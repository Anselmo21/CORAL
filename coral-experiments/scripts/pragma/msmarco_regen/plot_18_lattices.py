"""Focused plots for the 18 regen lattices:
  1. CF layer distribution (where the BFS-found CF lives in each lattice).
  2. Per-lattice cost (BFS / SRA / ERA pct_10), k=10 only.
  3. Per-lattice accuracy (BFS / SRA / ERA pct_10), k=10 only.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

DATASET_PATH = "data/sources/msmarco_pragma.json"
RESULTS_PATH = "data/msmarco_pragma/results.json"
OUT_DIR      = "plots/msmarco_pragma"

ERA_PCT_LABEL  = "era_pct_10"
ERA_B16_LABEL  = "era_fixed_16"

# Already excluded at sample-time (run_16_lattices.py). Kept here for safety
# in case results.json contains stale rows from a previous run.
EXCLUDE_QIDS: set[str] = {"480543", "51587"}


def _cf_layer_distribution() -> dict[int, int]:
    with open(DATASET_PATH) as f:
        d = json.load(f)
    return Counter(r["first_cf_layer"] for qid, r in d["lattices"].items()
                   if qid not in EXCLUDE_QIDS)


def plot_cf_layer_distribution() -> None:
    dist = _cf_layer_distribution()
    layers = sorted(set(range(2, 9)) | set(dist))
    counts = [dist.get(l, 0) for l in layers]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(layers, counts, color="#1f77b4", edgecolor="black", linewidth=0.6)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(count), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("first_cf_layer  (number of pragmas remaining at first CF)")
    ax.set_ylabel("# lattices")
    ax.set_xticks(layers)
    title_suffix = f" (excluded: {sorted(EXCLUDE_QIDS)})" if EXCLUDE_QIDS else ""
    ax.set_title(f"CF-layer distribution across {sum(counts)} regen lattices{title_suffix}",
                 fontweight="bold")
    ax.set_ylim(0, max(counts) + 1.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = f"{OUT_DIR}/cf_layer_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return dist


def _load_k10_results() -> list[dict]:
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    return [r for r in results
            if r["size"] == 10 and r["qid"] not in EXCLUDE_QIDS]


def plot_per_lattice_cost(k10: list[dict], cf_dist: dict[int, int]) -> None:
    """Side-by-side bars per lattice for BFS / SRA / ERA cost."""
    # Sort lattices by CF layer (asc) then qid for visual grouping
    with open(DATASET_PATH) as f:
        d = json.load(f)
    cf_layer = {qid: r["first_cf_layer"] for qid, r in d["lattices"].items()}
    rows = sorted(k10, key=lambda r: (cf_layer.get(r["qid"], 99), int(r["qid"])))

    qids = [r["qid"] for r in rows]
    bfs_c    = [r["bfs"]["calls"] for r in rows]
    sra_c    = [r["sra"]["calls"] + r["sra"].get("budget_used", 0) for r in rows]
    era_pc   = [r["era"][ERA_PCT_LABEL]["calls"] + r["era"][ERA_PCT_LABEL].get("budget_used", 0)
                for r in rows]
    era_16c  = [r["era"][ERA_B16_LABEL]["calls"] + r["era"][ERA_B16_LABEL].get("budget_used", 0)
                for r in rows]

    x = np.arange(len(qids))
    w = 0.20

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 1.5 * w, bfs_c,   w, color="black",   alpha=0.85, label="BFS (oracle)")
    ax.bar(x - 0.5 * w, sra_c,   w, color="#d62728", alpha=0.85, label="SRA")
    ax.bar(x + 0.5 * w, era_pc,  w, color="#1f77b4", alpha=0.85, label="ERA (pct_10)")
    ax.bar(x + 1.5 * w, era_16c, w, color="#9467bd", alpha=0.85, label="ERA (b=16)")

    # Annotate CF layer above each group
    for i, r in enumerate(rows):
        top = max(bfs_c[i], sra_c[i], era_pc[i], era_16c[i])
        ax.text(x[i], top + max(bfs_c) * 0.02,
                f"L{cf_layer[r['qid']]}", ha="center", va="bottom",
                fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(qids, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Query ID  (sorted by first_cf_layer asc; annotation Lx = CF layer)")
    ax.set_ylabel("LLM Calls (search + budget)")
    ax.set_title(f"Per-lattice cost on the full k=10 case  (n={len(rows)})",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = f"{OUT_DIR}/per_lattice_cost.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_per_lattice_accuracy(k10: list[dict]) -> None:
    """Per-lattice accuracy: did each method find the correct CF?
    BFS is always correct (oracle). SRA / ERA shown with green/red bars."""
    with open(DATASET_PATH) as f:
        d = json.load(f)
    cf_layer = {qid: r["first_cf_layer"] for qid, r in d["lattices"].items()}
    rows = sorted(k10, key=lambda r: (cf_layer.get(r["qid"], 99), int(r["qid"])))

    qids = [r["qid"] for r in rows]
    sra_correct    = [r["sra"]["outcome"] == "correct" for r in rows]
    era_pc_correct = [r["era"][ERA_PCT_LABEL]["outcome"] == "correct" for r in rows]
    era_16_correct = [r["era"][ERA_B16_LABEL]["outcome"] == "correct" for r in rows]

    # 3xN grid: rows = methods, cols = lattices, color = correctness
    grid = np.array([
        [1 if c else 0 for c in sra_correct],
        [1 if c else 0 for c in era_pc_correct],
        [1 if c else 0 for c in era_16_correct],
    ])

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d62728", "#2ca02c"])

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, "✓" if grid[i, j] else "✗",
                    ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(qids)))
    ax.set_xticklabels([f"{q}\nL{cf_layer[q]}" for q in qids],
                       rotation=0, ha="center", fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["SRA", "ERA pct_10", "ERA b=16"], fontsize=9)
    ax.set_xlabel("Query ID  (with first_cf_layer)")

    sra_acc    = 100 * sum(sra_correct)    / len(sra_correct)
    era_pc_acc = 100 * sum(era_pc_correct) / len(era_pc_correct)
    era_16_acc = 100 * sum(era_16_correct) / len(era_16_correct)
    ax.set_title(
        f"Per-lattice correctness (k=10) — SRA={sra_acc:.0f}%  "
        f"ERA(pct_10)={era_pc_acc:.0f}%  ERA(b=16)={era_16_acc:.0f}%   "
        f"(green=correct, red=incorrect)",
        fontweight="bold", fontsize=10,
    )
    # Add a right-side gutter showing per-row accuracy
    for i, acc in enumerate([sra_acc, era_pc_acc, era_16_acc]):
        ax.text(len(qids) + 0.2, i, f"{acc:.0f}%",
                ha="left", va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/per_lattice_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    cf_dist = plot_cf_layer_distribution()
    k10 = _load_k10_results()
    plot_per_lattice_cost(k10, cf_dist)
    plot_per_lattice_accuracy(k10)

    # Text summary
    with open(DATASET_PATH) as f:
        d = json.load(f)
    cf_layer = {qid: r["first_cf_layer"] for qid, r in d["lattices"].items()}
    print(f"\nCF-layer distribution: {dict(sorted(cf_dist.items()))}")
    print(f"\nPer-lattice (sorted by CF layer):")
    rows = sorted(k10, key=lambda r: (cf_layer[r["qid"]], int(r["qid"])))
    print(f"  {'qid':>9} {'L':>3} | {'BFS$':>5} | "
          f"{'SRA':<14} | {'ERA(pct_10)':<16} | {'ERA(b=16)':<16}")
    sra_correct = era_pc_correct = era_16_correct = 0
    for r in rows:
        qid = r["qid"]
        bfs_c    = r["bfs"]["calls"]
        sra_o    = r["sra"]["outcome"]
        era_pc_o = r["era"][ERA_PCT_LABEL]["outcome"]
        era_16_o = r["era"][ERA_B16_LABEL]["outcome"]
        sra_c    = r["sra"]["calls"] + r["sra"].get("budget_used", 0)
        era_pc_c = r["era"][ERA_PCT_LABEL]["calls"] + r["era"][ERA_PCT_LABEL].get("budget_used", 0)
        era_16_c = r["era"][ERA_B16_LABEL]["calls"] + r["era"][ERA_B16_LABEL].get("budget_used", 0)
        sra_correct    += sra_o == "correct"
        era_pc_correct += era_pc_o == "correct"
        era_16_correct += era_16_o == "correct"
        st = "✓" if sra_o == "correct" else "✗"
        pt = "✓" if era_pc_o == "correct" else "✗"
        bt = "✓" if era_16_o == "correct" else "✗"
        print(f"  {qid:>9} {cf_layer[qid]:>3} | {bfs_c:>5} | "
              f"{st} {sra_c:>4} {sra_o[:6]:<6} | "
              f"{pt} {era_pc_c:>4} {era_pc_o[:8]:<8} | "
              f"{bt} {era_16_c:>4} {era_16_o[:8]:<8}")
    n = len(rows)
    print(f"\n  Totals (n={n}):")
    print(f"    SRA          accuracy = {sra_correct}/{n} = {sra_correct/n*100:.0f}%")
    print(f"    ERA pct_10   accuracy = {era_pc_correct}/{n} = {era_pc_correct/n*100:.0f}%")
    print(f"    ERA b=16     accuracy = {era_16_correct}/{n} = {era_16_correct/n*100:.0f}%")
    bfs_total    = sum(r["bfs"]["calls"] for r in rows)
    sra_total    = sum(r["sra"]["calls"] + r["sra"].get("budget_used", 0) for r in rows)
    era_pc_total = sum(r["era"][ERA_PCT_LABEL]["calls"] + r["era"][ERA_PCT_LABEL].get("budget_used", 0) for r in rows)
    era_16_total = sum(r["era"][ERA_B16_LABEL]["calls"] + r["era"][ERA_B16_LABEL].get("budget_used", 0) for r in rows)
    print(f"\n  Avg LLM calls per lattice:")
    print(f"    BFS:        {bfs_total/n:>6.1f}")
    print(f"    SRA:        {sra_total/n:>6.1f}")
    print(f"    ERA pct_10: {era_pc_total/n:>6.1f}   ({era_pc_total/bfs_total*100:.0f}% of BFS)")
    print(f"    ERA b=16:   {era_16_total/n:>6.1f}   ({era_16_total/bfs_total*100:.0f}% of BFS)")


if __name__ == "__main__":
    main()
