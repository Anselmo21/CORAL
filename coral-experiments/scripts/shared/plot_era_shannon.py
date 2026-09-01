"""Compare ERA(shannon) vs ERA(softmax) vs SRA vs BFS across all 4 datasets.

Runs ERA with shannon_entropy at b=16 and pct_10, loads BFS/SRA/ERA(softmax)
from existing paper_results.json, and emits a 3-panel figure per dataset.

Outputs:
    plots/<dataset>/era_shannon.png

Usage:
    conda run -n coral python3 -m scripts.shared.plot_era_shannon
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from search.heuristics import shannon_entropy, softmax_entropy
from search.lattice import hop_distance
from search.rank import era_search

DATASETS = [
    ("msmarco_criteria", "scripts.criteria.msmarco.run_subsample", "MSMARCO Criteria"),
    ("curated_criteria",  "scripts.criteria.curated.run_subsample",  "Curated Criteria"),
    ("msmarco_pragma",   "scripts.pragma.msmarco.run_subsample",   "MSMARCO Pragma"),
    ("curated_pragma",    "scripts.pragma.curated.run_subsample",    "Curated Pragma"),
]

# Sequential colormaps: light→dark = low→high budget.
# Blues row = softmax, Oranges row = shannon.  BFS/SRA shared on both rows.
import matplotlib.cm as _cm
_blues   = _cm.Blues([0.35, 0.50, 0.65, 0.80, 0.95])
_oranges = _cm.Oranges([0.35, 0.50, 0.65, 0.80, 0.95])

# (col, label, marker, row)   — row 0 = softmax, row 1 = shannon
# colors are assigned by budget index at plot time
ERA_ROWS = [
    # row 0: softmax
    [
        ("era_softmax_b0",  "b=0",   "o"),
        ("era_softmax_b4",  "b=4",   "o"),
        ("era_softmax_b8",  "b=8",   "o"),
        ("era_b16",         "b=16",  "o"),
        ("era_pct10",       "10%",   "o"),
    ],
    # row 1: shannon
    [
        ("era_shannon_b0",   "b=0",  "o"),
        ("era_shannon_b4",   "b=4",  "o"),
        ("era_shannon_b8",   "b=8",  "o"),
        ("era_shannon_b16",  "b=16", "o"),
        ("era_shannon_pct10","10%",  "o"),
    ],
]
ROW_COLORS   = [_blues, _oranges]
ROW_LABELS   = ["ERA softmax", "ERA shannon"]

BUDGETS = [0, 4, 8, 16]  # fixed budgets to compute for both entropy fns


def _resolve_loader(mod):
    for name in ("_load_lattices", "_load_pragma_lattices", "_load_dataset"):
        fn = getattr(mod, name, None)
        if fn:
            return fn
    raise RuntimeError(f"No loader on {mod.__name__}")


def _run_era(criteria, rbc, budget, entropy_fn, start, bfs_for_proxy, bfs_key, score_proxy):
    res = era_search(criteria, rbc, budget=budget, entropy_fn=entropy_fn,
                     start_node=start, direction="down")
    cell = score_proxy(bfs_for_proxy, res, rbc, start)
    cell["hop_to_bfs"] = hop_distance(cell.get("proxy_key"), bfs_key, sorted(criteria))
    return cell


def _compute_all(dataset: str, modpath: str) -> list[dict]:
    """Run ERA(softmax) and ERA(shannon) at b=4/8 plus ERA(shannon) at b=16/pct_10."""
    with open(f"logged_results/llama31-8b/{dataset}/paper_results.json") as f:
        rows = json.load(f)

    mod = import_module(modpath)
    loader = _resolve_loader(mod)
    score_proxy = mod.score_proxy

    qids = sorted({r["qid"] for r in rows})
    lattices = loader(qids)

    augmented = []
    for i, row in enumerate(rows):
        qid, size, start = row["qid"], row["size"], row["start_node"]
        criteria, rbc = lattices[qid]
        bfs_for_proxy = {"key": row["bfs"]["key"], "size": row["bfs"]["layer"],
                         "found": row["bfs"]["found"]}
        bfs_key = row["bfs"].get("key")

        extra: dict = {}
        for b in BUDGETS:
            col_s = f"era_softmax_b{b}"
            col_h = f"era_shannon_b{b}"
            if col_s not in row:  # b=16 already in paper_results as era_b16; compute b=4,8
                extra[col_s] = _run_era(criteria, rbc, b, softmax_entropy,
                                        start, bfs_for_proxy, bfs_key, score_proxy)
            # shannon always recomputed
            extra[col_h] = _run_era(criteria, rbc, b, shannon_entropy,
                                    start, bfs_for_proxy, bfs_key, score_proxy)

        budget_pct = max(1, int(2**size * 10 / 100))
        extra["era_shannon_pct10"] = _run_era(criteria, rbc, budget_pct, shannon_entropy,
                                              start, bfs_for_proxy, bfs_key, score_proxy)

        augmented.append({**row, **extra})
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(rows)}] done", flush=True)

    return augmented


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
        hops = [c.get("hop_to_bfs") for c in cells if c.get("hop_to_bfs") is not None]
        return float(np.mean(hops)) if hops else None
    raise ValueError(kind)


def _series(rows_by: dict[int, list[dict]], col: str, kind: str, is_bfs: bool):
    xs, ys = [], []
    for size, bucket in rows_by.items():
        cells = [r.get(col) for r in bucket]
        v = _metric(cells, kind, is_bfs)
        if v is not None:
            xs.append(size)
            ys.append(v)
    return xs, ys


def _print_summary(rows: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'method':<28} | {'N':>4} | {'acc':>7} | {'cost':>8} | {'hop μ':>7}")
    print("  " + "-" * 64)
    all_cols = [("bfs", "BFS (oracle)", True), ("sra", "SRA", False)]
    for ri, row_variants in enumerate(ERA_ROWS):
        for col, blab, _ in row_variants:
            all_cols.append((col, f"{ROW_LABELS[ri]} {blab}", False))
    for col, display, is_bfs in all_cols:
        cells = [r[col] for r in rows if col in r and r[col] is not None]
        if not cells:
            continue
        n = len(cells)
        if is_bfs:
            acc = 100.0
            cost = float(np.mean([c["calls"] for c in cells]))
        else:
            acc = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / n
            cost = float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
        hops = [c.get("hop_to_bfs") for c in cells if c.get("hop_to_bfs") is not None]
        hop_mean = float(np.mean(hops)) if hops else float("nan")
        print(f"  {display:<28} | {n:>4} | {acc:>6.1f}% | {cost:>8.2f} | {hop_mean:>7.2f}")


def plot_dataset(rows: list[dict], dataset: str, nice: str) -> None:
    rows_by = _by_size(rows)
    sizes = sorted(rows_by)
    xlabel = ("Number of Available Pragmas" if "pragma" in dataset
              else "Number of Available Criteria")
    out_dir = f"plots/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    panels = ["Accuracy (%)", "Avg Cost (LLM Calls)", "Avg Hop Distance to BFS CF"]
    kinds  = ["acc", "cost", "hop"]
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7),
                             sharex=True, sharey="col")

    for ri, (variants, colors, row_label) in enumerate(
            zip(ERA_ROWS, ROW_COLORS, ROW_LABELS)):
        for ci, (ylabel, kind) in enumerate(zip(panels, kinds)):
            ax = axes[ri][ci]

            # BFS (cost panel only)
            if kind == "cost":
                xs, ys = _series(rows_by, "bfs", kind, True)
                ax.plot(xs, ys, "o--", color="black", linewidth=2,
                        markersize=6, label="BFS (oracle)")

            # SRA
            xs, ys = _series(rows_by, "sra", kind, False)
            ax.plot(xs, ys, "s-", color="#d62728", linewidth=1.6,
                    markersize=6, label="SRA")

            # ERA budget sweep for this row
            for bi, (col, blab, _) in enumerate(variants):
                if not any(col in r for r in rows):
                    continue
                xs, ys = _series(rows_by, col, kind, False)
                if xs:
                    ax.plot(xs, ys, "o-", color=colors[bi], linewidth=2,
                            markersize=6, label=blab)

            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.3)
            if kind == "acc":
                ax.set_ylim(0, 108)
            if ri == n_rows - 1:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel(f"{row_label}\n{ylabel}", labelpad=6)
            elif ri == 0:
                ax.set_title(ylabel, fontsize=10)

    # Shared legend: budget colours from the blue row; SRA + BFS separately
    budget_labels = [blab for _, blab, _ in ERA_ROWS[0]]
    legend_handles = (
        [plt.Line2D([0], [0], color="black",   ls="--", lw=2, label="BFS (oracle)"),
         plt.Line2D([0], [0], color="#d62728", ls="-",  lw=1.6, label="SRA")]
        + [plt.Line2D([0], [0], color=ROW_COLORS[0][i], ls="-", lw=2,
                      marker="o", markersize=6, label=budget_labels[i])
           for i in range(len(budget_labels))]
    )
    legend_labels = ["BFS (oracle)", "SRA"] + budget_labels
    fig.legend(legend_handles, legend_labels,
               loc="upper center", ncol=len(legend_labels),
               bbox_to_anchor=(0.5, 1.01), fontsize=9, frameon=True)
    fig.suptitle(nice, y=1.06, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"{out_dir}/era_shannon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main() -> None:
    apply_plot_style()
    for dataset, modpath, nice in DATASETS:
        print(f"\n[{nice}] running ERA(softmax/shannon) at b=4,8,16 + pct_10 …")
        rows = _compute_all(dataset, modpath)
        _print_summary(rows, nice)
        plot_dataset(rows, dataset, nice)


if __name__ == "__main__":
    main()
