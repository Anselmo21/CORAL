"""Unified per-dataset subsample plotter (replaces the four per-dataset variants).

Reads `data/<dataset>/results.json` for one or all of the four paper datasets
and writes the following PNGs to `plots/<dataset>/`:

  Standard (always):
    accuracy_by_size.png           two-panel line plot (fixed + pct ERA budgets)
    cost_by_size.png               two-panel line plot
    accuracy_heatmap.png           accuracy heatmap, rows = configs, cols = sizes
    cost_heatmap.png               cost heatmap
    era_{fixed,pct}_{accuracy,cost}.png    single-panel variants
    era_combined_{fixed,pct}.png   acc + cost side-by-side
    accuracy_cost_combined.png     2x2 grid (rows: fixed/pct; cols: acc/cost)

  Per-lattice (with --per-lattice; uses k=10 rows only):
    cf_layer_distribution.png      histogram of BFS-found CF layers per qid
    per_lattice_cost.png           bar group per qid, 4 series (BFS/SRA/ERA pct_10/ERA b=16)
    per_lattice_accuracy.png       3xN correctness heatmap

CLI:
    python -m scripts.shared.plot_subsample --dataset msmarco_criteria
    python -m scripts.shared.plot_subsample --all
    python -m scripts.shared.plot_subsample --all --per-lattice
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from scripts.shared.plot_style import apply_plot_style

DATASETS = ["msmarco_criteria", "sonnet_criteria", "msmarco_pragma", "sonnet_pragma"]

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT:   list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ERA_B16_LABEL = "era_fixed_16"
ERA_PCT10_LABEL = "era_pct_10"

REF_COLORS = {"BFS": "black", "SRA": "#d62728"}
_MARKERS = ["D", "s", "^", "o", "v", "p", "h", "P", "X", "*"]
_COLORS  = plt.colormaps["tab10"].colors


# ---------------------------------------------------------------------------
# Loading + helpers
# ---------------------------------------------------------------------------

def _load_results(dataset: str) -> list[dict]:
    """Load results.json for a dataset; handle both dict-wrapped (criteria) and flat-list (pragma) forms."""
    with open(f"data/{dataset}/results.json") as f:
        d = json.load(f)
    if isinstance(d, dict) and "results" in d:
        return d["results"]
    return d


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


# ---------------------------------------------------------------------------
# Standard plots — by size, all budgets
# ---------------------------------------------------------------------------

def _budget_panel(ax, sizes, by_size, labels_and_pickers, ylabel_kind: str,
                  show_sra: bool = True, show_bfs: bool = True) -> None:
    for i, (label, picker) in enumerate(labels_and_pickers):
        y = [_acc(by_size[s], picker) if ylabel_kind == "accuracy"
             else _cost(by_size[s], picker)
             for s in sizes]
        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(sizes, y, f"{marker}-", color=color, linewidth=1.7, markersize=6, label=label)

    if show_bfs:
        if ylabel_kind == "accuracy":
            bfs_y = [100.0 for _ in sizes]
        else:
            bfs_y = [float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
        ax.plot(sizes, bfs_y, "o:", color=REF_COLORS["BFS"], linewidth=2, markersize=7,
                label="BFS (baseline)")

    if show_sra:
        if ylabel_kind == "accuracy":
            sra_y = [_acc(by_size[s], lambda r: r.get("sra")) for s in sizes]
        else:
            sra_y = [_cost(by_size[s], lambda r: r.get("sra")) for s in sizes]
        ax.plot(sizes, sra_y, "s--", color=REF_COLORS["SRA"], linewidth=2, markersize=7,
                label="SRA")

    ax.set_xlabel("Number of Available Criteria")
    if ylabel_kind == "accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 108)
    else:
        ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)


def _make_pickers() -> tuple[list[tuple[str, callable]], list[tuple[str, callable]]]:
    fixed_specs = [(f"{b}",
                    (lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab)))
                   for b in ERA_BUDGETS_FIXED]
    pct_specs = [(f"{p}%",
                  (lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab)))
                 for p in ERA_BUDGETS_PCT]
    return fixed_specs, pct_specs


def plot_pair(results: list[dict], ylabel_kind: str, out_path: str) -> None:
    by_size = _by_size(results)
    sizes = sorted(by_size)
    fixed_specs, pct_specs = _make_pickers()
    show_sra = ylabel_kind != "accuracy"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _budget_panel(axes[0], sizes, by_size, fixed_specs, ylabel_kind, show_sra=show_sra)
    _budget_panel(axes[1], sizes, by_size, pct_specs, ylabel_kind, show_sra=show_sra)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def _heatmap_grid(results: list[dict], kind: str) -> tuple[list[str], list[int], np.ndarray]:
    by_size = _by_size(results)
    sizes = sorted(by_size)
    row_specs: list[tuple[str, callable]] = [("SRA", lambda r: r.get("sra"))]
    for b in ERA_BUDGETS_FIXED:
        row_specs.append((f"{b}",
                          (lambda r, lab=f"era_fixed_{b}": r.get("era", {}).get(lab))))
    for p in ERA_BUDGETS_PCT:
        row_specs.append((f"{p}%",
                          (lambda r, lab=f"era_pct_{p}": r.get("era", {}).get(lab))))
    matrix = np.zeros((len(row_specs), len(sizes)), dtype=float)
    for i, (_, picker) in enumerate(row_specs):
        for j, s in enumerate(sizes):
            matrix[i, j] = (_acc(by_size[s], picker) if kind == "accuracy"
                            else _cost(by_size[s], picker))
    return [r[0] for r in row_specs], sizes, matrix


def plot_heatmap(results: list[dict], kind: str, out_path: str) -> None:
    labels, sizes, matrix = _heatmap_grid(results, kind)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_name = "RdYlGn" if kind == "accuracy" else "YlOrRd"
    vmin, vmax = (0.0, 100.0) if kind == "accuracy" else (None, None)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Available Criteria")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = f"{v:.0f}" if kind == "accuracy" else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if (kind == "accuracy" and v >= 50) or
                                     (kind == "cost" and v < 0.6 * matrix.max())
                                  else "white")
    fig.colorbar(im, ax=ax).set_label("Avg cost" if kind == "cost" else "Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_combined(results: list[dict], out_dir: str) -> None:
    by_size = _by_size(results)
    sizes = sorted(by_size)
    fixed_specs, pct_specs = _make_pickers()
    for specs, name in [(fixed_specs, "fixed"), (pct_specs, "pct")]:
        for ylabel_kind in ("accuracy", "cost"):
            fig, ax = plt.subplots(figsize=(8, 3.2))
            _budget_panel(ax, sizes, by_size, specs, ylabel_kind, show_sra=False,
                          show_bfs=(ylabel_kind == "cost"))
            fig.tight_layout()
            out_path = f"{out_dir}/era_{name}_{ylabel_kind}.png"
            fig.savefig(out_path, dpi=150); plt.close(fig)
            print(f"  saved {out_path}")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        _budget_panel(axes[0], sizes, by_size, specs, "accuracy", show_sra=False, show_bfs=False)
        _budget_panel(axes[1], sizes, by_size, specs, "cost",     show_sra=False, show_bfs=True)
        fig.tight_layout()
        out_path = f"{out_dir}/era_combined_{name}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"  saved {out_path}")


def plot_accuracy_cost_combined(results: list[dict], out_path: str) -> None:
    by_size = _by_size(results)
    sizes = sorted(by_size)
    fixed_specs, pct_specs = _make_pickers()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _budget_panel(axes[0, 0], sizes, by_size, fixed_specs, "accuracy", show_sra=False, show_bfs=True)
    _budget_panel(axes[0, 1], sizes, by_size, fixed_specs, "cost",     show_sra=True,  show_bfs=True)
    _budget_panel(axes[1, 0], sizes, by_size, pct_specs,   "accuracy", show_sra=False, show_bfs=True)
    _budget_panel(axes[1, 1], sizes, by_size, pct_specs,   "cost",     show_sra=True,  show_bfs=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Per-lattice (k=10) plots — replaces plot_18_lattices.py
# ---------------------------------------------------------------------------

def _bfs_layer_per_qid(results: list[dict]) -> dict[str, int]:
    """Derive the BFS-found CF layer per qid from the k=10 rows."""
    return {r["qid"]: r["bfs"]["layer"]
            for r in results if r["size"] == 10 and r["bfs"].get("layer") is not None}


def plot_cf_layer_distribution(results: list[dict], out_path: str) -> None:
    cf_layer = _bfs_layer_per_qid(results)
    dist = Counter(cf_layer.values())
    layers = sorted(set(range(2, 11)) | set(dist))
    counts = [dist.get(l, 0) for l in layers]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(layers, counts, color="#1f77b4", edgecolor="black", linewidth=0.6)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(c), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("BFS-found CF layer at k=10")
    ax.set_ylabel("# lattices")
    ax.set_xticks(layers)
    ax.set_title(f"CF-layer distribution across {sum(counts)} lattices",
                 fontweight="bold")
    ax.set_ylim(0, max(counts) + 1.5 if counts else 1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_lattice_cost(results: list[dict], out_path: str) -> None:
    cf_layer = _bfs_layer_per_qid(results)
    rows = [r for r in results if r["size"] == 10]
    rows.sort(key=lambda r: (cf_layer.get(r["qid"], 99), str(r["qid"])))

    qids = [r["qid"] for r in rows]
    bfs_c   = [r["bfs"]["calls"] for r in rows]
    sra_c   = [r["sra"]["calls"] + r["sra"].get("budget_used", 0) for r in rows]
    era_pc  = [r["era"][ERA_PCT10_LABEL]["calls"] + r["era"][ERA_PCT10_LABEL].get("budget_used", 0)
               for r in rows if ERA_PCT10_LABEL in r["era"]]
    era_16c = [r["era"][ERA_B16_LABEL]["calls"] + r["era"][ERA_B16_LABEL].get("budget_used", 0)
               for r in rows if ERA_B16_LABEL in r["era"]]

    x = np.arange(len(qids)); w = 0.20
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - 1.5*w, bfs_c,   w, color="black",   alpha=0.85, label="BFS (oracle)")
    ax.bar(x - 0.5*w, sra_c,   w, color="#d62728", alpha=0.85, label="SRA")
    ax.bar(x + 0.5*w, era_pc,  w, color="#1f77b4", alpha=0.85, label="ERA (pct_10)")
    ax.bar(x + 1.5*w, era_16c, w, color="#9467bd", alpha=0.85, label="ERA (b=16)")
    for i, r in enumerate(rows):
        top = max(bfs_c[i], sra_c[i], era_pc[i], era_16c[i])
        ax.text(x[i], top + max(bfs_c) * 0.02,
                f"L{cf_layer.get(r['qid'], '?')}", ha="center", va="bottom",
                fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(qids, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Query ID  (sorted by BFS CF layer asc)")
    ax.set_ylabel("LLM Calls (search + budget)")
    ax.set_title(f"Per-lattice cost at k=10  (n={len(rows)})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_lattice_accuracy(results: list[dict], out_path: str) -> None:
    cf_layer = _bfs_layer_per_qid(results)
    rows = [r for r in results if r["size"] == 10]
    rows.sort(key=lambda r: (cf_layer.get(r["qid"], 99), str(r["qid"])))
    qids = [r["qid"] for r in rows]
    sra_ok  = [r["sra"]["outcome"] == "correct" for r in rows]
    pc_ok   = [r["era"].get(ERA_PCT10_LABEL, {}).get("outcome") == "correct" for r in rows]
    b16_ok  = [r["era"].get(ERA_B16_LABEL, {}).get("outcome") == "correct" for r in rows]
    grid = np.array([
        [1 if c else 0 for c in sra_ok],
        [1 if c else 0 for c in pc_ok],
        [1 if c else 0 for c in b16_ok],
    ])
    cmap = ListedColormap(["#d62728", "#2ca02c"])
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, "✓" if grid[i, j] else "✗",
                    ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(qids)))
    ax.set_xticklabels([f"{q}\nL{cf_layer.get(q, '?')}" for q in qids],
                       rotation=0, ha="center", fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["SRA", "ERA pct_10", "ERA b=16"], fontsize=9)
    sra_acc = 100*sum(sra_ok)/len(sra_ok)
    pc_acc  = 100*sum(pc_ok)/len(pc_ok)
    b16_acc = 100*sum(b16_ok)/len(b16_ok)
    ax.set_title(
        f"Per-lattice correctness (k=10) — SRA={sra_acc:.0f}%  "
        f"ERA(pct_10)={pc_acc:.0f}%  ERA(b=16)={b16_acc:.0f}%",
        fontweight="bold", fontsize=10,
    )
    for i, acc in enumerate([sra_acc, pc_acc, b16_acc]):
        ax.text(len(qids) + 0.2, i, f"{acc:.0f}%", ha="left", va="center",
                fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def plot_dataset(dataset: str, per_lattice: bool) -> None:
    out_dir = f"plots/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    results = _load_results(dataset)
    print(f"[{dataset}] {len(results)} cases")
    plot_pair(results, "accuracy", f"{out_dir}/accuracy_by_size.png")
    plot_pair(results, "cost",     f"{out_dir}/cost_by_size.png")
    plot_heatmap(results, "accuracy", f"{out_dir}/accuracy_heatmap.png")
    plot_heatmap(results, "cost",     f"{out_dir}/cost_heatmap.png")
    plot_combined(results, out_dir)
    plot_accuracy_cost_combined(results, f"{out_dir}/accuracy_cost_combined.png")
    if per_lattice:
        plot_cf_layer_distribution(results, f"{out_dir}/cf_layer_distribution.png")
        plot_per_lattice_cost(results, f"{out_dir}/per_lattice_cost.png")
        plot_per_lattice_accuracy(results, f"{out_dir}/per_lattice_accuracy.png")


def main() -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=DATASETS,
                   help="One of the four paper datasets.")
    g.add_argument("--all", action="store_true",
                   help="Plot all four datasets.")
    parser.add_argument("--per-lattice", action="store_true",
                        help="Also emit cf_layer / per-lattice cost + correctness plots (k=10 only).")
    args = parser.parse_args()

    targets = DATASETS if args.all else [args.dataset]
    for d in targets:
        plot_dataset(d, args.per_lattice)


if __name__ == "__main__":
    main()
