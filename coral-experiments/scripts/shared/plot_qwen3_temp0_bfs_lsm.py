"""BFS vs LSM vs MRA (b=4,8,16) across qwen3-32b temp=0 datasets.

Reads BFS from paper_results.json, computes MRA (softmax norm) at b=4/8/16 on
the fly against the temp=0 lattices, and reads LSM from
cc_results_budget_{4,8,16}.json under logged_results/qwen3-32b-temp0/<dataset>/.
Writes one combined figure plus one per-dataset figure to
plots/qwen3-32b-temp0/. Hop distance is the CF-size difference
|proxy_layer - bfs_layer| (0 for BFS itself), matching the convention in
scripts/shared/plot_qwen3_32b_paper.py.

Usage:
    conda run -n coral python3 -m scripts.shared.plot_qwen3_temp0_bfs_lsm
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.criteria.msmarco.run_subsample import score_proxy
from scripts.shared.plot_style import apply_plot_style, draw_hop_bars
from search.lattice import combo_key
from search.rank import era_search, mra_search

DATASETS = [
    ("msmarco_criteria", "msmarco-criteria-qwen3-32b-temp0.json",  "Qwen MSMARCO criteria"),
    ("msmarco_pragma",   "msmarco-pragmas-qwen3-32b-temp0.json",   "Qwen MSMARCO pragma"),
    # NOTE: sibling coral-data-gen still names the curated lattices "sonnet-*"; result dirs
    # were renamed to curated_*. Keep the on-disk data filename here until the source rename.
    ("curated_criteria", "sonnet-criteria-qwen3-32b-temp0.json",   "Qwen Curated criteria"),
    ("curated_pragma",   "sonnet-pragmas-qwen3-32b-temp0.json",    "Qwen Curated pragma"),
]

LSM_METHODS = [
    (4,  "LSM (b=4)",  "^", "#2ca02c"),   # green
    (8,  "LSM (b=8)",  "P", "#ff7f0e"),   # orange
    (16, "LSM (b=16)", "D", "#e377c2"),   # pink
]

ERA_METHODS = [
    (4,  "mra_b4",  "ERA (b=4)",  "X", "#1f77b4", "mra"),   # blue
    (8,  "mra_b8",  "ERA (b=8)",  "h", "#17becf", "mra"),   # cyan
    (16, "mra_b16", "ERA (b=16)", "D", "#9467bd", "mra"),   # purple
]


def _set_labels(label_stem: str) -> None:
    """Rename the ERA/MRA labels in place (e.g. 'ERA' → 'MRA')."""
    global ERA_METHODS
    ERA_METHODS = [(b, col, f"{label_stem} (b={b})", mk, color, kind)
                   for b, col, _, mk, color, kind in ERA_METHODS]

SRA_STYLE = ("sra", "SRA", "s", "#d62728")

BFS_STYLE = ("BFS", "o", "black")

RESULT_ROOT = "logged_results/qwen3-32b-temp0"
PLOT_ROOT = "presentation_plots/qwen3-32b-temp0"
_FILE_SUFFIX = ""  # set to "_mra" or "_era" depending on flags
_BACKEND = "mra"   # set to "era" via --backend era
QWEN_DATA_DIR = Path(__file__).resolve().parents[2].parent / "coral-data-gen" / "datasets" / "qwen3-32b-temp0"


def _load_lattices(fname: str) -> dict[str, tuple[list[str], dict]]:
    raw = json.load(open(QWEN_DATA_DIR / fname))
    out = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q.get("criteria") or [])
        rbc  = q.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _compute_era(rows: list[dict], lattices: dict, budget: int, col: str, kind: str = "mra") -> None:
    for row in rows:
        if col in row:
            continue
        qid, start = row["qid"], row["start_node"]
        if qid not in lattices:
            continue
        crit, rbc = lattices[qid]
        bfs_proxy = {"key": row["bfs"]["key"], "size": row["bfs"]["layer"],
                     "found": row["bfs"]["found"]}
        if kind == "era":
            result = era_search(crit, rbc, budget=budget, start_node=start,
                                direction="down", propagate_scores=True)
        else:
            # Paper Table I config: pass start_node as concepts (scopes Phase-1 to the
            # sub-lattice), normalize_fn=softmax, entropy_fn=shannon (mra_search default).
            result = mra_search(start, rbc, budget=budget, start_node=start,
                                direction="down", normalize_fn="softmax")
        row[col] = score_proxy(bfs_proxy, result, rbc, start)


def _hop(proxy_layer, bfs_layer):
    if proxy_layer is None or bfs_layer is None:
        return None
    return abs(proxy_layer - bfs_layer)


def _bfs_series(rows: list[dict], kind: str):
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_size[r["size"]].append(r)
    xs, ys = [], []
    for s in sorted(by_size):
        bucket = by_size[s]
        if kind == "acc":
            v = 100.0
        elif kind == "cost":
            v = float(np.mean([r["bfs"]["calls"] for r in bucket]))
        else:
            v = 0.0
        xs.append(s); ys.append(v)
    return xs, ys


def _era_series(rows: list[dict], col: str, kind: str):
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if col in r:
            by_size[r["size"]].append(r)
    xs, ys = [], []
    for s in sorted(by_size):
        bucket = by_size[s]
        cells = [r[col] for r in bucket]
        if kind == "acc":
            v = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)
        elif kind == "cost":
            # Paper Table I convention: cost = calls + budget_used (no preload adjustment).
            v = float(np.mean([c.get("calls", 0) + c.get("budget_used", 0)
                               for c in cells]))
        else:
            hops = [_hop(c.get("proxy_layer"), r["bfs"]["layer"])
                    for r, c in zip(bucket, cells)]
            hops = [h for h in hops if h is not None]
            v = float(np.mean(hops)) if hops else 0.0
        xs.append(s); ys.append(v)
    return xs, ys


def _era_hop_by_size(rows: list[dict], col: str, sizes: list[int]) -> dict[int, list[float]]:
    per_size: dict[int, list[float]] = {s: [] for s in sizes}
    for r in rows:
        if col not in r:
            continue
        cell = r[col]
        if cell.get("outcome") == "correct":
            continue
        h = _hop(cell.get("proxy_layer"), r["bfs"]["layer"])
        if h is not None:
            per_size[r["size"]].append(float(h))
    return per_size


def _cc_series(rows: list[dict], kind: str):
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_size[r["effective_size"]].append(r)
    xs, ys = [], []
    for s in sorted(by_size):
        bucket = by_size[s]
        if kind == "acc":
            v = 100.0 * sum(1 for r in bucket if r["outcome"] == "correct") / len(bucket)
        elif kind == "cost":
            # LSM cost = verification calls + ablation fits (num_ablations = budget)
            v = float(np.mean([r["calls"] + r.get("budget", 0) for r in bucket]))
        else:
            hops = [_hop(r.get("proxy_layer"), r.get("bfs_layer")) for r in bucket]
            hops = [h for h in hops if h is not None]
            v = float(np.mean(hops)) if hops else 0.0
        xs.append(s); ys.append(v)
    return xs, ys


def _cc_hop_by_size(rows: list[dict], sizes: list[int]) -> dict[int, list[float]]:
    per_size: dict[int, list[float]] = {s: [] for s in sizes}
    for r in rows:
        if r.get("outcome") == "correct":
            continue
        h = _hop(r.get("proxy_layer"), r.get("bfs_layer"))
        if h is not None:
            per_size[r["effective_size"]].append(float(h))
    return per_size


def _load(name: str, fname: str):
    rows = json.load(open(f"{RESULT_ROOT}/{name}/paper_results.json"))
    lattices = _load_lattices(fname)
    for budget, col, _lab, _mk, _color, kind in ERA_METHODS:
        # --backend era overrides the per-method kind; --backend mra respects it.
        effective_kind = "era" if _BACKEND == "era" else kind
        _compute_era(rows, lattices, budget, col, kind=effective_kind)
    cc = {b: json.load(open(f"{RESULT_ROOT}/{name}/cc_results_budget_{b}.json"))
          for b, *_ in LSM_METHODS}
    return rows, cc


def _plot_axes(axes_col, name: str, fname: str, nice: str, handles_by_label: dict):
    rows, cc = _load(name, fname)
    xlabel = "Number of Available Pragmas" if "pragma" in name else "Number of Available Criteria"
    _draw_panels(axes_col, rows, cc, xlabel, handles_by_label)


def _draw_panels(axes_col, rows, cc, xlabel: str, handles_by_label: dict):
    """Render the 3-panel plot given already-loaded rows + cc data."""
    all_sizes = sorted(set(r["size"] for r in rows))

    for ax, kind, ylabel in [(axes_col[0], "acc",  "Accuracy (%)"),
                              (axes_col[1], "cost", "Avg Cost (LLM Calls)")]:
        if kind != "acc":
            lab, mk, color = BFS_STYLE
            xs, ys = _bfs_series(rows, kind)
            line, = ax.plot(xs, ys, marker=mk, linestyle="--", color=color,
                            linewidth=2.0, markersize=7, label=lab)
            handles_by_label.setdefault(lab, line)

        sra_col, sra_lab, sra_mk, sra_color = SRA_STYLE
        xs, ys = _era_series(rows, sra_col, kind)
        if xs:
            line, = ax.plot(xs, ys, marker=sra_mk, linestyle="-", color=sra_color,
                            linewidth=1.8, markersize=7, label=sra_lab)
            handles_by_label.setdefault(sra_lab, line)

        for _, col, lab, mk, color, _k in ERA_METHODS:
            xs, ys = _era_series(rows, col, kind)
            if xs:
                line, = ax.plot(xs, ys, marker=mk, linestyle="-", color=color,
                                linewidth=1.8, markersize=7, label=lab)
                handles_by_label.setdefault(lab, line)

        for budget, lab, mk, color in LSM_METHODS:
            xs, ys = _cc_series(cc[budget], kind)
            if xs:
                line, = ax.plot(xs, ys, marker=mk, linestyle="-", color=color,
                                linewidth=1.8, markersize=7, label=lab)
                handles_by_label.setdefault(lab, line)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_sizes)
        ax.grid(True, alpha=0.3)
        if kind == "acc":
            ax.set_ylim(0, 108)

    ax_hop = axes_col[2]
    method_data = []
    method_data.append((SRA_STYLE[1], SRA_STYLE[3], _era_hop_by_size(rows, SRA_STYLE[0], all_sizes)))
    for _, col, lab, _mk, color, _k in ERA_METHODS:
        method_data.append((lab, color, _era_hop_by_size(rows, col, all_sizes)))
    for budget, lab, _mk, color in LSM_METHODS:
        method_data.append((lab, color, _cc_hop_by_size(cc[budget], all_sizes)))
    draw_hop_bars(ax_hop, all_sizes, method_data, handles_by_label)
    ax_hop.set_xlabel(xlabel)
    ax_hop.set_ylabel("Distance to True Counterfactual")
    ax_hop.set_xticks(all_sizes)
    ax_hop.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_hop.grid(True, alpha=0.3, axis="y")


FAMILIES = [
    ("msmarco", ["msmarco_criteria", "msmarco_pragma"], "Qwen MSMARCO"),
    ("curated", ["curated_criteria", "curated_pragma"], "Qwen Curated"),
]


def _load_family(family_datasets: list[str]) -> tuple[list[dict], dict]:
    """Pool rows + cc across every dataset in a family (criteria + pragma)."""
    all_rows: list[dict] = []
    all_cc: dict[int, list] = {b: [] for b, *_ in LSM_METHODS}
    for name in family_datasets:
        fname = next(fn for n, fn, _ in DATASETS if n == name)
        rows, cc = _load(name, fname)
        all_rows.extend(rows)
        for b in all_cc:
            all_cc[b].extend(cc[b])
    return all_rows, all_cc


def _plot_family(family: str, datasets_in_family: list[str], nice: str) -> None:
    rows, cc = _load_family(datasets_in_family)
    os.makedirs(PLOT_ROOT, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    handles: dict[str, object] = {}
    _draw_panels(axes, rows, cc, "Number of Available Concepts", handles)
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS] + [lab for _, lab, *_ in LSM_METHODS]
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.03), fontsize=14, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{PLOT_ROOT}/{family}{_FILE_SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def _plot_per_dataset(name: str, fname: str, nice: str) -> None:
    out_dir = f"{PLOT_ROOT}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    handles: dict[str, object] = {}
    _plot_axes(axes, name, fname, nice, handles)
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS] + [lab for _, lab, *_ in LSM_METHODS]
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.03), fontsize=14, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{out_dir}/{name}{_FILE_SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def _plot_combined() -> None:
    os.makedirs(PLOT_ROOT, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    handles: dict[str, object] = {}
    for col, (name, fname, nice) in enumerate(DATASETS):
        _plot_axes(axes[:, col], name, fname, nice, handles)
        axes[0, col].set_title(nice)
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS] + [lab for _, lab, *_ in LSM_METHODS]
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.0), fontsize=14, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{PLOT_ROOT}/bfs_lsm_mra_combined{_FILE_SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=("era", "mra"), default="era",
                    help="Legend label for the third method family. 'era' matches paper convention; 'mra' names the MRA algorithm truthfully.")
    ap.add_argument("--backend", choices=("mra", "era"), default="mra",
                    help="Which algorithm actually populates the ERA lines. 'mra' = mra_search(normalize_fn='softmax'), the paper default. 'era' = real era_search from search/rank.py.")
    ap.add_argument("--datasets", nargs="*",
                    help="Restrict to a subset of dataset names (e.g. msmarco_pragma).")
    ap.add_argument("--merge", action="store_true",
                    help="Pool criteria + pragma per family (msmarco, curated) into 2 plots instead of 4.")
    args = ap.parse_args()

    global _BACKEND, _FILE_SUFFIX, PLOT_ROOT
    _BACKEND = args.backend
    if args.backend == "era":
        _set_labels("ERA")
        PLOT_ROOT = "presentation_plots/true_era/qwen3-32b-temp0"  # folder denotes backend
        _FILE_SUFFIX = ""
    elif args.labels == "mra":
        _set_labels("MRA")
        _FILE_SUFFIX = "_mra"

    apply_plot_style()
    if args.merge:
        for family, ds_list, nice in FAMILIES:
            print(f"=== {nice} (criteria + pragma pooled) ===")
            _plot_family(family, ds_list, nice)
        return

    datasets = [d for d in DATASETS if not args.datasets or d[0] in args.datasets]
    for name, fname, nice in datasets:
        print(f"=== {nice} ===")
        _plot_per_dataset(name, fname, nice)
    if len(datasets) == len(DATASETS):
        print("=== combined ===")
        _plot_combined()


if __name__ == "__main__":
    main()
