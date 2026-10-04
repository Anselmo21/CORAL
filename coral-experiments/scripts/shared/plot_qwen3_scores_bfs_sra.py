"""BFS vs SRA vs MRA rank/score variants on the score/prob-elicited lattices.

Same layout and method conventions as plot_qwen3_temp0_bfs_lsm.py (see
PRESENTATION_PLOTS.md), but for the score-elicited runs under
coral-runs-lattices/<run>/. No ContextCite runs exist for this data, so
the LSM lines are replaced by score-fed ERA variants: same mra_search config
plus scores_by_combo (LLM scores or probs) with voter_norm="max".

There is no paper_results.json for this data — BFS, SRA and all six MRA
columns are computed fresh per case and cached to
logged_results/<out_name>/<dataset>/results.json. Delete that file to
force recompute. All runs share identical lattice inputs (same qids, atoms,
plausible answers), so the sub-lattice cases are always reused from
logged_results/qwen3-32b/<dataset>/cases.json for cross-run comparability.

Usage:
    conda run -n coral python3 -m scripts.shared.plot_qwen3_scores_bfs_sra [--run <run>]
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
from scripts.shared.run_cra_eval import load_score_lattices, load_cases
from search.bfs import bfs_search
from search.rank import mra_search, sra_search

# Run dirs under coral-runs-lattices/. "-100" runs elicit 0-100 answer scores
# (stored /100); "-percentage" runs elicit answer probabilities ("probs" key).
# qwen3-32b-100 is the original score run; its outputs keep the historical
# qwen3-32b-scores name so existing deck figures stay valid.
RUNS = {
    "qwen3-32b-100":         ("qwen3-32b-scores",       "Qwen3-32B (scores)"),
    "qwen3-32b-percentage":  ("qwen3-32b-percentage",   "Qwen3-32B (probs)"),
    "qwen35-122b-100":       ("qwen35-122b-100",        "Qwen3.5-122B (scores)"),
    "qwen35-122b-percentage": ("qwen35-122b-percentage", "Qwen3.5-122B (probs)"),
}

DATASET_SUBDIRS = [
    ("msmarco_criteria", "msmarco-criteria", "MSMARCO criteria"),
    ("msmarco_pragma",   "msmarco-pragmas",  "MSMARCO pragma"),
    ("curated_criteria", "curated-criteria", "Curated criteria"),
    ("curated_pragma",   "curated-pragmas",  "Curated pragma"),
]

# (budget, column, label, marker, color, scores?)
ERA_METHODS = [
    (4,  "mra_b4",         "ERA (b=4)",        "X", "#1f77b4", False),  # blue
    (8,  "mra_b8",         "ERA (b=8)",        "h", "#17becf", False),  # cyan
    (16, "mra_b16",        "ERA (b=16)",       "D", "#9467bd", False),  # purple
    (4,  "mra_scores_b4",  "ERA-scores (b=4)", "^", "#2ca02c", True),   # green
    (8,  "mra_scores_b8",  "ERA-scores (b=8)", "P", "#ff7f0e", True),   # orange
    (16, "mra_scores_b16", "ERA-scores (b=16)", "d", "#e377c2", True),  # pink
]

SRA_STYLE = ("sra", "SRA", "s", "#d62728")
BFS_STYLE = ("BFS", "o", "black")

# Set in main() from --run.
RESULT_ROOT = ""
PLOT_ROOT = ""


def _compute_rows(dataset: str, data_dir: str) -> list[dict]:
    lattices = load_score_lattices(data_dir)
    cases = load_cases("qwen3-32b", dataset)
    if cases is None:
        cases = [{"qid": qid, "size": len(atoms), "start_node": atoms}
                 for qid, (atoms, _, _) in sorted(lattices.items())]

    rows: list[dict] = []
    n_skipped = 0
    for case in cases:
        qid, start = case["qid"], case["start_node"]
        if qid not in lattices:
            n_skipped += 1
            continue
        atoms, rbc, sbc = lattices[qid]

        bfs_res = bfs_search(atoms, rbc, start_node=start, direction="down")
        if not bfs_res["found"]:
            n_skipped += 1
            continue
        row = {
            "qid": qid, "size": case["size"], "start_node": start,
            "bfs": {"found": True, "key": bfs_res["key"],
                    "layer": bfs_res["size"], "calls": bfs_res["calls"]},
        }
        bfs_proxy = {"key": bfs_res["key"], "size": bfs_res["size"], "found": True}

        sra_res = sra_search(atoms, rbc, start_node=start, direction="down")
        row["sra"] = score_proxy(bfs_proxy, sra_res, rbc, start)

        for budget, col, _lab, _mk, _color, use_scores in ERA_METHODS:
            # Paper Table I config (see PRESENTATION_PLOTS.md): concepts=start_node,
            # normalize_fn=softmax, entropy_fn=shannon (mra_search default).
            kwargs = dict(scores_by_combo=sbc, voter_norm="max") if use_scores else {}
            res = mra_search(start, rbc, budget=budget, start_node=start,
                             direction="down", normalize_fn="softmax", **kwargs)
            row[col] = score_proxy(bfs_proxy, res, rbc, start)
        rows.append(row)

    print(f"  {dataset}: {len(rows)} cases, {n_skipped} skipped")
    return rows


def _load(dataset: str, data_dir: str) -> list[dict]:
    cache = Path(f"{RESULT_ROOT}/{dataset}/results.json")
    if cache.exists():
        return json.load(open(cache))
    rows = _compute_rows(dataset, data_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(cache, "w"))
    return rows


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


def _cell_series(rows: list[dict], col: str, kind: str):
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
            v = float(np.mean([c.get("calls", 0) + c.get("budget_used", 0) for c in cells]))
        else:
            hops = [_hop(c.get("proxy_layer"), r["bfs"]["layer"])
                    for r, c in zip(bucket, cells)]
            hops = [h for h in hops if h is not None]
            v = float(np.mean(hops)) if hops else 0.0
        xs.append(s); ys.append(v)
    return xs, ys


def _hop_by_size(rows: list[dict], col: str, sizes: list[int]) -> dict[int, list[float]]:
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


def _draw_panels(axes_col, rows, xlabel: str, handles_by_label: dict):
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
        xs, ys = _cell_series(rows, sra_col, kind)
        if xs:
            line, = ax.plot(xs, ys, marker=sra_mk, linestyle="-", color=sra_color,
                            linewidth=1.8, markersize=7, label=sra_lab)
            handles_by_label.setdefault(sra_lab, line)

        for _b, col, lab, mk, color, _sc in ERA_METHODS:
            xs, ys = _cell_series(rows, col, kind)
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
    method_data = [(SRA_STYLE[1], SRA_STYLE[3], _hop_by_size(rows, SRA_STYLE[0], all_sizes))]
    for _b, col, lab, _mk, color, _sc in ERA_METHODS:
        method_data.append((lab, color, _hop_by_size(rows, col, all_sizes)))
    draw_hop_bars(ax_hop, all_sizes, method_data, handles_by_label)
    ax_hop.set_xlabel(xlabel)
    ax_hop.set_ylabel("Distance to True Counterfactual")
    ax_hop.set_xticks(all_sizes)
    ax_hop.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_hop.grid(True, alpha=0.3, axis="y")


def _legend_order(handles: dict) -> list[str]:
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS]
    return [l for l in order if l in handles]


def _plot_per_dataset(name: str, data_dir: str, nice: str) -> None:
    rows = _load(name, data_dir)
    out_dir = f"{PLOT_ROOT}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    handles: dict[str, object] = {}
    xlabel = "Number of Available Pragmas" if "pragma" in name else "Number of Available Criteria"
    _draw_panels(axes, rows, xlabel, handles)
    ordered = _legend_order(handles)
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.03), fontsize=13, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{out_dir}/{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def _plot_combined(datasets: list[tuple[str, str, str]]) -> None:
    os.makedirs(PLOT_ROOT, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    handles: dict[str, object] = {}
    for col, (name, data_dir, nice) in enumerate(datasets):
        rows = _load(name, data_dir)
        xlabel = "Number of Available Pragmas" if "pragma" in name else "Number of Available Criteria"
        _draw_panels(axes[:, col], rows, xlabel, handles)
        axes[0, col].set_title(nice)
    ordered = _legend_order(handles)
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.0), fontsize=13, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{PLOT_ROOT}/bfs_sra_mra_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(RUNS), default="qwen3-32b-100",
                    help="Which coral-runs-lattices/<run> to evaluate/plot.")
    ap.add_argument("--datasets", nargs="*",
                    help="Restrict to a subset of dataset names (e.g. msmarco_pragma).")
    args = ap.parse_args()

    global RESULT_ROOT, PLOT_ROOT
    out_name, label = RUNS[args.run]
    RESULT_ROOT = f"logged_results/{out_name}"
    PLOT_ROOT = f"presentation_plots/{out_name}"

    apply_plot_style()
    all_datasets = [(name, f"coral-runs-lattices/{args.run}/{sub}", f"{label} {nice}")
                    for name, sub, nice in DATASET_SUBDIRS]
    datasets = [d for d in all_datasets if not args.datasets or d[0] in args.datasets]
    for name, data_dir, nice in datasets:
        print(f"=== {nice} ===")
        _plot_per_dataset(name, data_dir, nice)
    if len(datasets) == len(all_datasets):
        print("=== combined ===")
        _plot_combined(datasets)


if __name__ == "__main__":
    main()
