"""BFS vs MRA vs LSM across Llama-3.1-8B datasets (labels display "ERA").

Reads BFS + SRA from paper_results.json (SRA = native combsum-Borda from
`search.rank.sra_search`), and MRA cells (b=4,8,16) from
mra_budget_ablation/results.json — hand-crafted Borda aggregation with
softmax normalize+entropy (`scripts.shared.run_mra_budget_ablation`).
LSM comes from cc_results_budget_{4,8,16}.json.
Writes one combined figure plus one per-dataset figure to plots/llama31-8b/.

Legend "ERA (b=*)" is MRA (see PRESENTATION_PLOTS.md).

  * MRA/SRA cost = calls + budget_used + free preloads (k singletons + empty set)
  * SRA skips the empty-set preload
  * LSM cost = calls + num_ablations (= budget)

Usage:
    conda run -n coral python3 -m scripts.shared.plot_llama_bfs_lsm
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style, draw_hop_bars
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, mra_search
from search.heuristics import shannon_entropy

MODELS = {
    "llama31-8b": {
        "datasets": [
            ("msmarco_criteria", "data/llama31-8b/msmarco-criteria-lattices.json", "Llama-3.1-8B MSMARCO criteria"),
            ("msmarco_pragma",   "data/llama31-8b/msmarco-pragmas-lattices.json",  "Llama-3.1-8B MSMARCO pragma"),
            ("curated_criteria", "data/llama31-8b/curated-criteria-lattices.json", "Llama-3.1-8B Curated criteria"),
            ("curated_pragma",   "data/llama31-8b/curated-pragmas-lattices.json",  "Llama-3.1-8B Curated pragma"),
        ],
        "result_root": "logged_results/llama31-8b",
        "plot_root": "presentation_plots/llama31-8b",
        "true_era_plot_root": "presentation_plots/true_era/llama31-8b",
        "families": [
            ("msmarco", ["msmarco_criteria", "msmarco_pragma"], "Llama-3.1-8B MSMARCO"),
            ("curated", ["curated_criteria", "curated_pragma"], "Llama-3.1-8B Curated"),
        ],
        "has_lsm": True,
        "has_mra_ablation": True,
    },
    "llama31-70b": {
        "datasets": [
            ("msmarco_criteria", "data/llama31-70b/msmarco-criteria-llama31-70b.json", "Llama-3.1-70B MSMARCO criteria"),
            ("msmarco_pragma",   "data/llama31-70b/msmarco-pragmas-llama31-70b.json",  "Llama-3.1-70B MSMARCO pragma"),
            ("curated_criteria", "data/llama31-70b/curated-criteria-llama31-70b.json", "Llama-3.1-70B Curated criteria"),
            ("curated_pragma",   "data/llama31-70b/curated-pragmas-llama31-70b.json",  "Llama-3.1-70B Curated pragma"),
        ],
        "result_root": "logged_results/llama31-70b",
        "plot_root": "presentation_plots/llama31-70b",
        "true_era_plot_root": "presentation_plots/true_era/llama31-70b",
        "families": [
            ("msmarco", ["msmarco_criteria", "msmarco_pragma"], "Llama-3.1-70B MSMARCO"),
            ("curated", ["curated_criteria", "curated_pragma"], "Llama-3.1-70B Curated"),
        ],
        "has_lsm": False,          # no CC runs for 70b yet
        "has_mra_ablation": False, # no MRA budget ablation for 70b yet
    },
    "llama33-70b": {
        "datasets": [
            ("msmarco_criteria", "data/llama33-70b/msmarco-criteria-llama33-70b.json", "Llama-3.3-70B MSMARCO criteria"),
            ("msmarco_pragma",   "data/llama33-70b/msmarco-pragmas-llama33-70b.json",  "Llama-3.3-70B MSMARCO pragma"),
            ("curated_criteria", "data/llama33-70b/curated-criteria-llama33-70b.json", "Llama-3.3-70B Curated criteria"),
            ("curated_pragma",   "data/llama33-70b/curated-pragmas-llama33-70b.json",  "Llama-3.3-70B Curated pragma"),
        ],
        "result_root": "logged_results/llama33-70b",
        "plot_root": "presentation_plots/llama33-70b",
        "true_era_plot_root": "presentation_plots/true_era/llama33-70b",
        "families": [
            ("msmarco", ["msmarco_criteria", "msmarco_pragma"], "Llama-3.3-70B MSMARCO"),
            ("curated", ["curated_criteria", "curated_pragma"], "Llama-3.3-70B Curated"),
        ],
        "has_lsm": False,
        "has_mra_ablation": False,
    },
}

# These four are set from MODELS[<model>] in main(). Defaults to 8b for module-level imports.
DATASETS = MODELS["llama31-8b"]["datasets"]
FAMILIES = MODELS["llama31-8b"]["families"]
_HAS_LSM = MODELS["llama31-8b"]["has_lsm"]
_HAS_MRA = MODELS["llama31-8b"]["has_mra_ablation"]

LSM_METHODS = [
    (4,  "LSM (b=4)",  "^", "#2ca02c"),
    (8,  "LSM (b=8)",  "P", "#ff7f0e"),
    (16, "LSM (b=16)", "D", "#e377c2"),
]

ERA_METHODS = [
    (4,  "era_b4",  "ERA (b=4)",  "X", "#1f77b4"),
    (8,  "era_b8",  "ERA (b=8)",  "h", "#17becf"),
    (16, "era_b16", "ERA (b=16)", "D", "#9467bd"),
]

MRA_COMPARE: list[tuple] = []

def _set_labels(label_stem: str) -> None:
    """Rename the ERA/MRA labels in place (e.g. 'ERA' → 'MRA')."""
    global ERA_METHODS
    ERA_METHODS = [(b, col, f"{label_stem} (b={b})", mk, color)
                   for b, col, _, mk, color in ERA_METHODS]

SRA_STYLE = ("sra", "SRA", "s", "#d62728")
BFS_STYLE = ("BFS", "o", "black")

RESULT_ROOT = MODELS["llama31-8b"]["result_root"]
PLOT_ROOT = MODELS["llama31-8b"]["plot_root"]
_FILE_SUFFIX = ""  # set to "_mra" or "_era" depending on flags
_BACKEND = "mra"   # set to "era" via --backend era


def _mra_index(name: str) -> dict[tuple[str, str], dict]:
    path = f"{RESULT_ROOT}/{name}/mra_budget_ablation/results.json"
    data = json.load(open(path))
    out: dict[tuple[str, str], dict] = {}
    for r in data["results"]:
        out[(r["qid"], combo_key(r["start_node"]))] = r
    return out


def _inject_mra(rows: list[dict], mra_by_key: dict[tuple[str, str], dict]) -> None:
    for row in rows:
        key = (row["qid"], combo_key(row["start_node"]))
        mra_row = mra_by_key.get(key)
        if mra_row is None:
            continue
        for _budget, col, _lab, _mk, _color in ERA_METHODS:
            cell = mra_row.get(col.replace("era_", ""))
            if cell is not None:
                row[col] = cell


def _load_lattices(data_path: str) -> dict[str, tuple[list[str], dict]]:
    """Read a llama31-8b lattice file → {qid: (criteria, raw_answers)}."""
    raw = json.load(open(data_path))
    out: dict[str, tuple[list[str], dict]] = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q.get("criteria") or [])
        rbc = q.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _score_proxy(bfs_key, proxy_result, rbc, start_node) -> dict:
    """Inline score_proxy — avoids importing per-dataset run_subsample modules
    (whose DATA_PATH constants point at renamed/nonexistent files)."""
    start_key = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")
    proxy_verified = (proxy_key is not None
                      and bool(rbc.get(proxy_key))
                      and rbc[proxy_key][0] != baseline_top1)
    if not proxy_result.get("found") or proxy_key is None:
        outcome = "not_found"
    elif not proxy_verified:
        outcome = "false_positive"
    else:
        outcome = "correct"  # layer-matching not needed here — bfs layer compared downstream
    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": proxy_verified,
        "outcome": outcome,
        "jaccard": jaccard_keys(bfs_key, proxy_key),
        "calls": proxy_result.get("calls", 0),
        "budget_used": proxy_result.get("budget_used", 0),
        "pruned_nodes": proxy_result.get("pruned_nodes", 0),
    }


def _compute_true_era(rows: list[dict], lattices: dict) -> None:
    """Fill missing era_b{4,8,16} cells by calling real era_search on each case."""
    for row in rows:
        if row["qid"] not in lattices:
            continue
        crit, rbc = lattices[row["qid"]]
        start = row["start_node"]
        bfs_key = row["bfs"].get("key")
        bfs_layer = row["bfs"]["layer"]
        for budget, col, *_ in ERA_METHODS:
            if col in row and row[col] is not None:
                continue  # already precomputed (e.g. era_b4/era_b8 in paper_results)
            result = era_search(crit, rbc, budget=budget, start_node=start,
                                direction="down", propagate_scores=True)
            scored = _score_proxy(bfs_key, result, rbc, start)
            # score against BFS layer to get "correct" vs "wrong_layer"
            if scored["outcome"] == "correct" and scored["proxy_layer"] != bfs_layer:
                scored["outcome"] = "wrong_layer"
            row[col] = scored


def _compute_mra_shannon(rows: list[dict], lattices: dict) -> None:
    """Fill era_b{4,8,16} cells by calling mra_search with the paper Table I config
    (normalize_fn='softmax', entropy_fn=shannon_entropy, concepts=start_node)."""
    for row in rows:
        if row["qid"] not in lattices:
            continue
        crit, rbc = lattices[row["qid"]]
        start = row["start_node"]
        bfs_key = row["bfs"].get("key")
        bfs_layer = row["bfs"]["layer"]
        for budget, col, *_ in ERA_METHODS:
            result = mra_search(start, rbc, budget=budget, start_node=start,
                                direction="down", normalize_fn="softmax",
                                entropy_fn=shannon_entropy)
            scored = _score_proxy(bfs_key, result, rbc, start)
            if scored["outcome"] == "correct" and scored["proxy_layer"] != bfs_layer:
                scored["outcome"] = "wrong_layer"
            row[col] = scored


def _compute_mra_compare(rows: list[dict], lattices: dict) -> None:
    """Fill mra_b{4,8,16} cells alongside existing era_b* for side-by-side comparison."""
    for row in rows:
        if row["qid"] not in lattices:
            continue
        crit, rbc = lattices[row["qid"]]
        start = row["start_node"]
        bfs_key = row["bfs"].get("key")
        bfs_layer = row["bfs"]["layer"]
        for budget, col, *_ in MRA_COMPARE:
            result = mra_search(start, rbc, budget=budget, start_node=start,
                                direction="down", normalize_fn="softmax",
                                entropy_fn=shannon_entropy)
            scored = _score_proxy(bfs_key, result, rbc, start)
            if scored["outcome"] == "correct" and scored["proxy_layer"] != bfs_layer:
                scored["outcome"] = "wrong_layer"
            row[col] = scored


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


def _load(name: str, data_path: str):
    rows = json.load(open(f"{RESULT_ROOT}/{name}/paper_results.json"))
    if _BACKEND == "era":
        # Let paper_results.json's era_b* columns stand; compute any missing budget
        # via real era_search on the fly. For 70b models all three are already there.
        need_compute = any(col not in rows[0] or rows[0][col] is None
                           for _, col, *_ in ERA_METHODS)
        if need_compute:
            _compute_true_era(rows, _load_lattices(data_path))
    else:
        # Compute MRA on the fly with paper Table I config (softmax norm + shannon
        # entropy). Overrides any era_b* cells already in paper_results.json.
        lats = _load_lattices(data_path)
        # Drop rows whose qid is no longer present in the lattice file (matches
        # paper Table I sample sizes — e.g. curated_pragma qid 45 is halted).
        rows = [r for r in rows if r["qid"] in lats]
        _compute_mra_shannon(rows, lats)

    if MRA_COMPARE:
        lats = _load_lattices(data_path)
        rows = [r for r in rows if r["qid"] in lats]
        _compute_mra_compare(rows, lats)

    if _HAS_LSM:
        cc = {b: json.load(open(f"{RESULT_ROOT}/{name}/cc_results_budget_{b}.json"))
              for b, *_ in LSM_METHODS}
    else:
        cc = {b: [] for b, *_ in LSM_METHODS}
    return rows, cc


def _plot_axes(axes_col, name: str, data_path: str, nice: str, handles_by_label: dict):
    rows, cc = _load(name, data_path)
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

        for _, col, lab, mk, color in ERA_METHODS:
            xs, ys = _era_series(rows, col, kind)
            if xs:
                line, = ax.plot(xs, ys, marker=mk, linestyle="-", color=color,
                                linewidth=1.8, markersize=7, label=lab)
                handles_by_label.setdefault(lab, line)

        for _, col, lab, mk, color in MRA_COMPARE:
            xs, ys = _era_series(rows, col, kind)
            if xs:
                line, = ax.plot(xs, ys, marker=mk, linestyle="--", color=color,
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
    for _, col, lab, _mk, color in ERA_METHODS:
        method_data.append((lab, color, _era_hop_by_size(rows, col, all_sizes)))
    for _, col, lab, _mk, color in MRA_COMPARE:
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
    ("msmarco", ["msmarco_criteria", "msmarco_pragma"], "Llama MSMARCO"),
    ("curated", ["curated_criteria", "curated_pragma"], "Llama Curated"),
]


def _load_family(family_datasets: list[str]) -> tuple[list[dict], dict]:
    """Pool rows + cc across every dataset in a family (criteria + pragma)."""
    all_rows: list[dict] = []
    all_cc: dict[int, list] = {b: [] for b, *_ in LSM_METHODS}
    for name in family_datasets:
        data_path = next(dp for n, dp, _ in DATASETS if n == name)
        rows, cc = _load(name, data_path)
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
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS] + [t[2] for t in MRA_COMPARE] + [lab for _, lab, *_ in LSM_METHODS]
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.03), fontsize=14, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{PLOT_ROOT}/{family}{_FILE_SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def _plot_per_dataset(name: str, data_path: str, nice: str) -> None:
    out_dir = f"{PLOT_ROOT}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    handles: dict[str, object] = {}
    _plot_axes(axes, name, data_path, nice, handles)
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS] + [t[2] for t in MRA_COMPARE] + [lab for _, lab, *_ in LSM_METHODS]
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
    for col, (name, data_path, nice) in enumerate(DATASETS):
        _plot_axes(axes[:, col], name, data_path, nice, handles)
        axes[0, col].set_title(nice)
    order = [BFS_STYLE[0], SRA_STYLE[1]] + [t[2] for t in ERA_METHODS] + [t[2] for t in MRA_COMPARE] + [lab for _, lab, *_ in LSM_METHODS]
    ordered = [l for l in order if l in handles]
    ncol = min(len(ordered), 5) if MRA_COMPARE else len(ordered)
    legend_font = 11 if MRA_COMPARE else 14
    top_margin = 0.93 if MRA_COMPARE else 0.96
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 1.0), fontsize=legend_font, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, top_margin))
    out = f"{PLOT_ROOT}/bfs_lsm_mra_combined{_FILE_SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(MODELS), default="llama31-8b",
                    help="Answering model. 70b variants have no LSM or MRA data, so they force --backend era.")
    ap.add_argument("--labels", choices=("era", "mra"), default="era",
                    help="Legend label for the third method family. 'era' matches paper convention; 'mra' names the MRA algorithm truthfully.")
    ap.add_argument("--backend", choices=("mra", "era"), default="mra",
                    help="Which algorithm actually populates the ERA lines. 'mra' = mra_search(normalize_fn='softmax'), the paper default. 'era' = real era_search from search/rank.py.")
    ap.add_argument("--datasets", nargs="*",
                    help="Restrict to a subset of dataset names (e.g. msmarco_pragma).")
    ap.add_argument("--merge", action="store_true",
                    help="Pool criteria + pragma per family (msmarco, curated) into 2 plots instead of 4.")
    ap.add_argument("--compare", action="store_true",
                    help="Plot MRA lines alongside ERA for side-by-side comparison. Forces --backend era.")
    args = ap.parse_args()

    global _BACKEND, _FILE_SUFFIX, PLOT_ROOT, RESULT_ROOT, DATASETS, FAMILIES, _HAS_LSM, _HAS_MRA, MRA_COMPARE
    cfg = MODELS[args.model]
    DATASETS = cfg["datasets"]
    FAMILIES = cfg["families"]
    RESULT_ROOT = cfg["result_root"]
    _HAS_LSM = cfg["has_lsm"]
    _HAS_MRA = cfg["has_mra_ablation"]

    if args.compare:
        args.backend = "era"

    _BACKEND = args.backend
    if args.backend == "era":
        _set_labels("ERA")  # always label 'ERA' when running real ERA
        PLOT_ROOT = cfg["true_era_plot_root"]
        _FILE_SUFFIX = ""
    else:
        PLOT_ROOT = cfg["plot_root"]
        if args.labels == "mra":
            _set_labels("MRA")
            _FILE_SUFFIX = "_mra"

    if args.compare:
        MRA_COMPARE = [
            (4,  "mra_b4",  "MRA (b=4)",  "v", "#2ca02c"),
            (8,  "mra_b8",  "MRA (b=8)",  "p", "#ff7f0e"),
            (16, "mra_b16", "MRA (b=16)", "d", "#bcbd22"),
        ]
        _FILE_SUFFIX = "_compare"

    apply_plot_style()
    if args.merge:
        for family, ds_list, nice in FAMILIES:
            print(f"=== {nice} (criteria + pragma pooled) ===")
            _plot_family(family, ds_list, nice)
        return

    datasets = [d for d in DATASETS if not args.datasets or d[0] in args.datasets]
    for name, data_path, nice in datasets:
        print(f"=== {nice} ===")
        _plot_per_dataset(name, data_path, nice)
    if len(datasets) == len(DATASETS):
        print("=== combined ===")
        _plot_combined()


if __name__ == "__main__":
    main()
