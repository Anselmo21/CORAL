"""Compute ERA/MRA b=4,8 results and regenerate combined plots for the 4 Llama datasets.

Adds era_b4, era_b8, mra_zscore_b4/b8, mra_softmax_b4/b8 to paper_results.json,
then writes plots/<dataset>/combined.png with those methods plus existing ones.

Usage:
    conda run -n coral python3 -m scripts.shared.plot_llama_paper_b4b8
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

from search.lattice import hop_distance
from search.rank import era_search, mra_search
from scripts.shared.plot_style import apply_plot_style, draw_hop_bars

DATASETS = [
    ("msmarco_criteria", "scripts.criteria.msmarco.run_subsample", "MSMARCO criteria"),
    ("msmarco_pragma",   "scripts.pragma.msmarco.run_subsample",   "MSMARCO pragma"),
    ("curated_criteria",  "scripts.criteria.curated.run_subsample",  "Curated criteria"),
    ("curated_pragma",    "scripts.pragma.curated.run_subsample",    "Curated pragma"),
]

NEW_METHODS = [
    ("era_b4",          lambda c, r, s: era_search(c, r, budget=4,  start_node=s, direction="down", propagate_scores=True)),
    ("era_b8",          lambda c, r, s: era_search(c, r, budget=8,  start_node=s, direction="down", propagate_scores=True)),
    ("mra_zscore_b4",   lambda c, r, s: mra_search(c, r, budget=4,  start_node=s, direction="down", normalize_fn="zscore")),
    ("mra_zscore_b8",   lambda c, r, s: mra_search(c, r, budget=8,  start_node=s, direction="down", normalize_fn="zscore")),
    ("mra_softmax_b4",  lambda c, r, s: mra_search(s, r, budget=4,  start_node=s, direction="down", normalize_fn="softmax")),
    ("mra_softmax_b8",  lambda c, r, s: mra_search(s, r, budget=8,  start_node=s, direction="down", normalize_fn="softmax")),
    ("mra_softmax_b16", lambda c, r, s: mra_search(s, r, budget=16, start_node=s, direction="down", normalize_fn="softmax")),
]

PLOT_METHODS = [
    ("bfs",             "BFS",        "o", "--", "black",   2.0, True),
    ("sra",             "SRA",        "s", "-",  "#d62728", 1.6, False),
    ("mra_softmax_b4",  "ERA (b=4)",  "X", "-",  "#9467bd", 1.8, False),
    ("mra_softmax_b8",  "ERA (b=8)",  "h", "-",  "#c5b0d5", 1.8, False),
    ("mra_softmax_b16", "ERA (b=16)", "D", "-",  "#6a3d9a", 1.8, False),
]

LSM_METHODS = [
    (4,  "LSM (b=4)",  "^", "-", "#2ca02c", 2.0),
    (8,  "LSM (b=8)",  "P", "-", "#98df8a", 2.0),
    (16, "LSM (b=16)", "D", "-", "#ff7f0e", 2.0),
]


def _compute_methods(name: str, modpath: str) -> None:
    mod = import_module(modpath)
    score_proxy = mod.score_proxy

    path = f"logged_results/{name}/paper_results.json"
    rows = json.load(open(path))

    qids = sorted({r["qid"] for r in rows})
    loader = next(getattr(mod, fn, None) for fn in ("_load_lattices", "_load_pragma_lattices", "_load_dataset") if getattr(mod, fn, None))
    lattices = loader(qids)

    for i, row in enumerate(rows):
        qid, start = row["qid"], row["start_node"]
        if qid not in lattices:
            continue
        crit, rbc = lattices[qid]
        bfs_key   = row["bfs"].get("key")
        bfs_proxy = {"key": bfs_key, "size": row["bfs"]["layer"], "found": row["bfs"]["found"]}
        for col, fn in NEW_METHODS:
            result = fn(crit, rbc, start)
            scored = score_proxy(bfs_proxy, result, rbc, start)
            scored["hop_to_bfs"] = hop_distance(scored.get("proxy_key"), bfs_key, sorted(crit))
            row[col] = scored
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  updated {path}")


def _load_cc_by_size(name: str, budget: int) -> dict[int, list[dict]] | None:
    path = f"logged_results/{name}/cc_results_budget_{budget}.json"
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("budget") == budget:
            out[r["effective_size"]].append(r)
    return dict(out) if out else None


def _cc_series(cc_by_size: dict[int, list[dict]] | None, kind: str, sizes: list[int]):
    if cc_by_size is None:
        return [], []
    xs, ys = [], []
    for s in sizes:
        bucket = cc_by_size.get(s)
        if not bucket:
            continue
        if kind == "acc":
            v = 100.0 * sum(1 for r in bucket if r["outcome"] == "correct") / len(bucket)
        elif kind == "cost":
            v = float(np.mean([r["calls"] for r in bucket]))
        else:
            v = None
        if v is not None:
            xs.append(s); ys.append(v)
    return xs, ys


def _cc_hop_values(cc_by_size: dict[int, list[dict]] | None, size: int) -> list[float]:
    if cc_by_size is None:
        return []
    bucket = cc_by_size.get(size) or []
    return [abs(r["proxy_layer"] - r["bfs_layer"])
            for r in bucket
            if r.get("outcome") != "correct"
            and r.get("proxy_layer") is not None and r.get("bfs_layer") is not None]


def _by_size(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return dict(sorted(out.items()))


def _metric(cells: list, kind: str, is_bfs: bool) -> float | None:
    cells = [c for c in cells if c]
    if not cells:
        return None
    if kind == "acc":
        return 100.0 if is_bfs else 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)
    if kind == "cost":
        if is_bfs:
            return float(np.mean([c["calls"] for c in cells]))
        return float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
    if kind == "hop":
        return None  # handled per-row in _series
    return None


def _series(rows_by: dict, col: str, kind: str, is_bfs: bool):
    xs, ys = [], []
    for size, bucket in rows_by.items():
        v = _metric([r.get(col) for r in bucket], kind, is_bfs)
        if v is not None:
            xs.append(size); ys.append(v)
    return xs, ys


def _hop_values(bucket: list[dict], col: str) -> list[float]:
    out: list[float] = []
    for r in bucket:
        cell = r.get(col)
        if not cell or cell.get("outcome") == "correct":
            continue
        pl = cell.get("proxy_layer")
        bl = r["bfs"]["layer"]
        if pl is not None:
            out.append(abs(pl - bl))
    return out


def _plot(name: str, nice: str) -> None:
    rows = json.load(open(f"logged_results/{name}/paper_results.json"))
    rows_by = _by_size(rows)
    xlabel = "Number of Available Pragmas" if "pragma" in name else "Number of Available Criteria"
    out_dir = f"plots/llama31/{name}"
    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted(rows_by.keys())
    cc_data = {budget: _load_cc_by_size(name, budget) for budget, *_ in LSM_METHODS}

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    handles_by_label: dict[str, object] = {}
    for ylabel, kind, ax in [("Accuracy (%)", "acc", axes[0]),
                              ("Avg Cost (LLM Calls)", "cost", axes[1]),
                              ("Avg. Hop Distance to BFS CF", "hop", axes[2])]:
        if kind == "hop":
            method_data = [
                (lab, color, {s: _hop_values(rows_by[s], col) for s in sizes})
                for col, lab, mk, ls, color, lw, is_bfs in PLOT_METHODS
                if not is_bfs
            ]
            method_data += [
                (lab, color, {s: _cc_hop_values(cc_data.get(budget), s) for s in sizes})
                for budget, lab, mk, ls, color, lw in LSM_METHODS
            ]
            draw_hop_bars(ax, sizes, method_data, handles_by_label)
        else:
            for col, lab, mk, ls, color, lw, is_bfs in PLOT_METHODS:
                if is_bfs and kind == "acc":
                    continue
                xs, ys = _series(rows_by, col, kind, is_bfs)
                if xs:
                    line, = ax.plot(xs, ys, marker=mk, linestyle=ls, color=color,
                                    linewidth=lw, markersize=7, label=lab)
                    handles_by_label.setdefault(lab, line)
            for budget, lab, mk, ls, color, lw in LSM_METHODS:
                xs, ys = _cc_series(cc_data.get(budget), kind, sizes)
                if xs:
                    line, = ax.plot(xs, ys, marker=mk, linestyle=ls, color=color,
                                    linewidth=lw, markersize=7, label=lab)
                    handles_by_label.setdefault(lab, line)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3, axis="y" if kind == "hop" else "both")
        if kind == "acc":
            ax.set_ylim(0, 108)

    method_order = [lab for _, lab, *_ in PLOT_METHODS] + [lab for _, lab, *_ in LSM_METHODS]
    ordered = [lab for lab in method_order if lab in handles_by_label]
    fig.legend([handles_by_label[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=True)
    fig.suptitle(nice, fontsize=11, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = f"{out_dir}/combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    for name, modpath, nice in DATASETS:
        print(f"\n=== {nice} ===")
        _compute_methods(name, modpath)

    apply_plot_style()
    for name, _, nice in DATASETS:
        _plot(name, nice)


if __name__ == "__main__":
    main()
