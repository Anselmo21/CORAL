"""CRA vs ERA/SRA/BFS comparison in the same 3×4 format as temperature ablation.

Plots accuracy, cost, and hop-distance vs lattice size for all 4 datasets,
comparing BFS, SRA, ERA (b=4,8,16) against CRA variants (b=4,8,16).

Usage:
    conda run -n coral python3 -m scripts.shared.plot_cra_comparison
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from search.lattice import combo_key, jaccard_keys
from search.rank import mra_search, cra_search
from search.heuristics import (
    shannon_entropy, cra_top1_disagreement, cra_score_variance,
    cra_kendall_tau, cra_distance_to_aggregate,
)
from scripts.shared.plot_style import apply_plot_style, draw_hop_bars, bootstrap_ci_mean
from matplotlib.ticker import MaxNLocator

MODELS = {
    "llama31-8b": {
        "datasets": [
            ("msmarco_criteria", "data/llama31-8b/msmarco-criteria-lattices.json", "MSMARCO criteria"),
            ("msmarco_pragma",   "data/llama31-8b/msmarco-pragmas-lattices.json",  "MSMARCO pragma"),
            ("curated_criteria", "data/llama31-8b/curated-criteria-lattices.json",  "Curated criteria"),
            ("curated_pragma",   "data/llama31-8b/curated-pragmas-lattices.json",   "Curated pragma"),
        ],
        "result_root": "logged_results/llama31-8b",
        "label": "Llama-3.1-8B",
    },
    "qwen3-32b": {
        "datasets": [
            ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-lattices.json",  "MSMARCO criteria"),
            ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-lattices.json",   "MSMARCO pragma"),
            ("curated_criteria", "data/qwen3-32b/curated-criteria-lattices.json",   "Curated criteria"),
            ("curated_pragma",   "data/qwen3-32b/curated-pragmas-lattices.json",    "Curated pragma"),
        ],
        "result_root": "logged_results/qwen3-32b",
        "label": "Qwen3-32B",
    },
}

BUDGETS = [4, 8, 16]

CRA_METHODS = [
    ("CRA-top1",     cra_top1_disagreement),
    ("CRA-variance", cra_score_variance),
    ("CRA-dist2agg", cra_distance_to_aggregate),
]


def _load_lattices(data_path: str) -> dict[str, tuple[list[str], dict]]:
    raw = json.load(open(data_path))
    out = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q.get("criteria") or [])
        rbc = q.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _score_proxy(bfs_result, proxy_result, rbc, start_node) -> dict:
    start_key = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    bfs_key = bfs_result.get("key")
    bfs_layer = bfs_result.get("size") or bfs_result.get("layer")
    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")
    proxy_verified = (proxy_key is not None
                      and bool(rbc.get(proxy_key))
                      and rbc[proxy_key][0] != baseline_top1)
    if not proxy_result.get("found") or proxy_key is None:
        outcome = "not_found"
    elif not proxy_verified:
        outcome = "false_positive"
    elif proxy_layer == bfs_layer:
        outcome = "correct"
    else:
        outcome = "wrong_layer"
    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": proxy_verified,
        "outcome": outcome,
        "jaccard": jaccard_keys(bfs_key, proxy_key),
        "calls": proxy_result.get("calls", 0),
        "budget_used": proxy_result.get("budget_used", 0),
    }


def evaluate_model(model: str):
    cfg = MODELS[model]
    result_root = cfg["result_root"]
    all_results = {}

    for ds_name, data_path, nice in cfg["datasets"]:
        print(f"  {model} / {nice}")
        lattices = _load_lattices(data_path)
        paper_path = f"{result_root}/{ds_name}/paper_results.json"
        rows = json.load(open(paper_path))
        rows = [r for r in rows if r["qid"] in lattices]

        for row in rows:
            qid = row["qid"]
            start = row["start_node"]
            if qid not in lattices:
                continue
            crit, rbc = lattices[qid]
            bfs_key = row["bfs"].get("key")
            bfs_layer = row["bfs"].get("layer")
            bfs_proxy = {"key": bfs_key, "size": bfs_layer, "found": row["bfs"]["found"]}

            for b in BUDGETS:
                era_col = f"era_b{b}"
                if era_col not in row or row[era_col] is None:
                    result = mra_search(start, rbc, budget=b, start_node=start,
                                        direction="down", normalize_fn="softmax",
                                        entropy_fn=shannon_entropy)
                    row[era_col] = _score_proxy(bfs_proxy, result, rbc, start)

                for cra_name, cra_fn in CRA_METHODS:
                    col = f"{cra_name}_b{b}"
                    result = cra_search(start, rbc, budget=b, start_node=start,
                                        direction="down", normalize_fn="zscore",
                                        informativeness_fn=cra_fn)
                    row[col] = _score_proxy(bfs_proxy, result, rbc, start)

        all_results[ds_name] = rows
        n = len(rows)
        for b in BUDGETS:
            era_acc = 100 * sum(1 for r in rows if r.get(f"era_b{b}", {}).get("outcome") == "correct") / n
            parts = [f"ERA={era_acc:.1f}%"]
            for cra_name, _ in CRA_METHODS:
                col = f"{cra_name}_b{b}"
                acc = 100 * sum(1 for r in rows if r.get(col, {}).get("outcome") == "correct") / n
                parts.append(f"{cra_name}={acc:.1f}%")
            print(f"    b={b}: {', '.join(parts)}")

    return all_results


def _by_size(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["size"]].append(r)
    return dict(sorted(out.items()))


def _series(rows_by, col, kind, is_bfs=False):
    xs, ys = [], []
    for size, bucket in rows_by.items():
        cells = [r.get(col) for r in bucket if r.get(col)]
        if not cells:
            continue
        if kind == "acc":
            if is_bfs:
                v = 100.0
            else:
                v = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)
        elif kind == "cost":
            if is_bfs:
                v = float(np.mean([c["calls"] for c in cells]))
            else:
                v = float(np.mean([c.get("calls", 0) + c.get("budget_used", 0) for c in cells]))
        else:
            v = 0.0
        xs.append(size)
        ys.append(v)
    return xs, ys


def _hop_by_size(rows, col, sizes):
    per_size = {s: [] for s in sizes}
    for r in rows:
        cell = r.get(col)
        if not cell or cell.get("outcome") == "correct":
            continue
        pl = cell.get("proxy_layer")
        bl = r["bfs"].get("layer")
        if pl is not None and bl is not None:
            per_size[r["size"]].append(float(abs(pl - bl)))
    return per_size


# ── visual config ────────────────────────────────────────────────────────────

ERA_COLORS = {4: "#1f77b4", 8: "#17becf", 16: "#9467bd"}
ERA_MARKERS = {4: "X", 8: "h", 16: "D"}

CRA_STYLE = {
    "CRA-top1": {
        "colors": {4: "#2ca02c", 8: "#98df8a", 16: "#006400"},
        "markers": {4: "^", 8: "P", 16: "v"},
        "linestyle": "-.",
    },
    "CRA-variance": {
        "colors": {4: "#ff7f0e", 8: "#ffbb78", 16: "#e377c2"},
        "markers": {4: ">", 8: "<", 16: "d"},
        "linestyle": ":",
    },
    "CRA-dist2agg": {
        "colors": {4: "#8c564b", 8: "#c49c94", 16: "#bcbd22"},
        "markers": {4: "p", 8: "H", 16: "*"},
        "linestyle": (0, (3, 1, 1, 1)),
    },
}


def plot_combined(model: str, all_results: dict):
    apply_plot_style()
    cfg = MODELS[model]
    datasets = cfg["datasets"]
    model_label = cfg["label"]
    plot_root = f"plots/cra_comparison/{model}"
    os.makedirs(plot_root, exist_ok=True)

    fig, axes = plt.subplots(3, len(datasets), figsize=(5 * len(datasets), 12))
    handles = {}

    for col_idx, (ds_name, _, nice) in enumerate(datasets):
        rows = all_results[ds_name]
        rows_by = _by_size(rows)
        all_sizes = sorted(rows_by.keys())
        xlabel = "Number of Available Pragmas" if "pragma" in ds_name else "Number of Available Criteria"

        for row_idx, (kind, ylabel) in enumerate([("acc", "Accuracy (%)"),
                                                    ("cost", "Avg Cost (LLM Calls)"),
                                                    ("hop", "Distance to True CF")]):
            ax = axes[row_idx, col_idx]

            if kind == "hop":
                method_data = []
                method_data.append(("SRA", "#d62728",
                                    _hop_by_size(rows, "sra", all_sizes)))
                for b in BUDGETS:
                    method_data.append((f"ERA (b={b})", ERA_COLORS[b],
                                        _hop_by_size(rows, f"era_b{b}", all_sizes)))
                for cra_name, _ in CRA_METHODS:
                    style = CRA_STYLE[cra_name]
                    for b in BUDGETS:
                        method_data.append((f"{cra_name} (b={b})", style["colors"][b],
                                            _hop_by_size(rows, f"{cra_name}_b{b}", all_sizes)))
                draw_hop_bars(ax, all_sizes, method_data, handles)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, alpha=0.3, axis="y")
            else:
                # BFS
                if kind == "cost":
                    xs, ys = _series(rows_by, "bfs", kind, is_bfs=True)
                    if xs:
                        line, = ax.plot(xs, ys, marker="o", linestyle="--", color="black",
                                        linewidth=2.0, markersize=7, label="BFS")
                        handles.setdefault("BFS", line)

                # SRA
                xs, ys = _series(rows_by, "sra", kind)
                if xs:
                    line, = ax.plot(xs, ys, marker="s", linestyle="-", color="#d62728",
                                    linewidth=1.8, markersize=7, label="SRA")
                    handles.setdefault("SRA", line)

                # ERA
                for b in BUDGETS:
                    lab = f"ERA (b={b})"
                    xs, ys = _series(rows_by, f"era_b{b}", kind)
                    if xs:
                        line, = ax.plot(xs, ys, marker=ERA_MARKERS[b], linestyle="-",
                                        color=ERA_COLORS[b], linewidth=1.5, markersize=6, label=lab)
                        handles.setdefault(lab, line)

                # CRA variants
                for cra_name, _ in CRA_METHODS:
                    style = CRA_STYLE[cra_name]
                    for b in BUDGETS:
                        lab = f"{cra_name} (b={b})"
                        xs, ys = _series(rows_by, f"{cra_name}_b{b}", kind)
                        if xs:
                            line, = ax.plot(xs, ys, marker=style["markers"][b],
                                            linestyle=style["linestyle"],
                                            color=style["colors"][b],
                                            linewidth=1.5, markersize=6, label=lab)
                            handles.setdefault(lab, line)

                ax.grid(True, alpha=0.3)
                if kind == "acc":
                    ax.set_ylim(0, 108)

            ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(all_sizes)
            if row_idx == 0:
                ax.set_title(nice)

    order = (["BFS", "SRA"]
             + [f"ERA (b={b})" for b in BUDGETS]
             + [f"{cra_name} (b={b})" for cra_name, _ in CRA_METHODS for b in BUDGETS])
    ordered = [l for l in order if l in handles]
    fig.legend([handles[l] for l in ordered], ordered,
               loc="upper center", ncol=min(len(ordered), 6),
               bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)
    fig.suptitle(f"{model_label} — CRA vs ERA/SRA Comparison", fontsize=15, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{plot_root}/combined_cra_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    for model in ["llama31-8b", "qwen3-32b"]:
        print(f"\n{'='*60}\n  {model}\n{'='*60}")
        results = evaluate_model(model)
        plot_combined(model, results)


if __name__ == "__main__":
    main()
