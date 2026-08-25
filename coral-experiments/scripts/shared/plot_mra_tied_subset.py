"""Plot ERA/MRA entropy ablation restricted to sub-lattices whose singleton
Borda aggregation has a tie for 1st at the start_node.

For each of the 4 Llama-3.1-8B datasets at a chosen budget, reads the existing
mra_entropy_ablation/results_b<N>.json, identifies which case_ids have a
>=2-way tie (singleton Borda) at the start_node, pools those cases across
all 4 datasets, and produces a 3-panel plot (accuracy / cost / hop distance)
grouped by lattice size — mirroring plot layout from run_mra_entropy_ablation.

Usage:
    conda run -n coral python3 -m scripts.shared.plot_mra_tied_subset --budget 16
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style, draw_hop_bars
from search.lattice import combo_key

DATASETS = [
    ("msmarco_criteria", "data/llama31-8b/msmarco-criteria-lattices.json",  "MSMARCO criteria"),
    ("msmarco_pragma",   "data/llama31-8b/msmarco-pragmas-lattices.json",   "MSMARCO pragma"),
    ("curated_criteria",  "data/llama31-8b/curated-criteria-lattices.json",   "Curated criteria"),
    ("curated_pragma",    "data/llama31-8b/curated-pragmas-lattices.json",    "Curated pragma"),
]

VARIANTS = ["softmax", "shannon", "renyi_0.5", "renyi_2", "weighted"]
COLORS   = ["#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]
MARKERS  = ["o", "s", "v", "D", "^"]


def _load_lattices(path: str) -> dict[str, tuple[list, dict]]:
    raw = json.load(open(path))
    out = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q["criteria"])
        rbc  = q["raw_answers"]
        if combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _singleton_borda_tie_width(start_node: list[str], rbc: dict) -> int:
    answers, voters = set(), {}
    for c in start_node:
        r = rbc.get(combo_key([c]))
        if r is None:
            continue
        voters[c] = r
        answers.update(r)
    answers = sorted(answers)
    n = len(answers)
    scores = {a: 0.0 for a in answers}
    for r in voters.values():
        for rank, ans in enumerate(r):
            if ans in scores:
                scores[ans] += float(n - rank)
    if not scores:
        return 0
    top = max(scores.values())
    return sum(1 for v in scores.values() if v == top)


def _coerce_start_node(v) -> list[str]:
    if isinstance(v, list):
        return v
    return ast.literal_eval(v)


def _collect(budget: int, min_tie: int) -> tuple[list[dict], dict[str, int]]:
    pooled: list[dict] = []
    per_ds_count: dict[str, int] = {}
    for ds, lat_path, _nice in DATASETS:
        lattices = _load_lattices(lat_path)
        res_path = f"logged_results/llama31-8b/{ds}/mra_entropy_ablation/results_b{budget}.json"
        if not os.path.exists(res_path):
            print(f"  skip {ds}: {res_path} missing")
            per_ds_count[ds] = 0
            continue
        results = json.load(open(res_path))["results"]
        kept = 0
        for r in results:
            qid = r["qid"]
            if qid not in lattices:
                continue
            _crit, rbc = lattices[qid]
            sn = _coerce_start_node(r["start_node"])
            w = _singleton_borda_tie_width(sn, rbc)
            if w >= min_tie:
                pooled.append(r)
                kept += 1
        per_ds_count[ds] = kept
    return pooled, per_ds_count


def _plot(pooled: list[dict], budget: int, min_tie: int, out_path: str, subtitle: str) -> None:
    apply_plot_style()
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in pooled:
        by_size[int(r["size"])].append(r)
    sizes = sorted(by_size)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = [
        ("acc",  "Accuracy (%)"),
        ("cost", "Avg Cost (LLM Calls)"),
        ("hop",  "Distance to True CF\n(incorrect cases only)"),
    ]
    handles_by_label: dict = {}

    for col_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[col_idx]
        if metric == "hop":
            method_data = []
            for vi, name in enumerate(VARIANTS):
                per_size = {}
                for s in sizes:
                    hops = [abs(r[name]["proxy_layer"] - r["bfs_layer"])
                            for r in by_size[s]
                            if r[name].get("outcome") != "correct"
                            and r[name].get("proxy_layer") is not None]
                    per_size[s] = hops
                method_data.append((name, COLORS[vi], per_size))
            draw_hop_bars(ax, sizes, method_data, handles_by_label)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Number of Available Criteria/Pragmas")
            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.3, axis="y")
            continue

        for vi, name in enumerate(VARIANTS):
            ys = []
            for s in sizes:
                bucket = by_size[s]
                if metric == "acc":
                    v = 100.0 * sum(1 for r in bucket if r[name]["outcome"] == "correct") / len(bucket)
                else:
                    v = float(np.mean([r[name]["calls"] + r[name].get("budget_used", 0) + r["size"]
                                       for r in bucket]))
                ys.append(v)
            line, = ax.plot(sizes, ys, marker=MARKERS[vi], linestyle="-",
                            color=COLORS[vi], linewidth=2, markersize=7, label=name)
            handles_by_label.setdefault(name, line)
        if metric == "cost":
            bfs_cost = [float(np.mean([r["bfs_calls"] for r in by_size[s]])) for s in sizes]
            bfs_line, = ax.plot(sizes, bfs_cost, marker="o", linestyle="--", color="black",
                                linewidth=2, markersize=7, label="BFS")
            handles_by_label.setdefault("BFS", bfs_line)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Number of Available Criteria/Pragmas")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
        if metric == "acc":
            ax.set_ylim(0, 108)

    fig.suptitle(f"ERA entropy ablation, tied-only subset ({subtitle}, budget={budget})",
                 fontsize=12, fontweight="bold")
    order = VARIANTS + ["BFS"]
    ordered = [l for l in order if l in handles_by_label]
    fig.legend([handles_by_label[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 0.98), fontsize=11, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",  type=int, default=16)
    parser.add_argument("--min-tie", type=int, default=2,
                        help="Minimum singleton-Borda tie width at start_node")
    args = parser.parse_args()

    pooled, per_ds = _collect(args.budget, args.min_tie)
    print(f"\nTied cases per dataset (>= {args.min_tie}-way Borda tie at start_node):")
    for ds, n in per_ds.items():
        print(f"  {ds}: {n}")
    print(f"  pooled total: {len(pooled)}")

    subtitle = f"singleton-Borda tie >={args.min_tie}-way at start_node, pooled 4 datasets"
    out_path = f"plots/llama31-8b/mra_tied_subset_b{args.budget}.png"
    _plot(pooled, args.budget, args.min_tie, out_path, subtitle)


if __name__ == "__main__":
    main()
