"""MRA budget ablation — fixed normalize_fn="softmax", entropy_fn=softmax_entropy.

Sweeps budget in [2, 4, 8, 16, 32] across the 4 Llama-3.1-8B datasets
(or Qwen3-32B with --model qwen3).

Results → logged_results/{ds}/mra_budget_ablation/results.json
           (qwen3: logged_results/qwen3-32b/{ds}/mra_budget_ablation/results.json)
Plot    → plots/llama31/mra_budget_ablation.png
           plots/qwen3-32b/mra_budget_ablation.png

Usage:
    conda run -n coral python3 -m scripts.shared.run_mra_budget_ablation
    conda run -n coral python3 -m scripts.shared.run_mra_budget_ablation --model qwen3
    conda run -n coral python3 -m scripts.shared.run_mra_budget_ablation --dataset msmarco_criteria
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.criteria.msmarco.run_subsample import score_proxy as _llama_score_proxy
from scripts.shared.plot_style import apply_plot_style
from search.bfs import bfs_search
from search.heuristics import softmax_entropy
from search.lattice import combo_key
from search.rank import mra_search

LLAMA_DATASETS = [
    ("msmarco_criteria", "scripts.criteria.msmarco.run_subsample", "MSMARCO criteria", "Criteria"),
    ("msmarco_pragma",   "scripts.pragma.msmarco.run_subsample",   "MSMARCO pragma",   "Pragmas"),
    ("sonnet_criteria",  "scripts.criteria.sonnet.run_subsample",  "Sonnet criteria",  "Criteria"),
    ("sonnet_pragma",    "scripts.pragma.sonnet.run_subsample",    "Sonnet pragma",    "Pragmas"),
]

QWEN3_DATASETS = [
    ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "Qwen3 MSMARCO criteria", "Criteria"),
    ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",  "Qwen3 MSMARCO pragma",   "Pragmas"),
    ("sonnet_criteria",  "data/qwen3-32b/sonnet-criteria-qwen3-32b.json",  "Qwen3 Sonnet criteria",  "Criteria"),
    ("sonnet_pragma",    "data/qwen3-32b/sonnet-pragmas-qwen3-32b.json",   "Qwen3 Sonnet pragma",    "Pragmas"),
]

BUDGETS = [2, 4, 8, 16, 32, 64]
COLORS  = ["#e377c2", "#9467bd", "#c5b0d5", "#1f77b4", "#aec7e8", "#17becf"]
MARKERS = ["D", "X", "h", "s", "o", "^"]


def _load_qwen3_lattices(data_path: str) -> dict[str, tuple[list, dict]]:
    raw = json.load(open(data_path))
    out = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q["criteria"])
        rbc  = q["raw_answers"]
        if combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _resolve_loader(mod):
    for name in ("_load_lattices", "_load_pragma_lattices"):
        fn = getattr(mod, name, None)
        if fn:
            return fn
    raise RuntimeError(f"no lattice loader on {mod.__name__}")


def _run(cases, lattices, score_proxy) -> list[dict]:
    results = []
    for i, case in enumerate(cases):
        qid        = case["qid"]
        start_node = case["start_node"]
        size       = case["size"]
        if qid not in lattices:
            continue
        criteria, rbc = lattices[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        row = {
            "case_id":    i,
            "qid":        qid,
            "size":       size,
            "start_node": start_node,
            "bfs_key":    bfs_res["key"],
            "bfs_layer":  bfs_res["size"],
            "bfs_calls":  bfs_res["calls"],
        }
        for budget in BUDGETS:
            res    = mra_search(start_node, rbc, budget=budget, start_node=start_node,
                                direction="down", normalize_fn="softmax",
                                entropy_fn=softmax_entropy)
            scored = score_proxy(bfs_res, res, rbc, start_node)
            scored["calls"]       = res["calls"]
            scored["budget_used"] = res.get("budget_used", 0)
            row[f"b{budget}"] = scored
        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)}", flush=True)
    return results


def _plot_all(all_results: dict[str, list[dict]], datasets: list,
              title: str, out_path: str) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    fig.suptitle(f"MRA budget ablation — normalize=softmax, entropy=softmax ({title})",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("acc",  "Accuracy (%)"),
        ("cost", "Avg Cost (LLM Calls)"),
        ("hop",  "Avg Hop Distance\n(incorrect cases only)"),
    ]

    for col_idx, (ds, _, nice, x_label) in enumerate(datasets):
        results = all_results.get(ds, [])
        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in results:
            by_size[r["size"]].append(r)
        sizes = sorted(by_size)

        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for bi, budget in enumerate(BUDGETS):
                col = f"b{budget}"
                ys = []
                for s in sizes:
                    bucket = by_size[s]
                    if metric == "acc":
                        v = 100.0 * sum(1 for r in bucket if r[col]["outcome"] == "correct") / len(bucket)
                    elif metric == "cost":
                        v = float(np.mean([r[col]["calls"] + r[col].get("budget_used", 0) for r in bucket]))
                    else:
                        hops = [abs(r[col]["proxy_layer"] - r["bfs_layer"])
                                for r in bucket
                                if r[col].get("outcome") != "correct"
                                and r[col].get("proxy_layer") is not None]
                        v = float(np.mean(hops)) if hops else float("nan")
                    ys.append(v)

                ax.plot(sizes, ys, marker=MARKERS[bi], linestyle="-",
                        color=COLORS[bi], linewidth=2, markersize=7,
                        label=f"b={budget}")

            if metric == "cost":
                bfs_cost = [float(np.mean([r["bfs_calls"] for r in by_size[s]])) for s in sizes]
                ax.plot(sizes, bfs_cost, marker="o", linestyle="--", color="black",
                        linewidth=2, markersize=7, label="BFS")

            if row_idx == 0:
                ax.set_title(nice, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel(f"Number of Available {x_label}" if row_idx == 2 else "")
            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if metric == "acc":
                ax.set_ylim(0, 108)

    handles = [plt.Line2D([0], [0], marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7)
               for i in range(len(BUDGETS))]
    handles += [plt.Line2D([0], [0], marker="o", linestyle="--", color="black", linewidth=2, markersize=7)]
    labels  = [f"b={b}" for b in BUDGETS] + ["BFS"]
    fig.legend(handles, labels, loc="upper center", ncol=len(BUDGETS) + 1,
               bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="llama31", choices=["llama31", "qwen3"])
    parser.add_argument("--dataset", default=None, help="run one dataset only")
    args = parser.parse_args()

    if args.model == "qwen3":
        datasets   = QWEN3_DATASETS
        log_prefix = "logged_results/qwen3-32b"
        plot_out   = "plots/qwen3-32b/mra_budget_ablation.png"
        title      = "Qwen3-32B"
    else:
        datasets   = LLAMA_DATASETS
        log_prefix = "logged_results"
        plot_out   = "plots/llama31/mra_budget_ablation.png"
        title      = "Llama-3.1-8B"

    targets = [d for d in datasets if args.dataset is None or d[0] == args.dataset]
    all_results: dict[str, list[dict]] = {}

    for ds, src, nice, x_label in targets:
        print(f"\n=== {nice} ===")

        if args.model == "qwen3":
            lattices    = _load_qwen3_lattices(src)
            score_proxy = _llama_score_proxy
            cases       = json.load(open(f"{log_prefix}/{ds}/cases.json"))
        else:
            mod         = import_module(src)
            lattices    = _resolve_loader(mod)(sorted({c["qid"] for c in json.load(open(f"{log_prefix}/{ds}/cases.json"))}))
            score_proxy = mod.score_proxy
            cases       = json.load(open(f"{log_prefix}/{ds}/cases.json"))

        print(f"  {len(cases)} cases, {len(lattices)} lattices")
        results = _run(cases, lattices, score_proxy)
        all_results[ds] = results

        out_dir = f"{log_prefix}/{ds}/mra_budget_ablation"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/results.json"
        json.dump({"dataset": ds, "budgets": BUDGETS,
                   "normalize_fn": "softmax", "entropy_fn": "softmax",
                   "results": results}, open(path, "w"), indent=2)
        print(f"  saved {path}")

    if args.dataset is None:
        _plot_all(all_results, datasets, title, plot_out)


if __name__ == "__main__":
    main()
