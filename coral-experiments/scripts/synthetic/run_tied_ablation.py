"""Run ERA/MRA entropy ablation on the synthetic tied dataset.

Samples sub-lattice cases with min_removed=1 (so layer-1 CFs qualify — the
common case when the baseline is tied), then runs mra_search for each of the
5 entropy variants over budgets [4, 8, 16], and produces a per-budget plot
plus a combined accuracy-vs-budget plot.

Also breaks accuracy out by query archetype (t2 / t3 / mix) so the effect of
2-way vs 3-way vs mixed ties is visible.

Outputs:
  logged_results/synthetic_tied/cases.json
  logged_results/synthetic_tied/mra_entropy_ablation/results_b{4,8,16}.json
  plots/synthetic_tied/mra_entropy_ablation_b{4,8,16}.png
  plots/synthetic_tied/accuracy_vs_budget.png
  plots/synthetic_tied/cost_vs_budget.png
  plots/synthetic_tied/accuracy_by_archetype.png

Usage:
    conda run -n coral python3 -m scripts.synthetic.run_tied_ablation
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.criteria.msmarco.run_subsample import score_proxy as llama_score_proxy
from scripts.shared.plot_style import apply_plot_style, draw_hop_bars
from scripts.shared.sampling import sample_cases
from search.bfs import bfs_search
from search.heuristics import renyi_entropy, shannon_entropy, weighted_entropy
from search.lattice import combo_key
from search.rank import mra_search

DATA_PATH = "data/synthetic/tied-lattices.json"
LOG_DIR   = "logged_results/synthetic_tied"
PLOT_DIR  = "plots/synthetic_tied"
BUDGETS   = [4, 8, 16]

# Same 5 variants as run_mra_entropy_ablation.
VARIANTS = [
    ("softmax",   shannon_entropy,                   "softmax"),
    ("shannon",   shannon_entropy,                   None),
    ("renyi_0.5", partial(renyi_entropy, alpha=0.5), None),
    ("renyi_2",   partial(renyi_entropy, alpha=2.0), None),
    ("weighted",  weighted_entropy,                  None),
]
COLORS  = ["#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]
MARKERS = ["o", "s", "v", "D", "^"]


def _load_lattices() -> dict[str, tuple[list, dict]]:
    raw = json.load(open(DATA_PATH))
    out = {}
    for qid, q in raw.items():
        if qid == "_meta" or q.get("halted"):
            continue
        crit = list(q["criteria"])
        rbc  = q["raw_answers"]
        if combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _sample_or_load_cases(lattices: dict) -> list[dict]:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = f"{LOG_DIR}/cases.json"
    if os.path.exists(path):
        cases = json.load(open(path))
        print(f"Loaded {len(cases)} cases from {path}")
        return cases
    # min_removed=2 so CF requires deeper search — makes the entropy-driven
    # Phase-1 budgeting materially affect accuracy, not just cost.
    cases = sample_cases(lattices, min_removed=2)
    json.dump(cases, open(path, "w"), indent=2)
    print(f"Sampled {len(cases)} cases → {path}")
    return cases


def _archetype(qid: str) -> str:
    return qid.split("_")[0]


def _run(cases: list[dict], lattices: dict, budget: int) -> list[dict]:
    results = []
    for i, case in enumerate(cases):
        qid, start_node, size = case["qid"], case["start_node"], case["size"]
        if qid not in lattices:
            continue
        criteria, rbc = lattices[qid]
        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        row = {
            "case_id":   i,
            "qid":       qid,
            "archetype": _archetype(qid),
            "size":      size,
            "start_node": start_node,
            "bfs_key":   bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "budget":    budget,
        }
        for name, entropy_fn, norm_fn in VARIANTS:
            res    = mra_search(start_node, rbc, budget=budget, start_node=start_node,
                                direction="down", normalize_fn=norm_fn, entropy_fn=entropy_fn)
            scored = llama_score_proxy(bfs_res, res, rbc, start_node)
            scored["calls"]       = res["calls"]
            scored["budget_used"] = res.get("budget_used", 0)
            row[name] = scored
        results.append(row)
    return results


def _plot_per_budget(results: list[dict], budget: int) -> None:
    apply_plot_style()
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)
    sizes = sorted(by_size)

    metrics = [
        ("acc",  "Accuracy (%)"),
        ("cost", "Avg Cost (LLM Calls)"),
        ("hop",  "Distance to True CF\n(incorrect cases only)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    handles_by_label: dict = {}

    for col_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[col_idx]
        if metric == "hop":
            method_data = []
            for vi, (name, _, _n) in enumerate(VARIANTS):
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
            ax.set_xlabel("Sub-lattice size")
            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.3, axis="y")
            continue

        for vi, (name, _, _n) in enumerate(VARIANTS):
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
        ax.set_xlabel("Sub-lattice size")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
        if metric == "acc":
            ax.set_ylim(0, 108)

    order = [n for n, _, _ in VARIANTS] + ["BFS"]
    ordered = [l for l in order if l in handles_by_label]
    fig.legend([handles_by_label[l] for l in ordered], ordered,
               loc="upper center", ncol=len(ordered),
               bbox_to_anchor=(0.5, 1.02), fontsize=11, frameon=True)
    fig.suptitle(f"Synthetic tied lattices — ERA entropy ablation (budget={budget})",
                 fontsize=12, fontweight="bold", y=1.09)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = f"{PLOT_DIR}/mra_entropy_ablation_b{budget}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_accuracy_vs_budget(all_results: dict[int, list[dict]]) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    budgets = sorted(all_results)
    for vi, (name, _, _n) in enumerate(VARIANTS):
        ys = []
        for b in budgets:
            rs = all_results[b]
            v = 100.0 * sum(1 for r in rs if r[name]["outcome"] == "correct") / len(rs)
            ys.append(v)
        ax.plot(budgets, ys, marker=MARKERS[vi], linestyle="-",
                color=COLORS[vi], linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Phase-1 budget")
    ax.set_ylabel("Accuracy (%)  — pooled over all sizes")
    ax.set_xticks(budgets)
    ax.set_ylim(0, 108)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Synthetic tied dataset — accuracy vs. budget", fontweight="bold")
    out = f"{PLOT_DIR}/accuracy_vs_budget.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_cost_vs_budget(all_results: dict[int, list[dict]]) -> None:
    """Two side-by-side panels: absolute LLM cost vs. budget, and % of BFS
    cost vs. budget. Pooled over all sizes."""
    apply_plot_style()
    budgets = sorted(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for vi, (name, _, _n) in enumerate(VARIANTS):
        ys = []
        for b in budgets:
            rs = all_results[b]
            v = float(np.mean([r[name]["calls"] + r[name].get("budget_used", 0) + r["size"]
                               for r in rs]))
            ys.append(v)
        ax.plot(budgets, ys, marker=MARKERS[vi], linestyle="-",
                color=COLORS[vi], linewidth=2, markersize=8, label=name)
    bfs_ys = [float(np.mean([r["bfs_calls"] for r in all_results[b]])) for b in budgets]
    ax.plot(budgets, bfs_ys, marker="o", linestyle="--", color="black",
            linewidth=2, markersize=8, label="BFS")
    ax.set_xlabel("Phase-1 budget")
    ax.set_ylabel("Avg LLM calls per case (pooled)")
    ax.set_xticks(budgets)
    ax.set_title("Absolute cost", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    for vi, (name, _, _n) in enumerate(VARIANTS):
        ys = []
        for b in budgets:
            rs = all_results[b]
            costs = [r[name]["calls"] + r[name].get("budget_used", 0) + r["size"]
                     for r in rs]
            bfs = [r["bfs_calls"] for r in rs]
            v = 100.0 * float(np.mean(costs)) / float(np.mean(bfs))
            ys.append(v)
        ax.plot(budgets, ys, marker=MARKERS[vi], linestyle="-",
                color=COLORS[vi], linewidth=2, markersize=8, label=name)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Phase-1 budget")
    ax.set_ylabel("Cost as % of BFS")
    ax.set_xticks(budgets)
    ax.set_title("Cost relative to BFS (lower = better)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Synthetic tied dataset — cost of each entropy variant",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"{PLOT_DIR}/cost_vs_budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_accuracy_by_archetype(all_results: dict[int, list[dict]]) -> None:
    apply_plot_style()
    archetypes = ["t2", "t3", "mix"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    budgets = sorted(all_results)
    for ai, arch in enumerate(archetypes):
        ax = axes[ai]
        for vi, (name, _, _n) in enumerate(VARIANTS):
            ys = []
            for b in budgets:
                rs = [r for r in all_results[b] if r["archetype"] == arch]
                if not rs:
                    ys.append(float("nan"))
                    continue
                v = 100.0 * sum(1 for r in rs if r[name]["outcome"] == "correct") / len(rs)
                ys.append(v)
            ax.plot(budgets, ys, marker=MARKERS[vi], linestyle="-",
                    color=COLORS[vi], linewidth=2, markersize=8, label=name)
        n_arch = sum(1 for r in all_results[budgets[0]] if r["archetype"] == arch)
        ax.set_title(f"{arch} archetype (n={n_arch})", fontweight="bold")
        ax.set_xlabel("Phase-1 budget")
        ax.set_xticks(budgets)
        ax.set_ylim(0, 108)
        ax.grid(True, alpha=0.3)
        if ai == 0:
            ax.set_ylabel("Accuracy (%)")
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Synthetic tied dataset — accuracy by archetype", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{PLOT_DIR}/accuracy_by_archetype.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()

    if args.resample:
        p = Path(f"{LOG_DIR}/cases.json")
        if p.exists():
            p.unlink()

    lattices = _load_lattices()
    print(f"Loaded {len(lattices)} synthetic lattices")
    cases = _sample_or_load_cases(lattices)

    all_results: dict[int, list[dict]] = {}
    for b in BUDGETS:
        print(f"\n=== budget={b} ===")
        results = _run(cases, lattices, b)
        all_results[b] = results
        out_dir = f"{LOG_DIR}/mra_entropy_ablation"
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"dataset": "synthetic_tied", "budget": b,
                   "variants": [n for n, _, _ in VARIANTS],
                   "results": results}, open(f"{out_dir}/results_b{b}.json", "w"), indent=2)
        print(f"  saved results_b{b}.json ({len(results)} cases)")
        _plot_per_budget(results, b)

    _plot_accuracy_vs_budget(all_results)
    _plot_cost_vs_budget(all_results)
    _plot_accuracy_by_archetype(all_results)

    # Console summary — one row per (variant, budget), showing accuracy and
    # avg cost pooled across all sizes.
    print("\n=== Pooled summary (variant × budget) ===")
    hdr = f"{'variant':<10} " + " ".join(f"b={b:<2} acc%   cost" for b in sorted(all_results))
    print(hdr)
    for name, _, _n in VARIANTS:
        row = f"{name:<10} "
        for b in sorted(all_results):
            rs = all_results[b]
            acc = 100.0 * sum(1 for r in rs if r[name]["outcome"] == "correct") / len(rs)
            cost = float(np.mean([r[name]["calls"] + r[name].get("budget_used", 0) + r["size"]
                                   for r in rs]))
            row += f"  {acc:5.1f}  {cost:5.1f}  "
        print(row)
    bfs_row = "BFS         "
    for b in sorted(all_results):
        cost = float(np.mean([r["bfs_calls"] for r in all_results[b]]))
        bfs_row += f"  {'—':>5}  {cost:5.1f}  "
    print(bfs_row)


if __name__ == "__main__":
    main()
