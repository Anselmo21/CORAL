"""Entropy function ablation on the MSMARCO pragma sub-lattice set.

Reuses the cases sampled by scripts.pragma.msmarco.run_subsample
(data/msmarco_pragma/cases.json) and runs ERA at the selected
pct_10 budget (max(1, int(0.10 * 2**size))) for each of the 5 entropy variants:
  softmax, shannon, renyi(α=0.5), renyi(α=2), weighted

Outputs:
  data/msmarco_pragma_entropy_ablation/results.json
  plots/msmarco_pragma_entropy_ablation/{accuracy_by_size,calls_by_size,outcomes}.png

CLI: conda run -n coral python3 run_msmarco_pragma_entropy_ablation.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from scripts.pragma.msmarco.run_subsample import score_proxy
from search.bfs import bfs_search
from search.heuristics import (
    renyi_entropy,
    shannon_entropy,
    softmax_entropy,
    weighted_entropy,
)
from search.rank import era_search

DATASET_PATH = "data/msmarco-pragma-final8-with-lattices.json"
CASES_PATH   = "data/msmarco_pragma/cases.json"
OUT_DIR      = "data/msmarco_pragma_entropy_ablation"
RESULTS_PATH = f"{OUT_DIR}/results.json"
PLOT_DIR     = "plots/msmarco_pragma_entropy_ablation"

BUDGET_PCT: int = 10  # percent of 2^size per case

ENTROPY_VARIANTS: list[tuple[str, object]] = [
    ("softmax",   softmax_entropy),
    ("shannon",   shannon_entropy),
    ("renyi_0.5", partial(renyi_entropy, alpha=0.5)),
    ("renyi_2",   partial(renyi_entropy, alpha=2.0)),
    ("weighted",  weighted_entropy),
]

COLORS  = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
MARKERS = ["s", "o", "^", "D", "v"]
DISPLAY_NAMES = {
    "renyi_0.5": "renyi (α=0.5)",
    "renyi_2":   "renyi (α=2)",
}


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _load_lattices() -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for rec in dataset:
        qid = str(rec["query_id"])
        criteria: list[str] = rec["lattice"]["config"]["criteria"]
        rbc: dict[str, list[str]] = rec["lattice"]["answers"]
        out[qid] = (criteria, rbc)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablation(
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
) -> list[dict]:
    results: list[dict] = []
    for i, case in enumerate(cases):
        qid = str(case["qid"])
        start_node = case["start_node"]
        size = case["size"]
        criteria, rbc = lattices[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")

        row: dict = {
            "case_id": i,
            "qid": qid,
            "size": size,
            "start_node": start_node,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
        }

        budget = max(1, int(2**size * BUDGET_PCT / 100))
        row["budget"] = budget
        row["budget_pct"] = BUDGET_PCT

        for name, entropy_fn in ENTROPY_VARIANTS:
            res = era_search(
                criteria, rbc,
                budget=budget,
                entropy_fn=entropy_fn,
                start_node=start_node,
                direction="down",
            )
            scored = score_proxy(bfs_res, res, rbc, start_node)
            scored["calls"] = res["calls"]
            scored["budget_used"] = res.get("budget_used", 0)
            row[name] = scored

        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cases)} cases done", flush=True)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    names = [n for n, _ in ENTROPY_VARIANTS]
    col_w = 16
    sep = f"  {'-'*4}-+-{'-'*3}-+" + f"-{'-'*(col_w-2)}-+" * len(names)

    print(f"\n{'=' * (30 + col_w * len(names))}")
    print(f"  ERA Entropy Ablation — MSMARCO pragma (budget=pct_{BUDGET_PCT}) — {len(results)} cases")
    print(f"{'=' * (30 + col_w * len(names))}")

    header = f"  {'Size':>4} | {'N':>3} |"
    for name in names:
        header += f" {name:>{col_w-2}} |"
    print(header)
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        line = f"  {size:>4} | {n:>3} |"
        for name in names:
            correct = sum(1 for r in bucket if r[name]["outcome"] == "correct")
            line += f" {correct/n:>4.0%} ({correct:>2}/{n:<2}) |"
        print(line)

    print(sep)
    total = len(results)
    totals = f"  {'All':>4} | {total:>3} |"
    for name in names:
        correct = sum(1 for r in results if r[name]["outcome"] == "correct")
        totals += f" {correct/total:>4.0%} ({correct:>2}/{total:<2}) |"
    print(totals)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_accuracy(results: list[dict], out_dir: str) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)
    sizes = sorted(by_size)
    names = [n for n, _ in ENTROPY_VARIANTS]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, name in enumerate(names):
        accs = [
            100.0 * sum(1 for r in by_size[s] if r[name]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]
        ax.plot(sizes, accs, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                linewidth=2, markersize=8)

    ax.set_xlabel("Number of Available Pragmas")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ordered = [_display(n) for n in ["shannon", "renyi_2", "softmax", "weighted", "renyi_0.5"]]
    order = [labels.index(n) for n in ordered]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, loc="lower left")
    fig.tight_layout()
    path = f"{out_dir}/accuracy_by_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_calls(results: list[dict], out_dir: str) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)
    sizes = sorted(by_size)
    names = [n for n, _ in ENTROPY_VARIANTS]

    # fig, ax = plt.subplots(figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, name in enumerate(names):
        avg_cost = [
            float(np.mean([r[name]["calls"] + r[name]["budget_used"] for r in by_size[s]]))
            for s in sizes
        ]
        ax.plot(sizes, avg_cost, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                linewidth=2, markersize=8)

    ax.set_xlabel("Number of Available Pragmas")
    ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    handles, labels = ax.get_legend_handles_labels()
    ordered = [_display(n) for n in ["shannon", "renyi_2", "softmax", "weighted", "renyi_0.5"]]
    order = [labels.index(n) for n in ordered]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{out_dir}/calls_by_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_outcomes(results: list[dict], out_dir: str) -> None:
    names = [n for n, _ in ENTROPY_VARIANTS]
    outcome_order  = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#aec7e8"]

    outcome_counts: dict[str, dict[str, int]] = {}
    for name in names:
        counts: dict[str, int] = defaultdict(int)
        for r in results:
            counts[r[name]["outcome"]] += 1
        outcome_counts[name] = dict(counts)

    x = np.arange(len(names))
    bottoms = np.zeros(len(names))

    fig, ax = plt.subplots(figsize=(8, 5))
    for outcome, oc in zip(outcome_order, outcome_colors):
        vals = [outcome_counts[name].get(outcome, 0) for name in names]
        ax.bar(x, vals, bottom=bottoms, color=oc, label=outcome)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([_display(n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("Cases")
    ax.legend(ncol=len(outcome_order), loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = f"{out_dir}/outcomes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_combined(results: list[dict], out_dir: str) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)
    sizes = sorted(by_size)
    names = [n for n, _ in ENTROPY_VARIANTS]
    ordered_names = ["shannon", "renyi_2", "softmax", "weighted", "renyi_0.5"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for i, name in enumerate(names):
        accs = [
            100.0 * sum(1 for r in by_size[s] if r[name]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]
        axes[0].plot(sizes, accs, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                     linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Available Pragmas")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(sizes)
    axes[0].grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    ordered = [_display(n) for n in ordered_names]
    order = [labels.index(n) for n in ordered]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, loc="lower left")

    for i, name in enumerate(names):
        avg_cost = [
            float(np.mean([r[name]["calls"] + r[name]["budget_used"] for r in by_size[s]]))
            for s in sizes
        ]
        axes[1].plot(sizes, avg_cost, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                     linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Available Pragmas")
    axes[1].set_ylabel("Average LLM Calls")
    axes[1].set_xticks(sizes)
    axes[1].grid(True, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    order = [labels.index(n) for n in ordered]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, loc="upper left")

    fig.tight_layout()
    path = f"{out_dir}/combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    apply_plot_style()
    os.makedirs(PLOT_DIR, exist_ok=True)

    if args.plot_only:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        results = data["results"]
        print(f"Loaded {len(results)} results from {RESULTS_PATH}")
        plot_accuracy(results, PLOT_DIR)
        plot_calls(results, PLOT_DIR)
        plot_outcomes(results, PLOT_DIR)
        plot_combined(results, PLOT_DIR)
        return

    with open(CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CASES_PATH}")

    lattices = _load_lattices()
    print(f"Loaded {len(lattices)} pragma lattices")

    print(f"\nRunning ERA (budget=pct_{BUDGET_PCT}) × {len(ENTROPY_VARIANTS)} entropy variants …")
    results = run_ablation(cases, lattices)

    print_summary(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "budget_pct": BUDGET_PCT,
            "entropy_variants": [n for n, _ in ENTROPY_VARIANTS],
            "n_cases": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")

    plot_accuracy(results, PLOT_DIR)
    plot_calls(results, PLOT_DIR)
    plot_outcomes(results, PLOT_DIR)


if __name__ == "__main__":
    main()
