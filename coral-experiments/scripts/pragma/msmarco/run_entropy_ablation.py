"""Entropy-function ablation on the 15-lattice MSMARCO regen.

Reuses the cases sampled by scripts.pragma.msmarco.run_15_lattices
(data/15_lattices_cf_layer2to8/cases.json) and runs ERA at pct_10 budget
(max(1, int(0.10 * 2**size))) for each of the 5 entropy variants:
  softmax, shannon, renyi(α=0.5), renyi(α=2), weighted

Outputs:
  data/15_lattices_entropy_ablation/results.json
  plots/15_lattices_entropy_ablation/{accuracy_by_size,calls_by_size,outcomes,combined}.png

CLI: conda run -n coral python3 -m scripts.pragma.msmarco.run_entropy_ablation
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
from scripts.pragma.msmarco.run_subsample import score_proxy, _load_lattices
from scripts.pragma.msmarco.run_16_lattices import _load_qids
from search.bfs import bfs_search
from search.heuristics import (
    renyi_entropy,
    shannon_entropy,
    softmax_entropy,
    weighted_entropy,
)
from search.rank import era_search

CASES_PATH   = "data/msmarco_pragma/cases.json"
OUT_DIR      = "data/16_lattices_entropy_ablation"
RESULTS_PATH = f"{OUT_DIR}/results.json"
PLOT_DIR     = "plots/16_lattices_entropy_ablation"

BUDGET_PCT: int = 10

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

        budget = max(1, int(2**size * BUDGET_PCT / 100))
        row: dict = {
            "case_id":   i,
            "qid":       qid,
            "size":      size,
            "start_node": start_node,
            "bfs_key":   bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "budget":    budget,
            "budget_pct": BUDGET_PCT,
        }

        for name, entropy_fn in ENTROPY_VARIANTS:
            res = era_search(
                criteria, rbc,
                budget=budget,
                entropy_fn=entropy_fn,
                start_node=start_node,
                direction="down",
            )
            scored = score_proxy(bfs_res, res, rbc, start_node)
            scored["calls"]       = res["calls"]
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
    print(f"  ERA Entropy Ablation — 15-lattice regen (budget=pct_{BUDGET_PCT}) — {len(results)} cases")
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

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
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
    ax.legend(ncol=2, loc="lower left", fontsize=8)
    ax.set_title("ERA entropy ablation — 15-lattice regen (pct_10)", fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, name in enumerate(names):
        avg_cost = [
            float(np.mean([r[name]["calls"] + r[name]["budget_used"] for r in by_size[s]]))
            for s in sizes
        ]
        ax.plot(sizes, avg_cost, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                linewidth=2, markersize=8)
    bfs_cost = [float(np.mean([r["bfs_calls"] for r in by_size[s]])) for s in sizes]
    ax.plot(sizes, bfs_cost, "o:", color="black", linewidth=2, markersize=7, label="BFS (oracle)")

    ax.set_xlabel("Number of Available Pragmas")
    ax.set_ylabel("Average LLM Calls (search + budget)")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, loc="upper left", fontsize=8)
    ax.set_title("ERA cost by entropy variant", fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, oc in zip(outcome_order, outcome_colors):
        vals = [outcome_counts[name].get(outcome, 0) for name in names]
        ax.bar(x, vals, bottom=bottoms, color=oc, label=outcome, edgecolor="black", linewidth=0.4)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([_display(n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("Cases")
    ax.legend(ncol=len(outcome_order), loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Outcome breakdown — entropy variants  (n={len(results)})",
                 fontweight="bold")
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

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(11, 4))

    for i, name in enumerate(names):
        accs = [
            100.0 * sum(1 for r in by_size[s] if r[name]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]
        ax_acc.plot(sizes, accs, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                    linewidth=2, markersize=8)
    ax_acc.set_xlabel("Number of Available Pragmas")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 108)
    ax_acc.set_xticks(sizes)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(ncol=2, loc="lower left", fontsize=8)

    for i, name in enumerate(names):
        avg_cost = [
            float(np.mean([r[name]["calls"] + r[name]["budget_used"] for r in by_size[s]]))
            for s in sizes
        ]
        ax_cost.plot(sizes, avg_cost, f"{MARKERS[i]}-", label=_display(name), color=COLORS[i],
                     linewidth=2, markersize=8)
    bfs_cost = [float(np.mean([r["bfs_calls"] for r in by_size[s]])) for s in sizes]
    ax_cost.plot(sizes, bfs_cost, "o:", color="black", linewidth=2, markersize=7, label="BFS")
    ax_cost.set_xlabel("Number of Available Pragmas")
    ax_cost.set_ylabel("Average LLM Calls")
    ax_cost.set_xticks(sizes)
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(ncol=2, loc="upper left", fontsize=8)

    fig.suptitle(f"ERA entropy ablation — 15-lattice regen, pct_10  (n={len(results)})",
                 fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_plot_style()

    with open(CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CASES_PATH}")

    qids = _load_qids()
    lattices = _load_lattices(qids)
    print(f"Loaded {len(lattices)} lattices")

    print(f"\nRunning ERA (budget=pct_{BUDGET_PCT}) × {len(ENTROPY_VARIANTS)} entropy variants ...")
    results = run_ablation(cases, lattices)

    print_summary(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "budget_pct":       BUDGET_PCT,
            "entropy_variants": [n for n, _ in ENTROPY_VARIANTS],
            "n_cases":          len(results),
            "results":          results,
        }, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_accuracy(results, PLOT_DIR)
    plot_calls(results, PLOT_DIR)
    plot_outcomes(results, PLOT_DIR)
    plot_combined(results, PLOT_DIR)


if __name__ == "__main__":
    main()
