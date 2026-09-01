"""ERA component ablation — sweeps one axis at a time across the four paper datasets.

Each axis fixes the other two components at their defaults and varies one:
  cutoff  — max_jump vs kneedle(S=0.5/1/2), fixed: softmax_entropy, norm=None
  norm    — none/minmax/zscore/rank/softmax, fixed: softmax_entropy, max_jump
  entropy — softmax/shannon/renyi_0.5/renyi_2/weighted, fixed: max_jump, norm=None

Outputs:
  logged_results/<dataset>/era_ablation/<axis>/results.json
  plots/<dataset>/era_ablation/<axis>/combined.png

CLI:
  python -m scripts.shared.run_era_ablation --dataset msmarco_criteria --axis cutoff
  python -m scripts.shared.run_era_ablation --all --axis norm
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from functools import partial
from importlib import import_module
from types import ModuleType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from search.bfs import bfs_search
from search.heuristics import (
    max_jump_cutoff,
    kneedle_cutoff,
    norm_minmax,
    norm_zscore,
    norm_rank,
    norm_softmax,
    renyi_entropy,
    shannon_entropy,
    softmax_entropy,
    weighted_entropy,
)
from search.rank import era_search

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

DATASETS: list[tuple[str, str, str]] = [
    ("msmarco_criteria", "scripts.criteria.msmarco.run_subsample", "Criteria"),
    ("curated_criteria",  "scripts.criteria.curated.run_subsample",  "Criteria"),
    ("msmarco_pragma",   "scripts.pragma.msmarco.run_subsample",   "Pragmas"),
    ("curated_pragma",    "scripts.pragma.curated.run_subsample",    "Pragmas"),
]

BUDGET_PCT: int = 10

# ---------------------------------------------------------------------------
# Variant definitions per axis
# ---------------------------------------------------------------------------

CUTOFF_VARIANTS: list[tuple[str, object]] = [
    ("max_jump",      max_jump_cutoff),
    ("kneedle_S0.5",  partial(kneedle_cutoff, S=0.5)),
    ("kneedle_S1",    partial(kneedle_cutoff, S=1)),
    ("kneedle_S2",    partial(kneedle_cutoff, S=2)),
]

NORM_VARIANTS: list[tuple[str, object | None]] = [
    ("none",     None),
    ("minmax",   norm_minmax),
    ("zscore",   norm_zscore),
    ("rank",     norm_rank),
    ("softmax",  norm_softmax),
]

ENTROPY_VARIANTS: list[tuple[str, object]] = [
    ("softmax",   softmax_entropy),
    ("shannon",   shannon_entropy),
    ("renyi_0.5", partial(renyi_entropy, alpha=0.5)),
    ("renyi_2",   partial(renyi_entropy, alpha=2.0)),
    ("weighted",  weighted_entropy),
]

AXES = {
    "cutoff":  CUTOFF_VARIANTS,
    "norm":    NORM_VARIANTS,
    "entropy": ENTROPY_VARIANTS,
}

# ---------------------------------------------------------------------------
# Plotting config
# ---------------------------------------------------------------------------

COLORS  = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
MARKERS = ["s", "o", "^", "D", "v"]

# ---------------------------------------------------------------------------
# Loader resolution
# ---------------------------------------------------------------------------

def _resolve_lattice_loader(mod: ModuleType):
    for name in ("_load_lattices", "_load_pragma_lattices"):
        fn = getattr(mod, name, None)
        if fn:
            return fn
    raise RuntimeError(f"No lattice loader on {mod.__name__}")

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_era_kwargs(axis: str, variant_fn) -> dict:
    """Build era_search kwargs for one variant given the axis being swept."""
    kwargs: dict = {}
    if axis == "cutoff":
        kwargs["cutoff_fn"] = variant_fn
        kwargs["entropy_fn"] = softmax_entropy
        # norm_fn defaults to None
    elif axis == "norm":
        kwargs["norm_fn"] = variant_fn  # None is valid (no-op)
        kwargs["entropy_fn"] = softmax_entropy
        kwargs["cutoff_fn"] = max_jump_cutoff
    elif axis == "entropy":
        kwargs["entropy_fn"] = variant_fn
        kwargs["cutoff_fn"] = max_jump_cutoff
        # norm_fn defaults to None
    return kwargs


def run_ablation(
    axis: str,
    variants: list[tuple[str, object | None]],
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
    score_proxy,
) -> list[dict]:
    results: list[dict] = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        start_node = case["start_node"]
        size = case["size"]
        criteria, rbc = lattices[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        budget = max(1, int(2**size * BUDGET_PCT / 100))

        row: dict = {
            "case_id": i,
            "qid": qid,
            "size": size,
            "start_node": start_node,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "budget": budget,
            "budget_pct": BUDGET_PCT,
        }

        for name, variant_fn in variants:
            era_kwargs = _build_era_kwargs(axis, variant_fn)
            res = era_search(
                criteria, rbc,
                budget=budget,
                start_node=start_node,
                direction="down",
                **era_kwargs,
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

def print_summary(results: list[dict], variants: list[tuple[str, object | None]], label: str, axis: str) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    names = [n for n, _ in variants]
    col_w = 18
    sep = f"  {'-'*4}-+-{'-'*3}-+" + f"-{'-'*(col_w-2)}-+" * len(names)

    print(f"\n{'=' * (30 + col_w * len(names))}")
    print(f"  {label} [{axis} axis] ERA ablation (budget=pct_{BUDGET_PCT}) — {len(results)} cases")
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

def plot_combined(
    results: list[dict],
    variants: list[tuple[str, object | None]],
    out_dir: str,
    x_label: str,
) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)
    sizes = sorted(by_size)
    names = [n for n, _ in variants]

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(10, 3.5))

    for i, name in enumerate(names):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        accs = [
            100.0 * sum(1 for r in by_size[s] if r[name]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]
        ax_acc.plot(sizes, accs, f"{marker}-", label=name, color=color, linewidth=2, markersize=8)

    ax_acc.set_xlabel(f"Number of Available {x_label}")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 108)
    ax_acc.set_xticks(sizes)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(ncol=2, loc="lower left", fontsize=8)

    for i, name in enumerate(names):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        avg_cost = [
            float(np.mean([r[name]["calls"] + r[name]["budget_used"] for r in by_size[s]]))
            for s in sizes
        ]
        ax_cost.plot(sizes, avg_cost, f"{marker}-", label=name, color=color, linewidth=2, markersize=8)

    bfs_cost = [float(np.mean([r["bfs_calls"] for r in by_size[s]])) for s in sizes]
    ax_cost.plot(sizes, bfs_cost, "o:", color="black", linewidth=2, markersize=7, label="BFS (oracle)")
    ax_cost.set_xlabel(f"Number of Available {x_label}")
    ax_cost.set_ylabel("Average LLM Calls")
    ax_cost.set_xticks(sizes)
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(ncol=2, loc="upper left", fontsize=8)

    fig.tight_layout()
    path = f"{out_dir}/combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

# ---------------------------------------------------------------------------
# Per-dataset entrypoint
# ---------------------------------------------------------------------------

def run_for_dataset(dataset: str, modpath: str, x_label: str, axis: str) -> None:
    variants = AXES[axis]
    cases_path   = f"logged_results/llama31-8b/{dataset}/cases.json"
    out_dir      = f"logged_results/llama31-8b/{dataset}/era_ablation/{axis}"
    plot_dir     = f"plots/{dataset}/era_ablation/{axis}"
    results_path = f"{out_dir}/results.json"

    with open(cases_path) as f:
        cases = json.load(f)
    print(f"[{dataset}] Loaded {len(cases)} cases from {cases_path}")

    mod = import_module(modpath)
    lattice_loader = _resolve_lattice_loader(mod)
    score_proxy = mod.score_proxy

    qids = sorted({c["qid"] for c in cases})
    lattices = lattice_loader(qids)
    print(f"[{dataset}] Loaded {len(lattices)} lattices")

    print(f"[{dataset}] Running ERA [{axis} axis] × {len(variants)} variants …")
    results = run_ablation(axis, variants, cases, lattices, score_proxy)

    print_summary(results, variants, label=dataset, axis=axis)

    os.makedirs(out_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "dataset": dataset,
            "axis": axis,
            "budget_pct": BUDGET_PCT,
            "variants": [n for n, _ in variants],
            "n_cases": len(results),
            "results": results,
        }, f, indent=2)
    print(f"[{dataset}] Results → {results_path}")

    os.makedirs(plot_dir, exist_ok=True)
    plot_combined(results, variants, plot_dir, x_label)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_plot_style()

    parser = argparse.ArgumentParser(description=__doc__)
    ds_group = parser.add_mutually_exclusive_group(required=True)
    ds_group.add_argument("--dataset", choices=[d for d, _, _ in DATASETS])
    ds_group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--axis", choices=list(AXES), required=True,
        help="Which ERA component to ablate",
    )
    args = parser.parse_args()

    targets = DATASETS if args.all else [d for d in DATASETS if d[0] == args.dataset]
    for dataset, modpath, x_label in targets:
        run_for_dataset(dataset, modpath, x_label, args.axis)


if __name__ == "__main__":
    main()
