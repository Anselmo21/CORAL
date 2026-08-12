"""Evaluate CRA (Conflict-based Rank Aggregation) vs MRA baselines.

Runs on data/qwen3-32b, data/llama31-8b, and coral-runs-lattices (score-based).
Reuses existing cases from logged_results/ when available; samples fresh otherwise.

Usage:
    conda run -n coral python3 -m scripts.shared.run_cra_eval
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from search.bfs import bfs_search
from search.lattice import combo_key
from search.rank import mra_search, cra_search, _normalize_max, _ranking_to_score_vector
from search.heuristics import (
    shannon_entropy,
    softmax_entropy,
    cra_top1_disagreement,
    cra_score_variance,
    cra_kendall_tau,
    cra_distance_to_aggregate,
)
from scripts.shared.sampling import sample_cases, SEED

ROOT = Path(__file__).resolve().parents[2]

BUDGETS = [4, 8, 16]
SIZES = list(range(3, 11))

# ── dataset configs ──────────────────────────────────────────────────────────

RANK_DATASETS = {
    "qwen3-32b": [
        ("curated_criteria",  "data/qwen3-32b/curated-criteria-lattices.json"),
        ("curated_pragma",    "data/qwen3-32b/curated-pragmas-lattices.json"),
        ("msmarco_criteria",  "data/qwen3-32b/msmarco-criteria-lattices.json"),
        ("msmarco_pragma",    "data/qwen3-32b/msmarco-pragmas-lattices.json"),
    ],
    "llama31-8b": [
        ("curated_criteria",  "data/llama31-8b/curated-criteria-lattices.json"),
        ("curated_pragma",    "data/llama31-8b/curated-pragmas-lattices.json"),
        ("msmarco_criteria",  "data/llama31-8b/msmarco-criteria-lattices.json"),
        ("msmarco_pragma",    "data/llama31-8b/msmarco-pragmas-lattices.json"),
    ],
}

SCORE_DATASETS = {
    "qwen3-32b-scores": [
        ("curated_criteria",  "coral-runs-lattices/qwen3-32b/curated-criteria"),
        ("curated_pragma",    "coral-runs-lattices/qwen3-32b/curated-pragmas"),
        ("msmarco_criteria",  "coral-runs-lattices/qwen3-32b/msmarco-criteria"),
        ("msmarco_pragma",    "coral-runs-lattices/qwen3-32b/msmarco-pragmas"),
    ],
}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_rank_lattices(data_path: str) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    with open(ROOT / data_path) as f:
        raw = json.load(f)
    out = {}
    for qid, lat in raw.items():
        if qid == "_meta":
            continue
        crit = lat["criteria"]
        rbc = lat["raw_answers"]
        start_key = combo_key(crit)
        if start_key not in rbc:
            continue
        out[qid] = (crit, rbc)
    return out


def load_score_lattices(
    data_dir: str,
) -> dict[str, tuple[list[str], dict[str, list[str]], dict[str, dict[str, float]]]]:
    base = ROOT / data_dir
    out = {}
    for qid_dir in sorted(base.iterdir()):
        if not qid_dir.is_dir():
            continue
        qid = qid_dir.name
        cfg = json.load(open(qid_dir / "config.json"))
        ans = json.load(open(qid_dir / "answers.json"))
        atoms = cfg["atoms"]
        pa = cfg["plausible_answers"]

        rbc: dict[str, list[str]] = {}
        sbc: dict[str, dict[str, float]] = {}
        for key, entry in ans.items():
            ranking_indices = entry["ranking"]
            rbc[key] = [pa[i] for i in ranking_indices]
            # "-100" runs store 0-1 scores under "scores"; "-percentage" runs
            # store probabilities under "probs". Both feed scores_by_combo.
            scores_raw = entry["scores"] if "scores" in entry else entry["probs"]
            sbc[key] = {pa[int(k)]: v for k, v in scores_raw.items()}

        start_key = combo_key(atoms)
        if start_key not in rbc:
            continue
        out[qid] = (atoms, rbc, sbc)
    return out


def load_cases(model: str, dataset: str) -> list[dict] | None:
    p = ROOT / "logged_results" / model / dataset / "cases.json"
    if p.exists():
        return json.load(open(p))
    return None


def load_results(model: str, dataset: str) -> dict | None:
    p = ROOT / "logged_results" / model / dataset / "results.json"
    if not p.exists():
        return None
    r = json.load(open(p))
    if isinstance(r, list):
        return {entry.get("case_id", i): entry for i, entry in enumerate(r)}
    return r


# ── scoring ──────────────────────────────────────────────────────────────────

def score_result(proxy_result, bfs_result, rbc, start_node):
    start_key = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    bfs_layer = bfs_result["size"] if isinstance(bfs_result.get("size"), int) else bfs_result.get("layer")

    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")

    if proxy_key is not None:
        ranking = rbc.get(proxy_key, [])
        proxy_verified = bool(ranking) and ranking[0] != baseline_top1
    else:
        proxy_verified = False

    if not proxy_result.get("found") or proxy_key is None:
        outcome = "not_found"
    elif not proxy_verified:
        outcome = "false_positive"
    elif proxy_layer != bfs_layer:
        outcome = "wrong_layer"
    else:
        outcome = "correct"

    gap = None
    if outcome == "wrong_layer" and bfs_layer is not None and proxy_layer is not None:
        gap = abs(proxy_layer - bfs_layer)

    return {
        "outcome": outcome,
        "calls": proxy_result.get("calls", 0),
        "budget_used": proxy_result.get("budget_used", 0),
        "gap": gap,
    }


# ── methods ──────────────────────────────────────────────────────────────────

CRA_VARIANTS = [
    ("CRA-top1",     cra_top1_disagreement,     "zscore"),
    ("CRA-variance", cra_score_variance,         "zscore"),
    ("CRA-kendall",  cra_kendall_tau,            "zscore"),
    ("CRA-dist2agg", cra_distance_to_aggregate,  "zscore"),
]

MRA_VARIANTS = [
    ("MRA-zscore",  "zscore", shannon_entropy),
]


def run_methods_rank(
    start_node, rbc, budget,
) -> dict[str, dict]:
    results = {}
    for tag, norm, ent_fn in MRA_VARIANTS:
        r = mra_search(start_node, rbc, budget=budget, start_node=start_node,
                       normalize_fn=norm, entropy_fn=ent_fn)
        results[tag] = r

    for tag, info_fn, norm in CRA_VARIANTS:
        r = cra_search(start_node, rbc, budget=budget, start_node=start_node,
                       normalize_fn=norm, informativeness_fn=info_fn)
        results[tag] = r
    return results


def run_methods_score(
    start_node, rbc, sbc, budget,
) -> dict[str, dict]:
    results = {}
    for tag, norm, ent_fn in MRA_VARIANTS:
        r = mra_search(start_node, rbc, budget=budget, start_node=start_node,
                       normalize_fn=norm, entropy_fn=ent_fn,
                       scores_by_combo=sbc, voter_norm="max")
        results[f"{tag}-scores"] = r

    for tag, info_fn, norm in CRA_VARIANTS:
        r = cra_search(start_node, rbc, budget=budget, start_node=start_node,
                       normalize_fn=norm, informativeness_fn=info_fn,
                       scores_by_combo=sbc, voter_norm="max")
        results[f"{tag}-scores"] = r
    return results


# ── evaluation loop ──────────────────────────────────────────────────────────

def evaluate_rank_dataset(
    model: str, dataset: str, data_path: str,
) -> dict[int, dict[str, list[dict]]]:
    lattices = load_rank_lattices(data_path)
    cases = load_cases(model, dataset)
    existing_results = load_results(model, dataset)

    if cases is None:
        # Full-lattice cases only (skip expensive sub-lattice sampling)
        cases = [{"qid": qid, "size": 10, "start_node": criteria}
                 for qid, (criteria, _) in sorted(lattices.items())]

    rows: dict[int, dict[str, list[dict]]] = {s: {} for s in SIZES}
    n_skipped = 0

    for i, case in enumerate(cases):
        qid = case["qid"]
        size = case["size"]
        start_node = case["start_node"]

        if qid not in lattices:
            n_skipped += 1
            continue
        criteria, rbc = lattices[qid]

        bfs_result = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_result["found"]:
            n_skipped += 1
            continue

        # BFS + SRA from existing results
        if existing_results and i in existing_results:
            stored = existing_results[i]
            bfs_cell = {"outcome": "correct" if stored["bfs"]["found"] else "not_found",
                        "calls": stored["bfs"]["calls"], "budget_used": 0}
            rows[size].setdefault("BFS", []).append(bfs_cell)
            if "sra" in stored:
                rows[size].setdefault("SRA", []).append(stored["sra"])

        for b in BUDGETS:
            method_results = run_methods_rank(start_node, rbc, b)
            for tag, proxy in method_results.items():
                scored = score_result(proxy, bfs_result, rbc, start_node)
                rows[size].setdefault(f"{tag}-b{b}", []).append(scored)

    print(f"  {model}/{dataset}: {len(cases)} cases, {n_skipped} skipped")
    return rows


def evaluate_score_dataset(
    model: str, dataset: str, data_dir: str,
) -> dict[int, dict[str, list[dict]]]:
    lattices_score = load_score_lattices(data_dir)

    # Reuse existing qwen3-32b cases (qids overlap)
    cases = load_cases("qwen3-32b", dataset)
    if cases is None:
        # Fallback: use full-lattice cases only (no expensive sampling)
        cases = [{"qid": qid, "size": 10, "start_node": atoms}
                 for qid, (atoms, _, _) in sorted(lattices_score.items())]

    rows: dict[int, dict[str, list[dict]]] = {s: {} for s in SIZES}
    n_skipped = 0

    for i, case in enumerate(cases):
        qid = case["qid"]
        size = case["size"]
        start_node = case["start_node"]

        if qid not in lattices_score:
            n_skipped += 1
            continue
        atoms, rbc, sbc = lattices_score[qid]

        bfs_result = bfs_search(atoms, rbc, start_node=start_node, direction="down")
        if not bfs_result["found"]:
            n_skipped += 1
            continue

        for b in BUDGETS:
            # rank-only methods for comparison
            rank_results = run_methods_rank(start_node, rbc, b)
            for tag, proxy in rank_results.items():
                scored = score_result(proxy, bfs_result, rbc, start_node)
                rows[size].setdefault(f"{tag}-b{b}", []).append(scored)

            # score-based methods
            score_results = run_methods_score(start_node, rbc, sbc, b)
            for tag, proxy in score_results.items():
                scored = score_result(proxy, bfs_result, rbc, start_node)
                rows[size].setdefault(f"{tag}-b{b}", []).append(scored)

    print(f"  {model}/{dataset}: {len(cases)} cases, {n_skipped} skipped")
    return rows


# ── aggregation helpers ──────────────────────────────────────────────────────

def _acc(cells: list[dict]) -> float:
    if not cells:
        return float("nan")
    return 100 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)


def _cost(cells: list[dict]) -> float:
    if not cells:
        return float("nan")
    return float(np.mean([c.get("calls", 0) + c.get("budget_used", 0) for c in cells]))


def _mean_gap(cells: list[dict]) -> float:
    gaps = [c["gap"] for c in cells if c.get("outcome") == "wrong_layer" and c.get("gap") is not None]
    return float(np.mean(gaps)) if gaps else float("nan")


# ── reporting ────────────────────────────────────────────────────────────────

def print_summary(all_rows: dict[int, dict[str, list[dict]]], title: str, budget: int):
    print(f"\n{'='*70}")
    print(f"  {title}  (budget={budget})")
    print(f"{'='*70}")

    def agg_cells(method_key: str) -> list[dict]:
        return [c for s in SIZES for c in all_rows.get(s, {}).get(method_key, [])]

    method_keys = set()
    for s in SIZES:
        method_keys.update(all_rows.get(s, {}).keys())

    budget_keys = sorted([k for k in method_keys if k.endswith(f"-b{budget}")])
    other_keys = sorted([k for k in method_keys if not any(k.endswith(f"-b{b}") for b in BUDGETS)])

    print(f"\n  {'Method':<35} {'Acc':>6} {'Cost':>6} {'Gap':>6} {'N':>5}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    for key in other_keys + budget_keys:
        cells = agg_cells(key)
        if not cells:
            continue
        acc = _acc(cells)
        cost = _cost(cells)
        gap = _mean_gap(cells)
        gap_str = f"{gap:.2f}" if not np.isnan(gap) else "  -"
        print(f"  {key:<35} {acc:>5.1f}% {cost:>6.1f} {gap_str:>6} {len(cells):>5}")


# ── plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "MRA-zscore":       "#e15759",
    "CRA-top1":         "#4e79a7",
    "CRA-variance":     "#f28e2b",
    "CRA-kendall":      "#59a14f",
    "CRA-dist2agg":     "#76b7b2",
}
MARKERS = {
    "MRA-zscore": "s",
    "CRA-top1": "o", "CRA-variance": "D", "CRA-kendall": "v", "CRA-dist2agg": "P",
}


def plot_budget_sweep(
    all_rows: dict[int, dict[str, list[dict]]],
    title: str,
    out_path: str,
    methods: list[str] | None = None,
):
    if methods is None:
        methods = list(COLORS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    def agg_cells(method_key: str) -> list[dict]:
        return [c for s in SIZES for c in all_rows.get(s, {}).get(method_key, [])]

    for method in methods:
        color = COLORS.get(method, "#888888")
        marker = MARKERS.get(method, "o")
        accs, costs, gaps = [], [], []
        for b in BUDGETS:
            cells = agg_cells(f"{method}-b{b}")
            accs.append(_acc(cells))
            costs.append(_cost(cells))
            gaps.append(_mean_gap(cells))
        kw = dict(color=color, marker=marker, linewidth=1.8, markersize=5, label=method)
        axes[0].plot(BUDGETS, accs, "-", **kw)
        axes[1].plot(BUDGETS, costs, "-", **kw)
        gap_pts = [(b, g) for b, g in zip(BUDGETS, gaps) if not np.isnan(g)]
        if gap_pts:
            axes[2].plot(*zip(*gap_pts), "-", **kw)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
    axes[1].set_ylabel("Avg total cost")
    axes[2].set_ylabel("Mean layer gap")
    for ax in axes:
        ax.set_xlabel("Budget")
        ax.set_xticks(BUDGETS)
        ax.legend(fontsize=6, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out_path}")
    plt.close(fig)


def plot_size_breakdown(
    all_rows: dict[int, dict[str, list[dict]]],
    title: str,
    out_path: str,
    budget: int = 8,
    methods: list[str] | None = None,
):
    if methods is None:
        methods = list(COLORS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{title} (budget={budget})", fontsize=12, fontweight="bold")

    for method in methods:
        color = COLORS.get(method, "#888888")
        marker = MARKERS.get(method, "o")
        key = f"{method}-b{budget}"
        accs = [_acc(all_rows.get(s, {}).get(key, [])) for s in SIZES]
        costs = [_cost(all_rows.get(s, {}).get(key, [])) for s in SIZES]
        gaps = [_mean_gap(all_rows.get(s, {}).get(key, [])) for s in SIZES]
        kw = dict(color=color, marker=marker, linewidth=1.8, markersize=5, label=method)
        axes[0].plot(SIZES, accs, "-", **kw)
        axes[1].plot(SIZES, costs, "-", **kw)
        gap_pts = [(s, g) for s, g in zip(SIZES, gaps) if not np.isnan(g)]
        if gap_pts:
            axes[2].plot(*zip(*gap_pts), "-", **kw)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
    axes[1].set_ylabel("Avg total cost")
    axes[2].set_ylabel("Mean layer gap")
    for ax in axes:
        ax.set_xlabel("Lattice size")
        ax.set_xticks(SIZES)
        ax.legend(fontsize=6, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out_path}")
    plt.close(fig)


def plot_combined_datasets(
    results_by_dataset: dict[str, dict[int, dict[str, list[dict]]]],
    model: str,
    out_dir: str,
    budget: int = 8,
    methods: list[str] | None = None,
):
    if methods is None:
        methods = list(COLORS.keys())

    ds_names = list(results_by_dataset.keys())
    n_ds = len(ds_names)
    fig, axes = plt.subplots(3, n_ds, figsize=(4 * n_ds, 9), squeeze=False)
    fig.suptitle(f"{model} — CRA vs MRA (budget={budget})", fontsize=13, fontweight="bold")

    for col, ds_name in enumerate(ds_names):
        rows = results_by_dataset[ds_name]
        for method in methods:
            color = COLORS.get(method, "#888888")
            marker = MARKERS.get(method, "o")
            key = f"{method}-b{budget}"
            accs = [_acc(rows.get(s, {}).get(key, [])) for s in SIZES]
            costs = [_cost(rows.get(s, {}).get(key, [])) for s in SIZES]
            gaps = [_mean_gap(rows.get(s, {}).get(key, [])) for s in SIZES]
            kw = dict(color=color, marker=marker, linewidth=1.8, markersize=4, label=method)
            axes[0, col].plot(SIZES, accs, "-", **kw)
            axes[1, col].plot(SIZES, costs, "-", **kw)
            gap_pts = [(s, g) for s, g in zip(SIZES, gaps) if not np.isnan(g)]
            if gap_pts:
                axes[2, col].plot(*zip(*gap_pts), "-", **kw)

        axes[0, col].set_title(ds_name.replace("_", " "), fontsize=10, fontweight="bold")
        axes[0, col].set_ylim(0, 105)
        axes[0, col].axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)

    for row in range(3):
        labels = ["Accuracy (%)", "Avg total cost", "Mean layer gap"]
        axes[row, 0].set_ylabel(labels[row])
        for col in range(n_ds):
            axes[row, col].set_xlabel("Lattice size" if row == 2 else "")
            axes[row, col].set_xticks(SIZES)
            axes[row, col].legend(fontsize=5, loc="best")
            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model}_cra_combined_b{budget}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(ROOT)
    out_dir = "plots/cra"
    zscore_methods = list(COLORS.keys())

    # ── Rank-only datasets ───────────────────────────────────────────────────
    for model, datasets in RANK_DATASETS.items():
        print(f"\n{'#'*70}")
        print(f"  {model} (rank-only)")
        print(f"{'#'*70}")

        model_results: dict[str, dict[int, dict[str, list[dict]]]] = {}

        for dataset, data_path in datasets:
            print(f"\n  --- {dataset} ---")
            rows = evaluate_rank_dataset(model, dataset, data_path)
            model_results[dataset] = rows

            for b in BUDGETS:
                print_summary(rows, f"{model}/{dataset}", b)

            plot_budget_sweep(
                rows, f"{model} {dataset}", f"{out_dir}/{model}_{dataset}_budget.png",
                methods=zscore_methods,
            )
            plot_size_breakdown(
                rows, f"{model} {dataset}", f"{out_dir}/{model}_{dataset}_size.png",
                budget=8, methods=zscore_methods,
            )

        plot_combined_datasets(model_results, model, out_dir, budget=8, methods=zscore_methods)

    # ── Score-based datasets ─────────────────────────────────────────────────
    for model, datasets in SCORE_DATASETS.items():
        print(f"\n{'#'*70}")
        print(f"  {model} (score-based)")
        print(f"{'#'*70}")

        model_results = {}
        for dataset, data_dir in datasets:
            print(f"\n  --- {dataset} ---")
            rows = evaluate_score_dataset(model, dataset, data_dir)
            model_results[dataset] = rows

            for b in BUDGETS:
                print_summary(rows, f"{model}/{dataset}", b)

        # score-based uses more method variants — show the -scores suffix ones
        score_methods = []
        sample_rows = next(iter(model_results.values()))
        for s in SIZES:
            for k in sample_rows.get(s, {}).keys():
                base = k.rsplit("-b", 1)[0] if "-b" in k else k
                if base not in score_methods:
                    score_methods.append(base)
        score_cra = [m for m in score_methods if "CRA" in m and "scores" in m and "raw" not in m]
        score_mra = [m for m in score_methods if "MRA" in m and "scores" in m and "raw" not in m]
        rank_baseline = [m for m in score_methods if m in COLORS]

        for ds_name, rows in model_results.items():
            # rank vs score comparison for each CRA variant
            for b in BUDGETS:
                print_summary(rows, f"{model}/{ds_name}", b)


if __name__ == "__main__":
    main()
