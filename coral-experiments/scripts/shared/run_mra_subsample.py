"""Evaluate MRA (zscore, minmax) on subsampled Qwen3-32B cases (sizes 3-10).

Loads existing BFS/SRA/ERA results from results.json, runs MRA on top,
then plots accuracy and cost vs lattice size for each dataset.

Usage:
    conda run -n coral python3 -m scripts.shared.run_mra_subsample
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import os

from search.heuristics import softmax_entropy
from search.lattice import combo_key
from search.rank import mra_search

DATASETS = [
    ("qwen3_msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "MSMARCO\ncriteria"),
    ("qwen3_msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",  "MSMARCO\npragma"),
    ("qwen3_curated_criteria",  "data/qwen3-32b/curated-criteria-qwen3-32b.json",   "Curated\ncriteria"),
    ("qwen3_curated_pragma",    "data/qwen3-32b/curated-pragmas-qwen3-32b.json",    "Curated\npragma"),
]

SIZES = list(range(3, 11))
MRA_BUDGETS = [4, 8, 16, 32, 64]
PLOT_BUDGET  = 8    # budget shown in the size-breakdown plot

COLORS = {
    "BFS":              "#4e79a7",
    "SRA":              "#f28e2b",
    "ERA":              "#59a14f",
    "MRA-zscore":       "#e15759",
    "MRA-max":          "#b07aa1",
    "MRA-zscore-softmax":  "#76b7b2",
    "MRA-max-softmax":     "#edc948",
    "MRA-raw-softmax":     "#ff9da7",
}
MARKERS = {"BFS": "D", "SRA": "v", "ERA": "o", "MRA-zscore": "s", "MRA-max": "^", "MRA-zscore-softmax": "P", "MRA-max-softmax": "X", "MRA-raw-softmax": "*"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(data_path: str) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    with open(data_path) as f:
        raw = json.load(f)
    out = {}
    for qid, q in raw.items():
        if qid == "_meta":
            continue
        out[qid] = (list(q["criteria"]), q["raw_answers"])
    return out


def _load_results(log_dir: str) -> tuple[list[dict], list[dict]]:
    with open(f"{log_dir}/cases.json") as f:
        cases = json.load(f)
    with open(f"{log_dir}/results.json") as f:
        r = json.load(f)
    results_by_id = {res["case_id"]: res for res in r["results"]}
    return cases, results_by_id


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_mra(proxy_result: dict, bfs: dict, rbc: dict, start_node: list[str]) -> dict:
    start_key     = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    bfs_layer     = bfs["layer"]
    proxy_key     = proxy_result.get("key")
    proxy_layer   = proxy_result.get("size")

    if proxy_key is not None:
        ranking       = rbc.get(proxy_key, [])
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

    gap: int | None = None
    if outcome == "wrong_layer" and bfs_layer is not None and proxy_layer is not None:
        gap = abs(proxy_layer - bfs_layer)

    return {
        "outcome":     outcome,
        "calls":       proxy_result.get("calls", 0),
        "budget_used": proxy_result.get("budget_used", 0),
        "gap":         gap,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_dataset(log_dir: str, data_path: str, label: str) -> dict:
    cases, results_by_id = _load_results(log_dir)
    lattices = _load_data(data_path)

    print(f"\n{'='*60}\n  {label}  ({len(cases)} cases)\n{'='*60}")

    # rows[size][budget_label] = list of {outcome, calls, budget_used}
    rows: dict[int, dict[str, list[dict]]] = {s: {} for s in SIZES}

    for i, case in enumerate(cases):
        case_id    = i
        stored     = results_by_id.get(case_id)
        if stored is None:
            continue

        qid        = case["qid"]
        size       = case["size"]
        start_node = case["start_node"]

        if qid not in lattices:
            continue
        criteria, rbc = lattices[qid]

        bfs = stored["bfs"]

        # Collect existing BFS/SRA/ERA results
        for method_label, cell in [
            ("BFS",  {"outcome": "correct" if bfs["found"] else "not_found",
                      "calls": bfs["calls"], "budget_used": 0}),
            ("SRA",  stored["sra"]),
        ]:
            rows[size].setdefault(method_label, []).append(cell)

        for b in MRA_BUDGETS:
            era_label = f"era_fixed_{b}"
            if era_label in stored.get("era", {}):
                rows[size].setdefault(f"ERA-b{b}", []).append(stored["era"][era_label])

        # Run MRA variants
        mra_variants = [
            ("zscore", None,            "MRA-zscore"),
            ("max",    None,            "MRA-max"),
            ("zscore", softmax_entropy, "MRA-zscore-softmax"),
            ("max",    softmax_entropy, "MRA-max-softmax"),
            (None,     softmax_entropy, "MRA-raw-softmax"),
        ]
        for norm, ent_fn, tag in mra_variants:
            for b in MRA_BUDGETS:
                r = mra_search(start_node, rbc, budget=b, start_node=start_node,
                               normalize_fn=norm, entropy_fn=ent_fn)
                scored = _score_mra(r, bfs, rbc, start_node)
                rows[size].setdefault(f"{tag}-b{b}", []).append(scored)

        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(cases)}] done", flush=True)

    print(f"  [{len(cases)}/{len(cases)}] done")
    return {"label": label, "rows": rows}


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _acc(cells: list[dict]) -> float:
    if not cells:
        return float("nan")
    return 100 * sum(1 for c in cells if c.get("outcome") == "correct") / len(cells)


def _cost(cells: list[dict]) -> float:
    if not cells:
        return float("nan")
    return float(np.mean([c.get("calls", 0) + c.get("budget_used", 0) for c in cells]))


def _mean_gap(cells: list[dict]) -> float:
    """Mean layer gap among wrong_layer cases only. NaN if none."""
    gaps = [c["gap"] for c in cells if c.get("outcome") == "wrong_layer" and c.get("gap") is not None]
    return float(np.mean(gaps)) if gaps else float("nan")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_size_breakdown(dataset_results: list[dict], budget: int, out: str) -> None:
    """3 rows (acc, cost, gap) × 4 cols (datasets), x-axis = size, fixed budget."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey="row")
    fig.suptitle(
        f"MRA vs ERA vs baselines — Qwen3-32B subsampled (budget={budget})",
        fontsize=12, fontweight="bold",
    )

    methods = [
        ("BFS",        "BFS"),
        ("SRA",        "SRA"),
        (f"ERA-b{budget}", "ERA"),
        (f"MRA-zscore-b{budget}",         "MRA-zscore"),
        (f"MRA-max-b{budget}",            "MRA-max"),
        (f"MRA-zscore-softmax-b{budget}", "MRA-zscore-softmax"),
        (f"MRA-max-softmax-b{budget}",    "MRA-max-softmax"),
        (f"MRA-raw-softmax-b{budget}",    "MRA-raw-softmax"),
    ]

    for col, ds in enumerate(dataset_results):
        rows = ds["rows"]
        ax_acc  = axes[0, col]
        ax_cost = axes[1, col]
        ax_gap  = axes[2, col]

        for method_key, display in methods:
            ls = "--" if display in ("BFS", "SRA") else "-"
            kw = dict(color=COLORS[display], marker=MARKERS[display],
                      linewidth=1.8, markersize=5, label=display)

            accs  = [_acc(rows[s].get(method_key, []))      for s in SIZES]
            costs = [_cost(rows[s].get(method_key, []))     for s in SIZES]
            gaps  = [_mean_gap(rows[s].get(method_key, [])) for s in SIZES]

            ax_acc.plot(SIZES, accs,  ls, **kw)
            ax_cost.plot(SIZES, costs, ls, **kw)
            gap_pts = [(s, g) for s, g in zip(SIZES, gaps) if not np.isnan(g)]
            if gap_pts:
                ax_gap.plot(*zip(*gap_pts), ls, **kw)

        ax_acc.set_title(ds["label"], fontsize=10, fontweight="bold")
        ax_acc.set_ylim(0, 115)
        ax_acc.axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_acc.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax_acc.set_xticks(SIZES)
        ax_acc.legend(fontsize=6, loc="lower left")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

        ax_cost.set_ylabel("Avg total cost" if col == 0 else "")
        ax_cost.set_xticks(SIZES)
        ax_cost.legend(fontsize=6, loc="upper left")
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)

        ax_gap.set_ylabel("Mean layer gap\n(wrong-layer only)" if col == 0 else "")
        ax_gap.set_xlabel("Lattice size")
        ax_gap.set_xticks(SIZES)
        ax_gap.legend(fontsize=6, loc="upper left")
        ax_gap.spines["top"].set_visible(False)
        ax_gap.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot → {out}")
    plt.close(fig)


def plot_budget_sweep(dataset_results: list[dict], out: str) -> None:
    """3 rows (acc, cost, gap) × 4 cols (datasets), x-axis = budget, aggregated over all sizes."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    fig.suptitle(
        "MRA vs ERA — budget sweep on Qwen3-32B subsampled (sizes 3–10 aggregated)",
        fontsize=12, fontweight="bold",
    )

    sweep_tags = [
        ("ERA",                "ERA"),
        ("MRA-zscore",         "MRA-zscore"),
        ("MRA-max",            "MRA-max"),
        ("MRA-zscore-softmax", "MRA-zscore-softmax"),
        ("MRA-max-softmax",    "MRA-max-softmax"),
        ("MRA-raw-softmax",    "MRA-raw-softmax"),
    ]

    for col, ds in enumerate(dataset_results):
        rows = ds["rows"]
        ax_acc  = axes[0, col]
        ax_cost = axes[1, col]
        ax_gap  = axes[2, col]

        def agg_cells(method_key: str) -> list[dict]:
            return [c for s in SIZES for c in rows[s].get(method_key, [])]

        sra_cells = agg_cells("SRA")
        bfs_cells = agg_cells("BFS")
        sra_acc, sra_cost = _acc(sra_cells), _cost(sra_cells)
        bfs_cost = _cost(bfs_cells)

        ax_acc.axhline(sra_acc, color=COLORS["SRA"], linewidth=1.2, linestyle="--",
                       alpha=0.85, label=f"SRA ({sra_acc:.0f}%)")
        ax_cost.axhline(sra_cost, color=COLORS["SRA"], linewidth=1.2, linestyle="--",
                        alpha=0.85, label=f"SRA ({sra_cost:.0f})")
        ax_cost.axhline(bfs_cost, color=COLORS["BFS"], linewidth=1.2, linestyle="--",
                        alpha=0.85, label=f"BFS ({bfs_cost:.0f})")

        for norm_tag, display in sweep_tags:
            accs, costs, gaps = [], [], []
            for b in MRA_BUDGETS:
                key = f"ERA-b{b}" if norm_tag == "ERA" else f"{norm_tag}-b{b}"
                cells = agg_cells(key)
                accs.append(_acc(cells))
                costs.append(_cost(cells))
                gaps.append(_mean_gap(cells))
            kw = dict(color=COLORS[display], marker=MARKERS[display],
                      linewidth=1.8, markersize=5, label=display)
            ax_acc.plot(MRA_BUDGETS, accs,  "-", **kw)
            ax_cost.plot(MRA_BUDGETS, costs, "-", **kw)
            gap_pts = [(b, g) for b, g in zip(MRA_BUDGETS, gaps) if not np.isnan(g)]
            if gap_pts:
                ax_gap.plot(*zip(*gap_pts), "-", **kw)

        ax_acc.set_title(ds["label"], fontsize=10, fontweight="bold")
        ax_acc.set_ylim(0, 115)
        ax_acc.axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_acc.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax_acc.legend(fontsize=6, loc="lower right")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

        ax_cost.set_ylabel("Avg total cost" if col == 0 else "")
        ax_cost.set_xticks(MRA_BUDGETS)
        ax_cost.legend(fontsize=6, loc="upper left")
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)

        ax_gap.set_ylabel("Mean layer gap\n(wrong-layer only)" if col == 0 else "")
        ax_gap.set_xlabel("Budget")
        ax_gap.set_xticks(MRA_BUDGETS)
        ax_gap.legend(fontsize=6, loc="upper left")
        ax_gap.spines["top"].set_visible(False)
        ax_gap.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    dataset_results = []
    for log_dir, data_path, label in DATASETS:
        ds = run_dataset(f"logged_results/{log_dir}", data_path, label)
        dataset_results.append(ds)

    plot_size_breakdown(dataset_results, PLOT_BUDGET, "plots/qwen3_mra_subsample_size.png")
    plot_budget_sweep(dataset_results, "plots/qwen3_mra_subsample_budget.png")


if __name__ == "__main__":
    main()
