"""Depth-stratified accuracy: SRA vs ERA vs MRA across criteria vs pragma datasets.

Loads BFS/SRA/ERA from results.json; runs MRA-raw-softmax on-the-fly.
CF depth = bfs.layer / case.size; quartile brackets Q1–Q4.
4 panels (one per quartile), grouped bars = criteria vs pragma.

Usage:
    conda run -n coral python3 -m scripts.shared.plot_depth_stratified
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from search.heuristics import softmax_entropy
from search.lattice import combo_key
from search.rank import mra_search

DATASETS = [
    ("qwen3_msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "criteria"),
    ("qwen3_sonnet_criteria",  "data/qwen3-32b/sonnet-criteria-qwen3-32b.json",   "criteria"),
    ("qwen3_msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",   "pragma"),
    ("qwen3_sonnet_pragma",    "data/qwen3-32b/sonnet-pragmas-qwen3-32b.json",    "pragma"),
]

MRA_BUDGETS = [4, 8]
QUARTILES = [
    ("Q1\n[0, 0.25)",    0.00, 0.25),
    ("Q2\n[0.25, 0.50)", 0.25, 0.50),
    ("Q3\n[0.50, 0.75)", 0.50, 0.75),
    ("Q4\n[0.75, 1.0]",  0.75, 1.01),
]
ALGORITHMS = ["SRA", "ERA-4", "ERA-8", "MRA-4", "MRA-8"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rbc(data_path: str) -> dict[str, dict[str, list[str]]]:
    with open(data_path) as f:
        raw = json.load(f)
    out: dict[str, dict[str, list[str]]] = {}
    for qid, q in raw.items():
        if qid == "_meta":
            continue
        out[qid] = q["raw_answers"]
    return out


def _load_results(log_dir: str) -> list[dict]:
    with open(f"{log_dir}/results.json") as f:
        r = json.load(f)
    return r["results"]


# ---------------------------------------------------------------------------
# MRA scoring (mirrors _score_mra in run_mra_subsample.py)
# ---------------------------------------------------------------------------

def _score_mra(proxy_result: dict, bfs: dict, rbc: dict, start_node: list[str]) -> str:
    start_key      = combo_key(start_node)
    baseline_top1  = (rbc.get(start_key) or [""])[0]
    bfs_layer      = bfs["layer"]
    proxy_key      = proxy_result.get("key")
    proxy_layer    = proxy_result.get("size")

    if proxy_key is not None:
        ranking        = rbc.get(proxy_key, [])
        proxy_verified = bool(ranking) and ranking[0] != baseline_top1
    else:
        proxy_verified = False

    if not proxy_result.get("found") or proxy_key is None:
        return "not_found"
    if not proxy_verified:
        return "false_positive"
    if proxy_layer != bfs_layer:
        return "wrong_layer"
    return "correct"


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(log_dir: str, data_path: str, dtype: str) -> list[dict]:
    results = _load_results(log_dir)
    rbc_by_qid = _load_rbc(data_path)

    rows = []
    for res in results:
        bfs = res["bfs"]
        if not bfs["found"]:
            continue
        size  = res["size"]
        depth = bfs["layer"] / size
        qid   = res["qid"]
        start = res["start_node"]
        rbc   = rbc_by_qid.get(qid, {})

        record: dict[str, object] = {"dtype": dtype, "depth": depth,
                                      "BFS_cost": float(bfs.get("calls", 0))}

        # SRA
        sra = res["sra"]
        record["SRA_acc"]  = sra.get("outcome") == "correct"
        record["SRA_cost"] = sra.get("calls", 0) + sra.get("budget_used", 0)

        # ERA
        for b in [4, 8]:
            label = f"era_fixed_{b}"
            era_r = res.get("era", {}).get(label, {})
            record[f"ERA-{b}_acc"]  = era_r.get("outcome") == "correct"
            record[f"ERA-{b}_cost"] = era_r.get("calls", 0) + era_r.get("budget_used", 0)

        # MRA raw-softmax
        for b in MRA_BUDGETS:
            mra_r   = mra_search(start, rbc, budget=b, start_node=start,
                                 normalize_fn=None, entropy_fn=softmax_entropy)
            outcome = _score_mra(mra_r, bfs, rbc, start)
            record[f"MRA-{b}_acc"]  = (outcome == "correct")
            record[f"MRA-{b}_cost"] = mra_r.get("calls", 0) + mra_r.get("budget_used", 0)

        rows.append(record)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(all_rows: list[dict], out: str) -> None:
    def _q(depth: float) -> int | None:
        for i, (_, lo, hi) in enumerate(QUARTILES):
            if lo <= depth < hi:
                return i
        return None

    # buckets[qi][alg][dtype] = {"acc": [bool,...], "cost": [float,...]}
    empty = lambda: {"criteria": {"acc": [], "cost": []}, "pragma": {"acc": [], "cost": []}}
    buckets: list[dict] = [{alg: empty() for alg in ALGORITHMS} for _ in QUARTILES]
    bfs_costs: list[dict] = [{"criteria": [], "pragma": []} for _ in QUARTILES]

    for row in all_rows:
        qi = _q(row["depth"])
        if qi is None:
            continue
        dtype = row["dtype"]
        bfs_costs[qi][dtype].append(row["BFS_cost"])
        for alg in ALGORITHMS:
            buckets[qi][alg][dtype]["acc"].append(bool(row[f"{alg}_acc"]))
            buckets[qi][alg][dtype]["cost"].append(float(row[f"{alg}_cost"]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharey="row")
    fig.suptitle(
        "Accuracy & cost by CF depth quartile — Qwen3-32B subsampled (criteria vs pragma)",
        fontsize=12, fontweight="bold",
    )

    x      = np.arange(len(ALGORITHMS))
    width  = 0.35
    colors = {"criteria": "#4e79a7", "pragma": "#f28e2b"}

    for qi, (qlabel, _, _) in enumerate(QUARTILES):
        ax_acc  = axes[0, qi]
        ax_cost = axes[1, qi]
        q = buckets[qi]

        crit_n = len(q[ALGORITHMS[0]]["criteria"]["acc"])
        prag_n = len(q[ALGORITHMS[0]]["pragma"]["acc"])

        for metric, ax, ylabel, ylim, fmt in [
            ("acc",  ax_acc,  "Accuracy (%)", (0, 120), lambda v: f"{v:.0f}"),
            ("cost", ax_cost, "Avg total cost", None,   lambda v: f"{v:.1f}"),
        ]:
            crit_vals = [100 * np.mean(q[a]["criteria"][metric]) if metric == "acc"
                         else np.mean(q[a]["criteria"][metric]) if q[a]["criteria"][metric]
                         else float("nan")
                         for a in ALGORITHMS]
            prag_vals = [100 * np.mean(q[a]["pragma"][metric]) if metric == "acc"
                         else np.mean(q[a]["pragma"][metric]) if q[a]["pragma"][metric]
                         else float("nan")
                         for a in ALGORITHMS]

            bars_c = ax.bar(x - width / 2, crit_vals, width, color=colors["criteria"],
                            label=f"criteria (n={crit_n})", alpha=0.85)
            bars_p = ax.bar(x + width / 2, prag_vals, width, color=colors["pragma"],
                            label=f"pragma (n={prag_n})", alpha=0.85)

            for bars in (bars_c, bars_p):
                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h):
                        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                                fmt(h), ha="center", va="bottom", fontsize=6)

            if ylim:
                ax.set_ylim(*ylim)
                ax.axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
            if metric == "cost":
                for dtype, color in colors.items():
                    vals = bfs_costs[qi][dtype]
                    if vals:
                        ax.axhline(np.mean(vals), color=color, linewidth=1.2,
                                   linestyle="--", alpha=0.7,
                                   label=f"BFS {dtype} ({np.mean(vals):.0f})")
            ax.set_xticks(x)
            ax.set_xticklabels(ALGORITHMS, fontsize=8)
            ax.set_ylabel(ylabel if qi == 0 else "")
            ax.legend(fontsize=6, loc="upper right" if metric == "acc" else "upper left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_acc.set_title(qlabel, fontsize=10, fontweight="bold")

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = []
    for log_name, data_path, dtype in DATASETS:
        log_dir = f"logged_results/{log_name}"
        print(f"Evaluating {log_name} ...")
        rows = evaluate_dataset(log_dir, data_path, dtype)
        print(f"  {len(rows)} BFS-found cases")
        all_rows.extend(rows)

    plot(all_rows, "plots/qwen3_depth_stratified.png")


if __name__ == "__main__":
    main()
