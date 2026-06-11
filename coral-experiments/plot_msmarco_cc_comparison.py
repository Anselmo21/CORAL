"""Compare BFS / SRA / ERA(pct=10) / CC(pct=10) on the 8 MSMARCO test queries at k=10.

Reads:
  data/msmarco_dataset.json             — lattices + criteria
  data/msmarco_subsample/split.json     — dev/test qid split
  data/msmarco_cc_results_pct10.json    — pre-run CC results

Runs BFS / SRA / ERA(pct=10) inline for the 8 test queries.

Writes:
  plots/msmarco_cc_comparison/accuracy_cost.png
  plots/msmarco_cc_comparison/outcomes.png
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from search.bfs import bfs_search
from search.rank import era_search, sra_search
from search.lattice import combo_key
from plot_style import apply_plot_style

DATA_PATH = "data/msmarco_dataset.json"
SPLIT_PATH = "data/msmarco_subsample/split.json"
CC_PATH = "data/msmarco_cc_results_pct10.json"
OUT_DIR = "plots/msmarco_cc_comparison"

PCT = 10
K = 10
BUDGET = max(1, int(2**K * PCT / 100))  # 102


def _outcome(proxy_key: str | None, proxy_found: bool,
             bfs_key: str, rbc: dict[str, list[str]]) -> str:
    if not proxy_found or proxy_key is None:
        return "not_found"
    ranking = rbc.get(proxy_key)
    if not ranking:
        return "not_found"
    baseline_top1 = rbc[bfs_key.split(", ")[0]] if False else rbc[combo_key(bfs_key.split(", "))]
    # baseline = top-1 at the full criteria node
    return "cf_found"  # simplified — verified below


def _run_methods(test_qids: list[str], lattices: dict) -> list[dict]:
    rows: list[dict] = []
    for qid in test_qids:
        lat = lattices.get(qid)
        if lat is None:
            continue
        criteria: list[str] = list(lat["criteria"])
        rbc: dict[str, list[str]] = lat["rankings"]
        full_key = combo_key(criteria)
        if full_key not in rbc or not rbc[full_key]:
            continue
        baseline_top1: str = rbc[full_key][0]

        bfs = bfs_search(criteria, rbc, start_node=criteria, direction="down")
        sra = sra_search(criteria, rbc, start_node=criteria, direction="down")
        era = era_search(criteria, rbc, budget=BUDGET, start_node=criteria, direction="down")

        def _verify_and_classify(result: dict, method_name: str) -> dict:
            found: bool = result["found"]
            key: str | None = result.get("key")
            calls: int = result.get("calls", 0)
            budget_used: int = result.get("budget_used", 0)

            if not found or key is None:
                return {"method": method_name, "outcome": "not_found",
                        "calls": calls, "total_cost": calls + budget_used}

            r = rbc.get(key, [])
            if not r:
                return {"method": method_name, "outcome": "false_positive",
                        "calls": calls, "total_cost": calls + budget_used}

            proxy_top1 = r[0]
            if proxy_top1 == baseline_top1:
                return {"method": method_name, "outcome": "false_positive",
                        "calls": calls, "total_cost": calls + budget_used}

            bfs_layer = len(bfs["key"].split(", ")) if bfs["found"] else None
            proxy_layer = len(key.split(", ")) if key else None

            if bfs_layer is not None and proxy_layer == bfs_layer:
                outcome = "correct"
            else:
                outcome = "wrong_layer"

            return {"method": method_name, "outcome": outcome,
                    "calls": calls, "total_cost": calls + budget_used}

        rows.append({
            "qid": qid,
            "bfs": _verify_and_classify(bfs, "BFS"),
            "sra": _verify_and_classify(sra, "SRA"),
            "era": _verify_and_classify(era, "ERA"),
        })
    return rows


def _load_cc(test_qids: list[str]) -> dict[str, dict]:
    with open(CC_PATH) as f:
        cc_raw = json.load(f)
    out: dict[str, dict] = {}
    for row in cc_raw:
        qid = str(row["query_id"])
        if qid in test_qids:
            out[qid] = {
                "method": "CC",
                "outcome": row["outcome"],
                "calls": row["calls"],
                "total_cost": row["calls"] + BUDGET,
            }
    return out


def plot_accuracy_cost(rows: list[dict], cc_by_qid: dict[str, dict], out_path: str) -> None:
    methods = ["BFS", "SRA", "ERA", "CC"]
    colors = {"BFS": "black", "SRA": "#d62728", "ERA": "#1f77b4", "CC": "#2ca02c"}

    acc: dict[str, float] = {}
    avg_cost: dict[str, float] = {}

    for method in methods:
        results: list[dict] = []
        for row in rows:
            if method == "CC":
                r = cc_by_qid.get(row["qid"])
            else:
                r = row[method.lower()]
            if r is not None:
                results.append(r)

        if not results:
            acc[method] = float("nan")
            avg_cost[method] = float("nan")
            continue

        acc[method] = 100.0 * sum(1 for r in results if r["outcome"] == "correct") / len(results)
        avg_cost[method] = float(np.mean([r["total_cost"] for r in results]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Accuracy
    ax = axes[0]
    bars = ax.bar(methods, [acc[m] for m in methods],
                  color=[colors[m] for m in methods], width=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Accuracy (n=8, pct={PCT}%)")
    for bar, m in zip(bars, methods):
        v = acc[m]
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    # Cost
    ax = axes[1]
    bars = ax.bar(methods, [avg_cost[m] for m in methods],
                  color=[colors[m] for m in methods], width=0.5)
    ax.set_ylabel("Average Total LLM Calls")
    ax.set_title(f"Cost (n=8, pct={PCT}%)")
    for bar, m in zip(bars, methods):
        v = avg_cost[m]
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_outcomes(rows: list[dict], cc_by_qid: dict[str, dict], out_path: str) -> None:
    methods = ["BFS", "SRA", "ERA", "CC"]
    outcome_labels = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = {"correct": "#2ca02c", "wrong_layer": "#ff7f0e",
                      "false_positive": "#9467bd", "not_found": "#d62728"}

    counts: dict[str, Counter] = {m: Counter() for m in methods}
    for row in rows:
        for method in methods:
            if method == "CC":
                r = cc_by_qid.get(row["qid"])
            else:
                r = row[method.lower()]
            if r is not None:
                counts[method][r["outcome"]] += 1

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    for label in outcome_labels:
        vals = [counts[m][label] for m in methods]
        bars = ax.bar(x, vals, bottom=bottoms, label=label,
                      color=outcome_colors[label], width=0.5)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + v / 2, str(v),
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Number of Cases")
    ax.set_title(f"Outcomes — MSMARCO test (n={n}, pct={PCT}%)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    split = json.load(open(SPLIT_PATH))
    test_qids: list[str] = split["test"]

    data = json.load(open(DATA_PATH))
    lattices = data["lattices"]

    print(f"Running BFS / SRA / ERA(pct={PCT}, budget={BUDGET}) on {len(test_qids)} test queries...")
    rows = _run_methods(test_qids, lattices)
    print(f"Done. {len(rows)} cases processed.")

    cc_by_qid = _load_cc(test_qids)
    print(f"Loaded CC results for {len(cc_by_qid)} queries.")

    plot_accuracy_cost(rows, cc_by_qid, f"{OUT_DIR}/accuracy_cost.png")
    plot_outcomes(rows, cc_by_qid, f"{OUT_DIR}/outcomes.png")


if __name__ == "__main__":
    main()
