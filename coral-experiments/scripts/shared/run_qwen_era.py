"""Evaluate BFS / SRA / ERA on the four Qwen3-32B datasets.

Ablates ERA over aggregator normalization functions (max-norm, z-score,
softmax, min-max) at budgets 4, 8, 16. Writes plots/qwen3_era_comparison.png.

Usage:
    conda run -n coral python3 -m scripts.shared.run_qwen_era
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from search.bfs import bfs_search
from search.heuristics import norm_minmax, norm_softmax, norm_zscore
from search.lattice import combo_key
from search.rank import _normalize_max, era_search, make_aggregator, sra_search

DATASETS = [
    ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "MSMARCO\ncriteria"),
    ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",  "MSMARCO\npragma"),
    ("curated_criteria",  "data/qwen3-32b/curated-criteria-qwen3-32b.json",   "Curated\ncriteria"),
    ("curated_pragma",    "data/qwen3-32b/curated-pragmas-qwen3-32b.json",    "Curated\npragma"),
]

ERA_BUDGETS = [4, 8, 16]

AGG_NORMS = [
    ("none",      lambda s: s),
    ("max-norm",  _normalize_max),
    ("z-score",   norm_zscore),
    ("softmax",   norm_softmax),
    ("min-max",   norm_minmax),
]

# Colors: SRA + one per agg norm
COLORS_SRA   = "#f28e2b"
COLORS_NORMS = ["#76b7b2", "#4e79a7", "#e15759", "#59a14f", "#b07aa1"]


def load_lattices(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    lattices = []
    for qid, q in data.items():
        if qid == "_meta" or q.get("halted"):
            continue
        criteria = list(q["criteria"])
        rbc: dict[str, list[str]] = q["raw_answers"]
        if combo_key(criteria) not in rbc:
            continue
        lattices.append({"qid": qid, "criteria": criteria,
                         "rankings_by_combo": rbc, "start_node": criteria})
    return lattices


def _score(proxy_result, lat, bfs_layer) -> tuple[bool, int]:
    key   = proxy_result.get("key")
    found = proxy_result.get("found", False)
    cost  = proxy_result.get("calls", 0) + proxy_result.get("budget_used", 0)
    if not found or key is None:
        return False, cost
    baseline = (lat["rankings_by_combo"].get(combo_key(lat["start_node"])) or [""])[0]
    ranking  = lat["rankings_by_combo"].get(key, [])
    verified = bool(ranking) and ranking[0] != baseline
    correct  = verified and proxy_result.get("size") == bfs_layer
    return correct, cost


def run_dataset(name: str, path: str, label: str) -> dict:
    lattices = load_lattices(path)
    valid, bfs_data = [], []
    for lat in lattices:
        bfs = bfs_search(lat["criteria"], lat["rankings_by_combo"],
                         start_node=lat["start_node"])
        if bfs["found"]:
            valid.append(lat)
            bfs_data.append(bfs)

    n = len(valid)
    print(f"\n{'='*72}\n  {name}  ({n} lattices)\n{'='*72}")

    # header
    col_w = 10
    header = f"  {'method':<22}"
    for norm_name, _ in AGG_NORMS:
        header += f"  {norm_name:>{col_w}}"
    print(header)
    print("  " + "-" * (22 + (col_w + 2) * len(AGG_NORMS)))

    result = {"label": label, "n": n, "bfs_cost": float(np.mean([b["calls"] for b in bfs_data]))}

    # SRA baseline
    sra_nc, sra_cc = 0, []
    for lat, bfs in zip(valid, bfs_data):
        r = sra_search(lat["criteria"], lat["rankings_by_combo"],
                       start_node=lat["start_node"])
        c, cost = _score(r, lat, bfs["size"])
        sra_nc += c; sra_cc.append(cost)
    result["sra"] = {"acc": 100 * sra_nc / n, "cost": float(np.mean(sra_cc))}
    print(f"  {'SRA (baseline)':<22}  {'acc':>{col_w}}  {'cost':>{col_w}}")
    print(f"  {'':22}  {result['sra']['acc']:>{col_w}.1f}%  {result['sra']['cost']:>{col_w}.1f}")

    # ERA ablation
    result["era"] = {}
    for b in ERA_BUDGETS:
        acc_row  = f"  {f'ERA b={b} — acc':<22}"
        cost_row = f"  {f'ERA b={b} — cost':<22}"
        result["era"][b] = {}
        for norm_name, norm_fn in AGG_NORMS:
            agg = make_aggregator(norm_fn)
            nc, cc = 0, []
            for lat, bfs in zip(valid, bfs_data):
                r = era_search(lat["criteria"], lat["rankings_by_combo"],
                               budget=b, start_node=lat["start_node"],
                               direction="down", propagate_scores=True,
                               aggregator_fn=agg)
                c, cost = _score(r, lat, bfs["size"])
                nc += c; cc.append(cost)
            acc  = 100 * nc / n
            cost = float(np.mean(cc))
            result["era"][b][norm_name] = {"acc": acc, "cost": cost}
            acc_row  += f"  {acc:>{col_w}.1f}%"
            cost_row += f"  {cost:>{col_w}.1f}"
        print(acc_row)
        print(cost_row)

    return result


def plot(results: list[dict], out: str) -> None:
    norm_names = [n for n, _ in AGG_NORMS]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    fig.suptitle(
        "ERA aggregator norm ablation — Qwen3-32B (full lattice, size=10)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    x      = np.arange(len(norm_names))
    width  = 0.22
    budget_styles = {4: dict(alpha=1.0), 8: dict(alpha=0.7), 16: dict(alpha=0.45)}

    for col, res in enumerate(results):
        ax_acc  = axes[0, col]
        ax_cost = axes[1, col]

        for row_ax, metric in [(ax_acc, "acc"), (ax_cost, "cost")]:
            for bi, b in enumerate(ERA_BUDGETS):
                vals = [res["era"][b][nm][metric] for nm in norm_names]
                offset = (bi - 1) * width
                bars = row_ax.bar(
                    x + offset, vals, width,
                    color=COLORS_NORMS, **budget_styles[b],
                    edgecolor="white", linewidth=0.4,
                )
                if metric == "acc":
                    for bar, v in zip(bars, vals):
                        row_ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=6,
                        )

            # SRA reference line
            sra_val = res["sra"][metric]
            row_ax.axhline(sra_val, color=COLORS_SRA, linewidth=1.2,
                           linestyle="--", alpha=0.8,
                           label=f"SRA ({sra_val:.0f}{'%' if metric=='acc' else ''})")

        ax_acc.set_title(res["label"], fontsize=10, fontweight="bold")
        ax_acc.set_ylim(0, 115)
        ax_acc.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax_acc.set_xticks(x); ax_acc.set_xticklabels(norm_names, fontsize=8)
        ax_acc.axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_acc.legend(fontsize=7, loc="lower right")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

        ax_cost.axhline(res["bfs_cost"], color="#4e79a7", linewidth=1.2,
                        linestyle="--", alpha=0.8,
                        label=f"BFS ({res['bfs_cost']:.0f})")
        ax_cost.set_ylabel("Avg total cost" if col == 0 else "")
        ax_cost.set_xticks(x); ax_cost.set_xticklabels(norm_names, fontsize=8)
        ax_cost.set_xlabel(f"n={res['n']}", fontsize=8)
        ax_cost.legend(fontsize=7, loc="upper right")
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)

    # shared legend for budget shading
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888", alpha=a, label=f"b={b}")
        for b, a in [(4, 1.0), (8, 0.7), (16, 0.45)]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02),
               title="ERA budget (darkness)", title_fontsize=8)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot → {out}")


def main() -> None:
    results = []
    for name, path, label in DATASETS:
        results.append(run_dataset(name, path, label))
    plot(results, "plots/qwen3_era_comparison.png")


if __name__ == "__main__":
    main()
