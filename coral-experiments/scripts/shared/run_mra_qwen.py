"""Compare MRA (zscore, minmax) against ERA, BFS, and SRA on Qwen3-32B datasets.

Sweeps budgets [4, 8, 16, 32, 64] for ERA and both MRA variants.
Plots accuracy and cost vs budget for all 4 datasets.

Usage:
    conda run -n coral python3 -m scripts.shared.run_mra_qwen
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from search.bfs import bfs_search
from search.lattice import combo_key
from search.rank import era_search, mra_search, sra_search

DATASETS = [
    ("msmarco_criteria", "data/qwen3-32b/msmarco-criteria-qwen3-32.json",  "MSMARCO\ncriteria"),
    ("msmarco_pragma",   "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",  "MSMARCO\npragma"),
    ("curated_criteria",  "data/qwen3-32b/curated-criteria-qwen3-32b.json",   "Curated\ncriteria"),
    ("curated_pragma",    "data/qwen3-32b/curated-pragmas-qwen3-32b.json",    "Curated\npragma"),
]

BUDGETS = [4, 8, 16, 32, 64]

COLORS = {
    "ERA":         "#59a14f",
    "MRA-zscore":  "#e15759",
    "MRA-minmax":  "#b07aa1",
    "SRA":         "#f28e2b",
    "BFS":         "#4e79a7",
}
MARKERS = {"ERA": "o", "MRA-zscore": "s", "MRA-minmax": "^"}


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


def _score(proxy_result: dict, lat: dict, bfs_layer: int) -> tuple[bool, int]:
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
    valid, bfs_results = [], []
    for lat in lattices:
        bfs = bfs_search(lat["criteria"], lat["rankings_by_combo"], start_node=lat["start_node"])
        if bfs["found"]:
            valid.append(lat)
            bfs_results.append(bfs)

    n = len(valid)
    print(f"\n{'='*60}\n  {name}  ({n} lattices)\n{'='*60}")

    bfs_cost = float(np.mean([b["calls"] for b in bfs_results]))

    sra_nc, sra_costs = 0, []
    for lat, bfs in zip(valid, bfs_results):
        r = sra_search(lat["criteria"], lat["rankings_by_combo"], start_node=lat["start_node"])
        c, cost = _score(r, lat, bfs["size"])
        sra_nc += c
        sra_costs.append(cost)
    sra_acc  = 100 * sra_nc / n
    sra_cost = float(np.mean(sra_costs))
    print(f"  SRA:  acc={sra_acc:.1f}%  cost={sra_cost:.1f}")

    era_acc, era_cost = [], []
    mra_z_acc, mra_z_cost = [], []
    mra_m_acc, mra_m_cost = [], []

    for b in BUDGETS:
        nc, cc = 0, []
        for lat, bfs in zip(valid, bfs_results):
            r = era_search(lat["criteria"], lat["rankings_by_combo"], budget=b,
                           start_node=lat["start_node"], direction="down", propagate_scores=True)
            c, cost = _score(r, lat, bfs["size"])
            nc += c; cc.append(cost)
        era_acc.append(100 * nc / n)
        era_cost.append(float(np.mean(cc)))

        nc, cc = 0, []
        for lat, bfs in zip(valid, bfs_results):
            r = mra_search(lat["criteria"], lat["rankings_by_combo"],
                           budget=b, start_node=lat["start_node"], normalize_fn="zscore")
            c, cost = _score(r, lat, bfs["size"])
            nc += c; cc.append(cost)
        mra_z_acc.append(100 * nc / n)
        mra_z_cost.append(float(np.mean(cc)))

        nc, cc = 0, []
        for lat, bfs in zip(valid, bfs_results):
            r = mra_search(lat["criteria"], lat["rankings_by_combo"],
                           budget=b, start_node=lat["start_node"], normalize_fn="minmax")
            c, cost = _score(r, lat, bfs["size"])
            nc += c; cc.append(cost)
        mra_m_acc.append(100 * nc / n)
        mra_m_cost.append(float(np.mean(cc)))

        print(f"  b={b:>3}:  ERA {era_acc[-1]:.0f}%/{era_cost[-1]:.1f}"
              f"  MRA-z {mra_z_acc[-1]:.0f}%/{mra_z_cost[-1]:.1f}"
              f"  MRA-m {mra_m_acc[-1]:.0f}%/{mra_m_cost[-1]:.1f}")

    return {
        "label": label,
        "n": n,
        "bfs_cost": bfs_cost,
        "sra": {"acc": sra_acc, "cost": sra_cost},
        "era":       {"acc": era_acc,   "cost": era_cost},
        "mra_zscore": {"acc": mra_z_acc, "cost": mra_z_cost},
        "mra_minmax": {"acc": mra_m_acc, "cost": mra_m_cost},
    }


def plot(results: list[dict], out: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    fig.suptitle(
        "MRA vs ERA vs baselines — Qwen3-32B (full lattice, size=10)",
        fontsize=12, fontweight="bold",
    )

    sweep_methods = [
        ("ERA",        "era"),
        ("MRA-zscore", "mra_zscore"),
        ("MRA-minmax", "mra_minmax"),
    ]

    for col, res in enumerate(results):
        ax_acc  = axes[0, col]
        ax_cost = axes[1, col]

        for label, key in sweep_methods:
            ax_acc.plot(BUDGETS, res[key]["acc"], color=COLORS[label],
                        marker=MARKERS[label], linewidth=1.8, markersize=5, label=label)
            ax_cost.plot(BUDGETS, res[key]["cost"], color=COLORS[label],
                         marker=MARKERS[label], linewidth=1.8, markersize=5, label=label)

        ax_acc.axhline(res["sra"]["acc"], color=COLORS["SRA"], linewidth=1.2,
                       linestyle="--", alpha=0.85, label=f"SRA ({res['sra']['acc']:.0f}%)")
        ax_cost.axhline(res["sra"]["cost"], color=COLORS["SRA"], linewidth=1.2,
                        linestyle="--", alpha=0.85, label=f"SRA ({res['sra']['cost']:.0f})")
        ax_cost.axhline(res["bfs_cost"], color=COLORS["BFS"], linewidth=1.2,
                        linestyle="--", alpha=0.85, label=f"BFS ({res['bfs_cost']:.0f})")

        ax_acc.set_title(f"{res['label']}\n(n={res['n']})", fontsize=9, fontweight="bold")
        ax_acc.set_ylim(0, 115)
        ax_acc.axhline(100, color="#aaa", linewidth=0.8, linestyle=":", alpha=0.6)
        ax_acc.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax_acc.legend(fontsize=7, loc="lower right")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

        ax_cost.set_ylabel("Avg total cost" if col == 0 else "")
        ax_cost.set_xlabel("Budget")
        ax_cost.set_xticks(BUDGETS)
        ax_cost.legend(fontsize=7, loc="upper left")
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot → {out}")
    plt.close(fig)


def main() -> None:
    results = []
    for name, path, label in DATASETS:
        results.append(run_dataset(name, path, label))
    plot(results, "plots/qwen3_mra_comparison.png")


if __name__ == "__main__":
    main()
