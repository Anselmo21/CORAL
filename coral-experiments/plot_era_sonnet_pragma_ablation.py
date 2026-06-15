"""Plot ERA budget ablation results for the Sonnet pragma dev set.

Reads data/sonnet_pragma/era_ablation_results.json.
Outputs:
  plots/sonnet_pragma/era_accuracy_by_budget.png  — fixed / pct panels
  plots/sonnet_pragma/era_cost_by_budget.png      — fixed / pct panels
  plots/sonnet_pragma/era_outcomes.png            — stacked bar

CLI: conda run -n coral python3 plot_era_sonnet_pragma_ablation.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

RESULTS_PATH = "data/sonnet_pragma/era_ablation_results.json"
OUT_DIR = "plots/sonnet_pragma"

BUDGETS_FIXED = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT   = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _acc(rows: list[dict], key: str) -> float:
    correct = sum(1 for r in rows if r[key]["outcome"] == "correct")
    return 100.0 * correct / len(rows)


def _cost(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key]["calls"] + r[key]["budget_used"] for r in rows]))


def plot_accuracy(fixed: list[dict], pct: list[dict], out_dir: str) -> None:
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(10, 3.5))

    accs_f = [_acc(fixed, f"budget_{b}") for b in BUDGETS_FIXED]
    ax_f.plot(BUDGETS_FIXED, accs_f, "o-", color="#1f77b4", linewidth=2, markersize=7)
    ax_f.set_xlabel("Budget (ablations)")
    ax_f.set_ylabel("Accuracy (%)")
    ax_f.set_title("Fixed budget")
    ax_f.set_xscale("log", base=2)
    ax_f.set_xticks(BUDGETS_FIXED)
    ax_f.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_f.set_ylim(0, 108)
    ax_f.grid(True, alpha=0.3)

    accs_p = [_acc(pct, f"pct_{p}") for p in BUDGETS_PCT]
    ax_p.plot(BUDGETS_PCT, accs_p, "s-", color="#ff7f0e", linewidth=2, markersize=7)
    ax_p.set_xlabel("Budget (% of lattice)")
    ax_p.set_title("Percent budget")
    ax_p.set_xticks(BUDGETS_PCT)
    ax_p.set_ylim(0, 108)
    ax_p.grid(True, alpha=0.3)

    fig.suptitle("ERA Accuracy — Sonnet pragma dev (n=8, k=10)", fontsize=11)
    fig.tight_layout()
    path = f"{out_dir}/era_accuracy_by_budget.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cost(fixed: list[dict], pct: list[dict], out_dir: str) -> None:
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(10, 3.5))

    costs_f = [_cost(fixed, f"budget_{b}") for b in BUDGETS_FIXED]
    ax_f.plot(BUDGETS_FIXED, costs_f, "o-", color="#1f77b4", linewidth=2, markersize=7)
    ax_f.set_xlabel("Budget (ablations)")
    ax_f.set_ylabel("Avg total LLM calls")
    ax_f.set_title("Fixed budget")
    ax_f.set_xscale("log", base=2)
    ax_f.set_xticks(BUDGETS_FIXED)
    ax_f.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_f.grid(True, alpha=0.3)

    costs_p = [_cost(pct, f"pct_{p}") for p in BUDGETS_PCT]
    ax_p.plot(BUDGETS_PCT, costs_p, "s-", color="#ff7f0e", linewidth=2, markersize=7)
    ax_p.set_xlabel("Budget (% of lattice)")
    ax_p.set_title("Percent budget")
    ax_p.set_xticks(BUDGETS_PCT)
    ax_p.grid(True, alpha=0.3)

    fig.suptitle("ERA Cost — Sonnet pragma dev (n=8, k=10)", fontsize=11)
    fig.tight_layout()
    path = f"{out_dir}/era_cost_by_budget.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_outcomes(fixed: list[dict], pct: list[dict], out_dir: str) -> None:
    outcome_order  = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#aec7e8"]

    labels_f = [str(b) for b in BUDGETS_FIXED]
    labels_p = [f"{p}%" for p in BUDGETS_PCT]
    keys_f   = [f"budget_{b}" for b in BUDGETS_FIXED]
    keys_p   = [f"pct_{p}"   for p in BUDGETS_PCT]

    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, rows, keys, labels, title in [
        (ax_f, fixed, keys_f, labels_f, "Fixed budget"),
        (ax_p, pct,   keys_p, labels_p, "Percent budget"),
    ]:
        x = np.arange(len(keys))
        bottoms = np.zeros(len(keys))
        for outcome, color in zip(outcome_order, outcome_colors):
            vals = [sum(1 for r in rows if r[k]["outcome"] == outcome) for k in keys]
            ax.bar(x, vals, bottom=bottoms, color=color, label=outcome)
            bottoms += np.array(vals, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Cases")
        ax.set_title(title)
        ax.legend(ncol=2, fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ERA Outcomes — Sonnet pragma dev (n=8, k=10)", fontsize=11)
    fig.tight_layout()
    path = f"{out_dir}/era_outcomes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    apply_plot_style()

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    fixed = data["fixed"]
    pct   = data["pct"]
    print(f"Loaded {len(fixed)} fixed rows, {len(pct)} pct rows")

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_accuracy(fixed, pct, OUT_DIR)
    plot_cost(fixed, pct, OUT_DIR)
    plot_outcomes(fixed, pct, OUT_DIR)


if __name__ == "__main__":
    main()
