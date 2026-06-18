"""Plot / summarize ContextCite dev results from data/cc_results_dev.json.

Default eval budget is 64 ablations (aligned with ERA64). Use --sweep to plot
the full budget grid from a multi-budget results file.

Usage:
  conda run -n coral python3 plot_cc_results_dev.py
  conda run -n coral python3 plot_cc_results_dev.py --sweep
"""

import argparse
import json
import os
from collections import Counter, defaultdict

CC_RESULTS_PATH = "data/cc_results_dev.json"
CC_EVAL_BUDGET = 64
OUT_DIR = "plots/cc_ablation_study/dev"


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def filter_budget(results: list[dict], budget: int) -> list[dict]:
    filtered = [r for r in results if r["budget"] == budget]
    if not filtered:
        available = sorted({r["budget"] for r in results})
        raise ValueError(
            f"No rows with budget={budget}. Available budgets: {available}"
        )
    return filtered


def print_summary(results: list[dict], *, budget: int | None = None) -> None:
    cases = len({r["case_id"] for r in results})
    label = f"budget={budget}" if budget is not None else "all budgets"
    print(f"\nCC dev results ({label}): {len(results)} rows, {cases} cases")

    if budget is not None:
        outcomes = Counter(r["outcome"] for r in results)
        correct = outcomes.get("correct", 0)
        n = len(results)
        jaccs = [r["jaccard"] for r in results if r.get("jaccard") is not None]
        mean_j = sum(jaccs) / len(jaccs) if jaccs else float("nan")
        print(f"  accuracy: {correct/n:.1%} ({correct}/{n})")
        print(f"  outcomes: {dict(outcomes)}")
        print(f"  mean_jaccard: {mean_j:.3f}")
        print(f"  mean_calls: {sum(r['calls'] for r in results)/n:.1f}")
        by_k: dict[int, list[dict]] = defaultdict(list)
        for r in results:
            by_k[r["effective_size"]].append(r)
        print("  by k:")
        for k in sorted(by_k):
            bucket = by_k[k]
            c = sum(1 for r in bucket if r["outcome"] == "correct")
            print(f"    k={k}: {c}/{len(bucket)} = {c/len(bucket):.0%}")
        return

    budgets = sorted({r["budget"] for r in results})
    print(f"Budgets: {budgets}\n")
    print(f"  {'budget':>8} | {'acc':>6} | outcomes")
    print(f"  {'-'*8}-+-{'-'*6}-+----------")
    for b in budgets:
        br = [r for r in results if r["budget"] == b]
        oc = Counter(r["outcome"] for r in br)
        correct = oc.get("correct", 0)
        acc = correct / len(br) if br else 0.0
        print(f"  {b:>8} | {acc:>5.0%} | {dict(oc)}")


def plot_eval(results: list[dict], out_dir: str, budget: int) -> None:
    """Single-budget eval plots (comparable to ERA dev / test eval)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.shared.plot_style import apply_plot_style

    apply_plot_style()
    os.makedirs(out_dir, exist_ok=True)

    def _save_dual(fig, filename: str) -> None:
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    direction = results[0]["direction"] if results else "down"
    dir_label = "Top-Down" if direction == "down" else "Bottom-Up"
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["effective_size"]].append(r)
    sizes = sorted(by_size)

    outcome_order = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = {
        "correct": "#2ca02c",
        "wrong_layer": "#ff7f0e",
        "false_positive": "#d62728",
        "not_found": "#aec7e8",
    }

    # Accuracy by k
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    accs = [
        sum(1 for r in by_size[s] if r["outcome"] == "correct") / len(by_size[s])
        for s in sizes
    ]
    ax1.plot(sizes, accs, "P-", color="#ff7f0e", linewidth=2, markersize=8,
             label=f"CC (ablations={budget})")
    ax1.set_xlabel("Number of Criteria")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(dir_label)
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    fig1.suptitle("ContextCite Accuracy by Effective Size", fontweight="bold")
    fig1.tight_layout()
    _save_dual(fig1, f"accuracy_by_size_{direction}.png")
    plt.close(fig1)

    # Outcomes stacked
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    xm = np.arange(len(sizes))
    bottoms = np.zeros(len(sizes))
    for outcome in outcome_order:
        props = np.array([
            sum(1 for r in by_size[s] if r["outcome"] == outcome) / len(by_size[s])
            for s in sizes
        ])
        ax2.bar(xm, props, bottom=bottoms, label=outcome, color=outcome_colors[outcome])
        bottoms += props
    ax2.set_xticks(xm)
    ax2.set_xticklabels([f"{s}\n(n={len(by_size[s])})" for s in sizes])
    ax2.set_xlabel("Number of Criteria")
    ax2.set_ylabel("Proportion")
    ax2.set_title(f"{dir_label} — ablations={budget}")
    ax2.set_ylim(0, 1.18)
    ax2.legend(ncol=len(outcome_order), loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.suptitle("ContextCite Outcome Breakdown", fontweight="bold")
    fig2.tight_layout()
    _save_dual(fig2, f"outcomes_by_size_{direction}.png")
    plt.close(fig2)

    # Jaccard by k
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    jacc = []
    for s in sizes:
        vals = [r["jaccard"] for r in by_size[s] if r.get("jaccard") is not None]
        jacc.append(float(np.mean(vals)) if vals else float("nan"))
    ax3.plot(sizes, jacc, "P-", color="#ff7f0e", linewidth=2, markersize=8)
    ax3.set_xlabel("Number of Criteria")
    ax3.set_ylabel("Mean Jaccard(proxy CF, BFS CF)")
    ax3.set_title(dir_label)
    ax3.set_xticks(sizes)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    fig3.suptitle("ContextCite Counterfactual Jaccard", fontweight="bold")
    fig3.tight_layout()
    _save_dual(fig3, f"jaccard_by_size_{direction}.png")
    plt.close(fig3)

    # Calls by k
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    calls = [float(np.mean([r["calls"] for r in by_size[s]])) for s in sizes]
    ax4.plot(sizes, calls, "P-", color="#ff7f0e", linewidth=2, markersize=8)
    ax4.set_xlabel("Number of Criteria")
    ax4.set_ylabel("Average Search Calls")
    ax4.set_title(dir_label)
    ax4.set_xticks(sizes)
    ax4.grid(True, alpha=0.3)
    fig4.suptitle("ContextCite Search Calls", fontweight="bold")
    fig4.tight_layout()
    _save_dual(fig4, f"search_calls_by_size_{direction}.png")
    plt.close(fig4)

    gaps = [
        r["gap"] for r in results
        if r["outcome"] == "wrong_layer" and r.get("gap") is not None
    ]
    if gaps:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        counts = Counter(gaps)
        max_gap = max(gaps)
        x = list(range(1, max_gap + 1))
        ax5.bar(x, [counts.get(g, 0) for g in x], color="#ff7f0e", edgecolor="white")
        ax5.set_xticks(x)
        ax5.set_xlabel("Layer Gap")
        ax5.set_ylabel("Count")
        ax5.set_title(f"{dir_label} — ablations={budget}")
        ax5.grid(True, axis="y", alpha=0.3)
        fig5.suptitle("ContextCite Layer Gap (wrong-layer only)", fontweight="bold")
        fig5.tight_layout()
        _save_dual(fig5, f"layer_gaps_{direction}.png")
        plt.close(fig5)


def plot_sweep(results: list[dict], out_dir: str) -> None:
    """Multi-budget ablation plots (original sweep layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.shared.plot_style import apply_plot_style

    apply_plot_style()
    sweep_dir = os.path.join(out_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    def _save_dual(fig, filename: str) -> None:
        path = os.path.join(sweep_dir, filename)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    budgets = sorted({r["budget"] for r in results})
    direction = results[0]["direction"] if results else "down"
    dir_label = "Top-Down" if direction == "down" else "Bottom-Up"
    sizes = sorted({r["effective_size"] for r in results})

    cmap = plt.cm.viridis
    n_b = max(1, len(budgets) - 1)
    budget_colors = {b: cmap(i / n_b) for i, b in enumerate(budgets)}
    _markers = ["o", "v", "s", "^", "D", "P", "X", "*", "h"]
    budget_markers = {b: _markers[i % len(_markers)] for i, b in enumerate(budgets)}

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for budget in budgets:
        accs = []
        for s in sizes:
            bucket = [r for r in results if r["effective_size"] == s and r["budget"] == budget]
            accs.append(sum(1 for r in bucket if r["outcome"] == "correct") / len(bucket))
        ax1.plot(sizes, accs, f"{budget_markers[budget]}-",
                 label=f"ablations={budget}", color=budget_colors[budget],
                 linewidth=2, markersize=7)
    ax1.set_xlabel("Number of Criteria")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(dir_label)
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(ncol=len(budgets), loc="lower left")
    ax1.grid(True, alpha=0.3)
    fig1.suptitle("ContextCite Accuracy by Budget (num_ablations)", fontweight="bold")
    fig1.tight_layout()
    _save_dual(fig1, f"accuracy_by_size_{direction}.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    size_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sizes)))
    for i, s in enumerate(sizes):
        accs = []
        for budget in budgets:
            bucket = [r for r in results if r["effective_size"] == s and r["budget"] == budget]
            accs.append(sum(1 for r in bucket if r["outcome"] == "correct") / len(bucket))
        ax2.plot(budgets, accs, "o-", label=f"n={s}", color=size_colors[i], linewidth=2, markersize=6)
    ax2.set_xlabel("Budget (num_ablations)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(dir_label)
    ax2.set_xticks(budgets)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.legend(title="Effective size", ncol=4, loc="lower right")
    ax2.grid(True, alpha=0.3)
    fig2.suptitle("ContextCite Accuracy vs Budget", fontweight="bold")
    fig2.tight_layout()
    _save_dual(fig2, f"accuracy_vs_budget_{direction}.png")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="Plot full budget sweep (default: eval budget only)")
    parser.add_argument("--budget", type=int, default=CC_EVAL_BUDGET,
                        help=f"Eval budget to plot (default {CC_EVAL_BUDGET})")
    parser.add_argument("--results", default=CC_RESULTS_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    all_results = load_results(args.results)
    if args.sweep:
        print_summary(all_results)
        plot_sweep(all_results, args.out_dir)
    else:
        eval_results = filter_budget(all_results, args.budget)
        print_summary(eval_results, budget=args.budget)
        plot_eval(eval_results, args.out_dir, args.budget)


if __name__ == "__main__":
    main()
