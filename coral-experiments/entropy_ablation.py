import argparse
import json
import os
import random
from collections import defaultdict
from functools import partial

from era_budget_ablation import load_dev_cases
from experiment import (
    DATA_PATH,
    SEED,
    TARGET_PER_BUCKET,
    generate_candidates,
    load_runs,
    sample_cases,
    score_proxy,
)
from search.bfs import bfs_search
from search.heuristics import (
    renyi_entropy,
    shannon_entropy,
    softmax_entropy,
    weighted_entropy,
)
from search.rank import era_search

DEV_LATTICES_PATH = "data/all_lattices.json"
DEV_CASES_PATH = "data/dev.jsonl"

ENTROPY_VARIANTS: list[tuple[str, object]] = [
    ("softmax", softmax_entropy),
    ("shannon", shannon_entropy),
    ("renyi_0.5", partial(renyi_entropy, alpha=0.5)),
    ("renyi_2", partial(renyi_entropy, alpha=2.0)),
    ("weighted", weighted_entropy),
]


def run_entropy_ablation(cases: list[dict]) -> list[dict]:
    results: list[dict] = []

    for case in cases:
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        eff = len(start_node) if direction == "down" else len(criteria) - len(start_node)
        budget = max(1, int((2**eff) * 0.1))

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)

        row: dict = {
            "query_id": case["query_id"],
            "effective_size": eff,
            "direction": direction,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
        }

        for name, entropy_fn in ENTROPY_VARIANTS:
            era_res = era_search(
                criteria, rbc,
                budget=budget,
                entropy_fn=entropy_fn,
                start_node=start_node,
                direction=direction,
            )
            scored = score_proxy(bfs_res, era_res, rbc, start_node)
            scored["calls"] = era_res["calls"]
            scored["budget_used"] = era_res.get("budget_used", 0)
            row[name] = scored

        results.append(row)

    return results


def print_summary(results: list[dict], direction: str) -> None:
    dir_results = [r for r in results if r["direction"] == direction]
    if not dir_results:
        return

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in dir_results:
        by_size[r["effective_size"]].append(r)

    label = "Down" if direction == "down" else "Up"
    names = [name for name, _ in ENTROPY_VARIANTS]
    col_w = 14
    print(f"\n{'=' * (30 + col_w * len(names))}")
    print(f"  {label} Search — ERA Accuracy by Entropy Function")
    print(f"{'=' * (30 + col_w * len(names))}")

    header = f"  {'Size':>4} | {'N':>3} |"
    for name in names:
        header += f" {name:>{col_w - 2}} |"
    print(header)

    sep = f"  {'-' * 4}-+-{'-' * 3}-+"
    for _ in names:
        sep += f"-{'-' * (col_w - 2)}-+"
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        line = f"  {size:>4} | {n:>3} |"
        for name in names:
            correct = sum(1 for r in bucket if r[name]["outcome"] == "correct")
            acc = correct / n
            line += f" {acc:>4.0%} ({correct:>2}/{n:<2}) |"
        print(line)

    total = len(dir_results)
    totals_line = f"  {'All':>4} | {total:>3} |"
    for name in names:
        correct = sum(1 for r in dir_results if r[name]["outcome"] == "correct")
        acc = correct / total
        totals_line += f" {acc:>4.0%} ({correct:>2}/{total:<2}) |"
    print(sep)
    print(totals_line)


def plot_results(
    results: list[dict],
    out_dir: str = "plots",
    directions: list[str] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from plot_style import apply_plot_style

    apply_plot_style()

    os.makedirs(out_dir, exist_ok=True)

    if directions is None:
        directions = ["down", "up"]
    directions = [d for d in directions if any(r["direction"] == d for r in results)]
    if not directions:
        print("No results to plot.")
        return

    dir_labels = {"down": "Top-Down", "up": "Bottom-Up"}
    dir_slug = {"down": "top_down", "up": "bottom_up"}
    names = [name for name, _ in ENTROPY_VARIANTS]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    markers = ["s", "o", "^", "D", "v"]

    for direction in directions:
        dir_results = [r for r in results if r["direction"] == direction]
        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in dir_results:
            by_size[r["effective_size"]].append(r)
        sizes = sorted(by_size)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(names):
            accs = [
                sum(1 for r in by_size[s] if r[name]["outcome"] == "correct") / len(by_size[s])
                for s in sizes
            ]
            ax.plot(sizes, accs, f"{markers[i]}-", label=name, color=colors[i],
                    linewidth=2, markersize=8)

        ax.set_xlabel("Number of Criteria")
        ax.set_ylabel("Accuracy")
        ax.set_title(dir_labels[direction])
        ax.set_xticks(sizes)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=len(names), loc="lower center",
                  bbox_to_anchor=(0.5, -0.28), frameon=True)

        fig.suptitle("ERA Accuracy by Entropy Function (budget=10%)", fontweight="bold")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        path = os.path.join(out_dir, f"entropy_ablation_accuracy_{dir_slug[direction]}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    n_panels = len(directions)
    fig2, axes2 = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), sharey=True, squeeze=False)
    for ax, direction in zip(axes2[0], directions):
        dir_results = [r for r in results if r["direction"] == direction]
        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in dir_results:
            by_size[r["effective_size"]].append(r)

        sizes = sorted(by_size)
        x = np.arange(len(sizes))
        n_bars = len(names)
        bar_width = 0.8 / n_bars

        for i, name in enumerate(names):
            total_calls = [
                sum(r[name]["calls"] + r[name]["budget_used"] for r in by_size[s]) / len(by_size[s])
                for s in sizes
            ]
            offset = (i - (n_bars - 1) / 2) * bar_width
            ax.bar(x + offset, total_calls, bar_width, label=name, color=colors[i])

        ax.set_xlabel("Effective Size")
        ax.set_ylabel("Average Total Calls")
        ax.set_title(dir_labels[direction])
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend(ncol=len(names), loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)

    fig2.suptitle("ERA Total Calls by Entropy Function (budget=10%)", fontweight="bold")
    fig2.tight_layout()
    calls_path = os.path.join(out_dir, "entropy_ablation_calls.png")
    fig2.savefig(calls_path, dpi=150)
    plt.close(fig2)
    print(f"Saved {calls_path}")

    fig3, axes3 = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5), squeeze=False)
    for ax, direction in zip(axes3[0], directions):
        dir_results = [r for r in results if r["direction"] == direction]
        outcome_counts: dict[str, dict[str, int]] = {}
        for name in names:
            counts: dict[str, int] = defaultdict(int)
            for r in dir_results:
                counts[r[name]["outcome"]] += 1
            outcome_counts[name] = dict(counts)

        outcome_order = ["correct", "wrong_layer", "false_positive", "not_found"]
        outcome_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#aec7e8"]
        x = np.arange(len(names))
        bottoms = np.zeros(len(names))

        for outcome, oc in zip(outcome_order, outcome_colors):
            vals = [outcome_counts[name].get(outcome, 0) for name in names]
            ax.bar(x, vals, bottom=bottoms, color=oc, label=outcome)
            bottoms += np.array(vals, dtype=float)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Cases")
        ax.set_title(dir_labels[direction])
        ax.legend(ncol=len(outcome_order), loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

    fig3.suptitle("Outcome Breakdown by Entropy Function", fontweight="bold")
    fig3.tight_layout()
    outcomes_path = os.path.join(out_dir, "entropy_ablation_outcomes.png")
    fig3.savefig(outcomes_path, dpi=150)
    plt.close(fig3)
    print(f"Saved {outcomes_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true",
                        help="Use dev.jsonl cases joined against all_lattices.json")
    parser.add_argument("--down-only", action="store_true",
                        help="Restrict to top-down lattices only")
    args = parser.parse_args()

    if args.dev:
        cases = load_dev_cases(DEV_LATTICES_PATH, DEV_CASES_PATH)
        if args.down_only:
            cases = [c for c in cases if c["direction"] == "down"]
        out_dir = "data/entropy_ablation_study/dev"
        plot_dir = "plots/entropy_ablation_study/dev"
        results_file = os.path.join(out_dir, "entropy_ablation_results.json")
        print(f"Loaded {len(cases)} dev cases from {DEV_CASES_PATH}")

        all_results = run_entropy_ablation(cases)

        directions = ["down"] if args.down_only else ["down", "up"]
        for direction in directions:
            print_summary(all_results, direction)

        os.makedirs(out_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_file}")

        plot_results(all_results, out_dir=plot_dir, directions=directions)
        return

    rng = random.Random(SEED)
    runs = load_runs(DATA_PATH)
    print(f"Loaded {len(runs)} runs from {DATA_PATH}")

    all_results: list[dict] = []
    directions = ["down"] if args.down_only else ["down", "up"]

    for direction in directions:
        for effective_size in range(3, 11):
            natives, sublattices = generate_candidates(runs, effective_size, direction)
            cases = sample_cases(natives, sublattices, TARGET_PER_BUCKET, rng)

            if not cases:
                print(f"  [{direction:>4} size={effective_size}] 0 cases (skipped)")
                continue

            native_count = sum(1 for c in cases if c["source"] == "native")
            print(
                f"  [{direction:>4} size={effective_size}] "
                f"{len(cases)} cases ({native_count} native, "
                f"{len(cases) - native_count} sublattice)"
            )

            results = run_entropy_ablation(cases)
            all_results.extend(results)

    for direction in directions:
        print_summary(all_results, direction)

    with open("data/entropy_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to data/entropy_ablation_results.json")

    plot_results(all_results, directions=directions)


if __name__ == "__main__":
    main()
