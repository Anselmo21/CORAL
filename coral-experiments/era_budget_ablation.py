import argparse
import json
import os
import random
from collections import defaultdict
from itertools import combinations

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
from search.lattice import combo_key
from search.rank import era_search

BASE_DATA_PATH = "data/base.json"
LATTICES_PATH = "data/all_lattices.json"
DEV_CASES_PATH = "data/dev.jsonl"
TEST_CASES_PATH = "data/test.jsonl"

BUDGETS: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGET_PCTS: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OUT_DIR = "data/era_ablation_study"
PLOT_BASE_DIR = "plots/era_ablation_study"


def effective_size(case: dict) -> int:
    start_node = case["start_node"]
    criteria = case["criteria"]
    if case["direction"] == "down":
        return len(start_node)
    return len(criteria) - len(start_node)


def lattice_size(eff: int) -> int:
    return 2**eff


def budget_from_pct(eff: int, pct: int) -> int:
    return max(1, int(lattice_size(eff) * pct / 100))


def result_key(mode: str, spec: int) -> str:
    return f"pct_{spec}" if mode == "percent" else f"budget_{spec}"


def load_runs_base(path: str) -> list[dict]:
    """Load runs from base.json (rows → baseline_lattice_runs format)."""
    with open(path) as f:
        data = json.load(f)
    runs: list[dict] = []
    for row in data["rows"]:
        for run in row["baseline_lattice_runs"]:
            runs.append({
                "query_id": row["query_id"],
                "criteria_used": run["criteria_used"],
                "rankings_by_combo": run["rankings_by_combo"],
            })
    return runs


def load_manifest_cases(
    lattices_path: str,
    cases_path: str,
    *,
    manifest_directions: list[str] | None = None,
) -> list[dict]:
    """Build per-case dicts from a jsonl manifest + data/all_lattices.json.

    Each case has its own *effective* lattice spanned by `toggleable_criteria`;
    `fixed_criteria` are pinned into every node. We project the full lattice's
    `rankings` down to a `rankings_by_combo` keyed on toggleable subsets by
    looking up `combo_key(sub + fixed)` in the full ranking dict.

    ``manifest_directions`` filters manifest rows (e.g. ``["top-down"]``).

    Returns case dicts compatible with run_era_budget_ablation().
    """
    with open(lattices_path) as f:
        lat_data = json.load(f)
    lattices: dict[str, dict] = lat_data["lattices"]

    cases: list[dict] = []
    with open(cases_path) as f:
        for line in f:
            rec = json.loads(line)
            if manifest_directions is not None and rec["direction"] not in manifest_directions:
                continue

            qid_str = str(rec["query_id"])
            lat = lattices.get(qid_str)
            if lat is None:
                continue

            fixed = list(rec["fixed_criteria"])
            toggleable = list(rec["toggleable_criteria"])
            full_rankings: dict[str, list[str]] = lat["rankings"]

            effective_rbc: dict[str, list[str]] = {}
            for r in range(len(toggleable) + 1):
                for sub in combinations(toggleable, r):
                    sub_list = list(sub)
                    sub_key = combo_key(sub_list)
                    full_key = combo_key(sub_list + fixed)
                    ranking = full_rankings.get(full_key)
                    if ranking is not None:
                        effective_rbc[sub_key] = ranking

            direction = "down" if rec["direction"] == "top-down" else "up"
            if direction == "down":
                start_node = sorted(toggleable)
            else:
                start_node = []

            cases.append({
                "query_id": rec["query_id"],
                "criteria": sorted(toggleable),
                "rankings_by_combo": effective_rbc,
                "start_node": start_node,
                "direction": direction,
                "k": rec["k"],
                "fixed_criteria": sorted(fixed),
                "toggleable_criteria": sorted(toggleable),
                "min_hops_to_counterfactual": rec.get("min_hops_to_counterfactual"),
            })
    return cases


def load_dev_cases(lattices_path: str, cases_path: str) -> list[dict]:
    return load_manifest_cases(lattices_path, cases_path)


def run_era_budget_ablation(
    cases: list[dict],
    *,
    mode: str = "absolute",
    specs: list[int] | None = None,
) -> list[dict]:
    if specs is None:
        specs = BUDGET_PCTS if mode == "percent" else BUDGETS

    results: list[dict] = []

    for case in cases:
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        eff = effective_size(case)
        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)

        row: dict = {
            "query_id": case["query_id"],
            "effective_size": eff,
            "lattice_size": lattice_size(eff),
            "direction": direction,
            "budget_mode": mode,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
        }

        for spec in specs:
            if mode == "percent":
                budget = budget_from_pct(eff, spec)
            else:
                budget = spec

            era_res = era_search(
                criteria, rbc,
                budget=budget,
                start_node=start_node,
                direction=direction,
            )
            scored = score_proxy(bfs_res, era_res, rbc, start_node)
            scored["calls"] = era_res["calls"]
            scored["budget_used"] = era_res.get("budget_used", 0)
            scored["budget_allocated"] = budget
            if mode == "percent":
                scored["budget_pct"] = spec
            row[result_key(mode, spec)] = scored

        results.append(row)

    return results


def print_summary(
    results: list[dict],
    direction: str,
    *,
    mode: str = "absolute",
    specs: list[int] | None = None,
) -> None:
    if specs is None:
        specs = BUDGET_PCTS if mode == "percent" else BUDGETS

    dir_results = [r for r in results if r["direction"] == direction]
    if not dir_results:
        return

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in dir_results:
        by_size[r["effective_size"]].append(r)

    label = "Top-Down" if direction == "down" else "Bottom-Up"
    bkeys = [result_key(mode, s) for s in specs]
    col_w = 16
    budget_title = (
        "ERA Accuracy by Budget (% of lattice)"
        if mode == "percent"
        else "ERA Accuracy by Budget Size"
    )
    print(f"\n{'=' * (28 + col_w * len(specs))}")
    print(f"  {label} — {budget_title}")
    print(f"{'=' * (28 + col_w * len(specs))}")

    header = f"  {'Size':>4} | {'N':>3} |"
    for spec in specs:
        col_label = f"{spec}%" if mode == "percent" else f"budget={spec}"
        header += f" {col_label:>{col_w - 2}} |"
    print(header)

    sep = f"  {'-' * 4}-+-{'-' * 3}-+"
    for _ in specs:
        sep += f"-{'-' * (col_w - 2)}-+"
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        line = f"  {size:>4} | {n:>3} |"
        for bkey in bkeys:
            correct = sum(1 for r in bucket if r[bkey]["outcome"] == "correct")
            acc = correct / n
            line += f" {acc:>4.0%} ({correct:>2}/{n:<2})    |"
        print(line)

    total = len(dir_results)
    print(sep)
    totals = f"  {'All':>4} | {total:>3} |"
    for bkey in bkeys:
        correct = sum(1 for r in dir_results if r[bkey]["outcome"] == "correct")
        acc = correct / total
        totals += f" {acc:>4.0%} ({correct:>2}/{total:<2})    |"
    print(totals)

    if mode == "percent":
        print("\n  Mean allocated Phase-1 budget (LLM calls) by pct:")
        alloc_line = f"  {'All':>4} | {total:>3} |"
        for spec in specs:
            bkey = result_key(mode, spec)
            allocs = [r[bkey]["budget_allocated"] for r in dir_results]
            alloc_line += f" {sum(allocs) / len(allocs):>{col_w - 2}.1f} |"
        print(alloc_line)

    print(f"\n  Mean Jaccard(proxy CF, BFS CF) by Budget:")
    jline = f"  {'All':>4} | {total:>3} |"
    for bkey in bkeys:
        jaccs = [r[bkey]["jaccard"] for r in dir_results if r[bkey].get("jaccard") is not None]
        avg = sum(jaccs) / len(jaccs) if jaccs else float("nan")
        jline += f" {avg:>{col_w - 2}.3f} |"
    print(jline)


def plot_results(
    results: list[dict],
    out_dir: str = PLOT_BASE_DIR,
    *,
    mode: str = "absolute",
    specs: list[int] | None = None,
) -> None:
    if specs is None:
        specs = BUDGET_PCTS if mode == "percent" else BUDGETS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from plot_style import apply_plot_style

    apply_plot_style()

    os.makedirs(out_dir, exist_ok=True)

    def _save_dual(fig, filename: str) -> None:
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    def _acc_ci(outcomes: list[int]) -> tuple[float, float]:
        n = len(outcomes)
        p = sum(outcomes) / n
        return p, 1.96 * np.sqrt(p * (1 - p) / n)

    def _mean_ci(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals, dtype=float)
        n = len(arr)
        return float(np.mean(arr)), float(1.96 * np.std(arr, ddof=1) / np.sqrt(n))

    cmap = plt.cm.viridis
    n_b = max(1, len(specs) - 1)
    budget_colors = {s: cmap(i / n_b) for i, s in enumerate(specs)}
    _markers = ["o", "v", "s", "^", "D", "P", "X", "*", "h"]
    budget_markers = {s: _markers[i % len(_markers)] for i, s in enumerate(specs)}
    dir_labels = {"down": "Top-Down", "up": "Bottom-Up"}
    pct_suffix = "_pct" if mode == "percent" else ""
    title_budget = (
        "ERA Accuracy by Budget (% of lattice)"
        if mode == "percent"
        else "ERA Accuracy by Budget Size"
    )
    xlabel_budget = (
        "Budget (% of lattice size 2^n)"
        if mode == "percent"
        else "Budget (# Phase-1 LLM calls)"
    )

    for direction in ["down", "up"]:
        dir_results = [r for r in results if r["direction"] == direction]
        if not dir_results:
            continue

        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in dir_results:
            by_size[r["effective_size"]].append(r)
        sizes = sorted(by_size)

        # -- Plot 1: Accuracy by effective size, one line per budget --
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for spec in specs:
            bkey = result_key(mode, spec)
            label = f"{spec}%" if mode == "percent" else f"budget={spec}"
            accs = [
                _acc_ci([1 if r[bkey]["outcome"] == "correct" else 0 for r in by_size[s]])[0]
                for s in sizes
            ]
            ax1.plot(sizes, accs, f"{budget_markers[spec]}-",
                     label=label, color=budget_colors[spec],
                     linewidth=2, markersize=7)
        ax1.set_xlabel("Number of Criteria")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(dir_labels[direction])
        ax1.set_xticks(sizes)
        ax1.set_ylim(0, 1.05)
        ax1.legend(ncol=len(specs), loc="lower left")
        ax1.grid(True, alpha=0.3)
        fig1.suptitle(title_budget, fontweight="bold")
        fig1.tight_layout()
        _save_dual(fig1, f"accuracy_by_size_{direction}{pct_suffix}.png")
        plt.close(fig1)

        # -- Plot 2: Accuracy vs budget (x=budget, one line per effective size) --
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        size_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sizes)))
        for i, s in enumerate(sizes):
            bucket = by_size[s]
            accs = [
                _acc_ci([
                    1 if r[result_key(mode, spec)]["outcome"] == "correct" else 0
                    for r in bucket
                ])[0]
                for spec in specs
            ]
            ax2.plot(specs, accs, "o-", label=f"n={s}", color=size_colors[i],
                     linewidth=2, markersize=6)
        ax2.set_xlabel(xlabel_budget)
        ax2.set_ylabel("Accuracy")
        ax2.set_title(dir_labels[direction])
        ax2.set_xticks(specs)
        if mode != "percent":
            ax2.set_xscale("log")
        ax2.set_ylim(0, 1.05)
        ax2.legend(title="Effective size", ncol=4, loc="lower right")
        ax2.grid(True, alpha=0.3)
        fig2.suptitle("ERA Accuracy vs Budget (by Effective Size)", fontweight="bold")
        fig2.tight_layout()
        _save_dual(fig2, f"accuracy_vs_budget_{direction}{pct_suffix}.png")
        plt.close(fig2)

        bfs_means = [_mean_ci([r["bfs_calls"] for r in by_size[s]])[0] for s in sizes]

        # -- Plot 3: Search calls by effective size, one line per budget --
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(sizes, bfs_means, "o:", label="BFS (baseline)", color="black",
                 linewidth=2, markersize=7)
        for spec in specs:
            bkey = result_key(mode, spec)
            label = f"{spec}%" if mode == "percent" else f"budget={spec}"
            means = [_mean_ci([r[bkey]["calls"] for r in by_size[s]])[0] for s in sizes]
            ax3.plot(sizes, means, f"{budget_markers[spec]}-",
                     label=label, color=budget_colors[spec],
                     linewidth=2, markersize=7)
        ax3.set_xlabel("Number of Criteria")
        ax3.set_ylabel("Average Search Calls")
        ax3.set_title(dir_labels[direction])
        ax3.set_xticks(sizes)
        ax3.legend(ncol=len(specs) + 1, loc="upper left")
        ax3.grid(True, alpha=0.3)
        fig3.suptitle("ERA Search Calls by Budget Size", fontweight="bold")
        fig3.tight_layout()
        _save_dual(fig3, f"search_calls_by_size_{direction}{pct_suffix}.png")
        plt.close(fig3)

        # -- Plot 4: Total calls (search + budget_used) by effective size --
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.plot(sizes, bfs_means, "o:", label="BFS (baseline)", color="black",
                 linewidth=2, markersize=7)
        for spec in specs:
            bkey = result_key(mode, spec)
            label = f"{spec}%" if mode == "percent" else f"budget={spec}"
            totals = [
                _mean_ci([r[bkey]["calls"] + r[bkey]["budget_used"] for r in by_size[s]])[0]
                for s in sizes
            ]
            ax4.plot(sizes, totals, f"{budget_markers[spec]}-",
                     label=label, color=budget_colors[spec],
                     linewidth=2, markersize=7)
        ax4.set_xlabel("Number of Criteria")
        ax4.set_ylabel("Average Total Calls (search + budget)")
        ax4.set_title(dir_labels[direction])
        ax4.set_xticks(sizes)
        ax4.legend(ncol=len(specs) + 1, loc="upper left")
        ax4.grid(True, alpha=0.3)
        fig4.suptitle("ERA Total Calls by Budget Size", fontweight="bold")
        fig4.tight_layout()
        _save_dual(fig4, f"total_calls_by_size_{direction}{pct_suffix}.png")
        plt.close(fig4)

        # -- Plot 4b: Mean Jaccard(proxy CF, BFS CF) by effective size --
        fig4b, ax4b = plt.subplots(figsize=(8, 5))
        for spec in specs:
            bkey = result_key(mode, spec)
            label = f"{spec}%" if mode == "percent" else f"budget={spec}"
            jacc_means = []
            for s in sizes:
                vals = [
                    r[bkey]["jaccard"] for r in by_size[s]
                    if r[bkey].get("jaccard") is not None
                ]
                jacc_means.append(float(np.mean(vals)) if vals else float("nan"))
            ax4b.plot(sizes, jacc_means, f"{budget_markers[spec]}-",
                      label=label, color=budget_colors[spec],
                      linewidth=2, markersize=7)
        ax4b.set_xlabel("Number of Criteria")
        ax4b.set_ylabel("Mean Jaccard(proxy CF, BFS CF)")
        ax4b.set_title(dir_labels[direction])
        ax4b.set_xticks(sizes)
        ax4b.set_ylim(0, 1.05)
        ax4b.legend(ncol=len(specs), loc="lower left")
        ax4b.grid(True, alpha=0.3)
        fig4b.suptitle("ERA Counterfactual Jaccard by Budget Size", fontweight="bold")
        fig4b.tight_layout()
        _save_dual(fig4b, f"jaccard_by_size_{direction}{pct_suffix}.png")
        plt.close(fig4b)

        # -- Plot 5: Normalized total calls (/ 2^n) by effective size --
        bfs_norm = [m / (2 ** s) for m, s in zip(bfs_means, sizes)]
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.plot(sizes, bfs_norm, "o:", label="BFS (baseline)", color="black",
                 linewidth=2, markersize=7)
        for spec in specs:
            bkey = result_key(mode, spec)
            label = f"{spec}%" if mode == "percent" else f"budget={spec}"
            norm = [
                _mean_ci([r[bkey]["calls"] + r[bkey]["budget_used"] for r in by_size[s]])[0] / (2 ** s)
                for s in sizes
            ]
            ax5.plot(sizes, norm, f"{budget_markers[spec]}-",
                     label=label, color=budget_colors[spec],
                     linewidth=2, markersize=7)
        ax5.set_xlabel("Number of Criteria")
        ax5.set_ylabel("Proportion of Lattice Visited")
        ax5.set_title(dir_labels[direction])
        ax5.set_xticks(sizes)
        ax5.set_ylim(0, 1.05)
        ax5.legend(ncol=len(specs) + 1, loc="upper left")
        ax5.grid(True, alpha=0.3)
        fig5.suptitle("ERA Lattice Coverage by Budget Size", fontweight="bold")
        fig5.tight_layout()
        _save_dual(fig5, f"normalized_calls_by_size_{direction}{pct_suffix}.png")
        plt.close(fig5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Use base.json (real dataset) instead of the skeleton dataset")
    parser.add_argument("--dev", action="store_true",
                        help="Use dev.jsonl cases joined against all_lattices.json")
    parser.add_argument("--test", action="store_true",
                        help="Use test.jsonl cases joined against all_lattices.json")
    parser.add_argument("--top-down", action="store_true",
                        help="Restrict --dev/--test to top-down manifest rows only")
    parser.add_argument(
        "--pct",
        action="store_true",
        help="Sweep Phase-1 budget as %% of lattice size (2^n): "
        + ", ".join(f"{p}%%" for p in BUDGET_PCTS),
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    mode = "percent" if args.pct else "absolute"

    if args.dev or args.test:
        cases_path = TEST_CASES_PATH if args.test else DEV_CASES_PATH
        split = "test" if args.test else "dev"
        manifest_directions = ["top-down"] if args.top_down else None
        cases = load_manifest_cases(
            LATTICES_PATH, cases_path, manifest_directions=manifest_directions
        )
        td_suffix = "_topdown" if args.top_down else ""
        if args.pct:
            subdir = f"{split}/pct{td_suffix}"
        else:
            subdir = f"{split}{td_suffix}"
        out_dir = os.path.join(OUT_DIR, subdir)
        results_name = (
            "era_budget_pct_ablation_results.json"
            if args.pct
            else "era_budget_ablation_results.json"
        )
        results_file = os.path.join(out_dir, results_name)
        td_note = " (top-down only)" if args.top_down else ""
        print(f"Loaded {len(cases)} {split} cases{td_note} from {cases_path}")
        if args.pct:
            print(f"Budget mode: {mode} — {BUDGET_PCTS}")

        all_results = run_era_budget_ablation(cases, mode=mode)

        directions = ["down"] if args.top_down else ["down", "up"]
        for direction in directions:
            print_summary(all_results, direction, mode=mode)

        os.makedirs(out_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_file}")

        plot_results(all_results, out_dir=out_dir.replace(OUT_DIR, PLOT_BASE_DIR, 1), mode=mode)
        return

    if args.real:
        runs = load_runs_base(BASE_DATA_PATH)
        subdir = "real/pct" if args.pct else "real"
        out_dir = os.path.join(OUT_DIR, subdir)
        results_name = (
            "era_budget_pct_ablation_results.json"
            if args.pct
            else "era_budget_ablation_results.json"
        )
        results_file = os.path.join(out_dir, results_name)
        print(f"Loaded {len(runs)} runs from {BASE_DATA_PATH}")
    else:
        runs = load_runs(DATA_PATH)
        out_dir = os.path.join(OUT_DIR, "pct") if args.pct else OUT_DIR
        results_name = (
            "era_budget_pct_ablation_results.json"
            if args.pct
            else "era_budget_ablation_results.json"
        )
        results_file = os.path.join(out_dir, results_name)
        print(f"Loaded {len(runs)} runs from {DATA_PATH}")

    if args.pct:
        print(f"Budget mode: {mode} — {BUDGET_PCTS}")

    all_results: list[dict] = []

    for direction in ["down", "up"]:
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

            results = run_era_budget_ablation(cases, mode=mode)
            all_results.extend(results)

    for direction in ["down", "up"]:
        print_summary(all_results, direction, mode=mode)

    os.makedirs(out_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    plot_results(all_results, out_dir=out_dir.replace(OUT_DIR, PLOT_BASE_DIR, 1), mode=mode)


if __name__ == "__main__":
    main()
