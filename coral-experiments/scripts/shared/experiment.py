import json
import random
from collections import Counter, defaultdict
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, sra_search

DATA_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"
QUERY_PATH = "data/base-q.json"
CC_CASES_PATH = "data/cc_cases.json"
CC_RESULTS_PATH = "data/cc_results_new.json"
TARGET_PER_BUCKET = 20
SEED = 42


def load_runs(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["runs"]


def generate_candidates(
    runs: list[dict], effective_size: int, direction: str
) -> tuple[list[dict], list[dict]]:
    natives: list[dict] = []
    sublattices: list[dict] = []

    for run in runs:
        criteria = run["criteria_used"]
        rbc = run["rankings_by_combo"]
        n = len(criteria)

        if direction == "down":
            if n == effective_size:
                natives.append({
                    "criteria": criteria,
                    "rankings_by_combo": rbc,
                    "start_node": list(criteria),
                    "direction": "down",
                    "source": "native",
                    "query_id": run["query_id"],
                    "parent_size": n,
                })
            elif n > effective_size:
                for combo in combinations(criteria, effective_size):
                    sublattices.append({
                        "criteria": criteria,
                        "rankings_by_combo": rbc,
                        "start_node": list(combo),
                        "direction": "down",
                        "source": "sublattice",
                        "query_id": run["query_id"],
                        "parent_size": n,
                    })

        elif direction == "up":
            k = n - effective_size
            if k >= 0 and k < n:
                for combo in combinations(criteria, k):
                    sublattices.append({
                        "criteria": criteria,
                        "rankings_by_combo": rbc,
                        "start_node": list(combo),
                        "direction": "up",
                        "source": "sublattice",
                        "query_id": run["query_id"],
                        "parent_size": n,
                    })

    return natives, sublattices


def has_cf(case: dict) -> bool:
    result = bfs_search(
        case["criteria"],
        case["rankings_by_combo"],
        start_node=case["start_node"],
        direction=case["direction"],
    )
    return result["found"]


def sample_cases(
    natives: list[dict],
    sublattices: list[dict],
    target: int,
    rng: random.Random,
) -> list[dict]:
    valid_natives = [c for c in natives if has_cf(c)]
    if len(valid_natives) >= target:
        return valid_natives[:target]

    remaining = target - len(valid_natives)
    rng.shuffle(sublattices)

    valid_subs: list[dict] = []
    for case in sublattices:
        if has_cf(case):
            valid_subs.append(case)
            if len(valid_subs) >= remaining:
                break

    return valid_natives + valid_subs


def score_proxy(
    bfs_result: dict,
    proxy_result: dict,
    rankings_by_combo: dict[str, list[str]],
    start_node: list[str],
) -> dict:
    start_key = combo_key(start_node)
    baseline_top1 = (rankings_by_combo.get(start_key) or [""])[0]

    bfs_key = bfs_result["key"]
    bfs_layer = bfs_result["size"]

    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")

    if proxy_key:
        ranking = rankings_by_combo.get(proxy_key, [])
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
    if outcome == "correct":
        gap = 0
    elif outcome == "wrong_layer" and bfs_layer is not None and proxy_layer is not None:
        gap = abs(bfs_layer - proxy_layer)

    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": proxy_verified,
        "outcome": outcome,
        "gap": gap,
        "jaccard": jaccard_keys(bfs_key, proxy_key),
    }


def run_experiment(cases: list[dict]) -> list[dict]:
    results: list[dict] = []

    for case in cases:
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        if direction == "down":
            eff = len(start_node)
        else:
            eff = len(criteria) - len(start_node)

        budget = max(1, int((2**eff) * 0.1))

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        sra_res = sra_search(criteria, rbc, start_node=start_node, direction=direction)
        era_res = era_search(
            criteria, rbc, budget=budget, start_node=start_node, direction=direction
        )
        era64_res = era_search(
            criteria, rbc, budget=64, start_node=start_node, direction=direction
        )

        sra_scored = score_proxy(bfs_res, sra_res, rbc, start_node)
        sra_scored["calls"] = sra_res["calls"]

        era_scored = score_proxy(bfs_res, era_res, rbc, start_node)
        era_scored["calls"] = era_res["calls"]

        era64_scored = score_proxy(bfs_res, era64_res, rbc, start_node)
        era64_scored["calls"] = era64_res["calls"]

        results.append({
            "query_id": case["query_id"],
            "effective_size": eff,
            "direction": direction,
            "source": case["source"],
            "parent_size": case["parent_size"],
            "start_node": start_node,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "sra": sra_scored,
            "era": era_scored,
            "era_budget": budget,
            "era_budget_used": era_res.get("budget_used", 0),
            "era_pruned": era_res.get("pruned_nodes", 0),
            "era64": era64_scored,
            "era64_budget_used": era64_res.get("budget_used", 0),
        })

    return results


def print_summary_table(results: list[dict], direction: str, has_cc: bool = False) -> None:
    dir_results = [r for r in results if r["direction"] == direction]
    if not dir_results:
        return

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in dir_results:
        by_size[r["effective_size"]].append(r)

    label = "Down" if direction == "down" else "Up"
    width = 114 + (24 if has_cc else 0)
    print(f"\n{'=' * width}")
    print(f"  {label} Search - Accuracy & Calls by Effective Size")
    print(f"{'=' * width}")
    header = (
        f"  {'Size':>4} | {'N':>3} | "
        f"{'BFS calls':>9} | "
        f"{'SRA Acc':>12} | {'SRA calls':>9} | "
        f"{'ERA Acc':>12} | {'ERA calls':>9} | "
        f"{'ERA64 Acc':>12} | {'ERA64 calls':>11}"
    )
    if has_cc:
        header += f" | {'CC Acc':>12} | {'CC calls':>9}"
    print(header)
    sep = (
        f"  {'-' * 4}-+-{'-' * 3}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 9}"
        f"-+-{'-' * 12}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 11}"
    )
    if has_cc:
        sep += f"-+-{'-' * 12}-+-{'-' * 9}"
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        sra_c = sum(1 for r in bucket if r["sra"]["outcome"] == "correct")
        era_c = sum(1 for r in bucket if r["era"]["outcome"] == "correct")
        era64_c = sum(1 for r in bucket if r.get("era64", {}).get("outcome") == "correct")
        bfs_avg = sum(r["bfs_calls"] for r in bucket) / n
        sra_avg = sum(r["sra"]["calls"] for r in bucket) / n
        era_avg = sum(r["era"]["calls"] for r in bucket) / n
        era64_avg = sum(r.get("era64", {}).get("calls", 0) for r in bucket) / n
        line = (
            f"  {size:>4} | {n:>3} | "
            f"{bfs_avg:>9.1f} | "
            f"{sra_c / n:>5.0%} ({sra_c:>2}/{n:<2}) | {sra_avg:>9.1f} | "
            f"{era_c / n:>5.0%} ({era_c:>2}/{n:<2}) | {era_avg:>9.1f} | "
            f"{era64_c / n:>5.0%} ({era64_c:>2}/{n:<2}) | {era64_avg:>11.1f}"
        )
        if has_cc:
            cc_bucket = [r for r in bucket if "cc" in r]
            if cc_bucket:
                cc_c = sum(1 for r in cc_bucket if r["cc"]["outcome"] == "correct")
                cc_n = len(cc_bucket)
                cc_avg = sum(r["cc"]["calls"] for r in cc_bucket) / cc_n
                line += f" | {cc_c / cc_n:>5.0%} ({cc_c:>2}/{cc_n:<2}) | {cc_avg:>9.1f}"
            else:
                line += f" | {'—':>12} | {'—':>9}"
        print(line)

    total = len(dir_results)
    sra_tot = sum(1 for r in dir_results if r["sra"]["outcome"] == "correct")
    era_tot = sum(1 for r in dir_results if r["era"]["outcome"] == "correct")
    era64_tot = sum(1 for r in dir_results if r.get("era64", {}).get("outcome") == "correct")
    bfs_all = sum(r["bfs_calls"] for r in dir_results) / total
    sra_all = sum(r["sra"]["calls"] for r in dir_results) / total
    era_all = sum(r["era"]["calls"] for r in dir_results) / total
    era64_all = sum(r.get("era64", {}).get("calls", 0) for r in dir_results) / total
    print(sep)
    totals = (
        f"  {'All':>4} | {total:>3} | "
        f"{bfs_all:>9.1f} | "
        f"{sra_tot / total:>5.0%} ({sra_tot:>2}/{total:<2}) | {sra_all:>9.1f} | "
        f"{era_tot / total:>5.0%} ({era_tot:>2}/{total:<2}) | {era_all:>9.1f} | "
        f"{era64_tot / total:>5.0%} ({era64_tot:>2}/{total:<2}) | {era64_all:>11.1f}"
    )
    if has_cc:
        cc_all = [r for r in dir_results if "cc" in r]
        if cc_all:
            cc_tot = sum(1 for r in cc_all if r["cc"]["outcome"] == "correct")
            cc_n = len(cc_all)
            cc_avg_all = sum(r["cc"]["calls"] for r in cc_all) / cc_n
            totals += f" | {cc_tot / cc_n:>5.0%} ({cc_tot:>2}/{cc_n:<2}) | {cc_avg_all:>9.1f}"
    print(totals)


def print_method_detail(results: list[dict], method: str, direction: str) -> None:
    dir_results = [r for r in results if r["direction"] == direction and method in r]
    if not dir_results:
        return

    label = "Down" if direction == "down" else "Up"
    total = len(dir_results)
    outcomes: dict[str, int] = defaultdict(int)
    for r in dir_results:
        outcomes[r[method]["outcome"]] += 1

    correct = outcomes.get("correct", 0)
    score = correct / total if total else 0.0

    print(f"\n  {method.upper()} ({label}) - {correct}/{total} ({score:.1%})")
    for outcome in ("correct", "wrong_layer", "false_positive", "not_found"):
        n = outcomes.get(outcome, 0)
        bar = "#" * n + "." * (total - n)
        print(f"    {outcome:<16}  {n:>3}  {bar[:40]}")

    wrong = [r for r in dir_results if r[method]["outcome"] == "wrong_layer"]
    if wrong:
        gaps = [r[method]["gap"] for r in wrong if r[method]["gap"] is not None]
        if gaps:
            avg = sum(gaps) / len(gaps)
            print(f"  Layer gap ({len(wrong)} wrong-layer): mean={avg:.2f} min={min(gaps)} max={max(gaps)}")

    verified = [r for r in dir_results if r[method].get("proxy_verified")]
    jaccs_v = [r[method]["jaccard"] for r in verified if r[method].get("jaccard") is not None]
    jaccs_all = [r[method]["jaccard"] for r in dir_results if r[method].get("jaccard") is not None]
    if jaccs_v or jaccs_all:
        msg = "  Jaccard(proxy CF, BFS CF):"
        if jaccs_v:
            msg += f"  verified mean={sum(jaccs_v) / len(jaccs_v):.3f} (n={len(jaccs_v)})"
        if jaccs_all:
            msg += f"  overall mean={sum(jaccs_all) / len(jaccs_all):.3f} (n={len(jaccs_all)})"
        print(msg)

    failures = [r for r in dir_results if r[method]["outcome"] != "correct"]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            tag = f"[{r[method]['outcome']}]"
            bfs = f"BFS->{r['bfs_key']} (size {r['bfs_layer']})"
            pkey = r[method].get("proxy_key")
            if pkey:
                prx = f"proxy->{pkey} (size {r[method]['proxy_layer']})"
            else:
                prx = "proxy->(none)"
            print(f"    Q{r['query_id']}  {tag:<18}  {bfs}  |  {prx}")


def plot_results(results: list[dict], has_cc: bool = False) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.shared.plot_style import apply_plot_style

    apply_plot_style()

    dir_labels = {"down": "Top-Down Search", "up": "Bottom-Up Search"}
    dir_folders = {"down": "plots/top_down", "up": "plots/bottom_up"}

    def _save_dual(fig, path: str) -> None:
        fig.savefig(path, dpi=150)

    outcome_order = ["correct", "wrong_layer", "not_found"]
    outcome_colors = {
        "correct": "#2ca02c",
        "wrong_layer": "#ff7f0e",
        "false_positive": "#d62728",
        "not_found": "#aec7e8",
    }
    methods = ["sra", "era", "era64"] + (["cc"] if has_cc else [])
    method_labels = {"sra": "SRA", "era": "ERA", "era64": "ERA64", "cc": "CC"}
    bar_colors = {
        "BFS": "#2ca02c", "SRA": "#d62728", "ERA": "#1f77b4",
        "ERA Total": "#aec7e8", "ERA64": "#9467bd", "ERA64 Total": "#c5b0d5",
        "CC": "#ff7f0e", "CC Total": "#ffbb78",
    }

    for direction in ["down", "up"]:
        out_dir = dir_folders[direction]
        os.makedirs(out_dir, exist_ok=True)

        dir_results = [r for r in results if r["direction"] == direction]
        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in dir_results:
            by_size[r["effective_size"]].append(r)
        sizes = sorted(by_size)

        def _acc_ci(outcomes: list[int]) -> tuple[float, float]:
            n = len(outcomes)
            p = sum(outcomes) / n
            return p * 100, 1.96 * np.sqrt(p * (1 - p) / n) * 100

        def _mean_ci(vals: list[float]) -> tuple[float, float]:
            arr = np.array(vals, dtype=float)
            n = len(arr)
            return float(np.mean(arr)), float(1.96 * np.std(arr, ddof=1) / np.sqrt(n))

        # -- Figure 1: Accuracy line plot with 95% CI bands --
        fig1, ax1 = plt.subplots(figsize=(7, 5))

        def _plot_acc_line(ax, xs, stats, marker, label, color):
            ys = [v for v, _ in stats]
            # cis = [c for _, c in stats]
            ax.plot(xs, ys, f"{marker}-", label=label, color=color, linewidth=2, markersize=7)
            # ax.errorbar(xs, ys, yerr=cis, fmt=f"{marker}-", label=label, color=color,
            #             linewidth=2, markersize=7, capsize=4, capthick=1.2, elinewidth=1.2)

        sra_acc_stats = [_acc_ci([1 if r["sra"]["outcome"] == "correct" else 0 for r in by_size[s]]) for s in sizes]
        era_acc_stats = [_acc_ci([1 if r["era"]["outcome"] == "correct" else 0 for r in by_size[s]]) for s in sizes]
        # era64_acc_stats = [_acc_ci([1 if r.get("era64", {}).get("outcome") == "correct" else 0 for r in by_size[s]]) for s in sizes]
        _plot_acc_line(ax1, sizes, sra_acc_stats, "o", "SRA", "#d62728")
        _plot_acc_line(ax1, sizes, era_acc_stats, "s", "ERA", "#1f77b4")
        # _plot_acc_line(ax1, sizes, era64_acc_stats, "D", "ERA (64)", "#9467bd")
        if has_cc:
            cc_sizes, cc_acc_stats = [], []
            for s in sizes:
                cc_bucket = [r for r in by_size[s] if "cc" in r]
                if cc_bucket:
                    cc_sizes.append(s)
                    cc_acc_stats.append(_acc_ci([1 if r["cc"]["outcome"] == "correct" else 0 for r in cc_bucket]))
            if cc_sizes:
                _plot_acc_line(ax1, cc_sizes, cc_acc_stats, "^", "CC", "#ff7f0e")
        ax1.set_xlabel("Number of Criteria")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(dir_labels[direction])
        ax1.set_xticks(sizes)
        ax1.set_ylim(0, 105)
        n_acc = 4 if has_cc else 3
        ax1.legend(ncol=n_acc, loc="lower left")
        ax1.grid(True, alpha=0.3)
        fig1.suptitle("Search Accuracy of Pruning Methods", fontweight="bold")
        fig1.tight_layout()
        path = f"{out_dir}/accuracy_by_size.png"
        _save_dual(fig1, path)
        plt.close(fig1)
        print(f"Saved {path}")

        # -- Pre-compute calls means + 95% CIs for Figures 2 and 2b --
        n_bars = 4 + (2 if has_cc else 0)
        bar_width = 0.8 / n_bars
        x = np.arange(len(sizes))

        bfs_stats   = [_mean_ci([r["bfs_calls"] for r in by_size[s]]) for s in sizes]
        sra_stats   = [_mean_ci([r["sra"]["calls"] for r in by_size[s]]) for s in sizes]
        era_stats   = [_mean_ci([r["era"]["calls"] for r in by_size[s]]) for s in sizes]
        era_tot_stats = [_mean_ci([r["era"]["calls"] + r["era_budget_used"] for r in by_size[s]]) for s in sizes]
        era64_stats  = [_mean_ci([r["era64"]["calls"] for r in by_size[s]]) for s in sizes]
        era64_tot_stats = [_mean_ci([r["era64"]["calls"] + r["era64_budget_used"] for r in by_size[s]]) for s in sizes]

        bfs_calls,   bfs_ci     = [v for v, _ in bfs_stats],   [c for _, c in bfs_stats]
        sra_calls,   sra_ci2    = [v for v, _ in sra_stats],   [c for _, c in sra_stats]
        era_calls,   era_ci2    = [v for v, _ in era_stats],   [c for _, c in era_stats]
        era_total,   era_tot_ci = [v for v, _ in era_tot_stats], [c for _, c in era_tot_stats]
        era64_calls, era64_ci2  = [v for v, _ in era64_stats],  [c for _, c in era64_stats]
        era64_total, era64_tot_ci = [v for v, _ in era64_tot_stats], [c for _, c in era64_tot_stats]

        cc_calls_list, cc_calls_ci, cc_total_list = [], [], []
        if has_cc:
            for s in sizes:
                cc_bucket = [r for r in by_size[s] if "cc" in r]
                if cc_bucket:
                    avg, ci = _mean_ci([r["cc"]["calls"] for r in cc_bucket])
                    cc_calls_list.append(avg)
                    cc_calls_ci.append(ci)
                    cc_total_list.append(avg + 64)
                else:
                    cc_calls_list.append(0.0)
                    cc_calls_ci.append(0.0)
                    cc_total_list.append(0.0)

        def _yerr(means: list[float], cis: list[float]):
            """Asymmetric yerr array clipped so the lower whisker never goes below 0."""
            lower = [min(ci, mean) for mean, ci in zip(means, cis)]
            upper = list(cis)
            return np.array([lower, upper])

        # -- Figure 2: Average calls line plot --
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.plot(sizes, bfs_calls, "o:", label="BFS (baseline)",       color="black",                 linewidth=2, markersize=7)
        ax2.plot(sizes, sra_calls, "s-", label="SRA",                  color=bar_colors["SRA"],       linewidth=2, markersize=7)
        ax2.plot(sizes, era_total, "D-", label="ERA", color=bar_colors["ERA"], linewidth=2, markersize=7)
        if has_cc:
            ax2.plot(sizes, cc_total_list, "^-", label="CC", color=bar_colors["CC"], linewidth=2, markersize=7)
        ax2.set_xlabel("Number of Criteria")
        ax2.set_ylabel("Average Calls")
        ax2.set_title(dir_labels[direction])
        ax2.set_xticks(sizes)
        n_calls = 4 if has_cc else 3
        ax2.legend(ncol=n_calls, loc="upper left")
        ax2.grid(True, alpha=0.3)
        fig2.suptitle("Average LLM Calls to Find First Counterfactual", fontweight="bold")
        fig2.tight_layout()
        path = f"{out_dir}/calls_by_size.png"
        _save_dual(fig2, path)
        plt.close(fig2)
        print(f"Saved {path}")

        # -- Figure 2b: Proportion of lattice visited — line plot --
        scale = [1.0 / (2 ** s) for s in sizes]
        fig2b, ax2b = plt.subplots(figsize=(9, 5))
        bfs_n   = [v * sc for v, sc in zip(bfs_calls, scale)]
        sra_n   = [v * sc for v, sc in zip(sra_calls, scale)]
        era_t_n = [v * sc for v, sc in zip(era_total, scale)]
        ax2b.plot(sizes, bfs_n,   "o:", label="BFS (baseline)",     color="black",                 linewidth=2, markersize=7)
        ax2b.plot(sizes, sra_n,   "s-", label="SRA",                color=bar_colors["SRA"],       linewidth=2, markersize=7)
        ax2b.plot(sizes, era_t_n, "D-", label="ERA", color=bar_colors["ERA"], linewidth=2, markersize=7)
        if has_cc:
            cc_t_n = [v * sc for v, sc in zip(cc_total_list, scale)]
            ax2b.plot(sizes, cc_t_n, "^-", label="CC", color=bar_colors["CC"], linewidth=2, markersize=7)
        ax2b.set_ylim(0, 1.0)
        ax2b.set_xlabel("Number of Criteria")
        ax2b.set_ylabel("Proportion of Lattice Visited")
        ax2b.set_title(dir_labels[direction])
        ax2b.set_xticks(sizes)
        ax2b.legend(ncol=n_calls, loc="upper right")
        ax2b.grid(True, alpha=0.3)
        fig2b.suptitle("Proportion of Lattice Visited by Lattice Size", fontweight="bold")
        fig2b.tight_layout()
        path = f"{out_dir}/calls_by_size_normalized.png"
        _save_dual(fig2b, path)
        plt.close(fig2b)
        print(f"Saved {path}")

        # -- Figure 3 helper: draw one outcomes subplot --
        def _draw_outcomes(ax, method: str) -> None:
            m_results = [r for r in dir_results if method != "cc" or "cc" in r]
            m_by_size: dict[int, list[dict]] = defaultdict(list)
            for r in m_results:
                m_by_size[r["effective_size"]].append(r)
            m_sizes = sorted(m_by_size)
            xm = np.arange(len(m_sizes))
            bottoms = np.zeros(len(m_sizes))
            for outcome in outcome_order:
                totals = np.array([len(m_by_size[s]) for s in m_sizes], dtype=float)
                counts = np.array([
                    sum(1 for r in m_by_size[s] if r[method]["outcome"] == outcome)
                    for s in m_sizes
                ], dtype=float)
                proportions = np.where(totals > 0, counts / totals, 0.0)
                ax.bar(xm, proportions, bottom=bottoms, label=outcome, color=outcome_colors[outcome])
                bottoms += proportions
            ax.set_title(f"{method_labels[method]} — {dir_labels[direction]}")
            ax.set_xticks(xm)
            ax.set_xticklabels([f"{s}\n(n={len(m_by_size[s])})" for s in m_sizes])
            ax.set_xlabel("Number of Criteria")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1.18)
            ax.legend(ncol=len(outcome_order), loc="upper right", framealpha=0.9)
            ax.grid(True, axis="y", alpha=0.3)

        # -- Figure 3a: SRA only --
        fig3a, ax3a = plt.subplots(figsize=(7, 3.5))
        _draw_outcomes(ax3a, "sra")
        fig3a.suptitle("Outcome Breakdown by Method and Effective Size", fontweight="bold")
        fig3a.tight_layout()
        path = f"{out_dir}/outcomes_sra.png"
        _save_dual(fig3a, path)
        plt.close(fig3a)
        print(f"Saved {path}")

        # -- Figure 3b: ERA / ERA64 / CC --
        other_methods = [m for m in methods if m != "sra"]
        fig3b, axes3b = plt.subplots(len(other_methods), 1, figsize=(7, 3.5 * len(other_methods)), sharey=False)
        if len(other_methods) == 1:
            axes3b = [axes3b]
        for ax, method in zip(axes3b, other_methods):
            _draw_outcomes(ax, method)
        fig3b.suptitle("Outcome Breakdown by Method and Effective Size", fontweight="bold")
        fig3b.tight_layout()
        path = f"{out_dir}/outcomes_by_size.png"
        _save_dual(fig3b, path)
        plt.close(fig3b)
        print(f"Saved {path}")

        # -- Figure 4: Layer gap distribution (line plot, one per method) --
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        method_specs = [
            ("sra",   "#d62728", "SRA",   "s"),
            ("era",   "#1f77b4", "ERA",   "D"),
            ("era64", "#9467bd", "ERA64", "D"),
        ] + ([("cc", "#ff7f0e", "CC", "^")] if has_cc else [])

        method_gap_counts: list[tuple[str, str, str, str, Counter]] = []
        max_gap = 0
        for method, color, label, marker in method_specs:
            gaps = [
                r[method]["gap"]
                for r in dir_results
                if method in r and r[method]["outcome"] == "wrong_layer" and r[method]["gap"] is not None
            ]
            if gaps:
                counts = Counter(gaps)
                method_gap_counts.append((method, color, label, marker, counts))
                max_gap = max(max_gap, max(gaps))

        if method_gap_counts:
            x = list(range(1, max_gap + 1))
            for method, color, label, marker, counts in method_gap_counts:
                y = [counts.get(g, 0) for g in x]
                n = sum(counts.values())
                ax4.plot(x, y, f"{marker}-", color=color, label=f"{label} (n={n})",
                         linewidth=2, markersize=7)
            ax4.set_xticks(x)
        ax4.set_xlabel("Layer Gap")
        ax4.set_ylabel("Count")
        ax4.set_title(dir_labels[direction])
        ax4.legend(ncol=len(method_gap_counts) or 1, loc="upper right")
        ax4.grid(True, alpha=0.3)
        fig4.suptitle("Layer Gap Distribution (wrong-layer cases only)", fontweight="bold")
        fig4.tight_layout()
        path = f"{out_dir}/layer_gaps.png"
        _save_dual(fig4, path)
        plt.close(fig4)
        print(f"Saved {path}")


def load_query_data(path: str) -> dict[int, dict]:
    with open(path) as f:
        data = json.load(f)
    lookup: dict[int, dict] = {}
    for run in data["runs"]:
        lookup[run["query_id"]] = {"query": run["query"], "sources": run["sources"]}
    return lookup


def export_cc_cases(all_results: list[dict], query_lookup: dict[int, dict]) -> None:
    cases: list[dict] = []
    for i, r in enumerate(all_results):
        qdata = query_lookup.get(r["query_id"], {})
        cases.append({
            "case_id": i,
            "query_id": r["query_id"],
            "effective_size": r["effective_size"],
            "direction": r["direction"],
            "source": r["source"],
            "parent_size": r["parent_size"],
            "start_node": r["start_node"],
            "query": qdata.get("query", ""),
            "sources": qdata.get("sources", ""),
        })
    with open(CC_CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Exported {len(cases)} CC cases to {CC_CASES_PATH}")


def load_cc_results() -> dict[int, dict] | None:
    try:
        with open(CC_RESULTS_PATH) as f:
            cc_data = json.load(f)
        return {r["case_id"]: r for r in cc_data}
    except FileNotFoundError:
        return None


def merge_cc_results(
    all_results: list[dict], cc_lookup: dict[int, dict]
) -> None:
    for i, r in enumerate(all_results):
        cc = cc_lookup.get(i)
        if cc:
            r["cc"] = {
                "proxy_key": cc.get("key"),
                "proxy_layer": cc.get("proxy_layer"),
                "proxy_verified": cc.get("proxy_verified", False),
                "outcome": cc.get("outcome", "not_found"),
                "gap": cc.get("gap"),
                "calls": cc.get("calls", 0),
            }


def main() -> None:
    rng = random.Random(SEED)
    runs = load_runs(DATA_PATH)
    print(f"Loaded {len(runs)} runs from {DATA_PATH}")

    all_cases: list[dict] = []
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

            all_cases.extend(cases)
            results = run_experiment(cases)
            all_results.extend(results)

    # Export cases for CC runner (skipped when the legacy query file is absent)
    try:
        query_lookup = load_query_data(QUERY_PATH)
        export_cc_cases(all_results, query_lookup)
    except FileNotFoundError:
        print(f"Skipping CC case export — {QUERY_PATH} not present")

    # Merge CC results if available
    methods = ["sra", "era", "era64"]
    cc_lookup = load_cc_results()
    if cc_lookup:
        merge_cc_results(all_results, cc_lookup)
        methods.append("cc")
        print(f"Loaded {len(cc_lookup)} CC results from {CC_RESULTS_PATH}")
    else:
        print(f"No CC results found at {CC_RESULTS_PATH} (run run_cc.py to generate)")

    for direction in ["down", "up"]:
        print_summary_table(all_results, direction, has_cc="cc" in methods)
        for method in methods:
            print_method_detail(all_results, method, direction)

    output_path = "data/experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    plot_results(all_results, has_cc="cc" in methods)


if __name__ == "__main__":
    main()
