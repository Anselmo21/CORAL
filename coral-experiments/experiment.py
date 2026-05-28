import json
import random
from collections import defaultdict
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key
from search.rank import era_search, sra_search

DATA_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"
QUERY_PATH = "data/base-q.json"
CC_CASES_PATH = "data/cc_cases.json"
CC_RESULTS_PATH = "data/cc_results.json"
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
            if k >= 1 and k < n:
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

        sra_scored = score_proxy(bfs_res, sra_res, rbc, start_node)
        sra_scored["calls"] = sra_res["calls"]

        era_scored = score_proxy(bfs_res, era_res, rbc, start_node)
        era_scored["calls"] = era_res["calls"]

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
    cc_col = " | {'CC Acc':>12} | {'CC calls':>9}" if has_cc else ""
    cc_sep = f"-+-{'-' * 12}-+-{'-' * 9}" if has_cc else ""
    width = 88 + (24 if has_cc else 0)
    print(f"\n{'=' * width}")
    print(f"  {label} Search - Accuracy & Calls by Effective Size")
    print(f"{'=' * width}")
    header = (
        f"  {'Size':>4} | {'N':>3} | "
        f"{'BFS calls':>9} | "
        f"{'SRA Acc':>12} | {'SRA calls':>9} | "
        f"{'ERA Acc':>12} | {'ERA calls':>9}"
    )
    if has_cc:
        header += f" | {'CC Acc':>12} | {'CC calls':>9}"
    print(header)
    sep = f"  {'-' * 4}-+-{'-' * 3}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 9}"
    if has_cc:
        sep += f"-+-{'-' * 12}-+-{'-' * 9}"
    print(sep)

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        sra_c = sum(1 for r in bucket if r["sra"]["outcome"] == "correct")
        era_c = sum(1 for r in bucket if r["era"]["outcome"] == "correct")
        bfs_avg = sum(r["bfs_calls"] for r in bucket) / n
        sra_avg = sum(r["sra"]["calls"] for r in bucket) / n
        era_avg = sum(r["era"]["calls"] for r in bucket) / n
        line = (
            f"  {size:>4} | {n:>3} | "
            f"{bfs_avg:>9.1f} | "
            f"{sra_c / n:>5.0%} ({sra_c:>2}/{n:<2}) | {sra_avg:>9.1f} | "
            f"{era_c / n:>5.0%} ({era_c:>2}/{n:<2}) | {era_avg:>9.1f}"
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
    bfs_all = sum(r["bfs_calls"] for r in dir_results) / total
    sra_all = sum(r["sra"]["calls"] for r in dir_results) / total
    era_all = sum(r["era"]["calls"] for r in dir_results) / total
    print(sep)
    totals = (
        f"  {'All':>4} | {total:>3} | "
        f"{bfs_all:>9.1f} | "
        f"{sra_tot / total:>5.0%} ({sra_tot:>2}/{total:<2}) | {sra_all:>9.1f} | "
        f"{era_tot / total:>5.0%} ({era_tot:>2}/{total:<2}) | {era_all:>9.1f}"
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
    dir_results = [r for r in results if r["direction"] == direction]
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directions = ["down", "up"]
    dir_labels = {"down": "Down Search", "up": "Up Search"}

    # -- Figure 1: Accuracy line plot --
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, direction in zip(axes1, directions):
        dir_results = [r for r in results if r["direction"] == direction]
        by_size: dict[int, list[dict]] = defaultdict(list)
        for r in dir_results:
            by_size[r["effective_size"]].append(r)

        sizes = sorted(by_size)
        sra_acc = [
            sum(1 for r in by_size[s] if r["sra"]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]
        era_acc = [
            sum(1 for r in by_size[s] if r["era"]["outcome"] == "correct") / len(by_size[s])
            for s in sizes
        ]

        ax.plot(sizes, [a * 100 for a in sra_acc], "o-", label="SRA", color="#d62728", linewidth=2, markersize=7)
        ax.plot(sizes, [a * 100 for a in era_acc], "s-", label="ERA", color="#1f77b4", linewidth=2, markersize=7)

        if has_cc:
            cc_acc = []
            cc_sizes = []
            for s in sizes:
                cc_bucket = [r for r in by_size[s] if "cc" in r]
                if cc_bucket:
                    cc_sizes.append(s)
                    cc_acc.append(sum(1 for r in cc_bucket if r["cc"]["outcome"] == "correct") / len(cc_bucket) * 100)
            if cc_sizes:
                ax.plot(cc_sizes, cc_acc, "^-", label="CC", color="#ff7f0e", linewidth=2, markersize=7)

        ax.set_xlabel("Effective Size")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(dir_labels[direction])
        ax.set_xticks(sizes)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig1.suptitle("Search Accuracy by Effective Lattice Size", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("data/accuracy_by_size.png", dpi=150)
    print("Saved data/accuracy_by_size.png")

    # -- Figure 2: Calls bar chart --
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    n_bars = 5 if has_cc else 4
    bar_width = 0.8 / n_bars
    colors = {
        "BFS": "#2ca02c", "SRA": "#d62728", "ERA": "#1f77b4",
        "ERA Total": "#aec7e8", "CC": "#ff7f0e",
    }

    for ax, direction in zip(axes2, directions):
        dir_results = [r for r in results if r["direction"] == direction]
        by_size = defaultdict(list)
        for r in dir_results:
            by_size[r["effective_size"]].append(r)

        sizes = sorted(by_size)
        x = np.arange(len(sizes))

        bfs_calls = [sum(r["bfs_calls"] for r in by_size[s]) / len(by_size[s]) for s in sizes]
        sra_calls = [sum(r["sra"]["calls"] for r in by_size[s]) / len(by_size[s]) for s in sizes]
        era_calls = [sum(r["era"]["calls"] for r in by_size[s]) / len(by_size[s]) for s in sizes]
        era_total = [
            sum(r["era"]["calls"] + r["era_budget_used"] for r in by_size[s]) / len(by_size[s])
            for s in sizes
        ]

        offset = -(n_bars - 1) / 2
        ax.bar(x + offset * bar_width, bfs_calls, bar_width, label="BFS", color=colors["BFS"])
        ax.bar(x + (offset + 1) * bar_width, sra_calls, bar_width, label="SRA", color=colors["SRA"])
        ax.bar(x + (offset + 2) * bar_width, era_calls, bar_width, label="ERA (search)", color=colors["ERA"])
        ax.bar(x + (offset + 3) * bar_width, era_total, bar_width, label="ERA (total)", color=colors["ERA Total"], edgecolor=colors["ERA"], linewidth=0.8)

        if has_cc:
            cc_calls = []
            for s in sizes:
                cc_bucket = [r for r in by_size[s] if "cc" in r]
                if cc_bucket:
                    cc_calls.append(sum(r["cc"]["calls"] for r in cc_bucket) / len(cc_bucket))
                else:
                    cc_calls.append(0)
            ax.bar(x + (offset + 4) * bar_width, cc_calls, bar_width, label="CC", color=colors["CC"])

        ax.set_xlabel("Effective Size")
        ax.set_ylabel("Average Calls")
        ax.set_title(dir_labels[direction])
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig2.suptitle("Average Calls by Effective Lattice Size", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("data/calls_by_size.png", dpi=150)
    print("Saved data/calls_by_size.png")


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

    # Export cases for CC runner
    query_lookup = load_query_data(QUERY_PATH)
    export_cc_cases(all_results, query_lookup)

    # Merge CC results if available
    methods = ["sra", "era"]
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
