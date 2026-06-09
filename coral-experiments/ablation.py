import json
import os
from collections import defaultdict

from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search

DATA_PATH = "data/base.json"
BUDGET_PCTS = [5, 10, 15, 20, 25, 30]


def load_lattices(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    lattices: list[dict] = []
    for row in data["rows"]:
        for run in row["baseline_lattice_runs"]:
            lattices.append({
                "query_id": row["query_id"],
                "criteria": run["criteria_used"],
                "rankings_by_combo": run["rankings_by_combo"],
                "start_node": run["criteria_used"],
                "n_criteria": len(run["criteria_used"]),
            })
    return lattices


def score_era(
    bfs_result: dict,
    era_result: dict,
    rankings_by_combo: dict[str, list[str]],
    start_node: list[str],
) -> str:
    start_key = combo_key(start_node)
    baseline_top1 = (rankings_by_combo.get(start_key) or [""])[0]

    proxy_key = era_result.get("key")
    if not era_result.get("found") or proxy_key is None:
        return "not_found"

    ranking = rankings_by_combo.get(proxy_key, [])
    if not ranking or ranking[0] == baseline_top1:
        return "false_positive"

    if era_result.get("size") != bfs_result["size"]:
        return "wrong_layer"

    return "correct"


def run_ablation(lattices: list[dict], budget_pcts: list[int]) -> list[dict]:
    bfs_cache: dict[int, dict] = {}
    for i, lat in enumerate(lattices):
        bfs_cache[i] = bfs_search(
            lat["criteria"], lat["rankings_by_combo"],
            start_node=lat["start_node"], direction="down",
        )

    all_results: list[dict] = []

    for pct in budget_pcts:
        for i, lat in enumerate(lattices):
            bfs_res = bfs_cache[i]
            if not bfs_res["found"]:
                continue

            search_space = 2 ** lat["n_criteria"]
            budget = max(1, int(search_space * pct / 100))

            era_res = era_search(
                lat["criteria"], lat["rankings_by_combo"],
                budget=budget, start_node=lat["start_node"], direction="down",
            )

            outcome = score_era(bfs_res, era_res, lat["rankings_by_combo"], lat["start_node"])
            total_calls = era_res["calls"] + era_res.get("budget_used", 0)
            jacc = jaccard_keys(bfs_res.get("key"), era_res.get("key"))

            all_results.append({
                "query_id": lat["query_id"],
                "n_criteria": lat["n_criteria"],
                "budget_pct": pct,
                "budget": budget,
                "budget_used": era_res.get("budget_used", 0),
                "era_calls": era_res["calls"],
                "era_total": total_calls,
                "bfs_calls": bfs_res["calls"],
                "outcome": outcome,
                "jaccard": jacc,
            })

    return all_results


def print_summary(results: list[dict], budget_pcts: list[int]) -> None:
    print(f"\n{'=' * 89}")
    print(f"  ERA Budget Ablation — Accuracy & Calls by Budget %")
    print(f"{'=' * 89}")
    print(
        f"  {'Budget':>6} | {'N':>3} | {'Accuracy':>12} | "
        f"{'ERA search':>10} | {'ERA total':>10} | {'BFS':>10} | "
        f"{'Savings':>7} | {'Jaccard':>7}"
    )
    print(
        f"  {'-' * 6}-+-{'-' * 3}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 7}-+-{'-' * 7}"
    )

    for pct in budget_pcts:
        bucket = [r for r in results if r["budget_pct"] == pct]
        n = len(bucket)
        correct = sum(1 for r in bucket if r["outcome"] == "correct")
        acc = correct / n if n else 0.0
        era_search_avg = sum(r["era_calls"] for r in bucket) / n
        era_total_avg = sum(r["era_total"] for r in bucket) / n
        bfs_avg = sum(r["bfs_calls"] for r in bucket) / n
        savings = (1 - era_total_avg / bfs_avg) * 100 if bfs_avg > 0 else 0.0
        jaccs = [r["jaccard"] for r in bucket if r.get("jaccard") is not None]
        jacc_avg = sum(jaccs) / len(jaccs) if jaccs else float("nan")
        print(
            f"  {pct:>5}% | {n:>3} | "
            f"{acc:>5.0%} ({correct:>2}/{n:<2}) | "
            f"{era_search_avg:>10.1f} | {era_total_avg:>10.1f} | {bfs_avg:>10.1f} | "
            f"{savings:>6.1f}% | {jacc_avg:>7.3f}"
        )

    # Outcome breakdown per budget %
    print(f"\n  Outcome breakdown:")
    print(f"  {'Budget':>6} | {'correct':>7} | {'wrong_layer':>11} | {'false_pos':>9} | {'not_found':>9}")
    print(f"  {'-' * 6}-+-{'-' * 7}-+-{'-' * 11}-+-{'-' * 9}-+-{'-' * 9}")
    for pct in budget_pcts:
        bucket = [r for r in results if r["budget_pct"] == pct]
        outcomes: dict[str, int] = defaultdict(int)
        for r in bucket:
            outcomes[r["outcome"]] += 1
        print(
            f"  {pct:>5}% | {outcomes.get('correct', 0):>7} | "
            f"{outcomes.get('wrong_layer', 0):>11} | "
            f"{outcomes.get('false_positive', 0):>9} | "
            f"{outcomes.get('not_found', 0):>9}"
        )


def plot_ablation(results: list[dict], budget_pcts: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import apply_plot_style

    apply_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    accs: list[float] = []
    era_search_avgs: list[float] = []
    era_total_avgs: list[float] = []
    bfs_avg = 0.0

    for pct in budget_pcts:
        bucket = [r for r in results if r["budget_pct"] == pct]
        n = len(bucket)
        correct = sum(1 for r in bucket if r["outcome"] == "correct")
        accs.append(correct / n * 100 if n else 0.0)
        era_search_avgs.append(sum(r["era_calls"] for r in bucket) / n)
        era_total_avgs.append(sum(r["era_total"] for r in bucket) / n)
        bfs_avg = sum(r["bfs_calls"] for r in bucket) / n

    # Accuracy vs budget %
    ax1.plot(budget_pcts, accs, "s-", color="#1f77b4", linewidth=2, markersize=8)
    ax1.set_xlabel("Budget (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("ERA Accuracy vs Budget")
    ax1.set_xticks(budget_pcts)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Calls vs budget %
    ax2.plot(budget_pcts, era_search_avgs, "o-", label="ERA (search)", color="#1f77b4", linewidth=2, markersize=7)
    ax2.plot(budget_pcts, era_total_avgs, "s-", label="ERA (total)", color="#aec7e8", linewidth=2, markersize=7)
    ax2.axhline(y=bfs_avg, color="#2ca02c", linestyle="--", linewidth=2, label=f"BFS ({bfs_avg:.1f})")
    ax2.set_xlabel("Budget (%)")
    ax2.set_ylabel("Mean Calls")
    ax2.set_title("ERA Calls vs Budget")
    ax2.set_xticks(budget_pcts)
    ax2.legend(ncol=3, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("ERA Budget Ablation (base.json, 19 lattices)", fontweight="bold")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/ablation_budget.png", dpi=150)
    print("Saved plots/ablation_budget.png")


def main() -> None:
    lattices = load_lattices(DATA_PATH)
    print(f"Loaded {len(lattices)} lattices from {DATA_PATH}")

    has_cf = sum(
        1 for lat in lattices
        if bfs_search(lat["criteria"], lat["rankings_by_combo"],
                      start_node=lat["start_node"], direction="down")["found"]
    )
    print(f"  {has_cf} have a counterfactual (used in ablation)")

    results = run_ablation(lattices, BUDGET_PCTS)
    print_summary(results, BUDGET_PCTS)
    plot_ablation(results, BUDGET_PCTS)

    with open("data/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved data/ablation_results.json")


if __name__ == "__main__":
    main()
