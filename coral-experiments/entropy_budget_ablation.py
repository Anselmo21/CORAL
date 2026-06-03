import json
from collections import defaultdict
from functools import partial

from search.bfs import bfs_search
from search.heuristics import (
    renyi_entropy,
    shannon_entropy,
    softmax_entropy,
    weighted_entropy,
)
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search

DATA_PATH = "data/base.json"
BUDGET_PCTS = [5, 10, 15, 20, 25, 30]

ENTROPY_VARIANTS: list[tuple[str, object]] = [
    ("softmax", softmax_entropy),
    ("shannon", shannon_entropy),
    ("renyi_0.5", partial(renyi_entropy, alpha=0.5)),
    ("renyi_2", partial(renyi_entropy, alpha=2.0)),
    ("weighted", weighted_entropy),
]


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
        for ent_name, entropy_fn in ENTROPY_VARIANTS:
            for i, lat in enumerate(lattices):
                bfs_res = bfs_cache[i]
                if not bfs_res["found"]:
                    continue

                search_space = 2 ** lat["n_criteria"]
                budget = max(1, int(search_space * pct / 100))

                era_res = era_search(
                    lat["criteria"], lat["rankings_by_combo"],
                    budget=budget, entropy_fn=entropy_fn,
                    start_node=lat["start_node"], direction="down",
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
                    "entropy": ent_name,
                    "jaccard": jacc,
                })

    return all_results


def print_summary(results: list[dict], budget_pcts: list[int]) -> None:
    names = [name for name, _ in ENTROPY_VARIANTS]

    print(f"\n{'=' * (20 + 14 * len(names))}")
    print(f"  ERA Accuracy by Budget % and Entropy Function")
    print(f"{'=' * (20 + 14 * len(names))}")

    header = f"  {'Budget':>6} | {'N':>3} |"
    for name in names:
        header += f" {name:>12} |"
    print(header)

    sep = f"  {'-' * 6}-+-{'-' * 3}-+"
    for _ in names:
        sep += f"-{'-' * 12}-+"
    print(sep)

    for pct in budget_pcts:
        row_by_ent: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            if r["budget_pct"] == pct:
                row_by_ent[r["entropy"]].append(r)

        n = len(row_by_ent[names[0]]) if names[0] in row_by_ent else 0
        line = f"  {pct:>5}% | {n:>3} |"
        for name in names:
            bucket = row_by_ent.get(name, [])
            correct = sum(1 for r in bucket if r["outcome"] == "correct")
            nn = len(bucket)
            acc = correct / nn if nn else 0
            line += f" {acc:>4.0%} ({correct:>2}/{nn:<2}) |"
        print(line)

    print(f"\n  Total calls (search + budget) by Budget % and Entropy:")
    header2 = f"  {'Budget':>6} |"
    for name in names:
        header2 += f" {name:>12} |"
    print(header2)
    print(sep[:-1])

    for pct in budget_pcts:
        row_by_ent: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            if r["budget_pct"] == pct:
                row_by_ent[r["entropy"]].append(r)

        line = f"  {pct:>5}% |"
        for name in names:
            bucket = row_by_ent.get(name, [])
            avg = sum(r["era_total"] for r in bucket) / len(bucket) if bucket else 0
            line += f" {avg:>12.1f} |"
        print(line)

    bfs_avg = 0.0
    bfs_buckets = [r for r in results if r["entropy"] == names[0]]
    if bfs_buckets:
        pct0 = budget_pcts[0]
        bfs_subset = [r for r in bfs_buckets if r["budget_pct"] == pct0]
        bfs_avg = sum(r["bfs_calls"] for r in bfs_subset) / len(bfs_subset) if bfs_subset else 0
    print(f"  BFS avg: {bfs_avg:.1f}")

    print(f"\n  Mean Jaccard(proxy CF, BFS CF) by Budget % and Entropy:")
    print(header2)
    print(sep[:-1])
    for pct in budget_pcts:
        row_by_ent: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            if r["budget_pct"] == pct:
                row_by_ent[r["entropy"]].append(r)

        line = f"  {pct:>5}% |"
        for name in names:
            bucket = row_by_ent.get(name, [])
            jaccs = [r["jaccard"] for r in bucket if r.get("jaccard") is not None]
            avg = sum(jaccs) / len(jaccs) if jaccs else float("nan")
            line += f" {avg:>12.3f} |"
        print(line)


def plot_results(results: list[dict], budget_pcts: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import apply_plot_style

    apply_plot_style()

    names = [name for name, _ in ENTROPY_VARIANTS]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    markers = ["s", "o", "^", "D", "v"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bfs_avg = 0.0

    for i, name in enumerate(names):
        accs: list[float] = []
        totals: list[float] = []
        for pct in budget_pcts:
            bucket = [r for r in results if r["budget_pct"] == pct and r["entropy"] == name]
            n = len(bucket)
            correct = sum(1 for r in bucket if r["outcome"] == "correct")
            accs.append(correct / n * 100 if n else 0)
            totals.append(sum(r["era_total"] for r in bucket) / n if n else 0)
            if i == 0:
                bfs_avg = sum(r["bfs_calls"] for r in bucket) / n if n else bfs_avg

        ax1.plot(budget_pcts, accs, f"{markers[i]}-", label=name, color=colors[i],
                 linewidth=2, markersize=8)
        ax2.plot(budget_pcts, totals, f"{markers[i]}-", label=name, color=colors[i],
                 linewidth=2, markersize=8)

    ax1.set_xlabel("Budget (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("ERA Accuracy vs Budget")
    ax1.set_xticks(budget_pcts)
    ax1.set_ylim(0, 105)
    ax1.legend(ncol=len(names), loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=bfs_avg, color="gray", linestyle="--", linewidth=2, label=f"BFS ({bfs_avg:.1f})")
    ax2.set_xlabel("Budget (%)")
    ax2.set_ylabel("Mean Total Calls")
    ax2.set_title("ERA Total Calls vs Budget")
    ax2.set_xticks(budget_pcts)
    ax2.legend(ncol=len(names) + 1, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("ERA Budget Ablation by Entropy Function (base.json, 19 lattices)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("data/entropy_budget_ablation.png", dpi=150)
    plt.close(fig)
    print("Saved data/entropy_budget_ablation.png")


def main() -> None:
    lattices = load_lattices(DATA_PATH)
    print(f"Loaded {len(lattices)} lattices from {DATA_PATH}")

    has_cf = sum(
        1 for lat in lattices
        if bfs_search(lat["criteria"], lat["rankings_by_combo"],
                      start_node=lat["start_node"], direction="down")["found"]
    )
    print(f"  {has_cf} have a counterfactual")

    results = run_ablation(lattices, BUDGET_PCTS)
    print_summary(results, BUDGET_PCTS)
    plot_results(results, BUDGET_PCTS)

    with open("data/entropy_budget_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved data/entropy_budget_ablation_results.json")


if __name__ == "__main__":
    main()
