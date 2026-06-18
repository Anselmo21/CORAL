"""Compare BFS / SRA / ERA(budget=64) / ERA(budget=10%) / CC(budget=64) on test top-down cases.

Reads existing ERA + CC results (both embed BFS) and runs SRA inline on the
same 200 test top-down cases (deterministic case ordering ensures alignment).
Writes SRA results to disk, then plots accuracy, call-cost, and CF hop distance by effective size.

Run:
    conda run -n coral python plot_test_top_down.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.era_budget_ablation import (
    LATTICES_PATH,
    TEST_CASES_PATH,
    effective_size,
    lattice_size,
    load_manifest_cases,
)
from scripts.shared.experiment import score_proxy
from scripts.shared.plot_style import apply_plot_style
from search.bfs import bfs_search
from search.lattice import hop_distance
from search.rank import sra_search


ERA64_PATH = "data/era_ablation_study/test/top_down_budget_64/era_budget_ablation_results.json"
ERA10_PATH = "data/era_ablation_study/test/top_down_budget_10pct/era_budget_pct_ablation_results.json"
CC64_PATH = "data/era_ablation_study/test/cc_results_test_topdown.json"
SRA_OUT = "data/era_ablation_study/test/top_down_sra/sra_results.json"
PLOT_DIR = "plots/era_ablation_study/test/comparison"


def run_sra(cases: list[dict]) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        sra_res = sra_search(criteria, rbc, start_node=start_node, direction=direction)
        scored = score_proxy(bfs_res, sra_res, rbc, start_node)
        scored["calls"] = sra_res["calls"]

        results.append({
            "query_id": case["query_id"],
            "effective_size": effective_size(case),
            "direction": direction,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "sra": scored,
        })
    return results


def _load_or_run_sra(cases: list[dict]) -> list[dict]:
    if os.path.exists(SRA_OUT):
        with open(SRA_OUT) as f:
            data = json.load(f)
        if len(data) == len(cases):
            print(f"Loaded {len(data)} SRA rows from {SRA_OUT}")
            return data
        print(f"SRA cache row count mismatch ({len(data)} vs {len(cases)}); recomputing")
    print(f"Running SRA on {len(cases)} cases...")
    results = run_sra(cases)
    os.makedirs(os.path.dirname(SRA_OUT), exist_ok=True)
    with open(SRA_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved SRA results to {SRA_OUT}")
    return results


def main() -> None:
    apply_plot_style()

    with open(ERA64_PATH) as f:
        era64_rows = json.load(f)
    with open(ERA10_PATH) as f:
        era10_rows = json.load(f)
    with open(CC64_PATH) as f:
        cc64_rows = json.load(f)
    print(f"Loaded ERA(64): {len(era64_rows)} rows")
    print(f"Loaded ERA(10%): {len(era10_rows)} rows")
    print(f"Loaded CC(64):  {len(cc64_rows)} rows")

    cases = load_manifest_cases(LATTICES_PATH, TEST_CASES_PATH, manifest_directions=["top-down"])
    sra_rows = _load_or_run_sra(cases)

    # all_lattices.json can drift between runs — restrict to the (query_id, effective_size)
    # tuples present in SRA (which is freshly regenerated from the current lattices file).
    sra_keys = [(r["query_id"], r["effective_size"]) for r in sra_rows]
    if len(set(sra_keys)) != len(sra_keys):
        print(f"WARN: SRA keys not unique ({len(sra_keys)} vs {len(set(sra_keys))} unique)")
    keep_set = set(sra_keys)
    def _filter(rows: list[dict], name: str) -> list[dict]:
        kept = [r for r in rows if (r["query_id"], r["effective_size"]) in keep_set]
        if len(kept) != len(rows):
            print(f"  {name}: dropped {len(rows) - len(kept)} rows not in SRA case set "
                  f"(kept {len(kept)}/{len(rows)})")
        return kept
    era64_rows = _filter(era64_rows, "ERA(64)")
    era10_rows = _filter(era10_rows, "ERA(10%)")
    cc64_rows = _filter(cc64_rows, "CC(64)")

    assert len(sra_rows) == len(era64_rows) == len(era10_rows) == len(cc64_rows), \
        (len(sra_rows), len(era64_rows), len(era10_rows), len(cc64_rows))

    # Index by effective_size
    sizes = sorted({r["effective_size"] for r in era64_rows})

    def by_size(rows: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            out[r["effective_size"]].append(r)
        return out

    era64_by = by_size(era64_rows)
    era10_by = by_size(era10_rows)
    sra_by = by_size(sra_rows)
    cc64_by = by_size(cc64_rows)

    # Accuracy curves
    def acc(rows: list[dict], result_key: str) -> float:
        if not rows:
            return float("nan")
        return sum(1 for r in rows if r[result_key]["outcome"] == "correct") / len(rows)

    bfs_acc = [1.0] * len(sizes)
    sra_acc = [acc(sra_by[s], "sra") for s in sizes]
    era64_acc = [acc(era64_by[s], "budget_64") for s in sizes]
    era10_acc = [acc(era10_by[s], "pct_10") for s in sizes]
    cc64_acc = [
        sum(1 for r in cc64_by[s] if r["outcome"] == "correct") / len(cc64_by[s])
        for s in sizes
    ]

    # Calls curves (means)
    def mean_calls(rows: list[dict], call_fn) -> float:
        vals = [call_fn(r) for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    bfs_calls = [mean_calls(era64_by[s], lambda r: r["bfs_calls"]) for s in sizes]
    sra_calls = [mean_calls(sra_by[s], lambda r: r["sra"]["calls"]) for s in sizes]
    era64_calls = [
        mean_calls(era64_by[s], lambda r: r["budget_64"]["calls"] + r["budget_64"]["budget_used"])
        for s in sizes
    ]
    era10_calls = [
        mean_calls(era10_by[s], lambda r: r["pct_10"]["calls"] + r["pct_10"]["budget_used"])
        for s in sizes
    ]
    # CC total cost: search calls + num_ablations spent fitting the surrogate.
    cc64_calls = [
        mean_calls(cc64_by[s], lambda r: r["calls"] + r["budget"])
        for s in sizes
    ]

    os.makedirs(PLOT_DIR, exist_ok=True)

    def _save_dual(fig, name: str) -> None:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    # ─── Accuracy plot ────────────────────────────────────────────────────
    # Slightly smaller canvas; fonts unchanged via apply_plot_style().
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sizes, bfs_acc, "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax1.plot(sizes, sra_acc, "s-", label="SRA",            color="#d62728", linewidth=2, markersize=7)
    # ax1.plot(sizes, era10_acc, "D-", label="ERA (10% budget)", color="#1f77b4", linewidth=2, markersize=7)
    ax1.plot(sizes, era64_acc, "D-", label="ERA",  color="#9467bd", linewidth=2, markersize=7)
    ax1.plot(sizes, cc64_acc, "^-", label="CC",    color="#ff7f0e", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Available Criteria")
    ax1.set_ylabel("Average Accuracy")
    ax1.set_xticks(sizes)
    ax1.set_ylim(0, 1.05)
    ax1.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.31), frameon=True)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.subplots_adjust(bottom=0.22)
    _save_dual(fig1, "accuracy_by_size.png")
    plt.close(fig1)

    # ─── Calls plot ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(sizes, bfs_calls, "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax2.plot(sizes, sra_calls, "s-", label="SRA",            color="#d62728", linewidth=2, markersize=7)
    # ax2.plot(sizes, era10_calls, "D-", label="ERA (10% budget)", color="#1f77b4", linewidth=2, markersize=7)
    ax2.plot(sizes, era64_calls, "D-", label="ERA",  color="#9467bd", linewidth=2, markersize=7)
    ax2.plot(sizes, cc64_calls, "^-", label="CC",    color="#ff7f0e", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Available Criteria")
    ax2.set_ylabel("Average Number of LLM Calls")
    # ax2.set_title("Top-Down Test")
    ax2.set_xticks(sizes)
    ax2.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.36), frameon=True)
    ax2.grid(True, alpha=0.3)
    fig2.suptitle("LLM Call Cost to Find First Counterfactual (test, top-down)", fontweight="bold")
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.22)
    _save_dual(fig2, "calls_by_size.png")
    plt.close(fig2)

    # ─── Normalized calls (total LLM cost / 2^k) ─────────────────────────
    def norm_calls(raw: list[float]) -> list[float]:
        return [v / float(lattice_size(k)) for v, k in zip(raw, sizes)]

    bfs_calls_norm = norm_calls(bfs_calls)
    sra_calls_norm = norm_calls(sra_calls)
    era64_calls_norm = norm_calls(era64_calls)
    cc64_calls_norm = norm_calls(cc64_calls)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(sizes, bfs_calls_norm, "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax3.plot(sizes, sra_calls_norm, "s-", label="SRA", color="#d62728", linewidth=2, markersize=7)
    ax3.plot(sizes, era64_calls_norm, "D-", label="ERA", color="#9467bd", linewidth=2, markersize=7)
    ax3.plot(sizes, cc64_calls_norm, "^-", label="CC", color="#ff7f0e", linewidth=2, markersize=8)
    ax3.set_xlabel("Number of Available Criteria")
    ax3.set_ylabel("Average LLM Calls / Lattice Size")
    ax3.set_xticks(sizes)
    ax3.set_ylim(0, 1)
    ax3.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.36), frameon=True)
    ax3.grid(True, alpha=0.3)
    fig3.suptitle(
        "Normalized LLM Call Cost to Find First Counterfactual (test, top-down)",
        fontweight="bold",
    )
    fig3.tight_layout()
    fig3.subplots_adjust(bottom=0.22)
    _save_dual(fig3, "calls_by_size_normalized.png")
    plt.close(fig3)

    # ─── CF hop distance to BFS's minimal CF ───────────────────────────────
    def case_key(r: dict) -> tuple[str, int]:
        return (r["query_id"], r["effective_size"])

    sra_by_case = {case_key(r): r for r in sra_rows}
    cc_by_case = {case_key(r): r for r in cc64_rows}

    dist_by_method: dict[str, dict[int, list[int]]] = {
        "SRA": defaultdict(list),
        "ERA": defaultdict(list),
        "CC": defaultdict(list),
    }
    for r in era64_rows:
        k = case_key(r)
        s = r["effective_size"]
        bfs_key = r["bfs_key"]
        sr = sra_by_case.get(k)
        if sr is not None:
            d = hop_distance(sr["sra"]["proxy_key"], bfs_key)
            if d is not None:
                dist_by_method["SRA"][s].append(d)
        d = hop_distance(r["budget_64"]["proxy_key"], bfs_key)
        if d is not None:
            dist_by_method["ERA"][s].append(d)
        cc = cc_by_case.get(k)
        if cc is not None:
            d = hop_distance(cc.get("proxy_key"), bfs_key)
            if d is not None:
                dist_by_method["CC"][s].append(d)

    def mean_dist(by_s: dict[int, list[int]]) -> list[float]:
        return [float(np.mean(by_s[s])) if by_s[s] else float("nan") for s in sizes]

    sra_dist = mean_dist(dist_by_method["SRA"])
    era_dist = mean_dist(dist_by_method["ERA"])
    cc_dist = mean_dist(dist_by_method["CC"])
    bfs_dist = [0.0] * len(sizes)

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    # ax4.plot(sizes, bfs_dist, "o:", label="BFS (baseline)", color="black", linewidth=2, markersize=7)
    ax4.plot(sizes, sra_dist, "s-", label="SRA", color="#d62728", linewidth=2, markersize=7)
    ax4.plot(sizes, era_dist, "D-", label="ERA", color="#9467bd", linewidth=2, markersize=7)
    ax4.plot(sizes, cc_dist, "^-", label="CC", color="#ff7f0e", linewidth=2, markersize=8)
    ax4.set_xlabel("Number of Available Criteria")
    ax4.set_ylabel("Average Distance to True CF")
    ax4.set_xticks(sizes)
    ax4.set_ylim(bottom=0)
    ax4.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.31), frameon=True)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.subplots_adjust(bottom=0.22)
    _save_dual(fig4, "cf_distance_by_size.png")
    plt.close(fig4)

    all_dists: dict[str, list[int]] = {
        "SRA": [d for ds in dist_by_method["SRA"].values() for d in ds],
        "ERA": [d for ds in dist_by_method["ERA"].values() for d in ds],
        "CC": [d for ds in dist_by_method["CC"].values() for d in ds],
    }
    max_d = max((d for dl in all_dists.values() for d in dl), default=0)
    bins = np.arange(-0.5, max_d + 1.5, 1)
    hist_methods = [
        ("SRA", "#d62728", "s", all_dists["SRA"]),
        ("ERA", "#9467bd", "D", all_dists["ERA"]),
        ("CC", "#ff7f0e", "^", all_dists["CC"]),
    ]
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    n_hist = len(hist_methods)
    width = 0.8 / n_hist
    for j, (label, color, marker, dl) in enumerate(hist_methods):
        counts, _ = np.histogram(dl, bins=bins)
        proportions = counts / max(1, len(dl))
        xs = np.arange(max_d + 1) + (j - (n_hist - 1) / 2) * width
        ax5.bar(xs, proportions, width=width, label=label, color=color)
    ax5.set_xlabel("Hop Distance to BFS CF")
    ax5.set_ylabel("Fraction of Cases")
    ax5.set_xticks(range(max_d + 1))
    ax5.legend(ncol=3, loc="upper right", frameon=True)
    ax5.grid(True, axis="y", alpha=0.3)
    fig5.tight_layout()
    _save_dual(fig5, "cf_distance_hist.png")
    plt.close(fig5)

    # ─── Summary table ───────────────────────────────────────────────────
    print()
    print(
        f"{'k':>3} | {'N':>3} | "
        f"{'BFS acc':>8} | {'SRA acc':>8} | {'ERA10 acc':>10} | {'ERA64 acc':>10} | {'CC64 acc':>9} || "
        f"{'BFS calls':>10} | {'SRA calls':>10} | {'ERA10 calls':>12} | {'ERA64 calls':>12} | {'CC64 calls':>11}"
    )
    print("-" * 165)
    for s in sizes:
        n = len(era64_by[s])
        cc_correct = sum(1 for r in cc64_by[s] if r["outcome"] == "correct")
        cc_acc_s = cc_correct / len(cc64_by[s])
        print(
            f"{s:>3} | {n:>3} | "
            f"{1.0:>8.1%} | "
            f"{acc(sra_by[s], 'sra'):>8.1%} | "
            f"{acc(era10_by[s], 'pct_10'):>10.1%} | "
            f"{acc(era64_by[s], 'budget_64'):>10.1%} | "
            f"{cc_acc_s:>9.1%} || "
            f"{mean_calls(era64_by[s], lambda r: r['bfs_calls']):>10.1f} | "
            f"{mean_calls(sra_by[s], lambda r: r['sra']['calls']):>10.1f} | "
            f"{mean_calls(era10_by[s], lambda r: r['pct_10']['calls'] + r['pct_10']['budget_used']):>12.1f} | "
            f"{mean_calls(era64_by[s], lambda r: r['budget_64']['calls'] + r['budget_64']['budget_used']):>12.1f} | "
            f"{mean_calls(cc64_by[s], lambda r: r['calls'] + r['budget']):>11.1f}"
        )


if __name__ == "__main__":
    main()
