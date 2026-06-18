"""Per-method distance to BFS's CF (test, top-down).

For each case and each method, computes the lattice hop distance between the
proxy's returned key and BFS's CF key:

    hop_distance(P, B) = |P △ B| = |P ∪ B| - |P ∩ B|

i.e. the number of single-criterion toggles needed to walk from P to B in
the Hasse diagram. 0 means the proxy hit BFS's exact node; larger means
further away.

Cases where the proxy returned no key (`not_found`) are excluded from the
distance plot; they are reported separately.

Run:
    conda run -n coral python plot_cf_distance_test.py
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from search.lattice import hop_distance


ERA64_PATH         = "data/era_ablation_study/test/top_down_budget_64/era_budget_ablation_results.json"
ERA10_PATH         = "data/era_ablation_study/test/top_down_budget_10pct/era_budget_pct_ablation_results.json"
ERA_WEIGHTED_PATH  = "data/era_ablation_study/test/top_down_weighted_10pct/era_weighted_10pct_results.json"
ERA_WEIGHTED64_PATH = "data/era_ablation_study/test/top_down_weighted_64/era_weighted_64_results.json"
SRA_PATH           = "data/era_ablation_study/test/top_down_sra/sra_results.json"
CC64_PATH          = "data/era_ablation_study/test/cc_results_test_topdown.json"
PLOT_DIR           = "plots/era_ablation_study/test/comparison/cf_distance"


def main() -> None:
    apply_plot_style()

    with open(ERA64_PATH) as f:
        era64 = json.load(f)
    with open(ERA10_PATH) as f:
        era10 = json.load(f)
    with open(ERA_WEIGHTED_PATH) as f:
        erawt = json.load(f)
    with open(ERA_WEIGHTED64_PATH) as f:
        erawt64 = json.load(f)
    with open(SRA_PATH) as f:
        sra = json.load(f)
    cc64 = None
    if os.path.exists(CC64_PATH):
        with open(CC64_PATH) as f:
            cc64 = json.load(f)

    n = len(era64)
    assert all(len(x) == n for x in [era10, erawt, erawt64, sra]), \
        (len(era10), len(erawt), len(erawt64), len(sra))
    if cc64 is not None and len(cc64) != n:
        print(f"WARN: CC has {len(cc64)} rows vs {n} for others — skipping CC")
        cc64 = None

    sizes = sorted({r["effective_size"] for r in era64})

    # Methods: (label, color, marker, extractor of (bfs_key, proxy_key, found))
    methods = [
        ("SRA",                    "#d62728", "s", lambda i: (sra[i]["bfs_key"], sra[i]["sra"]["proxy_key"])),
        ("ERA softmax (10%)",      "#1f77b4", "D", lambda i: (era10[i]["bfs_key"], era10[i]["pct_10"]["proxy_key"])),
        ("ERA weighted (10%)",     "#17becf", "D", lambda i: (erawt[i]["bfs_key"], erawt[i]["pct_10"]["proxy_key"])),
        ("ERA softmax (budget=64)", "#9467bd", "D", lambda i: (era64[i]["bfs_key"], era64[i]["budget_64"]["proxy_key"])),
        ("ERA weighted (budget=64)", "#8c564b", "D", lambda i: (erawt64[i]["bfs_key"], erawt64[i]["budget_64"]["proxy_key"])),
    ]
    if cc64 is not None:
        methods.append(("CC (budget=64)", "#ff7f0e", "^", lambda i: (cc64[i].get("bfs_key"), cc64[i].get("proxy_key"))))

    # Compute hop distances per (method, case)
    distances: dict[str, list[int | None]] = {m[0]: [] for m in methods}
    for i in range(n):
        for label, _, _, get in methods:
            b_key, p_key = get(i)
            distances[label].append(hop_distance(p_key, b_key))

    os.makedirs(PLOT_DIR, exist_ok=True)

    def _save_dual(fig, name: str) -> None:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")

    # ─── Plot 1: Histogram of hop distances per method (all k pooled) ─────
    max_d = max(d for dl in distances.values() for d in dl if d is not None)
    bins = np.arange(-0.5, max_d + 1.5, 1)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    n_methods = len(methods)
    width = 0.8 / n_methods
    for j, (label, color, _, _) in enumerate(methods):
        dl = [d for d in distances[label] if d is not None]
        counts, _ = np.histogram(dl, bins=bins)
        proportions = counts / max(1, len(dl))
        xs = np.arange(max_d + 1) + (j - (n_methods - 1) / 2) * width
        ax1.bar(xs, proportions, width=width, label=label, color=color)
    ax1.set_xlabel("Lattice hop distance from BFS's CF")
    ax1.set_ylabel("Fraction of cases")
    ax1.set_title(f"Test top-down (N={n})")
    ax1.set_xticks(range(max_d + 1))
    ax1.legend(ncol=2, fontsize=10, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)
    fig1.suptitle("Distance from Proxy's CF to BFS's Minimal CF (test, top-down)", fontweight="bold")
    fig1.tight_layout()
    _save_dual(fig1, "cf_distance_hist.png")
    plt.close(fig1)

    # ─── Plot 2: Mean hop distance by k ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for label, color, marker, _ in methods:
        means = []
        for s in sizes:
            idx = [i for i in range(n) if era64[i]["effective_size"] == s]
            dl = [distances[label][i] for i in idx if distances[label][i] is not None]
            means.append(float(np.mean(dl)) if dl else float("nan"))
        ax2.plot(sizes, means, f"{marker}-", label=label, color=color, linewidth=2, markersize=7)
    ax2.set_xlabel("Number of Available Criteria")
    ax2.set_ylabel("Mean hop distance to BFS's CF")
    ax2.set_title(f"Test top-down (N={n})")
    ax2.set_xticks(sizes)
    ax2.legend(ncol=2, fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.suptitle("Mean Distance to BFS's CF by Lattice Size (test, top-down)", fontweight="bold")
    fig2.tight_layout()
    _save_dual(fig2, "cf_distance_by_size.png")
    plt.close(fig2)

    # ─── Summary table ─────────────────────────────────────────────────
    print()
    print(f"{'method':>26} | {'N w/ key':>8} | {'mean d':>6} | {'d=0':>5} | {'d=1':>5} | {'d=2':>5} | {'d>=3':>5} | not_found")
    print("-" * 105)
    for label, _, _, _ in methods:
        dl = distances[label]
        valid = [d for d in dl if d is not None]
        c = Counter(min(d, 3) for d in valid)  # bin >=3 together for the print
        print(
            f"{label:>26} | {len(valid):>8} | "
            f"{(sum(valid)/len(valid) if valid else float('nan')):>6.2f} | "
            f"{c.get(0,0):>5} | {c.get(1,0):>5} | {c.get(2,0):>5} | {c.get(3,0):>5} | "
            f"{sum(1 for d in dl if d is None):>9}"
        )


if __name__ == "__main__":
    main()
