"""Compare ERA at softmax T=1 vs T=3 on the 16-lattice case set.

Reads:
  data/16_lattices_cf_layer2to8/results.json      (T=1)
  data/16_lattices_cf_layer2to8/results_T3.json   (T=3)

Output:
  plots/16_lattices_cf_layer2to8/T1_vs_T3.png
  plots/16_lattices_cf_layer2to8/T1_vs_T3.txt
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

T1_PATH = "data/16_lattices_cf_layer2to8/results.json"
T3_PATH = "data/16_lattices_cf_layer2to8/results_T3.json"
OUT_DIR = "plots/16_lattices_cf_layer2to8"

CONFIGS = [
    ("era_pct_10",  "pct_10"),
    ("era_fixed_16", "b=16"),
]


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _acc(bucket: list[dict], label: str) -> float:
    cells = [r["era"].get(label) for r in bucket]
    cells = [c for c in cells if c is not None]
    if not cells:
        return float("nan")
    return 100.0 * sum(1 for c in cells if c["outcome"] == "correct") / len(cells)


def _cost(bucket: list[dict], label: str) -> float:
    cells = [r["era"].get(label) for r in bucket]
    cells = [c for c in cells if c is not None]
    if not cells:
        return float("nan")
    return float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))


def _bfs_cost(bucket: list[dict]) -> float:
    return float(np.mean([r["bfs"]["calls"] for r in bucket]))


def _sra_acc(bucket: list[dict]) -> float:
    return 100.0 * sum(1 for r in bucket if r["sra"]["outcome"] == "correct") / len(bucket)


def _sra_cost(bucket: list[dict]) -> float:
    return float(np.mean([r["sra"]["calls"] + r["sra"].get("budget_used", 0) for r in bucket]))


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(T1_PATH) as f:
        t1 = json.load(f)
    with open(T3_PATH) as f:
        t3 = json.load(f)

    t1_by = _by_size(t1)
    t3_by = _by_size(t3)
    sizes = sorted(set(t1_by) | set(t3_by))

    bfs_cost   = [_bfs_cost(t1_by[s]) for s in sizes]
    sra_acc    = [_sra_acc(t1_by[s])  for s in sizes]
    sra_cost   = [_sra_cost(t1_by[s]) for s in sizes]

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(11, 4))
    ax_acc.plot(sizes, [100.0]*len(sizes), "o:", color="black", linewidth=2, markersize=7, label="BFS (oracle)")
    ax_acc.plot(sizes, sra_acc, "s--", color="#d62728", linewidth=2, markersize=7, label="SRA")

    ax_cost.plot(sizes, bfs_cost, "o:", color="black", linewidth=2, markersize=7, label="BFS")
    ax_cost.plot(sizes, sra_cost, "s--", color="#d62728", linewidth=2, markersize=7, label="SRA")

    style_map = {
        ("era_pct_10",  "T1"): ("D-",  "#1f77b4"),
        ("era_pct_10",  "T3"): ("D--", "#7fbfff"),
        ("era_fixed_16", "T1"): ("v-",  "#9467bd"),
        ("era_fixed_16", "T3"): ("v--", "#cea3f0"),
    }

    table_rows: list[str] = []
    table_rows.append(f"  {'k':>3} {'n':>3} | "
                      f"{'ERA pct_10 T1':>13} {'ERA pct_10 T3':>13} {'Δ':>5} | "
                      f"{'ERA b=16 T1':>13} {'ERA b=16 T3':>13} {'Δ':>5}")

    for label, nice in CONFIGS:
        for tag, by, lstyle, label_suffix in [
            ("T1", t1_by, style_map[(label, "T1")], "T=1"),
            ("T3", t3_by, style_map[(label, "T3")], "T=3"),
        ]:
            accs  = [_acc(by[s], label)  for s in sizes]
            costs = [_cost(by[s], label) for s in sizes]
            ax_acc.plot(sizes, accs, lstyle[0], color=lstyle[1], linewidth=2, markersize=7,
                        label=f"ERA ({nice}, {label_suffix})")
            ax_cost.plot(sizes, costs, lstyle[0], color=lstyle[1], linewidth=2, markersize=7,
                         label=f"ERA ({nice}, {label_suffix})")

    for s in sizes:
        n = len(t1_by[s])
        pct_t1 = _acc(t1_by[s], "era_pct_10")
        pct_t3 = _acc(t3_by[s], "era_pct_10")
        b16_t1 = _acc(t1_by[s], "era_fixed_16")
        b16_t3 = _acc(t3_by[s], "era_fixed_16")
        table_rows.append(
            f"  {s:>3} {n:>3} | "
            f"{pct_t1:>13.1f} {pct_t3:>13.1f} {pct_t3-pct_t1:>+5.1f} | "
            f"{b16_t1:>13.1f} {b16_t3:>13.1f} {b16_t3-b16_t1:>+5.1f}"
        )

    for ax in (ax_acc, ax_cost):
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.3)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 108)
    ax_acc.set_xlabel("Number of Available Pragmas")
    ax_acc.legend(loc="lower left", fontsize=7, ncol=2)
    ax_cost.set_ylabel("Avg LLM Calls")
    ax_cost.set_xlabel("Number of Available Pragmas")
    ax_cost.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle("ERA softmax T=1 vs T=3 — 16-lattice regen", fontweight="bold")
    fig.tight_layout()
    out = f"{OUT_DIR}/T1_vs_T3.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    print("\nAccuracy table (T1 vs T3):")
    for r in table_rows:
        print(r)

    with open(f"{OUT_DIR}/T1_vs_T3.txt", "w") as f:
        f.write("\n".join(table_rows) + "\n")

    # Aggregate totals
    print("\nAggregate per-config accuracy:")
    for label, nice in CONFIGS:
        t1_correct = sum(1 for r in t1 if r["era"].get(label, {}).get("outcome") == "correct")
        t3_correct = sum(1 for r in t3 if r["era"].get(label, {}).get("outcome") == "correct")
        n = len(t1)
        print(f"  {nice:<10}: T=1: {t1_correct}/{n} = {100*t1_correct/n:.1f}%   "
              f"T=3: {t3_correct}/{n} = {100*t3_correct/n:.1f}%   "
              f"Δ = {100*(t3_correct-t1_correct)/n:+.1f}pp")


if __name__ == "__main__":
    main()
