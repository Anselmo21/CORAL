"""Headline plot: BFS / SRA / ERA(pct_10) accuracy + cost by effective lattice size.

Reads a per-dataset results.json (produced by any run_subsample.py) and emits
one PNG with two panels (accuracy | avg LLM calls).

CLI:
  conda run -n coral python3 -m scripts.shared.plot_headline \\
    --results data/msmarco_criteria/results.json \\
    --out plots/msmarco_criteria/headline.png \\
    --title "MSMARCO (criteria)"
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style

ERA_PCT: int = 10


def _normalize(raw: dict | list) -> list[dict]:
    """run_subsample.py outputs either {results: [...]} (msmarco/sonnet) or just [...] (pragma)."""
    if isinstance(raw, list):
        return raw
    return raw["results"]


def _era_pct10(r: dict) -> dict | None:
    return r.get("era", {}).get(f"era_pct_{ERA_PCT}")


def _era_by_label(label: str):
    def pick(r: dict) -> dict | None:
        return r.get("era", {}).get(label)
    return pick


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _acc(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m["outcome"] == "correct") / len(ms)


def _cost(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r)) is not None]
    if not ms:
        return float("nan")
    return float(np.mean([m["calls"] + m.get("budget_used", 0) for m in ms]))


def _bfs_calls(bucket: list[dict]) -> float:
    return float(np.mean([r["bfs"]["calls"] for r in bucket]))


def plot_headline(
    results_path: str,
    out_path: str,
    title: str,
    exclude_qids: set[str] | None = None,
    extra_era_labels: list[str] | None = None,
) -> None:
    """`extra_era_labels` — list of additional ERA config labels to overlay
    (e.g. ['era_fixed_16']). Each appears as another line on both panels."""
    apply_plot_style()
    with open(results_path) as f:
        results = _normalize(json.load(f))
    if exclude_qids:
        before = len(results)
        results = [r for r in results if str(r.get("qid")) not in exclude_qids]
        print(f"  excluded {before - len(results)} cases for qids={sorted(exclude_qids)}")

    by_size = _by_size(results)
    sizes = sorted(by_size)
    n_by_size = {s: len(by_size[s]) for s in sizes}

    # Reference + primary ERA series
    bfs_acc = [100.0] * len(sizes)
    sra_acc = [_acc(by_size[s], lambda r: r.get("sra")) for s in sizes]
    era_acc = [_acc(by_size[s], _era_pct10) for s in sizes]

    bfs_cost = [_bfs_calls(by_size[s]) for s in sizes]
    sra_cost = [_cost(by_size[s], lambda r: r.get("sra")) for s in sizes]
    era_cost = [_cost(by_size[s], _era_pct10) for s in sizes]

    # Extra ERA series (e.g. era_fixed_16)
    extra_colors = ["#9467bd", "#2ca02c", "#ff7f0e", "#17becf"]
    extra_markers = ["v", "^", "P", "X"]
    extra_series: list[dict] = []
    for i, label in enumerate(extra_era_labels or []):
        picker = _era_by_label(label)
        nice = label.replace("era_", "ERA (").replace("_", "=") + ")"
        # 'ERA (fixed=16)' is a bit verbose; clean up common cases
        nice = nice.replace("ERA (fixed=", "ERA (b=").replace("ERA (pct=", "ERA (pct_")
        extra_series.append({
            "label": nice,
            "color": extra_colors[i % len(extra_colors)],
            "marker": extra_markers[i % len(extra_markers)],
            "acc":   [_acc(by_size[s], picker)  for s in sizes],
            "cost":  [_cost(by_size[s], picker) for s in sizes],
        })

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax_acc.plot(sizes, bfs_acc, "o:", color="black",   linewidth=2, markersize=7, label="BFS (oracle)")
    ax_acc.plot(sizes, sra_acc, "s--", color="#d62728", linewidth=2, markersize=7, label="SRA")
    ax_acc.plot(sizes, era_acc, "D-", color="#1f77b4", linewidth=2, markersize=7, label=f"ERA (pct_{ERA_PCT})")
    for s in extra_series:
        ax_acc.plot(sizes, s["acc"], f"{s['marker']}-", color=s["color"],
                    linewidth=2, markersize=7, label=s["label"])
    ax_acc.set_xlabel("Number of Available Criteria")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 108)
    ax_acc.set_xticks(sizes)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower left", fontsize=8, ncol=2)

    ax_cost.plot(sizes, bfs_cost, "o:", color="black",   linewidth=2, markersize=7, label="BFS")
    ax_cost.plot(sizes, sra_cost, "s--", color="#d62728", linewidth=2, markersize=7, label="SRA")
    ax_cost.plot(sizes, era_cost, "D-", color="#1f77b4", linewidth=2, markersize=7, label=f"ERA (pct_{ERA_PCT})")
    for s in extra_series:
        ax_cost.plot(sizes, s["cost"], f"{s['marker']}-", color=s["color"],
                     linewidth=2, markersize=7, label=s["label"])
    ax_cost.set_xlabel("Number of Available Criteria")
    ax_cost.set_ylabel("Avg LLM Calls (search + budget)")
    ax_cost.set_xticks(sizes)
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(loc="upper left", fontsize=8, ncol=2)

    n_total = sum(n_by_size.values())
    fig.suptitle(f"{title}  —  n={n_total} cases  ({', '.join(f'k={s}:{n_by_size[s]}' for s in sizes)})",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    extra_header = "".join(f" {s['label'][:10]:>10}" for s in extra_series)
    print(f"\n  {title} — summary:")
    print(f"  {'k':>3} {'n':>4} | {'BFS':>5} {'SRA':>6} {'ERA':>6}{extra_header} | "
          f"{'BFS$':>6} {'SRA$':>6} {'ERA$':>6}"
          + "".join(f" {s['label'][:10]:>10}$" for s in extra_series))
    for i, s in enumerate(sizes):
        extra_acc_str  = "".join(f" {ex['acc'][i]:>10.1f}"  for ex in extra_series)
        extra_cost_str = "".join(f" {ex['cost'][i]:>10.1f}" for ex in extra_series)
        print(f"  {s:>3} {n_by_size[s]:>4} | "
              f"{bfs_acc[i]:>5.0f} {sra_acc[i]:>6.1f} {era_acc[i]:>6.1f}{extra_acc_str} | "
              f"{bfs_cost[i]:>6.1f} {sra_cost[i]:>6.1f} {era_cost[i]:>6.1f}{extra_cost_str}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out",     required=True)
    p.add_argument("--title",   required=True)
    p.add_argument("--exclude-qids", default="",
                   help="Comma-separated qids to drop from analysis.")
    p.add_argument("--extra-era", default="",
                   help="Comma-separated extra ERA labels to overlay (e.g. 'era_fixed_16').")
    args = p.parse_args()
    exclude = {q.strip() for q in args.exclude_qids.split(",") if q.strip()}
    extra = [s.strip() for s in args.extra_era.split(",") if s.strip()]
    plot_headline(args.results, args.out, args.title,
                  exclude_qids=exclude or None,
                  extra_era_labels=extra or None)


if __name__ == "__main__":
    main()
