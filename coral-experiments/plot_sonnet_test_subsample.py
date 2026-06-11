"""Plot BFS / SRA / ERA(b=16) / CC(optional) for sonnet test sub-lattices.

Reads data/sonnet_test_subsample/results.json.
Merges CC results from data/sonnet_test_subsample/cc_results.json if present.
Writes plots/sonnet_test_subsample/{accuracy_by_size,cost_by_size,outcomes}.png
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style

RESULTS_PATH = "data/sonnet_test_subsample/results.json"
CC_RESULTS_PATH = "data/sonnet_test_subsample/cc_results_pct10.json"
OUT_DIR = "plots/sonnet_test_subsample"

CC_BUDGET = 10  # pct_10: budget = max(1, int(2^k * 10 / 100))


def _load() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _load_cc() -> dict[int, dict] | None:
    """Load CC results keyed by case_id. Returns None if file absent."""
    if not os.path.exists(CC_RESULTS_PATH):
        return None
    with open(CC_RESULTS_PATH) as f:
        raw = json.load(f)
    results = raw if isinstance(raw, list) else raw.get("results", [])
    return {r["case_id"]: r for r in results}


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


def plot_accuracy(results: list[dict], cc_by_id: dict[int, dict] | None, out_dir: str) -> None:
    by_size = _by_size(results)
    sizes = sorted(by_size)

    methods = [
        ("BFS",       "black",   "o:", lambda r: {"outcome": "correct", "calls": r["bfs"]["calls"], "budget_used": 0}),
        ("SRA",       "#d62728", "s--", lambda r: r.get("sra")),
        ("ERA b=16",  "#1f77b4", "D-",  lambda r: r.get("era")),
    ]
    if cc_by_id is not None:
        def _cc_pick(r):
            cc = cc_by_id.get(r["case_id"])
            return cc
        methods.append(("LSM b=10%", "#ff7f0e", "^-", _cc_pick))

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, style, picker in methods:
        if label == "BFS":
            y = [100.0] * len(sizes)
        else:
            y = [_acc(by_size[s], picker) for s in sizes]
        ax.plot(sizes, y, style, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Available Criteria")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=4)
    fig.tight_layout()
    path = f"{out_dir}/accuracy_by_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cost(results: list[dict], cc_by_id: dict[int, dict] | None, out_dir: str) -> None:
    by_size = _by_size(results)
    sizes = sorted(by_size)

    methods = [
        ("BFS",       "black",   "o:", lambda r: {"calls": r["bfs"]["calls"], "budget_used": 0}),
        ("SRA",       "#d62728", "s--", lambda r: r.get("sra")),
        ("ERA b=16",  "#1f77b4", "D-",  lambda r: r.get("era")),
    ]
    if cc_by_id is not None:
        def _cc_pick(r):
            cc = cc_by_id.get(r["case_id"])
            if cc is None:
                return None
            return {"calls": cc.get("calls", 0), "budget_used": cc.get("budget", CC_BUDGET), "outcome": cc.get("outcome")}
        methods.append(("LSM b=10%", "#ff7f0e", "^-", _cc_pick))

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, style, picker in methods:
        if label == "BFS":
            y = [float(np.mean([r["bfs"]["calls"] for r in by_size[s]])) for s in sizes]
        else:
            y = [_cost(by_size[s], picker) for s in sizes]
        ax.plot(sizes, y, style, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Available Criteria")
    ax.set_ylabel("Average LLM Calls")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    path = f"{out_dir}/cost_by_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Normalized version: cost / 2^k per case, then averaged
    def _cost_norm(bucket: list[dict], picker, s: int) -> float:
        ms = [m for r in bucket if (m := picker(r)) is not None]
        if not ms:
            return float("nan")
        return float(np.mean([(m["calls"] + m.get("budget_used", 0)) / (2 ** s) for m in ms]))

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, style, picker in methods:
        if label == "BFS":
            y = [float(np.mean([r["bfs"]["calls"] / (2 ** s) for r in by_size[s]])) for s in sizes]
        else:
            y = [_cost_norm(by_size[s], picker, s) for s in sizes]
        ax.plot(sizes, y, style, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Available Criteria")
    ax.set_ylabel("Average LLM Calls / Lattice Size")
    ax.set_ylim(0, 1)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    path = f"{out_dir}/cost_by_size_normalized.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_outcomes(results: list[dict], cc_by_id: dict[int, dict] | None, out_dir: str) -> None:
    methods_data = [("SRA", lambda r: r.get("sra")), ("ERA b=16", lambda r: r.get("era"))]
    if cc_by_id is not None:
        methods_data.append(("LSM b=10%", lambda r: cc_by_id.get(r["case_id"])))

    outcome_order = ["correct", "wrong_layer", "false_positive", "not_found"]
    outcome_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#aec7e8"]

    counts: dict[str, dict[str, int]] = {}
    for label, picker in methods_data:
        c: dict[str, int] = defaultdict(int)
        for r in results:
            m = picker(r)
            if m:
                c[m["outcome"]] += 1
        counts[label] = dict(c)

    x = np.arange(len(methods_data))
    bottoms = np.zeros(len(methods_data))

    fig, ax = plt.subplots(figsize=(7, 5))
    for outcome, color in zip(outcome_order, outcome_colors):
        vals = [counts[label].get(outcome, 0) for label, _ in methods_data]
        ax.bar(x, vals, bottom=bottoms, color=color, label=outcome)
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in methods_data])
    ax.set_ylabel("Cases")
    ax.legend(ncol=len(outcome_order), loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Outcome breakdown — sonnet test sub-lattices", fontweight="bold")
    fig.tight_layout()
    path = f"{out_dir}/outcomes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    d = _load()
    results = d["results"]
    cc_by_id = _load_cc()
    if cc_by_id is not None:
        print(f"Loaded {len(cc_by_id)} CC results from {CC_RESULTS_PATH}")
    else:
        print(f"No CC results found at {CC_RESULTS_PATH} — plotting BFS/SRA/ERA only")

    plot_accuracy(results, cc_by_id, OUT_DIR)
    plot_cost(results, cc_by_id, OUT_DIR)
    plot_outcomes(results, cc_by_id, OUT_DIR)


if __name__ == "__main__":
    main()
