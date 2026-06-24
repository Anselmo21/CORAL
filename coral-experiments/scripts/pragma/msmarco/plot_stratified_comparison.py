"""Compare stratified vs original msmarco-pragma results.

Reads:
  data/msmarco_pragma/results.json                  (original)
  data/msmarco_pragma_stratified/results.json       (stratified)
  data/msmarco_pragma_stratified/sampling_diag.json (per-case plurality_share)

Writes plots/msmarco_pragma_stratified/:
  accuracy_compare.png         — BFS / SRA / ERA(pct_10) accuracy by k, original vs stratified
  cost_compare.png             — avg LLM calls by k, original vs stratified
  plurality_share_compare.png  — distribution of plurality_share across cases
  summary.txt                  — printed per-size delta table
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from scripts.pragma.msmarco.run_subsample import _load_all_qids, _load_lattices
from scripts.pragma.msmarco.run_stratified import _class_map, _plurality_share

ORIG_RESULTS  = "data/msmarco_pragma/results.json"
STRAT_RESULTS = "data/msmarco_pragma_stratified/results.json"
STRAT_DIAG    = "data/msmarco_pragma_stratified/sampling_diag.json"
ORIG_CASES    = "data/msmarco_pragma/cases.json"

OUT_DIR = "plots/msmarco_pragma_stratified"
SUMMARY_PATH = f"{OUT_DIR}/summary.txt"

ERA_PCT = 10


def _load(path: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d["results"] if isinstance(d, dict) else d


def _by_size(results: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        out[r["size"]].append(r)
    return out


def _era(r: dict) -> dict:
    return r["era"].get(f"era_pct_{ERA_PCT}", {})


def _acc(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r))]
    if not ms:
        return float("nan")
    return 100.0 * sum(1 for m in ms if m.get("outcome") == "correct") / len(ms)


def _cost(bucket: list[dict], picker) -> float:
    ms = [m for r in bucket if (m := picker(r))]
    if not ms:
        return float("nan")
    return float(np.mean([m["calls"] + m.get("budget_used", 0) for m in ms]))


def _bfs_cost(bucket: list[dict]) -> float:
    return float(np.mean([r["bfs"]["calls"] for r in bucket]))


def plot_accuracy(orig: list[dict], strat: list[dict]) -> None:
    o, s = _by_size(orig), _by_size(strat)
    sizes = sorted(set(o) | set(s))

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, by, ls, alpha in [("original", o, "--", 0.55), ("stratified", s, "-", 1.0)]:
        sra = [_acc(by[k], lambda r: r.get("sra")) for k in sizes]
        era = [_acc(by[k], _era) for k in sizes]
        ax.plot(sizes, sra, f"s{ls}", color="#d62728", linewidth=2, markersize=7,
                alpha=alpha, label=f"SRA ({label})")
        ax.plot(sizes, era, f"D{ls}", color="#1f77b4", linewidth=2, markersize=7,
                alpha=alpha, label=f"ERA pct_10 ({label})")
    ax.plot(sizes, [100.0] * len(sizes), "o:", color="black", linewidth=2,
            markersize=7, label="BFS (oracle)")

    ax.set_xlabel("Number of Available Pragmas")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_title("MSMARCO-pragma: stratified vs original (n≈20 cases / k)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/accuracy_compare.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/accuracy_compare.png")


def plot_cost(orig: list[dict], strat: list[dict]) -> None:
    o, s = _by_size(orig), _by_size(strat)
    sizes = sorted(set(o) | set(s))

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, by, ls, alpha in [("original", o, "--", 0.55), ("stratified", s, "-", 1.0)]:
        bfs_c = [_bfs_cost(by[k]) for k in sizes]
        sra_c = [_cost(by[k], lambda r: r.get("sra")) for k in sizes]
        era_c = [_cost(by[k], _era) for k in sizes]
        ax.plot(sizes, bfs_c, f"o{ls}", color="black", linewidth=2, markersize=7,
                alpha=alpha, label=f"BFS ({label})")
        ax.plot(sizes, sra_c, f"s{ls}", color="#d62728", linewidth=2, markersize=7,
                alpha=alpha, label=f"SRA ({label})")
        ax.plot(sizes, era_c, f"D{ls}", color="#1f77b4", linewidth=2, markersize=7,
                alpha=alpha, label=f"ERA pct_10 ({label})")

    ax.set_xlabel("Number of Available Pragmas")
    ax.set_ylabel("Avg LLM Calls (search + budget)")
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("MSMARCO-pragma: cost comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cost_compare.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/cost_compare.png")


def plot_plurality(orig_cases: list[dict], strat_diag: list[dict],
                   lattices: dict[str, tuple[list[str], dict[str, list[str]]]]) -> None:
    """Recompute plurality_share for original cases (not stored), use stored values for stratified."""
    qid_class_maps = {qid: _class_map(crit, rbc) for qid, (crit, rbc) in lattices.items()}

    orig_shares: list[float] = []
    for c in orig_cases:
        if c["size"] == 10:
            continue   # skip full-lattice cases (identical in both regimes)
        cm = qid_class_maps[c["qid"]]
        orig_shares.append(_plurality_share(c["start_node"], cm))

    strat_shares = [d["plurality_share"] for d in strat_diag if d["size"] != 10]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(orig_shares, bins=bins, alpha=0.55, color="#d62728",
            label=f"original (n={len(orig_shares)})", edgecolor="black", linewidth=0.4)
    ax.hist(strat_shares, bins=bins, alpha=0.7, color="#1f77b4",
            label=f"stratified (n={len(strat_shares)})", edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6, label="balanced (0.5)")
    ax.set_xlabel("Plurality-class share within sub-lattice")
    ax.set_ylabel("Number of sampled sub-lattice cases")
    ax.set_title("Stratification effect on within-case voter balance (k∈3..9)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/plurality_share_compare.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/plurality_share_compare.png")

    print(f"\n  Plurality share (k∈3..9, full-lattice cases excluded):")
    print(f"    original:   median={np.median(orig_shares):.3f}  mean={np.mean(orig_shares):.3f}")
    print(f"    stratified: median={np.median(strat_shares):.3f}  mean={np.mean(strat_shares):.3f}")


def print_summary(orig: list[dict], strat: list[dict]) -> str:
    o, s = _by_size(orig), _by_size(strat)
    sizes = sorted(set(o) | set(s))

    lines: list[str] = []
    lines.append(f"\n  MSMARCO-pragma: stratified vs original")
    lines.append(f"  {'k':>3} {'n_o':>4} {'n_s':>4} | "
                 f"{'SRA_o':>6} {'SRA_s':>6} {'ΔSRA':>5} | "
                 f"{'ERA_o':>6} {'ERA_s':>6} {'ΔERA':>5} | "
                 f"{'BFS$_o':>7} {'BFS$_s':>7} | "
                 f"{'ERA$_o':>7} {'ERA$_s':>7} {'Δ$':>6}")
    for k in sizes:
        bo, bs = o[k], s[k]
        sra_o = _acc(bo, lambda r: r.get("sra"))
        sra_s = _acc(bs, lambda r: r.get("sra"))
        era_o = _acc(bo, _era)
        era_s = _acc(bs, _era)
        bfs_o = _bfs_cost(bo) if bo else float("nan")
        bfs_s = _bfs_cost(bs) if bs else float("nan")
        era_co = _cost(bo, _era)
        era_cs = _cost(bs, _era)
        lines.append(
            f"  {k:>3} {len(bo):>4} {len(bs):>4} | "
            f"{sra_o:>5.1f} {sra_s:>5.1f}  {sra_s-sra_o:>+5.1f} | "
            f"{era_o:>5.1f} {era_s:>5.1f}  {era_s-era_o:>+5.1f} | "
            f"{bfs_o:>7.1f} {bfs_s:>7.1f} | "
            f"{era_co:>7.1f} {era_cs:>7.1f} {era_cs-era_co:>+6.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    orig = _load(ORIG_RESULTS)
    strat = _load(STRAT_RESULTS)
    with open(STRAT_DIAG) as f:
        strat_diag = json.load(f)
    with open(ORIG_CASES) as f:
        orig_cases = json.load(f)
    print(f"Loaded {len(orig)} original results, {len(strat)} stratified results")

    plot_accuracy(orig, strat)
    plot_cost(orig, strat)

    lats = _load_lattices(_load_all_qids())
    plot_plurality(orig_cases, strat_diag, lats)

    summary = print_summary(orig, strat)
    print(summary)
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
