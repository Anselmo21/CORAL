"""Extend compare_mra_borda_impls.py: sweep pyflagr Borda + CombSUM + CombMNZ
against the built-in MRA aggregator.

Focused run on Llama-3.1-8B + msmarco_criteria only. Uses the same cases from
logged_results/msmarco_criteria/cases.json and the same score_proxy pipeline
so results are directly comparable to the existing SUMMARY.md.

Usage:
    conda run -n coral python3 -m scripts.shared.compare_mra_flagr_comb [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import search.rank as sr
from scripts.criteria.msmarco.run_subsample import score_proxy
from search.bfs import bfs_search
from search.flagr import make_era_aggregator
from search.lattice import combo_key, hop_distance
from search.rank import mra_search

BUDGETS = (4, 8, 16)
NORMALIZE_FN = "softmax"
METHODS = ["borda", "combsum", "combmnz"]
DATA_PATH = Path("data/llama31-8b/msmarco-criteria-lattices.json")
CASES_PATH = Path("logged_results/msmarco_criteria/cases.json")
OUT_DIR = Path("logged_results/mra_flagr_comb_compare")


def _load_dataset(path: Path):
    raw = json.loads(path.read_text())
    out = {}
    for qid, lat in raw.items():
        if qid == "_meta" or lat.get("halted"):
            continue
        crit = list(lat.get("criteria") or [])
        rbc = lat.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


class _AggregatorSwap:
    def __init__(self, replacement):
        self._new = replacement
        self._orig = None

    def __enter__(self):
        self._orig = sr._mra_aggregate_borda
        sr._mra_aggregate_borda = self._new
        return self

    def __exit__(self, *exc):
        sr._mra_aggregate_borda = self._orig


def _run(criteria, rbc, start_node, budget, replacement):
    if replacement is None:
        return mra_search(criteria, rbc, budget=budget, start_node=start_node,
                          direction="down", normalize_fn=NORMALIZE_FN)
    with _AggregatorSwap(replacement):
        return mra_search(criteria, rbc, budget=budget, start_node=start_node,
                          direction="down", normalize_fn=NORMALIZE_FN)


def _score(res, bfs, rbc, start_node):
    bfs_for_proxy = {"key": bfs["key"], "size": bfs["size"], "found": bfs["found"]}
    scored = score_proxy(bfs_for_proxy, res, rbc, start_node)
    scored["hop_to_bfs"] = hop_distance(scored.get("proxy_key"), bfs["key"], start_node)
    return scored


def _pool(rows, col):
    cells = [r[col] for r in rows if col in r]
    n = len(cells)
    if n == 0:
        return None
    acc = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / n
    cost = float(np.mean([c["calls"] + c.get("budget_used", 0) for c in cells]))
    miss_hops = [c["hop_to_bfs"] for c in cells
                 if c.get("outcome") != "correct" and c.get("hop_to_bfs") is not None]
    miss_hop = float(np.mean(miss_hops)) if miss_hops else float("nan")
    return {"acc": acc, "cost": cost, "hop_miss": miss_hop, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on cases (evenly spaced) for a fast pilot run.")
    args = ap.parse_args()

    lattices = _load_dataset(DATA_PATH)
    cases = json.loads(CASES_PATH.read_text())
    if args.limit and args.limit < len(cases):
        step = len(cases) // args.limit
        cases = cases[::step][:args.limit]
    print(f"loaded {len(cases)} cases, {len(lattices)} lattices", flush=True)

    # Build replacements once.
    replacements = {m: make_era_aggregator(m) for m in METHODS}

    rows = []
    t0 = time.time()
    for i, case in enumerate(cases):
        qid = str(case["qid"])
        if qid not in lattices:
            continue
        crit, rbc = lattices[qid]
        start = case["start_node"]
        bfs = bfs_search(crit, rbc, start_node=start, direction="down")
        row = {"case_id": i, "qid": qid, "size": case["size"], "start_node": start,
               "bfs": {"key": bfs["key"], "size": bfs["size"], "found": bfs["found"],
                       "calls": bfs["calls"]}}
        for b in BUDGETS:
            row[f"builtin_b{b}"] = _score(_run(crit, rbc, start, b, None), bfs, rbc, start)
            for m in METHODS:
                res = _run(crit, rbc, start, b, replacements[m])
                row[f"{m}_b{b}"] = _score(res, bfs, rbc, start)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:>3}/{len(cases)}] {time.time()-t0:.0f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "msmarco_criteria_llama31_8b.json").write_text(json.dumps(rows, indent=2))

    lines = ["# MRA: built-in vs pyflagr comb methods (Llama-3.1-8B, msmarco_criteria)\n",
             f"- normalize_fn=\"{NORMALIZE_FN}\", n={len(rows)}",
             "- pyflagr methods collapse each estimated child's score vector to a "
             "ranking, run the classical aggregator, and re-normalize.\n",
             "| Budget | Impl | Acc | Cost | Hop-miss μ |",
             "|---:|---|---:|---:|---:|"]
    for b in BUDGETS:
        for col, label in [(f"builtin_b{b}", "built-in"),
                            (f"borda_b{b}",   "pyflagr borda"),
                            (f"combsum_b{b}", "pyflagr combsum"),
                            (f"combmnz_b{b}", "pyflagr combmnz")]:
            s = _pool(rows, col)
            if s is None:
                continue
            lines.append(f"| {b} | {label} | {s['acc']:.1f}% | {s['cost']:.1f} | {s['hop_miss']:.2f} |")
        lines.append("")

    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\n→ {OUT_DIR/'SUMMARY.md'}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
