"""Compare MRA using our built-in Borda aggregator vs pyflagr's Borda.

For each (model, dataset), reuses the existing sampled cases from
logged_results/<...>/cases.json and re-runs mra_search twice per case per
budget: once with the built-in _mra_aggregate_borda, once with pyflagr's
Linear.BordaCount plugged in via search.flagr.make_era_aggregator.

Normalization is fixed to normalize_fn="softmax" to match the canonical MRA
config used by plot_qwen3_temp0_bfs_lsm.py. Both runs are scored with the same
score_proxy from scripts.criteria.msmarco.run_subsample so accuracy/cost/hop
numbers are directly comparable.

Writes per-case rows to
    logged_results/mra_borda_compare/<model>/<dataset>.json
and a pooled summary table to
    logged_results/mra_borda_compare/SUMMARY.md

Usage:
    conda run -n coral python3 -m scripts.shared.compare_mra_borda_impls
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
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
OUT_ROOT = Path("logged_results/mra_borda_compare")


# ---------------------------------------------------------------------------
# Model / dataset registry
# ---------------------------------------------------------------------------

def _load_llama31_8b_lattices():
    root = Path("data/llama31-8b")
    files = {
        "msmarco_criteria": "msmarco-criteria-lattices.json",
        "msmarco_pragma":   "msmarco-pragmas-lattices.json",
        "curated_criteria":  "curated-criteria-lattices.json",
        "curated_pragma":    "curated-pragmas-lattices.json",
    }
    cases_root = Path("logged_results")
    return {ds: {"data": root / fn, "cases": cases_root / ds / "cases.json"}
            for ds, fn in files.items()}


def _load_llama31_70b_lattices():
    root = Path("data/llama31-70b")
    files = {
        "msmarco_criteria": "msmarco-criteria-llama31-70b.json",
        "msmarco_pragma":   "msmarco-pragmas-llama31-70b.json",
        "curated_criteria":  "curated-criteria-llama31-70b.json",
        "curated_pragma":    "curated-pragmas-llama31-70b.json",
    }
    cases_root = Path("logged_results/llama31-70b")
    return {ds: {"data": root / fn, "cases": cases_root / ds / "cases.json"}
            for ds, fn in files.items()}


def _load_qwen3_32b_lattices():
    root = Path(os.path.expanduser("~/coral-data-gen/datasets/qwen3-32b-temp0"))
    files = {
        "msmarco_criteria": "msmarco-criteria-qwen3-32b-temp0.json",
        "msmarco_pragma":   "msmarco-pragmas-qwen3-32b-temp0.json",
        "curated_criteria":  "curated-criteria-qwen3-32b-temp0.json",
        "curated_pragma":    "curated-pragmas-qwen3-32b-temp0.json",
    }
    cases_root = Path("logged_results/qwen3-32b-temp0")
    return {ds: {"data": root / fn, "cases": cases_root / ds / "cases.json"}
            for ds, fn in files.items()}


MODELS = [
    ("llama31-8b",  _load_llama31_8b_lattices),
    ("llama31-70b", _load_llama31_70b_lattices),
    ("qwen3-32b",   _load_qwen3_32b_lattices),
]


# ---------------------------------------------------------------------------
# Lattice loading (flat qid → raw_answers)
# ---------------------------------------------------------------------------

def _load_dataset(data_path: Path) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    raw = json.loads(data_path.read_text())
    out = {}
    for qid, lat in raw.items():
        if qid == "_meta" or lat.get("halted"):
            continue
        crit = list(lat.get("criteria") or [])
        rbc = lat.get("raw_answers") or {}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


# ---------------------------------------------------------------------------
# Aggregator swap
# ---------------------------------------------------------------------------

_pyflagr_agg = make_era_aggregator("borda")


def _pyflagr_mra_aggregate(voter_scores, answers):
    return _pyflagr_agg(voter_scores, answers)


class _AggregatorSwap:
    """Temporarily replace search.rank._mra_aggregate_borda."""

    def __init__(self, replacement):
        self._new = replacement
        self._orig = None

    def __enter__(self):
        self._orig = sr._mra_aggregate_borda
        sr._mra_aggregate_borda = self._new
        return self

    def __exit__(self, *exc):
        sr._mra_aggregate_borda = self._orig


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def _run_one(criteria, rbc, start_node, budget, use_pyflagr):
    if use_pyflagr:
        with _AggregatorSwap(_pyflagr_mra_aggregate):
            return mra_search(criteria, rbc, budget=budget, start_node=start_node,
                              direction="down", normalize_fn=NORMALIZE_FN)
    return mra_search(criteria, rbc, budget=budget, start_node=start_node,
                      direction="down", normalize_fn=NORMALIZE_FN)


def _score(mra_res, bfs_res, rbc, start_node):
    bfs_for_proxy = {"key": bfs_res["key"], "size": bfs_res["size"], "found": bfs_res["found"]}
    scored = score_proxy(bfs_for_proxy, mra_res, rbc, start_node)
    scored["hop_to_bfs"] = hop_distance(scored.get("proxy_key"), bfs_res["key"], start_node)
    return scored


def _run_dataset(model_name, ds_name, data_path, cases_path):
    print(f"\n[{model_name} / {ds_name}]")
    if not data_path.exists() or not cases_path.exists():
        print(f"  skip: missing {data_path if not data_path.exists() else cases_path}")
        return None
    lattices = _load_dataset(data_path)
    cases = json.loads(cases_path.read_text())
    print(f"  {len(cases)} cases, {len(lattices)} lattices in data")

    rows = []
    t0 = time.time()
    for i, case in enumerate(cases):
        qid = case["qid"]
        if qid not in lattices:
            continue
        crit, rbc = lattices[qid]
        start = case["start_node"]
        bfs = bfs_search(crit, rbc, start_node=start, direction="down")
        row = {"case_id": i, "qid": qid, "size": case["size"], "start_node": start,
               "bfs": {"key": bfs["key"], "size": bfs["size"], "found": bfs["found"],
                       "calls": bfs["calls"]}}
        for b in BUDGETS:
            builtin = _run_one(crit, rbc, start, b, use_pyflagr=False)
            pyflagr = _run_one(crit, rbc, start, b, use_pyflagr=True)
            row[f"builtin_b{b}"] = _score(builtin, bfs, rbc, start)
            row[f"pyflagr_b{b}"] = _score(pyflagr, bfs, rbc, start)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"    [{i+1:>3}/{len(cases)}] elapsed={time.time()-t0:.0f}s", flush=True)

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds_name}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"  → {out_path}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    return rows


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def _pool_stats(rows, col):
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


def _pool_agreement(rows, b):
    """How often do builtin and pyflagr end at the same CF node?"""
    a_col, b_col = f"builtin_b{b}", f"pyflagr_b{b}"
    n_same_key = n_same_layer = n = 0
    for r in rows:
        if a_col not in r or b_col not in r:
            continue
        n += 1
        ka, kb = r[a_col].get("proxy_key"), r[b_col].get("proxy_key")
        la, lb = r[a_col].get("proxy_layer"), r[b_col].get("proxy_layer")
        if ka is not None and ka == kb:
            n_same_key += 1
        if la is not None and la == lb:
            n_same_layer += 1
    if n == 0:
        return None
    return {"same_key_pct": 100.0 * n_same_key / n,
            "same_layer_pct": 100.0 * n_same_layer / n, "n": n}


def _summarize_all(all_results):
    lines = []
    lines.append("# MRA: Built-in Borda vs pyflagr Borda\n")
    lines.append(f"- Normalization: `normalize_fn=\"{NORMALIZE_FN}\"` (both runs, matches "
                 "`plot_qwen3_temp0_bfs_lsm.py`).")
    lines.append(f"- Budgets: {list(BUDGETS)}.")
    lines.append("- Cases: reused from each model's existing `cases.json` (156 per dataset when full).\n")
    lines.append("**Built-in Borda** = `search.rank._mra_aggregate_borda`: unnormalized Borda-count "
                 "voter score vectors summed entry-wise across children.\n")
    lines.append("**pyflagr Borda** = `pyflagr.Linear.BordaCount` via `search.flagr.make_era_aggregator("
                 "\"borda\")`: each child voter's score vector is collapsed to a ranking, written to a "
                 "CSV, aggregated by FLAGR's Borda, then max-normalized. Rank-only aggregator → "
                 "positional scores.\n")

    for model, per_ds in all_results.items():
        lines.append(f"\n## {model}\n")
        for ds, rows in per_ds.items():
            if not rows:
                continue
            lines.append(f"### {ds} (n={len(rows)})\n")
            lines.append("| Budget | Impl | Acc | Cost | Hop-miss μ | Same CF node | Same layer |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|")
            for b in BUDGETS:
                bs = _pool_stats(rows, f"builtin_b{b}")
                ps = _pool_stats(rows, f"pyflagr_b{b}")
                ag = _pool_agreement(rows, b)
                if bs is None or ps is None or ag is None:
                    continue
                lines.append(f"| {b} | built-in | {bs['acc']:.1f}% | {bs['cost']:.1f} | "
                             f"{bs['hop_miss']:.2f} | {ag['same_key_pct']:.1f}% | "
                             f"{ag['same_layer_pct']:.1f}% |")
                lines.append(f"| {b} | pyflagr  | {ps['acc']:.1f}% | {ps['cost']:.1f} | "
                             f"{ps['hop_miss']:.2f} |  |  |")
            lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\n→ {OUT_ROOT/'SUMMARY.md'}")


def main() -> None:
    all_results: dict[str, dict[str, list]] = {}
    for model_name, loader in MODELS:
        all_results[model_name] = {}
        for ds_name, paths in loader().items():
            rows = _run_dataset(model_name, ds_name, paths["data"], paths["cases"])
            if rows is not None:
                all_results[model_name][ds_name] = rows
    _summarize_all(all_results)


if __name__ == "__main__":
    main()
