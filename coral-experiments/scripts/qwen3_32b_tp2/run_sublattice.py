"""Sub-lattice sweep on coral-runs qwen3-32b data with score-aware vs rank-only voters.

Loads the coral-runs per-qid format via scripts.qwen3_32b_tp2.loader, samples
cases with the shared sample_cases recipe (sizes 3..9 + full k=10), then runs:

  BFS (ground truth)
  SRA (rank voters) vs SRA (model score voters)
  ERA (rank voters) vs ERA (model score voters)  — budgets 4, 8, 16
  MRA (rank voters) vs MRA (model score voters)  — budgets 4, 8, 16

Datasets (under coral-runs-lattices/qwen3-32b/):
  curated-criteria, curated-pragmas, msmarco-criteria, msmarco-pragmas

Invoke from repo root:
  conda run -n coral python3 -m scripts.qwen3_32b_tp2.run_sublattice --dataset curated-criteria
  conda run -n coral python3 -m scripts.qwen3_32b_tp2.run_sublattice --all
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from scripts.qwen3_32b_tp2.loader import load_all
from scripts.shared.sampling import SEED, SIZES, TARGET_PER_SIZE, sample_cases
from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, mra_search, sra_search

CORAL_RUNS_DIR = Path("coral-runs-lattices/qwen3-32b")
ALL_DATASETS = ["curated-criteria", "curated-pragmas", "msmarco-criteria", "msmarco-pragmas"]
RESULTS_BASE = "logged_results/qwen3-32b-tp2"

BUDGETS = [4, 8, 16]


def _out_dir(dataset: str) -> str:
    slug = dataset.replace("-", "_")
    return f"{RESULTS_BASE}/{slug}"


def _score_proxy(bfs_res, proxy_res, rbc, start_node) -> dict:
    start_key = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    bfs_key, bfs_layer = bfs_res["key"], bfs_res["size"]
    proxy_key = proxy_res.get("key")
    proxy_layer = proxy_res.get("size")
    if proxy_key is not None:
        r = rbc.get(proxy_key, [])
        verified = bool(r) and r[0] != baseline_top1
    else:
        verified = False
    if not proxy_res.get("found") or proxy_key is None:
        outcome = "not_found"
    elif not verified:
        outcome = "false_positive"
    elif proxy_layer != bfs_layer:
        outcome = "wrong_layer"
    else:
        outcome = "correct"
    gap = 0 if outcome == "correct" else (
        abs(bfs_layer - proxy_layer) if outcome == "wrong_layer" and bfs_layer and proxy_layer else None
    )
    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": verified,
        "outcome": outcome,
        "gap": gap,
        "jaccard": jaccard_keys(bfs_key, proxy_key),
        "calls": proxy_res.get("calls", 0),
        "budget_used": proxy_res.get("budget_used", 0),
        "pruned_nodes": proxy_res.get("pruned_nodes", 0),
    }


def _run_case(case_id: int, case: dict, lat: dict) -> dict:
    crit = list(lat["atoms"])
    rbc = lat["rankings_by_combo"]
    sbc = lat["scores_by_combo"]
    pa = lat["plausible_answers"]
    start = case["start_node"]

    bfs = bfs_search(crit, rbc, start_node=start, direction="down")
    bfs_for_proxy = {"key": bfs["key"], "size": bfs["size"]}

    out = {
        "case_id": case_id,
        "qid": case["qid"],
        "size": case["size"],
        "start_node": start,
        "bfs": {"found": bfs["found"], "key": bfs["key"], "layer": bfs["size"], "calls": bfs["calls"]},
    }

    modes = (
        ("rank",       None, "max"),
        ("scores",     sbc,  "max"),
        ("scores_raw", sbc,  "none"),
    )
    for mode, sbc_arg, vnorm in modes:
        out[f"sra_{mode}"] = _score_proxy(
            bfs_for_proxy,
            sra_search(crit, rbc, start_node=start, direction="down",
                       scores_by_combo=sbc_arg, voter_norm=vnorm),
            rbc, start,
        )
        out[f"era_{mode}"] = {}
        out[f"mra_{mode}"] = {}
        for b in BUDGETS:
            out[f"era_{mode}"][f"b{b}"] = _score_proxy(
                bfs_for_proxy,
                era_search(crit, rbc, budget=b, start_node=start, direction="down",
                           propagate_scores=True, scores_by_combo=sbc_arg, voter_norm=vnorm),
                rbc, start,
            )
            out[f"mra_{mode}"][f"b{b}"] = _score_proxy(
                bfs_for_proxy,
                mra_search(crit, rbc, plausible_answers=pa, start_node=start, budget=b,
                           normalize_fn="softmax", scores_by_combo=sbc_arg, voter_norm=vnorm),
                rbc, start,
            )
    return out


def _summary(results: list[dict], dataset: str) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    def fmt(bucket, pick):
        ms = [pick(r) for r in bucket if pick(r) is not None]
        if not ms:
            return f"{'—':>12} | {'—':>9}"
        c = sum(1 for m in ms if m["outcome"] == "correct")
        cost = sum(m["calls"] + m.get("budget_used", 0) for m in ms) / len(ms)
        return f"{c/len(ms):>4.0%} ({c:>2}/{len(ms):<2}) | {cost:>9.1f}"

    print(f"\n{'=' * 90}")
    print(f"  {dataset} — {len(results)} cases across sizes {sorted(by_size)}")
    print(f"{'=' * 90}")
    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        bfs_avg = sum(r["bfs"]["calls"] for r in bucket) / n
        print(f"\n  --- size={size} (n={n}) --- BFS avg calls = {bfs_avg:.1f}")
        print(f"    {'Method':<28} | {'Acc':>12} | {'Avg cost':>9}")
        print(f"    {'-'*28}-+-{'-'*12}-+-{'-'*9}")
        modes = ("rank", "scores", "scores_raw")
        for mode in modes:
            print(f"    {f'SRA ({mode})':<28} | {fmt(bucket, lambda r, m=mode: r.get(f'sra_{m}'))}")
        for kind in ("era", "mra"):
            for mode in modes:
                for b in BUDGETS:
                    label = f"{kind.upper()} b={b} ({mode})"
                    print(f"    {label:<28} | "
                          f"{fmt(bucket, lambda r, k=kind, m=mode, bb=b: r.get(f'{k}_{m}', {}).get(f'b{bb}'))}")


def _run_dataset(dataset: str, resample: bool) -> None:
    data_dir = CORAL_RUNS_DIR / dataset
    if not data_dir.is_dir():
        print(f"SKIP {dataset}: {data_dir} not found")
        return

    out_dir = _out_dir(dataset)
    cases_path = f"{out_dir}/cases.json"
    results_path = f"{out_dir}/results.json"
    os.makedirs(out_dir, exist_ok=True)

    lats = load_all(data_dir=data_dir)
    print(f"[{dataset}] Loaded {len(lats)} qids from {data_dir}")

    lat_pairs = {qid: (list(l["atoms"]), l["rankings_by_combo"]) for qid, l in lats.items()}

    if resample or not os.path.exists(cases_path):
        cases = sample_cases(lat_pairs)
        with open(cases_path, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Sampled {len(cases)} cases → {cases_path}")
    else:
        with open(cases_path) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cached cases from {cases_path}")

    results = []
    for i, case in enumerate(cases):
        r = _run_case(i, case, lats[case["qid"]])
        results.append(r)
        if (i + 1) % 10 == 0 or i == len(cases) - 1:
            print(f"  [{i+1:>3}/{len(cases)}] qid={case['qid']} size={case['size']} bfs_layer={r['bfs']['layer']}",
                  flush=True)

    with open(results_path, "w") as f:
        json.dump({
            "source_dir": str(data_dir),
            "dataset": dataset,
            "seed": SEED,
            "sizes": sorted(set(SIZES + [10])),
            "target_per_size": TARGET_PER_SIZE,
            "budgets": BUDGETS,
            "n_cases": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\nResults → {results_path}")

    _summary(results, dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", choices=ALL_DATASETS,
                        help="one or more datasets to run")
    parser.add_argument("--all", action="store_true",
                        help="run all four datasets")
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()

    if args.all:
        datasets = ALL_DATASETS
    elif args.dataset:
        datasets = args.dataset
    else:
        parser.error("specify --dataset <name> [<name> ...] or --all")

    for ds in datasets:
        print(f"\n{'#' * 90}")
        print(f"# Dataset: {ds}")
        print(f"{'#' * 90}\n")
        _run_dataset(ds, args.resample)


if __name__ == "__main__":
    main()
