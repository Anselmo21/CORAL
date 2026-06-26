"""Sub-lattice ERA budget ablation sweep on the Sonnet pragma dataset.

Iterates every record in data/sonnet_pragma.json (no split). For each
effective size k in 3..9 samples 20 sub-lattices that:
  (a) have a BFS-found CF, AND
  (b) the CF requires removal of ≥2 pragmas (BFS CF layer ≤ k-2)
Also appends every full k=10 lattice as a size-10 case.

Runs BFS / SRA / ERA(fixed-budget grid) / ERA(percent-of-lattice grid) on each
case with incremental per-case checkpointing.

Outputs:
  logged_results/sonnet_pragma/cases.json                — sampled cases (full set)
  logged_results/sonnet_pragma/per_case/case_<i>.json    — per-case results (resumable)
  logged_results/sonnet_pragma/results.json              — aggregate results

CLI:
  conda run -n coral python3 -m scripts.pragma.sonnet.run_subsample
  conda run -n coral python3 -m scripts.pragma.sonnet.run_subsample --resample
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from scripts.shared.sampling import SEED, SIZES, TARGET_PER_SIZE, sample_cases
from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, sra_search

DATASET_PATH = "data/sonnet_pragma.json"
OUT_DIR      = "logged_results/sonnet_pragma"
CASES_PATH   = f"{OUT_DIR}/cases.json"
RESULTS_PATH = f"{OUT_DIR}/results.json"

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT:  list[int]  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def _load_all_qids() -> list[str]:
    """Return every non-halted qid in the dataset, sorted for determinism."""
    data = _load_dataset()
    qids = [str(r["id"]) for r in data if not r.get("halted")]
    return sorted(qids, key=lambda q: int(q) if q.isdigit() else q)


def _load_pragma_lattices(
    qids: set[str] | list[str] | None = None,
) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    """Return {qid: (criteria, rankings_by_combo)}. If qids is None, load all."""
    data = _load_dataset()
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    qid_set = set(qids) if qids is not None else None
    for rec in data:
        if rec.get("halted"):
            continue
        qid = str(rec["id"])
        if qid_set is not None and qid not in qid_set:
            continue
        cfg = rec["lattice"]["config"]
        criteria: list[str] = cfg["criteria"]
        rbc: dict[str, list[str]] = rec["lattice"]["answers"]
        if combo_key(criteria) not in rbc:
            print(f"  Skipping qid={qid}: start_key missing from answers")
            continue
        out[qid] = (criteria, rbc)
    return out


# ---------------------------------------------------------------------------
# Sub-lattice restriction
# ---------------------------------------------------------------------------

def _restrict_rbc(
    rbc: dict[str, list[str]],
    start_node: list[str],
) -> dict[str, list[str]]:
    """Return only the rankings_by_combo entries that are subsets of start_node."""
    pragma_set = set(start_node)
    out: dict[str, list[str]] = {}
    for key, ranking in rbc.items():
        parts = [p for p in key.split(", ") if p] if key else []
        if all(p in pragma_set for p in parts):
            out[key] = ranking
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_proxy(
    bfs_result: dict,
    proxy_result: dict,
    rbc: dict[str, list[str]],
    start_node: list[str],
) -> dict:
    start_key = combo_key(start_node)
    baseline_top1 = (rbc.get(start_key) or [""])[0]
    bfs_key = bfs_result["key"]
    bfs_layer = bfs_result["size"]
    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")
    if proxy_key:
        ranking = rbc.get(proxy_key, [])
        proxy_verified = bool(ranking) and ranking[0] != baseline_top1
    else:
        proxy_verified = False
    if not proxy_result.get("found") or proxy_key is None:
        outcome = "not_found"
    elif not proxy_verified:
        outcome = "false_positive"
    elif proxy_layer != bfs_layer:
        outcome = "wrong_layer"
    else:
        outcome = "correct"
    gap: int | None = None
    if outcome == "correct":
        gap = 0
    elif outcome == "wrong_layer" and bfs_layer is not None and proxy_layer is not None:
        gap = abs(bfs_layer - proxy_layer)
    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": proxy_verified,
        "outcome": outcome,
        "gap": gap,
        "jaccard": jaccard_keys(bfs_key, proxy_key),
        "calls": proxy_result.get("calls", 0),
        "budget_used": proxy_result.get("budget_used", 0),
        "pruned_nodes": proxy_result.get("pruned_nodes", 0),
    }


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def _per_case_path(case_id: int) -> str:
    return f"{OUT_DIR}/per_case/case_{case_id:04d}.json"


def _checkpoint(case_id: int, partial: dict) -> None:
    path = _per_case_path(case_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(partial, f, indent=2)
    os.replace(tmp, path)


def _load_partial(case_id: int) -> dict | None:
    path = _per_case_path(case_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_one_case(
    case_id: int,
    case: dict,
    criteria: list[str],
    rbc: dict[str, list[str]],
) -> dict:
    start_node = case["start_node"]
    size = case["size"]

    partial = _load_partial(case_id) or {
        "case_id": case_id,
        "qid": case["qid"],
        "size": size,
        "start_node": start_node,
    }

    if "bfs" not in partial:
        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        partial["bfs"] = {
            "found": bfs_res["found"],
            "key": bfs_res["key"],
            "layer": bfs_res["size"],
            "calls": bfs_res["calls"],
        }
        _checkpoint(case_id, partial)

    bfs_for_proxy = {"key": partial["bfs"]["key"], "size": partial["bfs"]["layer"]}

    if "sra" not in partial:
        sra_res = sra_search(criteria, rbc, start_node=start_node, direction="down")
        partial["sra"] = score_proxy(bfs_for_proxy, sra_res, rbc, start_node)
        _checkpoint(case_id, partial)

    partial.setdefault("era", {})

    for b in ERA_BUDGETS_FIXED:
        label = f"era_fixed_{b}"
        if label in partial["era"]:
            continue
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down",
                         propagate_scores=True)
        partial["era"][label] = score_proxy(bfs_for_proxy, res, rbc, start_node)
        _checkpoint(case_id, partial)

    for p in ERA_BUDGETS_PCT:
        label = f"era_pct_{p}"
        if label in partial["era"]:
            continue
        budget = max(1, int(2**size * p / 100))
        res = era_search(criteria, rbc, budget=budget, start_node=start_node, direction="down",
                         propagate_scores=True)
        partial["era"][label] = score_proxy(bfs_for_proxy, res, rbc, start_node)
        _checkpoint(case_id, partial)

    _checkpoint(case_id, partial)
    return partial


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    era_labels = [f"era_fixed_{b}" for b in ERA_BUDGETS_FIXED[:4]]

    print(f"\n{'='*60}")
    print(f"  SONNET pragma sub-lattice sweep — {len(results)} cases (full set)")
    print(f"{'='*60}")
    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        bfs_calls = sum(r["bfs"]["calls"] for r in bucket) / n
        sra_acc = 100 * sum(1 for r in bucket if r["sra"]["outcome"] == "correct") / n
        print(f"\n  k={size}  n={n}  BFS avg_calls={bfs_calls:.0f}  SRA acc={sra_acc:.0f}%")
        for label in era_labels:
            acc = 100 * sum(1 for r in bucket if r["era"][label]["outcome"] == "correct") / n
            print(f"    {label}: {acc:.0f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", action="store_true",
                        help="Re-roll sampled cases (clears per-case cache)")
    args = parser.parse_args()

    qids = _load_all_qids()
    print(f"Qids ({len(qids)}): {qids}")
    lattices = _load_pragma_lattices(qids)
    print(f"Loaded {len(lattices)} pragma lattices")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.resample or not os.path.exists(CASES_PATH):
        print("\nSampling cases ...")
        cases = sample_cases(lattices, seed=SEED)
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Saved {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} existing cases from {CASES_PATH}")

    print(f"\nRunning BFS/SRA/ERA on {len(cases)} cases ...")
    results: list[dict] = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        criteria, rbc = lattices[qid]
        print(f"  [{i+1:>3}/{len(cases)}] qid={qid} k={case['size']} ...", end=" ", flush=True)
        result = run_one_case(i, case, criteria, rbc)
        results.append(result)
        bfs_layer = result["bfs"]["layer"]
        sra_ok = "OK" if result["sra"]["outcome"] == "correct" else result["sra"]["outcome"]
        print(f"BFS_layer={bfs_layer} SRA={sra_ok}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")
    print_summary(results)


if __name__ == "__main__":
    main()
