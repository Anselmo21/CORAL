"""Sub-lattice ERA budget ablation sweep on the Sonnet pragma dataset.

Uses the existing 8/8 dev/test split from data/sonnet_pragma/split.json.
For each split, samples 20 sub-lattices per effective size k in 3..9 that:
  (a) have a BFS-found CF, AND
  (b) the CF requires removal of ≥2 pragmas (BFS CF layer ≤ k-2)

Runs BFS / SRA / ERA(fixed-budget grid) / ERA(percent-of-lattice grid) on
each dev case with incremental per-case checkpointing.

Outputs:
  data/sonnet_pragma/cases_dev.json             — sampled dev cases
  data/sonnet_pragma/cases_test.json            — sampled test cases
  data/sonnet_pragma/per_case/case_<i>.json     — per-case results (resumable)
  data/sonnet_pragma/results.json               — aggregate results

CLI:
  conda run -n coral python3 run_sonnet_pragma_subsample.py
  conda run -n coral python3 run_sonnet_pragma_subsample.py --resample
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, sra_search

DATASET_PATH = "data/sonnet-pragma-dataset.json"
SPLIT_PATH   = "data/sonnet_pragma/split.json"
OUT_DIR      = "data/sonnet_pragma"

SIZES: list[int] = list(range(3, 11))        # 3..10
TARGET_PER_SIZE: int = 20
ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT:  list[int]  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEED: int = 42
MIN_PRAGMAS_REMOVED: int = 2   # CF must require removal of ≥ this many pragmas


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pragma_lattices(
    dataset_path: str,
    qids: set[str],
) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    """Return {qid: (criteria, rankings_by_combo)} for the requested qids."""
    with open(dataset_path) as f:
        data = json.load(f)
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for rec in data:
        qid = str(rec["id"])
        if qid not in qids:
            continue
        cfg = rec["lattice"]["config"]
        criteria: list[str] = cfg["criteria"]
        rbc: dict[str, list[str]] = rec["lattice"]["answers"]
        if combo_key(criteria) not in rbc:
            print(f"  Skipping qid={qid}: start_key missing from answers")
            continue
        out[qid] = (criteria, rbc)
    return out


def _load_split() -> tuple[list[str], list[str]]:
    with open(SPLIT_PATH) as f:
        s = json.load(f)
    return s["dev"], s["test"]


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
# Case sampling
# ---------------------------------------------------------------------------

def _qualifies(
    rbc: dict[str, list[str]],
    criteria: list[str],
    start_node: list[str],
) -> bool:
    """BFS finds a CF that requires removal of ≥ MIN_PRAGMAS_REMOVED pragmas."""
    res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
    if not res["found"]:
        return False
    k = len(start_node)
    cf_layer = res["size"]  # number of pragmas in the CF node
    pragmas_removed = k - cf_layer
    return pragmas_removed >= MIN_PRAGMAS_REMOVED


def sample_cases(
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
    seed: int = SEED,
    label: str = "",
) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    for size in SIZES:
        candidates: list[dict] = []
        for qid, (criteria, _) in lattices.items():
            for combo in combinations(criteria, size):
                candidates.append({"qid": qid, "size": size, "start_node": list(combo)})
        rng.shuffle(candidates)
        accepted: list[dict] = []
        for cand in candidates:
            criteria, rbc = lattices[cand["qid"]]
            if _qualifies(rbc, criteria, cand["start_node"]):
                accepted.append(cand)
                if len(accepted) >= TARGET_PER_SIZE:
                    break
        tag = f" [{label}]" if label else ""
        print(f"  size={size}{tag}: pool={len(candidates):>5}  accepted={len(accepted):>3}/{TARGET_PER_SIZE}")
        cases.extend(accepted)
    return cases


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
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down")
        partial["era"][label] = score_proxy(bfs_for_proxy, res, rbc, start_node)
        _checkpoint(case_id, partial)

    for p in ERA_BUDGETS_PCT:
        label = f"era_pct_{p}"
        if label in partial["era"]:
            continue
        budget = max(1, int(2**size * p / 100))
        res = era_search(criteria, rbc, budget=budget, start_node=start_node, direction="down")
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

    era_labels = (
        [f"era_fixed_{b}" for b in ERA_BUDGETS_FIXED]
        + [f"era_pct_{p}" for p in ERA_BUDGETS_PCT]
    )

    print(f"\n{'='*60}")
    print(f"  Pragma sub-lattice sweep — {len(results)} cases")
    print(f"{'='*60}")
    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        bfs_calls = sum(r["bfs"]["calls"] for r in bucket) / n
        sra_acc = 100 * sum(1 for r in bucket if r["sra"]["outcome"] == "correct") / n
        print(f"\n  k={size}  n={n}  BFS avg_calls={bfs_calls:.0f}  SRA acc={sra_acc:.0f}%")
        for label in era_labels[:4]:  # just first few for brevity
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

    dev_qids, test_qids = _load_split()
    print(f"Split: {len(dev_qids)} dev, {len(test_qids)} test")

    dev_lattices  = _load_pragma_lattices(DATASET_PATH, set(dev_qids))
    test_lattices = _load_pragma_lattices(DATASET_PATH, set(test_qids))
    print(f"Loaded {len(dev_lattices)} dev lattices, {len(test_lattices)} test lattices")

    cases_dev_path  = f"{OUT_DIR}/cases_dev.json"
    cases_test_path = f"{OUT_DIR}/cases_test.json"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Sample dev cases
    if args.resample or not os.path.exists(cases_dev_path):
        print("\nSampling dev cases ...")
        cases_dev = sample_cases(dev_lattices, seed=SEED, label="dev")
        with open(cases_dev_path, "w") as f:
            json.dump(cases_dev, f, indent=2)
        print(f"Saved {len(cases_dev)} dev cases → {cases_dev_path}")
    else:
        with open(cases_dev_path) as f:
            cases_dev = json.load(f)
        print(f"Loaded {len(cases_dev)} existing dev cases from {cases_dev_path}")

    # Sample test cases
    if args.resample or not os.path.exists(cases_test_path):
        print("\nSampling test cases ...")
        cases_test = sample_cases(test_lattices, seed=SEED + 1, label="test")
        with open(cases_test_path, "w") as f:
            json.dump(cases_test, f, indent=2)
        print(f"Saved {len(cases_test)} test cases → {cases_test_path}")
    else:
        with open(cases_test_path) as f:
            cases_test = json.load(f)
        print(f"Loaded {len(cases_test)} existing test cases from {cases_test_path}")

    # Run BFS/SRA/ERA on dev cases
    print(f"\nRunning BFS/SRA/ERA on {len(cases_dev)} dev cases ...")
    results: list[dict] = []
    for i, case in enumerate(cases_dev):
        qid = case["qid"]
        criteria, rbc = dev_lattices[qid]
        print(f"  [{i+1:>3}/{len(cases_dev)}] qid={qid} k={case['size']} ...", end=" ", flush=True)
        result = run_one_case(i, case, criteria, rbc)
        results.append(result)
        bfs_layer = result["bfs"]["layer"]
        sra_ok = "OK" if result["sra"]["outcome"] == "correct" else result["sra"]["outcome"]
        print(f"BFS_layer={bfs_layer} SRA={sra_ok}")

    results_path = f"{OUT_DIR}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {results_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
