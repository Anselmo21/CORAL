"""Sub-lattice sweep on the MSMARCO test set.

Uses the 8 test qids from data/msmarco_subsample/split.json. For each size k in
3..9, samples 20 sub-lattices with a BFS-found CF and runs BFS / SRA /
ERA(budget=16). Exports CC-compatible cases to
data/msmarco_test_subsample/cc_cases.json so CC can be run on a GPU machine.

Outputs:
  data/msmarco_test_subsample/cases.json      — sampled (qid, size, start_node)
  data/msmarco_test_subsample/cc_cases.json   — CC-ready export (sub-lattice rbc + sources)
  data/msmarco_test_subsample/per_case/       — per-case results (resumable)
  data/msmarco_test_subsample/results.json    — aggregate BFS/SRA/ERA results

CLI: conda run -n coral python3 run_msmarco_test_subsample.py [--resample]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, sra_search
from run_msmarco_subsample import _load_lattices, _load_split, score_proxy

DATA_PATH = "data/msmarco_dataset.json"
OUT_DIR = "data/msmarco_test_subsample"
CASES_PATH = f"{OUT_DIR}/cases.json"
CC_CASES_PATH = f"{OUT_DIR}/cc_cases.json"
RESULTS_PATH = f"{OUT_DIR}/results.json"

SIZES: list[int] = list(range(3, 10))
TARGET_PER_SIZE: int = 20
ERA_BUDGET: int = 16
SEED: int = 42
MAX_RETRIES_PER_CASE: int = 8


# ---------------------------------------------------------------------------
# Case sampling
# ---------------------------------------------------------------------------

def _has_cf(rbc: dict[str, list[str]], criteria: list[str], start_node: list[str]) -> bool:
    return bfs_search(criteria, rbc, start_node=start_node, direction="down")["found"]


def sample_cases(test_qids: list[str], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    lattices = _load_lattices(test_qids)
    print(f"Loaded {len(lattices)} test lattices.")

    cases: list[dict] = []
    for size in SIZES:
        candidates: list[dict] = []
        for qid, (criteria, _) in lattices.items():
            for combo in combinations(criteria, size):
                candidates.append({"qid": qid, "size": size, "start_node": list(combo)})
        rng.shuffle(candidates)
        accepted: list[dict] = []
        for cand in candidates:
            crit, rbc = lattices[cand["qid"]]
            if _has_cf(rbc, crit, cand["start_node"]):
                accepted.append(cand)
                if len(accepted) >= TARGET_PER_SIZE:
                    break
        print(f"  size={size}: pool={len(candidates):>6}  accepted={len(accepted):>3}/{TARGET_PER_SIZE}")
        cases.extend(accepted)
    return cases


# ---------------------------------------------------------------------------
# CC export
# ---------------------------------------------------------------------------

def _sublattice_rbc(
    rbc: dict[str, list[str]],
    start_node: list[str],
) -> dict[str, list[str]]:
    """Return rbc restricted to subsets of start_node.

    MSMARCO rankings use pre-sorted comma-separated keys, so we can parse by
    splitting on ', ' directly (no greedy parser needed).
    """
    start_set = set(start_node)
    out: dict[str, list[str]] = {}
    for k, v in rbc.items():
        if not k:
            out[k] = v
            continue
        parts = [p.strip() for p in k.split(", ")]
        if set(parts) <= start_set:
            out[k] = v
    return out


def export_cc_cases(
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
) -> list[dict]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    raw_lats = data["lattices"]

    cc_cases: list[dict] = []
    for case_id, case in enumerate(cases):
        qid = case["qid"]
        start_node = case["start_node"]
        criteria, rbc = lattices[qid]
        q = raw_lats[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            continue

        sub_rbc = _sublattice_rbc(rbc, start_node)
        cc_cases.append({
            "case_id": case_id,
            "query_id": qid,
            "query": q["question"],
            "sources": q.get("sources") or "",
            "plausible_answers": list(q.get("plausible_answers") or []),
            "criteria": start_node,
            "start_node": start_node,
            "direction": "down",
            "effective_size": len(start_node),
            "k": len(start_node),
            "rankings_by_combo": sub_rbc,
        })

    return cc_cases


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

    if "era" not in partial:
        era_res = era_search(
            criteria, rbc,
            budget=ERA_BUDGET,
            start_node=start_node,
            direction="down",
        )
        partial["era"] = score_proxy(bfs_for_proxy, era_res, rbc, start_node)
        _checkpoint(case_id, partial)

    partial["complete"] = True
    _checkpoint(case_id, partial)
    return partial


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------

def _run_subprocess_single(case_id: int) -> int:
    with open(CASES_PATH) as f:
        cases = json.load(f)
    if case_id < 0 or case_id >= len(cases):
        print(f"case_id={case_id} out of range", file=sys.stderr)
        return 2
    case = cases[case_id]
    _, test_qids = _load_split()
    lattices = _load_lattices(test_qids)
    if case["qid"] not in lattices:
        print(f"qid={case['qid']} not loadable", file=sys.stderr)
        return 2
    criteria, rbc = lattices[case["qid"]]
    run_one_case(case_id, case, criteria, rbc)
    return 0


def _completed_method_count(partial: dict | None) -> int:
    if not partial:
        return 0
    return sum(k in partial for k in ("bfs", "sra", "era"))


def run_subprocess_case(case_id: int) -> tuple[dict | None, str]:
    cached = _load_partial(case_id)
    if cached and cached.get("complete"):
        return cached, "cached"
    last_status = "no-attempts"
    result = cached
    for attempt in range(1, MAX_RETRIES_PER_CASE + 1):
        prior = _completed_method_count(_load_partial(case_id))
        cmd = [sys.executable, __file__, "--single-case", str(case_id)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result = _load_partial(case_id)
        n_methods = _completed_method_count(result)
        if proc.returncode == 0 and result and result.get("complete"):
            return result, "ok" if attempt == 1 else f"ok (retry x{attempt-1})"
        if proc.returncode != 0:
            tag = "crash" if proc.returncode < 0 or "Abort" in (proc.stderr or "") else f"err rc={proc.returncode}"
            last_status = f"{tag} attempt={attempt} methods={n_methods}"
            if n_methods > prior:
                continue
            if attempt >= 2:
                break
        else:
            last_status = f"rc=0 no-complete attempt={attempt}"
            break
    return result, last_status


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    print(f"\n{'=' * 80}")
    print(f"  MSMARCO TEST SUB-LATTICE SWEEP — {len(results)} cases, ERA budget={ERA_BUDGET}")
    print(f"{'=' * 80}\n")

    def fmt(bucket: list[dict], pick) -> str:
        ms = [pick(r) for r in bucket if pick(r) is not None]
        if not ms:
            return f"{'—':>14} | {'—':>9}"
        c = sum(1 for m in ms if m["outcome"] == "correct")
        cost = sum(m["calls"] + m.get("budget_used", 0) for m in ms) / len(ms)
        return f"{c/len(ms):>5.0%} ({c:>2}/{len(ms):<2}) | {cost:>9.1f}"

    for size in sorted(by_size):
        bucket = by_size[size]
        bfs_avg = sum(r["bfs"]["calls"] for r in bucket) / len(bucket)
        print(f"  --- size={size} (n={len(bucket)}) ---  BFS avg calls = {bfs_avg:.1f}")
        print(f"    {'Method':<12} | {'Accuracy':>14} | {'Avg cost':>9}")
        print(f"    {'-'*12}-+-{'-'*14}-+-{'-'*9}")
        print(f"    {'SRA':<12} | {fmt(bucket, lambda r: r.get('sra'))}")
        print(f"    {'ERA b=16':<12} | {fmt(bucket, lambda r: r.get('era'))}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case", type=int)
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()

    if args.single_case is not None:
        sys.exit(_run_subprocess_single(args.single_case))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/per_case", exist_ok=True)

    _, test_qids = _load_split()
    print(f"Test ({len(test_qids)}): {test_qids}")

    if args.resample or not os.path.exists(CASES_PATH):
        cases = sample_cases(test_qids)
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Sampled {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cached cases from {CASES_PATH}")

    # Export CC cases (regenerated each run to stay in sync with cases.json)
    lattices = _load_lattices(test_qids)
    cc_cases = export_cc_cases(cases, lattices)
    with open(CC_CASES_PATH, "w") as f:
        json.dump(cc_cases, f, indent=2)
    print(f"Exported {len(cc_cases)} CC cases → {CC_CASES_PATH}")

    results: list[dict] = []
    failed: list[tuple[int, str]] = []
    for i, case in enumerate(cases):
        res, status = run_subprocess_case(i)
        tag = "ok" if status in ("ok", "cached") else status
        print(f"  [{i+1:>3}/{len(cases)}] case_id={i} qid={case['qid']} size={case['size']} -> {tag}",
              flush=True)
        if res is not None:
            results.append(res)
        else:
            failed.append((i, status))

    if failed:
        print(f"\nFailed: {len(failed)}")
        for fid, st in failed[:20]:
            print(f"  case_id={fid}: {st}")

    if results:
        print_summary(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "data_path": DATA_PATH,
            "test_qids": test_qids,
            "sizes": SIZES,
            "target_per_size": TARGET_PER_SIZE,
            "era_budget": ERA_BUDGET,
            "seed": SEED,
            "n_cases": len(results),
            "n_failed": len(failed),
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
