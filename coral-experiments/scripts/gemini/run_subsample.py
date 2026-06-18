"""Sub-lattice sweep on data/gemini_datasets.json.

For each effective lattice size k in 3..9, samples 20 sub-lattices that
contain a counterfactual (verified by BFS), then runs BFS / SRA / ERA(b=16).

A sub-lattice of size k is constructed by picking a `start_node` of length k
from a query's 10 criteria — the top-down lattice underneath has 2**k nodes.
This generalization sweep tests whether the budget=16 sweet spot from
run_gemini_eval.py holds at smaller lattices too.

Outputs:
  data/gemini_subsample/cases.json                  — sampled (qid, size, start_node) cases
  data/gemini_subsample/per_case/case_<i>.json      — per-case results (resumable)
  data/gemini_subsample/gemini_subsample_results.json — aggregate
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

DATA_PATH = "data/gemini_datasets.json"
OUT_DIR = "data/gemini_subsample"
CASES_PATH = f"{OUT_DIR}/cases.json"
RESULTS_PATH = f"{OUT_DIR}/gemini_subsample_results.json"

SIZES: list[int] = list(range(3, 10))   # 3..9 inclusive
TARGET_PER_SIZE: int = 20
ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50]
SEED: int = 42

MAX_RETRIES_PER_CASE: int = 8


def _normalize_lattice(q: dict) -> tuple[list[str], dict[str, list[str]]] | None:
    if q.get("halted"):
        return None
    raw = q.get("raw_answers") or {}
    criteria = list(q["criteria"])
    rbc: dict[str, list[str]] = {}
    for k, v in raw.items():
        parts = [p.strip() for p in k.split(",") if p.strip()]
        rbc[combo_key(parts)] = v
    if combo_key(criteria) not in rbc:
        return None
    return criteria, rbc


def _load_lattices() -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    """Returns {qid: (criteria, rankings_by_combo)} for usable lattices."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for qid, q in data.items():
        if qid == "_meta":
            continue
        norm = _normalize_lattice(q)
        if norm is None:
            continue
        out[qid] = norm
    return out


def _has_cf(rbc: dict[str, list[str]], criteria: list[str], start_node: list[str]) -> bool:
    res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
    return res["found"]


def sample_cases(seed: int = SEED) -> list[dict]:
    """For each size in SIZES, sample TARGET_PER_SIZE sub-lattices with a CF.

    Strategy: enumerate every (qid, start_node) of the requested size across
    all usable lattices, shuffle, then walk and accept the first 20 that have
    a BFS-found CF.
    """
    rng = random.Random(seed)
    lattices = _load_lattices()
    print(f"Loaded {len(lattices)} usable lattices.")

    cases: list[dict] = []
    for size in SIZES:
        candidates: list[dict] = []
        for qid, (criteria, rbc) in lattices.items():
            for combo in combinations(criteria, size):
                candidates.append({
                    "qid": qid,
                    "size": size,
                    "start_node": list(combo),
                })
        rng.shuffle(candidates)
        accepted: list[dict] = []
        for cand in candidates:
            criteria, rbc = lattices[cand["qid"]]
            if _has_cf(rbc, criteria, cand["start_node"]):
                accepted.append(cand)
                if len(accepted) >= TARGET_PER_SIZE:
                    break
        print(f"  size={size}: pool={len(candidates):>6}  accepted={len(accepted):>3}/{TARGET_PER_SIZE}")
        cases.extend(accepted)
    return cases


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


def era_label(kind: str, value: int) -> str:
    return f"era_{kind}_{value}"


def run_one_case(case_id: int, case: dict, criteria: list[str], rbc: dict[str, list[str]]) -> dict:
    """Run BFS + SRA + all ERA budget configs, checkpoint after each method.

    Reuses on-disk checkpoint when present so a crash mid-sweep is resumable.
    Each method writes a checkpoint; in-progress partial files are valid input
    to the next subprocess attempt.
    """
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
        label = era_label("fixed", b)
        if label in partial["era"]:
            continue
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down")
        partial["era"][label] = score_proxy(bfs_for_proxy, res, rbc, start_node)
        _checkpoint(case_id, partial)

    for pct in ERA_BUDGETS_PCT:
        label = era_label("pct", pct)
        if label in partial["era"]:
            continue
        b = max(1, int(2**size * pct / 100))
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down")
        scored = score_proxy(bfs_for_proxy, res, rbc, start_node)
        scored["budget"] = b
        scored["budget_pct"] = pct
        partial["era"][label] = scored
        _checkpoint(case_id, partial)

    partial["complete"] = True
    _checkpoint(case_id, partial)
    return partial


def _run_subprocess_single(case_id: int) -> int:
    """Used by --single-case to run one case in an isolated process."""
    with open(CASES_PATH) as f:
        cases = json.load(f)
    if case_id < 0 or case_id >= len(cases):
        print(f"case_id={case_id} out of range", file=sys.stderr)
        return 2
    case = cases[case_id]
    lattices = _load_lattices()
    if case["qid"] not in lattices:
        print(f"qid={case['qid']} not loadable", file=sys.stderr)
        return 2
    criteria, rbc = lattices[case["qid"]]
    run_one_case(case_id, case, criteria, rbc)
    return 0


def run_subprocess_case(case_id: int) -> tuple[dict | None, str]:
    cached = _load_partial(case_id)
    if cached and cached.get("complete"):
        return cached, "cached"
    last_status = "no-attempts"
    result = cached
    for attempt in range(1, MAX_RETRIES_PER_CASE + 1):
        prior_methods = _completed_method_count(_load_partial(case_id))
        cmd = [sys.executable, __file__, "--single-case", str(case_id)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result = _load_partial(case_id)
        result_methods = _completed_method_count(result)
        if proc.returncode == 0 and result and result.get("complete"):
            return result, "ok" if attempt == 1 else f"ok (retry x{attempt - 1})"
        if proc.returncode != 0:
            tag = "crash" if proc.returncode < 0 or "Abort" in (proc.stderr or "") else f"err rc={proc.returncode}"
            last_status = f"{tag} attempt={attempt} methods={result_methods}"
            if result_methods > prior_methods:
                continue
            if attempt >= 2:
                break
        else:
            last_status = f"rc=0 no-complete attempt={attempt}"
            break
    return result, last_status


def _completed_method_count(partial: dict | None) -> int:
    if not partial:
        return 0
    n = 0
    if "bfs" in partial:
        n += 1
    if "sra" in partial:
        n += 1
    n += len(partial.get("era", {}))
    return n


def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    era_labels = [era_label("fixed", b) for b in ERA_BUDGETS_FIXED] + \
                 [era_label("pct", p) for p in ERA_BUDGETS_PCT]

    print(f"\n{'=' * 110}")
    print(f"  GEMINI SUB-LATTICE SWEEP — {len(results)} cases across sizes {SIZES}")
    print(f"  ERA fixed budgets: {ERA_BUDGETS_FIXED}")
    print(f"  ERA percent budgets: {ERA_BUDGETS_PCT}")
    print(f"{'=' * 110}\n")

    def fmt_method(bucket: list[dict], pick: callable) -> str:
        ms = [pick(r) for r in bucket if pick(r) is not None]
        if not ms:
            return f"{'—':>14} | {'—':>9}"
        c = sum(1 for m in ms if m["outcome"] == "correct")
        cost = sum(m["calls"] + m.get("budget_used", 0) for m in ms) / len(ms)
        return f"{c/len(ms):>5.0%} ({c:>2}/{len(ms):<2}) | {cost:>9.1f}"

    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        bfs_avg = sum(r["bfs"]["calls"] for r in bucket if "bfs" in r) / n
        print(f"\n  --- size={size} (n={n}) ---  BFS avg calls = {bfs_avg:.1f}")
        print(f"    {'Method':<22} | {'Accuracy':>14} | {'Avg cost':>9}")
        print(f"    {'-'*22}-+-{'-'*14}-+-{'-'*9}")
        print(f"    {'SRA':<22} | {fmt_method(bucket, lambda r: r.get('sra'))}")
        for b in ERA_BUDGETS_FIXED:
            label = era_label("fixed", b)
            print(f"    {f'ERA (budget={b})':<22} | {fmt_method(bucket, lambda r, l=label: r.get('era', {}).get(l))}")
        for pct in ERA_BUDGETS_PCT:
            label = era_label("pct", pct)
            b = max(1, int(2**size * pct / 100))
            print(f"    {f'ERA ({pct}% → b={b})':<22} | {fmt_method(bucket, lambda r, l=label: r.get('era', {}).get(l))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case", type=int, help="Run one sampled case in subprocess mode.")
    parser.add_argument("--resample", action="store_true", help="Re-sample cases (ignores cases.json).")
    args = parser.parse_args()

    if args.single_case is not None:
        sys.exit(_run_subprocess_single(args.single_case))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/per_case", exist_ok=True)

    if args.resample or not os.path.exists(CASES_PATH):
        cases = sample_cases()
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Sampled {len(cases)} cases, saved to {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cached cases from {CASES_PATH}")

    results: list[dict] = []
    failed: list[tuple[int, str]] = []
    for i, case in enumerate(cases):
        res, status = run_subprocess_case(i)
        tag = "ok" if status == "ok" else status
        print(f"  [{i+1:>3}/{len(cases)}] case_id={i} qid={case['qid']} size={case['size']} -> {tag}",
              flush=True)
        if res is not None:
            results.append(res)
        else:
            failed.append((i, status))

    if failed:
        print(f"\nFailed: {len(failed)}")
        for i, st in failed[:20]:
            print(f"  case_id={i}: {st}")

    if results:
        print_summary(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "data_path": DATA_PATH,
            "sizes": SIZES,
            "target_per_size": TARGET_PER_SIZE,
            "era_budgets_fixed": ERA_BUDGETS_FIXED,
            "era_budgets_pct": ERA_BUDGETS_PCT,
            "seed": SEED,
            "n_cases": len(results),
            "n_failed": len(failed),
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
