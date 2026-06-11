"""Sub-lattice ERA budget ablation on data/haiku_dataset_clean.json.

Sampling strategy:
  - Sizes 3-9: sample 15 sub-lattices per size from the pool of all usable
    lattices (each sub-lattice is a top-down search from a start_node of
    length k, so the lattice has 2**k nodes).
  - Size 10: randomly select 3 full lattices from the pool.

ERA configs:
  - Fixed budgets: [4, 8, 16, 32, 64]
  - Percent-of-lattice budgets: [10, 20, 30, 40, 50] % of 2**k

Outputs:
  data/haiku_subsample/cases.json               — (qid, size, start_node) for every sampled case
  data/haiku_subsample/per_case/case_<i>.json   — per-case results (resumable)
  data/haiku_subsample/haiku_subsample_results.json — aggregate
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

DATA_PATH = "data/haiku_dataset_clean.json"
OUT_DIR = "data/haiku_subsample"
CASES_PATH = f"{OUT_DIR}/cases.json"
RESULTS_PATH = f"{OUT_DIR}/haiku_subsample_results.json"

_IS_TEST: bool = False  # set to True in main() when --test is passed

SIZES: list[int] = list(range(3, 10))   # 3..9 inclusive
SIZE_10_COUNT: int = 3                  # full lattices used at size 10
TARGET_PER_SIZE: int = 15
ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64]
ERA_BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50]
SEED: int = 42

# 7/7 train-test split of the 14 clean lattices (random.Random(42).shuffle).
# Sub-lattices for sizes 3-9 are sampled exclusively from DEV_QIDS so that
# TEST_QIDS are fully held out for evaluation.
DEV_QIDS: set[str] = {"44", "46", "53"}
TEST_QIDS: set[str] = {"15", "20", "21", "30", "32", "36", "38", "5", "51", "52", "9"}

MAX_RETRIES_PER_CASE: int = 8


def _normalize_lattice(entry: dict) -> tuple[list[str], dict[str, list[str]]] | None:
    """Return (criteria, rbc) for a usable lattice, or None.

    Keys in raw_answers are already in combo_key format (sorted, ", "-separated),
    so we use them directly rather than re-parsing. This avoids breakage when a
    criterion name contains a comma (e.g. "field-specific requirements (labs, clinicals)").
    """
    if entry.get("halted"):
        return None
    raw = entry.get("raw_answers") or {}
    criteria = list(entry["criteria"])
    rbc: dict[str, list[str]] = dict(raw)
    full_key = combo_key(criteria)
    if full_key not in rbc:
        return None
    return criteria, rbc


def _sub_rbc(rbc: dict[str, list[str]], start_node: list[str]) -> dict[str, list[str]]:
    """Extract sub-lattice rbc by enumerating all subsets of start_node.

    Enumeration avoids parsing keys by comma, which breaks when criterion names
    contain commas.
    """
    sub: dict[str, list[str]] = {}
    for size in range(len(start_node) + 1):
        for combo in combinations(start_node, size):
            key = combo_key(list(combo))
            if key in rbc:
                sub[key] = rbc[key]
    return sub


def _load_lattices() -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for qid, entry in data.items():
        if qid == "_meta":
            continue
        norm = _normalize_lattice(entry)
        if norm is None:
            continue
        out[qid] = norm
    return out


def _has_cf(rbc: dict[str, list[str]], start_node: list[str]) -> bool:
    sub = _sub_rbc(rbc, start_node)
    res = bfs_search(start_node, sub, start_node=start_node, direction="down")
    return res["found"]


def sample_cases(seed: int = SEED) -> list[dict]:
    """Sample sub-lattice cases with a BFS-confirmed CF.

    For sizes 3-9: TARGET_PER_SIZE cases each, drawn from the full lattice pool.
    For size 10: SIZE_10_COUNT full lattices, randomly selected from the pool.
    """
    rng = random.Random(seed)
    lattices = _load_lattices()
    pool_qids = TEST_QIDS if _IS_TEST else DEV_QIDS
    dev_lattices = {qid: v for qid, v in lattices.items() if qid in pool_qids}
    split_label = "test" if _IS_TEST else "dev"
    print(f"Loaded {len(lattices)} usable lattices (using {len(dev_lattices)} {split_label} lattices).")

    cases: list[dict] = []

    # Sizes 3-9: sample only from dev lattices
    for size in SIZES:
        candidates: list[dict] = []
        for qid, (criteria, _) in dev_lattices.items():
            for combo in combinations(criteria, size):
                candidates.append({"qid": qid, "size": size, "start_node": list(combo)})
        rng.shuffle(candidates)
        accepted: list[dict] = []
        for cand in candidates:
            criteria, rbc = dev_lattices[cand["qid"]]
            if _has_cf(rbc, cand["start_node"]):
                accepted.append(cand)
                if len(accepted) >= TARGET_PER_SIZE:
                    break
        print(f"  size={size}: pool={len(candidates):>6}  accepted={len(accepted):>3}/{TARGET_PER_SIZE}")
        cases.extend(accepted)

    # Size 10: full dev lattices (random subset)
    dev_qids = list(dev_lattices.keys())
    rng.shuffle(dev_qids)
    size10_qids = dev_qids[:SIZE_10_COUNT]
    for qid in size10_qids:
        criteria, _ = dev_lattices[qid]
        cases.append({"qid": qid, "size": 10, "start_node": criteria})
    print(f"  size=10: selected {SIZE_10_COUNT} full lattices: {size10_qids}")

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
    start_node = case["start_node"]
    size = case["size"]
    sub = _sub_rbc(rbc, start_node)

    partial = _load_partial(case_id) or {
        "case_id": case_id,
        "qid": case["qid"],
        "size": size,
        "start_node": start_node,
    }

    # Pass start_node as criteria so search functions only enumerate the k
    # criteria in scope — sub only contains subsets of start_node.
    if "bfs" not in partial:
        bfs_res = bfs_search(start_node, sub, start_node=start_node, direction="down")
        partial["bfs"] = {
            "found": bfs_res["found"],
            "key": bfs_res["key"],
            "layer": bfs_res["size"],
            "calls": bfs_res["calls"],
        }
        _checkpoint(case_id, partial)

    bfs_for_proxy = {"key": partial["bfs"]["key"], "size": partial["bfs"]["layer"]}

    if "sra" not in partial:
        sra_res = sra_search(start_node, sub, start_node=start_node, direction="down")
        partial["sra"] = score_proxy(bfs_for_proxy, sra_res, sub, start_node)
        _checkpoint(case_id, partial)

    partial.setdefault("era", {})

    for b in ERA_BUDGETS_FIXED:
        label = era_label("fixed", b)
        if label in partial["era"]:
            continue
        res = era_search(start_node, sub, budget=b, start_node=start_node, direction="down")
        partial["era"][label] = score_proxy(bfs_for_proxy, res, sub, start_node)
        _checkpoint(case_id, partial)

    for pct in ERA_BUDGETS_PCT:
        label = era_label("pct", pct)
        if label in partial["era"]:
            continue
        b = max(1, int(2**size * pct / 100))
        res = era_search(start_node, sub, budget=b, start_node=start_node, direction="down")
        scored = score_proxy(bfs_for_proxy, res, sub, start_node)
        scored["budget"] = b
        scored["budget_pct"] = pct
        partial["era"][label] = scored
        _checkpoint(case_id, partial)

    partial["complete"] = True
    _checkpoint(case_id, partial)
    return partial


def _run_subprocess_single(case_id: int) -> int:
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


def run_subprocess_case(case_id: int) -> tuple[dict | None, str]:
    cached = _load_partial(case_id)
    if cached and cached.get("complete"):
        return cached, "cached"
    last_status = "no-attempts"
    result = cached
    for attempt in range(1, MAX_RETRIES_PER_CASE + 1):
        prior_methods = _completed_method_count(_load_partial(case_id))
        cmd = [sys.executable, __file__, "--single-case", str(case_id)]
        if _IS_TEST:
            cmd.append("--test")
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


def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    print(f"\n{'=' * 110}")
    print(f"  HAIKU SUB-LATTICE ERA ABLATION — {len(results)} cases")
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
    global OUT_DIR, CASES_PATH, RESULTS_PATH, _IS_TEST

    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case", type=int)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run on held-out test set (TEST_QIDS).")
    args = parser.parse_args()

    if args.test:
        _IS_TEST = True
        OUT_DIR = "data/haiku_subsample_test"
        CASES_PATH = f"{OUT_DIR}/cases.json"
        RESULTS_PATH = f"{OUT_DIR}/haiku_subsample_test_results.json"

    if args.single_case is not None:
        sys.exit(_run_subprocess_single(args.single_case))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/per_case", exist_ok=True)

    if args.resample or not os.path.exists(CASES_PATH):
        cases = sample_cases()
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Sampled {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cached cases from {CASES_PATH}")

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
        for i, st in failed[:20]:
            print(f"  case_id={i}: {st}")

    if results:
        print_summary(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "data_path": DATA_PATH,
            "split": "test" if _IS_TEST else "dev",
            "qids_used": sorted(TEST_QIDS if _IS_TEST else DEV_QIDS),
            "sizes": SIZES + [10],
            "target_per_size": TARGET_PER_SIZE,
            "size_10_count": SIZE_10_COUNT,
            "era_budgets_fixed": ERA_BUDGETS_FIXED,
            "era_budgets_pct": ERA_BUDGETS_PCT,
            "seed": SEED,
            "n_cases": len(results),
            "n_failed": len(failed),
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
