"""Sub-lattice sampling and ERA sweep on Qwen3-32B datasets.

Only lattices whose size-10 BFS counterfactual requires removing 2-8 concepts
(CF size in [2, 8]) are included. For each effective size k in 3..9, samples
20 sub-lattices where BFS also finds a CF requiring removing >= 2 items from
that sub-start-node. Also appends the full size-10 lattice for every qualifying
qid. Runs BFS / SRA / ERA(fixed-budget grid) / ERA(percent-of-lattice grid)
on each case with subprocess isolation + incremental checkpointing.

Usage:
    conda run -n coral python3 -m scripts.shared.run_qwen3_subsample --dataset msmarco_criteria
    conda run -n coral python3 -m scripts.shared.run_qwen3_subsample --dataset msmarco_pragma
    conda run -n coral python3 -m scripts.shared.run_qwen3_subsample --dataset curated_criteria
    conda run -n coral python3 -m scripts.shared.run_qwen3_subsample --dataset curated_pragma

Outputs (per dataset):
    logged_results/qwen3-32b-prelim/<dataset>/cases.json
    logged_results/qwen3-32b-prelim/<dataset>/cc_cases.json
    logged_results/qwen3-32b-prelim/<dataset>/per_case/case_<i>.json
    logged_results/qwen3-32b-prelim/<dataset>/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

from scripts.shared.sampling import SEED, SIZES, TARGET_PER_SIZE, sample_cases
from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys, restrict_rbc
from search.rank import era_search, sra_search

DATASETS: dict[str, tuple[str, str, int | None]] = {
    "msmarco_criteria": (
        "data/qwen3-32b/msmarco-criteria-qwen3-32.json",
        "logged_results/qwen3-32b-prelim/msmarco_criteria",
        20,
    ),
    "msmarco_pragma": (
        "data/qwen3-32b/msmarco-pragmas-qwen3-32b.json",
        "logged_results/qwen3-32b-prelim/msmarco_pragma",
        20,
    ),
    "curated_criteria": (
        "data/qwen3-32b/curated-criteria-qwen3-32b.json",
        "logged_results/qwen3-32b-prelim/curated_criteria",
        20,
    ),
    "curated_pragma": (
        "data/qwen3-32b/curated-pragmas-qwen3-32b.json",
        "logged_results/qwen3-32b-prelim/curated_pragma",
        20,
    ),
}

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
MAX_RETRIES_PER_CASE: int = 8

# Set in main() from --dataset; used by subprocess helpers.
DATASET_NAME: str = ""
DATA_PATH: str = ""
OUT_DIR: str = ""
CASES_PATH: str = ""
CC_CASES_PATH: str = ""
RESULTS_PATH: str = ""
MAX_LATTICES: int | None = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_lattices(qid_filter: set[str] | None = None) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    """Load and filter Qwen3 lattices: exclude halted, missing start key, and
    cases where fewer than 2 concepts must be removed to reach the CF (i.e.
    CF size > n-2). When qid_filter is set, restrict to exactly those QIDs.
    If MAX_LATTICES is set and the pool exceeds it, keep the top MAX_LATTICES
    by ERA b=16 performance (correct first, then by cost reduction vs BFS)."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    bfs_costs: dict[str, int] = {}
    skipped_halted = skipped_no_key = skipped_cf = 0
    for qid, q in data.items():
        if qid == "_meta":
            continue
        if qid_filter is not None and qid not in qid_filter:
            continue
        if q.get("halted"):
            skipped_halted += 1
            continue
        criteria = list(q["criteria"])
        rbc: dict[str, list[str]] = q["raw_answers"]
        if combo_key(criteria) not in rbc:
            skipped_no_key += 1
            continue
        bfs = bfs_search(criteria, rbc, start_node=criteria)
        n = len(criteria)
        if not bfs["found"] or bfs["size"] > n - 2:
            skipped_cf += 1
            continue
        out[qid] = (criteria, rbc)
        bfs_costs[qid] = bfs["calls"]

    if MAX_LATTICES is not None and len(out) > MAX_LATTICES:
        scored = []
        for qid, (criteria, rbc) in out.items():
            baseline = (rbc.get(combo_key(criteria)) or [""])[0]
            res = era_search(criteria, rbc, budget=16, start_node=criteria,
                             direction="down", propagate_scores=True)
            era_cost = res.get("calls", 0) + res.get("budget_used", 0)
            cf_key = res.get("key")
            correct = (res.get("found", False) and cf_key is not None and
                       (rbc.get(cf_key) or [""])[0] != baseline)
            reduction = bfs_costs[qid] - era_cost
            scored.append((qid, correct, reduction))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        keep = {qid for qid, _, _ in scored[:MAX_LATTICES]}
        dropped = len(out) - MAX_LATTICES
        for qid in list(out):
            if qid not in keep:
                del out[qid]
        print(f"  ERA-selected top {MAX_LATTICES} from {MAX_LATTICES + dropped} (dropped {dropped})")

    print(
        f"  Loaded {len(out)} qualifying lattices "
        f"(skipped: {skipped_halted} halted, {skipped_no_key} no-key, "
        f"{skipped_cf} CF outside [2,8])"
    )
    return out


def _load_single_lattice(qid: str) -> tuple[list[str], dict[str, list[str]]]:
    """Load one qid directly — no BFS filter. Used by subprocess path."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    q = data[qid]
    return list(q["criteria"]), q["raw_answers"]


# ---------------------------------------------------------------------------
# CC export
# ---------------------------------------------------------------------------

def export_cc_cases(
    cases: list[dict],
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
) -> list[dict]:
    with open(DATA_PATH) as f:
        data = json.load(f)

    cc_cases: list[dict] = []
    for case_id, case in enumerate(cases):
        qid = case["qid"]
        if qid not in lattices:
            continue
        start_node = case["start_node"]
        criteria, rbc = lattices[qid]
        q = data[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            continue

        sub_rbc = restrict_rbc(rbc, start_node, criteria)
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
    if proxy_key is not None:
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
        "jaccard": jaccard_keys(bfs_key, proxy_key, sorted(start_node)),
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


def era_label(kind: str, value: int) -> str:
    return f"era_{kind}_{value}"


def run_one_case(case_id: int, case: dict, criteria: list[str], rbc: dict[str, list[str]]) -> dict:
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
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down",
                         propagate_scores=True)
        partial["era"][label] = score_proxy(bfs_for_proxy, res, rbc, start_node)
        _checkpoint(case_id, partial)

    for pct in ERA_BUDGETS_PCT:
        label = era_label("pct", pct)
        if label in partial["era"]:
            continue
        b = max(1, int(2**size * pct / 100))
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down",
                         propagate_scores=True)
        scored = score_proxy(bfs_for_proxy, res, rbc, start_node)
        scored["budget"] = b
        scored["budget_pct"] = pct
        partial["era"][label] = scored
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
    criteria, rbc = _load_single_lattice(case["qid"])
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
        cmd = [
            sys.executable, "-m", "scripts.shared.run_qwen3_subsample",
            "--dataset", DATASET_NAME,
            "--single-case", str(case_id),
        ]
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    all_sizes = sorted(set(SIZES + [10]))
    print(f"\n{'=' * 110}")
    print(f"  QWEN3 {DATASET_NAME.upper()} — {len(results)} cases across sizes {all_sizes}")
    print(f"  ERA fixed budgets: {ERA_BUDGETS_FIXED}")
    print(f"  ERA percent budgets: {ERA_BUDGETS_PCT}")
    print(f"{'=' * 110}\n")

    def fmt_method(bucket: list[dict], pick) -> str:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global DATASET_NAME, DATA_PATH, OUT_DIR, CASES_PATH, CC_CASES_PATH, RESULTS_PATH, MAX_LATTICES

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS),
        help="Which Qwen3 dataset to process.",
    )
    parser.add_argument("--single-case", type=int, help="Run one case in subprocess mode.")
    parser.add_argument("--resample", action="store_true", help="Re-sample cases even if cases.json exists.")
    parser.add_argument("--qid-filter", nargs="+", metavar="QID",
                        help="Restrict lattice pool to these QIDs exactly (bypasses CF range filter).")
    args = parser.parse_args()

    DATASET_NAME = args.dataset
    DATA_PATH, OUT_DIR, MAX_LATTICES = DATASETS[DATASET_NAME]
    CASES_PATH = f"{OUT_DIR}/cases.json"
    CC_CASES_PATH = f"{OUT_DIR}/cc_cases.json"
    RESULTS_PATH = f"{OUT_DIR}/results.json"

    if args.single_case is not None:
        sys.exit(_run_subprocess_single(args.single_case))

    qid_filter = set(args.qid_filter) if args.qid_filter else None

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/per_case", exist_ok=True)

    print(f"Loading {DATASET_NAME} from {DATA_PATH} ...")
    lattices = _load_lattices(qid_filter=qid_filter)
    qids = sorted(lattices)
    print(f"Qids ({len(qids)}): {qids}")

    if args.resample or not os.path.exists(CASES_PATH):
        import shutil
        per_case_dir = f"{OUT_DIR}/per_case"
        if os.path.exists(per_case_dir):
            shutil.rmtree(per_case_dir)
            os.makedirs(per_case_dir)
        cases = sample_cases(lattices)
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Sampled {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cached cases from {CASES_PATH}")

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
        for i, st in failed[:20]:
            print(f"  case_id={i}: {st}")

    if results:
        print_summary(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "dataset": DATASET_NAME,
            "data_path": DATA_PATH,
            "qids": qids,
            "sizes": sorted(set(SIZES + [10])),
            "target_per_size": TARGET_PER_SIZE,
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
