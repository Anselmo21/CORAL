"""Evaluate BFS vs SRA vs ERA on data/gemini_datasets.json.

Each query in the dataset is a 10-criteria lattice with `raw_answers` already
populated for all 1024 combos (the empty key "" plus every comma-separated
sorted subset). We treat each query as one top-down case starting at the full
criteria set and report:

  * BFS  - exact baseline (ground-truth lookups)
  * SRA  - PyFlagr CombSUM proxy
  * ERA  - budgeted entropy-guided search, swept across:
      - fixed budgets:   [4, 8, 16, 32, 64, 128, 256]
      - percent budgets: [10, 20, 30, 40, 50]% of 2**k  (k = effective size = 10)

Each case runs in a subprocess via `--single-case <qid>` because PyFlagr's
linear aggregators can segfault (Abort trap 6) on certain inputs; subprocess
isolation lets us record a failure and continue.

Results JSON: data/gemini_eval/gemini_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from search.bfs import bfs_search
from search.lattice import combo_key, jaccard_keys
from search.rank import era_search, sra_search

DATA_PATH = "data/gemini_datasets.json"
OUT_DIR = "data/gemini_eval"
OUT_PATH = f"{OUT_DIR}/gemini_eval_results.json"

ERA_BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
ERA_BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50]


def load_cases(path: str) -> list[dict]:
    """Load gemini_datasets.json and yield top-down cases for non-halted queries."""
    with open(path) as f:
        data = json.load(f)

    cases: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for qid, q in data.items():
        if qid == "_meta":
            continue
        if q.get("halted"):
            skipped.append((qid, q.get("halted_reason", "halted")))
            continue
        raw = q.get("raw_answers") or {}
        criteria = list(q["criteria"])
        # The dataset's raw_answers keys are NOT lex-sorted (gemini emits them
        # in arbitrary order). Re-key with `combo_key()` so search lookups hit.
        rbc: dict[str, list[str]] = {}
        for k, v in raw.items():
            parts = [p.strip() for p in k.split(",") if p.strip()]
            rbc[combo_key(parts)] = v
        full_key = combo_key(criteria)
        if full_key not in rbc:
            skipped.append((qid, f"missing full-set key {full_key!r}"))
            continue
        cases.append({
            "query_id": qid,
            "question": q.get("question", ""),
            "criteria": criteria,
            "rankings_by_combo": rbc,
            "start_node": list(criteria),
            "direction": "down",
            "effective_size": len(criteria),
            "n_combos": len(raw),
        })
    return cases, skipped


def score_proxy(
    bfs_result: dict,
    proxy_result: dict,
    rankings_by_combo: dict[str, list[str]],
    start_node: list[str],
) -> dict:
    """Same outcome logic as experiment.py:score_proxy."""
    start_key = combo_key(start_node)
    baseline_top1 = (rankings_by_combo.get(start_key) or [""])[0]

    bfs_key = bfs_result["key"]
    bfs_layer = bfs_result["size"]

    proxy_key = proxy_result.get("key")
    proxy_layer = proxy_result.get("size")

    if proxy_key:
        ranking = rankings_by_combo.get(proxy_key, [])
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


def era_config_label(kind: str, value: int) -> str:
    return f"era_{kind}_{value}"


def _checkpoint(qid: str, partial: dict) -> None:
    """Write partial result to disk so a crash mid-case is resumable."""
    path = _per_case_path(qid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(partial, f, indent=2)
    os.replace(tmp, path)


def run_one_case(case: dict) -> dict:
    """Run BFS + SRA + all ERA configs, checkpointing after each method.

    If a per-case file already exists, resumes from it (skips completed methods).
    On crash mid-sweep, the most-recent checkpoint is preserved; rerunning the
    subprocess picks up where it left off.
    """
    qid = case["query_id"]
    criteria = case["criteria"]
    rbc = case["rankings_by_combo"]
    start_node = case["start_node"]
    direction = case["direction"]
    eff = case["effective_size"]

    path = _per_case_path(qid)
    if os.path.exists(path):
        with open(path) as f:
            partial = json.load(f)
    else:
        partial = {
            "query_id": qid,
            "question": case["question"],
            "effective_size": eff,
            "direction": direction,
            "start_node": start_node,
        }

    # BFS
    if "bfs" not in partial:
        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        partial["bfs"] = {
            "found": bfs_res["found"],
            "key": bfs_res["key"],
            "layer": bfs_res["size"],
            "calls": bfs_res["calls"],
        }
        _checkpoint(qid, partial)
    bfs_proxy_input = {
        "key": partial["bfs"]["key"],
        "size": partial["bfs"]["layer"],
    }

    # SRA
    if "sra" not in partial:
        sra_res = sra_search(criteria, rbc, start_node=start_node, direction=direction)
        partial["sra"] = score_proxy(bfs_proxy_input, sra_res, rbc, start_node)
        _checkpoint(qid, partial)

    partial.setdefault("era", {})

    for b in ERA_BUDGETS_FIXED:
        label = era_config_label("fixed", b)
        if label in partial["era"]:
            continue
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction=direction)
        partial["era"][label] = score_proxy(bfs_proxy_input, res, rbc, start_node)
        _checkpoint(qid, partial)

    for pct in ERA_BUDGETS_PCT:
        label = era_config_label("pct", pct)
        if label in partial["era"]:
            continue
        b = max(1, int(2**eff * pct / 100))
        res = era_search(criteria, rbc, budget=b, start_node=start_node, direction=direction)
        scored = score_proxy(bfs_proxy_input, res, rbc, start_node)
        scored["budget"] = b
        scored["budget_pct"] = pct
        partial["era"][label] = scored
        _checkpoint(qid, partial)

    partial["complete"] = True
    _checkpoint(qid, partial)
    return partial


def print_summary(results: list[dict]) -> None:
    # Tolerate partial results: each method is reported over only the cases that have it.
    has_bfs = [r for r in results if "bfs" in r]
    n = len(has_bfs)
    bfs_found = [r for r in has_bfs if r["bfs"]["found"]]
    print(f"\n{'='*100}")
    print(f"  GEMINI EVAL — {len(results)} queries (top-down from full 10-criteria set)")
    print(f"  BFS available in {n}/{len(results)} cases; BFS found CF in {len(bfs_found)}/{n}.")
    print(f"{'='*100}\n")

    def _stats(method_results: list[dict]) -> tuple[int, int, float, float]:
        """Returns (correct, total, mean_calls, mean_jaccard_verified)."""
        total = len(method_results)
        if total == 0:
            return 0, 0, float("nan"), float("nan")
        correct = sum(1 for m in method_results if m["outcome"] == "correct")
        mean_calls = sum(m["calls"] + m.get("budget_used", 0) for m in method_results) / total
        verified = [m for m in method_results if m.get("proxy_verified") and m.get("jaccard") is not None]
        mean_j = sum(m["jaccard"] for m in verified) / len(verified) if verified else float("nan")
        return correct, total, mean_calls, mean_j

    def _collect(label: str) -> list[dict]:
        out = []
        for r in results:
            if label == "sra":
                if "sra" in r:
                    out.append(r["sra"])
            else:
                if r.get("era", {}).get(label) is not None:
                    out.append(r["era"][label])
        return out

    # BFS row
    bfs_avg = sum(r["bfs"]["calls"] for r in has_bfs) / n if n else float("nan")
    print(f"  {'Method':<22} | {'Accuracy':>16} | {'Avg cost*':>10} | {'Mean Jaccard':>14} | {'N':>3}")
    print(f"  {'-'*22}-+-{'-'*16}-+-{'-'*10}-+-{'-'*14}-+-{'-'*3}")
    print(f"  {'BFS (baseline)':<22} | {'—':>16} | {bfs_avg:>10.1f} | {'—':>14} | {n:>3}")
    print(f"  {'-'*22}-+-{'-'*16}-+-{'-'*10}-+-{'-'*14}-+-{'-'*3}")

    sra_method = _collect("sra")
    c, t, mc, mj = _stats(sra_method)
    if t > 0:
        print(f"  {'SRA':<22} | {c/t:>5.0%} ({c:>3}/{t:<3}) | {mc:>10.1f} | {mj:>14.3f} | {t:>3}")
    print(f"  {'-'*22}-+-{'-'*16}-+-{'-'*10}-+-{'-'*14}-+-{'-'*3}")

    for b in ERA_BUDGETS_FIXED:
        label = era_config_label("fixed", b)
        method = _collect(label)
        c, t, mc, mj = _stats(method)
        if t > 0:
            print(f"  {f'ERA (budget={b})':<22} | {c/t:>5.0%} ({c:>3}/{t:<3}) | {mc:>10.1f} | {mj:>14.3f} | {t:>3}")

    print(f"  {'-'*22}-+-{'-'*16}-+-{'-'*10}-+-{'-'*14}-+-{'-'*3}")
    for pct in ERA_BUDGETS_PCT:
        label = era_config_label("pct", pct)
        method = _collect(label)
        c, t, mc, mj = _stats(method)
        b = max(1, int(2**10 * pct / 100))
        if t > 0:
            print(f"  {f'ERA ({pct}% → b={b})':<22} | {c/t:>5.0%} ({c:>3}/{t:<3}) | {mc:>10.1f} | {mj:>14.3f} | {t:>3}")

    print()
    print("  * cost = search calls + ERA budget_used (i.e. total LLM-equivalent operations)")
    print("  N = number of cases that completed this method (not all cases have all methods if pyflagr crashed)")

    # Outcome breakdown for SRA + each ERA config
    print(f"\n  Outcome breakdown:")
    outcomes_order = ["correct", "wrong_layer", "false_positive", "not_found"]
    methods_to_breakdown: list[tuple[str, list[dict]]] = [("SRA", sra_method)]
    for b in ERA_BUDGETS_FIXED:
        methods_to_breakdown.append((f"ERA b={b}", _collect(era_config_label("fixed", b))))
    for pct in ERA_BUDGETS_PCT:
        methods_to_breakdown.append((f"ERA {pct}%", _collect(era_config_label("pct", pct))))

    header = f"  {'Method':<14} | {'N':>4} | " + " | ".join(f"{o:>15}" for o in outcomes_order)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, ms in methods_to_breakdown:
        if not ms:
            continue
        counts = {o: sum(1 for m in ms if m["outcome"] == o) for o in outcomes_order}
        row = f"  {label:<14} | {len(ms):>4} | " + " | ".join(f"{counts[o]:>15}" for o in outcomes_order)
        print(row)


def _per_case_path(qid: str) -> str:
    return f"{OUT_DIR}/per_case/qid_{qid}.json"


MAX_RETRIES_PER_CASE = 8


def _load_partial(qid: str) -> dict | None:
    path = _per_case_path(qid)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_subprocess_case(qid: str) -> tuple[dict | None, str]:
    """Run a single case (possibly across multiple subprocess attempts).

    Each subprocess call resumes from the on-disk checkpoint. We re-launch on
    crash up to MAX_RETRIES_PER_CASE times, each time hoping to skip past the
    last segfaulting method by virtue of the checkpoint already advancing past
    the prior method's completion.

    Returns (partial_or_complete_result, status_string).
    """
    cached = _load_partial(qid)
    if cached and cached.get("complete"):
        return cached, "cached"

    last_status = "no-attempts"
    for attempt in range(1, MAX_RETRIES_PER_CASE + 1):
        prior = _load_partial(qid)
        prior_methods = _completed_method_count(prior)

        cmd = [sys.executable, __file__, "--single-case", qid]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result = _load_partial(qid)
        result_methods = _completed_method_count(result)

        if proc.returncode == 0 and result and result.get("complete"):
            return result, "ok" if attempt == 1 else f"ok (retry x{attempt - 1})"

        if proc.returncode != 0:
            tag = "crash" if proc.returncode < 0 or "Abort" in (proc.stderr or "") else f"err rc={proc.returncode}"
            last_status = f"{tag} attempt={attempt} methods={result_methods}"
            # If the checkpoint advanced, give it another try.
            if result_methods > prior_methods:
                continue
            # No progress this attempt — try once more, then give up.
            if attempt >= 2:
                break
        else:
            # rc==0 but no complete flag (shouldn't normally happen) — break.
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


def _load_single_case(qid: str) -> dict | None:
    """Load only one query's lattice — avoids parsing the whole dataset on every retry."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    q = data.get(qid)
    if q is None or q.get("halted"):
        return None
    raw = q.get("raw_answers") or {}
    criteria = list(q["criteria"])
    rbc: dict[str, list[str]] = {}
    for k, v in raw.items():
        parts = [p.strip() for p in k.split(",") if p.strip()]
        rbc[combo_key(parts)] = v
    if combo_key(criteria) not in rbc:
        return None
    return {
        "query_id": qid,
        "question": q.get("question", ""),
        "criteria": criteria,
        "rankings_by_combo": rbc,
        "start_node": list(criteria),
        "direction": "down",
        "effective_size": len(criteria),
        "n_combos": len(rbc),
    }


def _run_and_write_single(qid: str) -> int:
    """Used by the --single-case subprocess to run one case and write its result."""
    case = _load_single_case(qid)
    if case is None:
        print(f"qid={qid} not loadable", file=sys.stderr)
        return 2
    run_one_case(case)  # checkpoints internally
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case", help="Run a single qid (subprocess mode) and write its JSON.")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached per-case results and re-run.")
    args = parser.parse_args()

    if args.single_case is not None:
        sys.exit(_run_and_write_single(args.single_case))

    cases, skipped = load_cases(DATA_PATH)
    print(f"Loaded {len(cases)} usable cases from {DATA_PATH}.")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for qid, reason in skipped:
            print(f"  qid={qid}: {reason[:120]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/per_case", exist_ok=True)

    if args.fresh:
        for qid in [c["query_id"] for c in cases]:
            p = _per_case_path(qid)
            if os.path.exists(p):
                os.remove(p)

    results: list[dict] = []
    crashed: list[tuple[str, str]] = []
    for i, case in enumerate(cases, 1):
        qid = case["query_id"]
        res, status = run_subprocess_case(qid)
        tag = "ok" if status == "ok" else status
        print(f"  [{i}/{len(cases)}] qid={qid} -> {tag}", flush=True)
        if res is not None:
            results.append(res)
        else:
            crashed.append((qid, status))

    if crashed:
        print(f"\nCrashed/errored: {len(crashed)}")
        for qid, st in crashed:
            print(f"  qid={qid}: {st}")

    if results:
        print_summary(results)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "data_path": DATA_PATH,
            "n_cases": len(results),
            "n_crashed": len(crashed),
            "crashed": crashed,
            "skipped": skipped,
            "era_budgets_fixed": ERA_BUDGETS_FIXED,
            "era_budgets_pct": ERA_BUDGETS_PCT,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
