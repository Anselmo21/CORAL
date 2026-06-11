"""ContextCite evaluation on data/gemini_datasets.json (top-down, budget=64).

Phase 1 — export cases locally:
  conda run -n coral python3 run_cc_gemini_eval.py --export

Phase 2 — GPU run:
  conda run -n coral python3 run_cc_gemini_eval.py --run [--model PATH] [--batch-size 1]

Writes:
  data/gemini_eval/cc_cases.json  (default; override with --out)
  data/gemini_eval/cc_results_gemini_topdown.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from cc_budget_ablation_dev import CC_EVAL_NUM_ABLATIONS, score_cc
from run_gemini_eval import DATA_PATH, load_cases
from search.bfs import bfs_search

CC_CASES_GEMINI_PATH = "data/gemini_eval/cc_cases.json"
CC_RESULTS_GEMINI_PATH = "data/gemini_eval/cc_results_gemini_topdown.json"


def build_cases() -> list[dict]:
    cases, skipped = load_cases(DATA_PATH)
    with open(DATA_PATH) as f:
        data = json.load(f)

    cc_cases: list[dict] = []
    missing_sources: list[str] = []
    for case_id, case in enumerate(cases):
        qid = case["query_id"]
        q = data.get(qid, {})
        sources = q.get("sources") or ""
        if not sources:
            missing_sources.append(qid)
        cc_cases.append({
            "case_id": case_id,
            "query_id": qid,
            "query": case["question"],
            "sources": sources,
            "plausible_answers": list(q.get("plausible_answers") or []),
            "criteria": case["criteria"],
            "start_node": case["start_node"],
            "direction": case["direction"],
            "effective_size": case["effective_size"],
            "k": case["effective_size"],
            "rankings_by_combo": case["rankings_by_combo"],
        })

    if skipped:
        print(f"Skipped {len(skipped)} queries upstream "
              f"(halted / missing full-set key):")
        for qid, reason in skipped:
            print(f"  qid={qid}: {reason}")
    if missing_sources:
        print(f"WARN: {len(missing_sources)} cases have empty sources "
              f"(they will be skipped by --run): {missing_sources}")

    return cc_cases


def export_cases(out_path: str) -> None:
    cc_cases = build_cases()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cc_cases, f, indent=2)

    with_sources = sum(1 for c in cc_cases if c.get("sources"))
    print(f"Exported {len(cc_cases)} cases from {DATA_PATH} to {out_path}")
    print(f"  {with_sources} cases have sources ({len(cc_cases) - with_sources} will be skipped on --run)")


def print_summary(results: list[dict], budget: int) -> None:
    n = len(results)
    if n == 0:
        print("\nNo CC results to summarize.")
        return
    outcomes = Counter(r["outcome"] for r in results)
    correct = outcomes.get("correct", 0)
    jaccs = [r["jaccard"] for r in results if r.get("jaccard") is not None]
    mean_j = sum(jaccs) / len(jaccs) if jaccs else float("nan")
    print(f"\nCC gemini (top-down, budget={budget}): {n} cases")
    print(f"  accuracy: {correct/n:.1%} ({correct}/{n})")
    print(f"  outcomes: {dict(outcomes)}")
    print(f"  mean_jaccard: {mean_j:.3f}")
    print(f"  mean_calls: {sum(r['calls'] for r in results)/n:.1f}")
    print(f"  mean_total_cost: {sum(r['calls'] + r['budget'] for r in results)/n:.1f}")

    by_k: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_k[r["effective_size"]].append(r)
    print("  by k:")
    for k in sorted(by_k):
        bucket = by_k[k]
        c = sum(1 for r in bucket if r["outcome"] == "correct")
        print(f"    k={k}: {c}/{len(bucket)} = {c/len(bucket):.0%}")


def run_eval(args: argparse.Namespace) -> None:
    from load_cc_model import load_model_and_tokenizer
    from search.contextcite import contextcite_search

    budget = args.budget
    cases_path = args.cases
    with open(cases_path) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {cases_path}")
    print(f"Eval budget (num_ablations): {budget}")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(CC_RESULTS_GEMINI_PATH) or ".", exist_ok=True)
    try:
        with open(CC_RESULTS_GEMINI_PATH) as f:
            results: list[dict] = json.load(f)
        done = {r["case_id"] for r in results}
        print(f"Resuming: {len(done)} cases already done")
    except FileNotFoundError:
        results = []
        done = set()

    for case in cases:
        cid = case["case_id"]
        if cid in done:
            continue
        if not case.get("sources"):
            continue

        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        if not bfs_res["found"]:
            continue

        print(
            f"  [c{cid:>3}] Q{case['query_id']} k={case['effective_size']} {direction}...",
            end=" ", flush=True,
        )
        t0 = time.time()
        cc_res = contextcite_search(
            criteria,
            rbc,
            case["query"],
            case["sources"],
            case["plausible_answers"],
            model,
            tokenizer,
            num_ablations=budget,
            batch_size=args.batch_size,
            start_node=start_node,
            direction=direction,
        )
        elapsed = time.time() - t0

        scored = score_cc(bfs_res, cc_res, rbc, start_node)
        row = {
            "case_id": cid,
            "query_id": case["query_id"],
            "effective_size": case["effective_size"],
            "direction": direction,
            "budget": budget,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "elapsed": round(elapsed, 2),
            **scored,
        }
        results.append(row)
        done.add(cid)

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s, {scored['calls']} calls)")

        with open(CC_RESULTS_GEMINI_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} rows in {CC_RESULTS_GEMINI_PATH}")
    print_summary(results, budget)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true",
                      help="Build data/gemini_eval/cc_cases.json from gemini_datasets.json")
    mode.add_argument("--run", action="store_true",
                      help="Run ContextCite at fixed budget on exported cases (GPU)")
    from load_cc_model import add_model_args
    add_model_args(parser)
    parser.add_argument(
        "--budget", type=int, default=CC_EVAL_NUM_ABLATIONS,
        help=f"num_ablations (default {CC_EVAL_NUM_ABLATIONS})",
    )
    parser.add_argument(
        "--out", default=CC_CASES_GEMINI_PATH,
        help=f"Cases JSON path for --export (default {CC_CASES_GEMINI_PATH})",
    )
    parser.add_argument(
        "--cases", default=CC_CASES_GEMINI_PATH,
        help=f"Cases JSON path for --run (default {CC_CASES_GEMINI_PATH})",
    )
    args = parser.parse_args()

    if args.export:
        export_cases(args.out)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
