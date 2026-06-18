"""ContextCite evaluation on data/test.jsonl or data/test_hard.jsonl (top-down rows, budget=64).

Phase 1 — export cases locally:
  conda run -n coral python3 run_cc_test_eval.py --export          # test.jsonl
  conda run -n coral python3 run_cc_test_eval.py --export --hard   # test_hard.jsonl

Phase 2 — GPU run:
  conda run -n coral python3 run_cc_test_eval.py --run [--hard] [--model PATH] [--batch-size 1]

Writes (regular):
  data/cc_cases_test.json
  data/era_ablation_study/test/cc_results_test_topdown.json

Writes (--hard):
  data/cc_cases_test_hard.json
  data/era_ablation_study/test_hard/cc_results_test_hard_topdown.json
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from scripts.shared.cc_budget_ablation_dev import (
    CC_EVAL_NUM_ABLATIONS,
    build_manifest_cc_cases,
    score_cc,
)
from scripts.shared.era_budget_ablation import LATTICES_PATH, TEST_CASES_PATH

TEST_HARD_CASES_PATH = "data/test_hard.jsonl"

CC_CASES_TEST_PATH = "data/cc_cases_test.json"
CC_RESULTS_TEST_PATH = "data/era_ablation_study/test/cc_results_test_topdown.json"

CC_CASES_TEST_HARD_PATH = "data/cc_cases_test_hard.json"
CC_RESULTS_TEST_HARD_PATH = "data/era_ablation_study/test_hard/cc_results_test_hard_topdown.json"


def _paths(hard: bool) -> tuple[str, str, str]:
    """Return (manifest_path, cases_out_path, results_out_path) for the chosen split."""
    if hard:
        return TEST_HARD_CASES_PATH, CC_CASES_TEST_HARD_PATH, CC_RESULTS_TEST_HARD_PATH
    return TEST_CASES_PATH, CC_CASES_TEST_PATH, CC_RESULTS_TEST_PATH


def export_cases(args: argparse.Namespace) -> None:
    sources_override: dict[int, str] | None = None
    if args.sources:
        with open(args.sources) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            sources_override = {
                int(k): (v["sources"] if isinstance(v, dict) else v)
                for k, v in raw.items()
            }
        else:
            sources_override = {int(r["query_id"]): r["sources"] for r in raw}
        print(f"Loaded sources override for {len(sources_override)} queries")

    manifest_path, cases_out, _ = _paths(args.hard)
    directions: set[str] | None = {"down"} if args.down_only else None
    cases = build_manifest_cc_cases(
        LATTICES_PATH,
        manifest_path,
        sources_override,
        directions=directions,
    )
    os.makedirs(os.path.dirname(cases_out) or ".", exist_ok=True)
    with open(cases_out, "w") as f:
        json.dump(cases, f, indent=2)
    with_sources = sum(1 for c in cases if c.get("sources"))
    print(f"Exported {len(cases)} cases from {manifest_path} to {cases_out}")
    print(f"  {with_sources} cases have sources ({len(cases) - with_sources} will be skipped on --run)")


def print_summary(results: list[dict], budget: int) -> None:
    n = len(results)
    outcomes = Counter(r["outcome"] for r in results)
    correct = outcomes.get("correct", 0)
    jaccs = [r["jaccard"] for r in results if r.get("jaccard") is not None]
    mean_j = sum(jaccs) / len(jaccs) if jaccs else float("nan")
    print(f"\nCC test (top-down, budget={budget}): {n} cases")
    print(f"  accuracy: {correct/n:.1%} ({correct}/{n})")
    print(f"  outcomes: {dict(outcomes)}")
    print(f"  mean_jaccard: {mean_j:.3f}")
    print(f"  mean_calls: {sum(r['calls'] for r in results)/n:.1f}")

    by_k: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_k[r["effective_size"]].append(r)
    print("  by k:")
    for k in sorted(by_k):
        bucket = by_k[k]
        c = sum(1 for r in bucket if r["outcome"] == "correct")
        print(f"    k={k}: {c}/{len(bucket)} = {c/len(bucket):.0%}")


def run_eval(args: argparse.Namespace) -> None:
    from scripts.shared.load_cc_model import load_model_and_tokenizer
    from search.bfs import bfs_search
    from search.contextcite import contextcite_search

    budget = args.budget
    _, cases_path, results_path = _paths(args.hard)
    with open(cases_path) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {cases_path}")
    print(f"Eval budget (num_ablations): {budget}")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    try:
        with open(results_path) as f:
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

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} rows in {results_path}")
    print_summary(results, budget)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true",
                      help="Build data/cc_cases_test.json from test.jsonl")
    mode.add_argument("--run", action="store_true",
                      help="Run ContextCite at fixed budget on exported cases (GPU)")

    parser.add_argument(
        "--all-directions", action="store_true",
        help="Include bottom-up rows (default: top-down only)",
    )
    parser.add_argument(
        "--hard", action="store_true",
        help="Use data/test_hard.jsonl (default: data/test.jsonl)",
    )
    parser.add_argument("--sources", help="Optional sources override JSON")
    from scripts.shared.load_cc_model import add_model_args
    add_model_args(parser)
    parser.add_argument(
        "--budget", type=int, default=CC_EVAL_NUM_ABLATIONS,
        help=f"num_ablations (default {CC_EVAL_NUM_ABLATIONS})",
    )
    args = parser.parse_args()
    args.down_only = not args.all_directions

    if args.export:
        export_cases(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
