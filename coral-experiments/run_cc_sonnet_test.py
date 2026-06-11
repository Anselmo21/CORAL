"""ContextCite evaluation on the sonnet test sub-lattice cases (GPU).

Reads the pre-exported data/sonnet_test_subsample/cc_cases.json and runs
ContextCite at a fixed budget (default 64 ablations) on every case that has
sources and a BFS-found CF. Resumes from data/sonnet_test_subsample/cc_results.json
if it already exists.

After running, copy cc_results.json back and run plot_sonnet_test_subsample.py
to merge CC into the comparison plots.

Usage:
  conda run -n coral python3 run_cc_sonnet_test.py [--model PATH] [--budget 64] [--batch-size 1]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from cc_budget_ablation_dev import CC_EVAL_NUM_ABLATIONS, score_cc
from search.bfs import bfs_search

CC_CASES_PATH = "data/sonnet_test_subsample/cc_cases.json"
CC_RESULTS_PATH = "data/sonnet_test_subsample/cc_results.json"


def print_summary(results: list[dict], budget: int) -> None:
    n = len(results)
    if n == 0:
        print("No results yet.")
        return
    outcomes = Counter(r["outcome"] for r in results)
    correct = outcomes.get("correct", 0)
    print(f"\nCC sonnet test (top-down, budget={budget}): {n} cases")
    print(f"  accuracy:   {correct/n:.1%} ({correct}/{n})")
    print(f"  outcomes:   {dict(outcomes)}")
    print(f"  mean_calls: {sum(r['calls'] for r in results)/n:.1f}")
    print(f"  mean_cost:  {sum(r['calls'] + budget for r in results)/n:.1f}")

    by_k: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_k[r["effective_size"]].append(r)
    print("  by k:")
    for k in sorted(by_k):
        bucket = by_k[k]
        c = sum(1 for r in bucket if r["outcome"] == "correct")
        print(f"    k={k}: {c}/{len(bucket)} = {c/len(bucket):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--budget", type=int, default=CC_EVAL_NUM_ABLATIONS,
                        help=f"num_ablations (default {CC_EVAL_NUM_ABLATIONS})")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    with open(CC_CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from search.contextcite import contextcite_search

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )
    print("Model loaded.")

    os.makedirs(os.path.dirname(CC_RESULTS_PATH), exist_ok=True)
    try:
        with open(CC_RESULTS_PATH) as f:
            results: list[dict] = json.load(f)
        done = {r["case_id"] for r in results}
        print(f"Resuming: {len(done)} cases already done")
    except FileNotFoundError:
        results = []
        done = set()

    skipped = 0
    for case in cases:
        cid = case["case_id"]
        if cid in done:
            continue
        if not case.get("sources"):
            skipped += 1
            continue

        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        print(
            f"  [c{cid:>3}] qid={case['query_id']} k={case['effective_size']} ...",
            end=" ", flush=True,
        )
        t0 = time.time()
        try:
            cc_res = contextcite_search(
                criteria,
                rbc,
                case["query"],
                case["sources"],
                case["plausible_answers"],
                model,
                tokenizer,
                num_ablations=args.budget,
                batch_size=args.batch_size,
                start_node=start_node,
                direction="down",
            )
        except Exception as e:
            print(f"ERROR: {e}")
            with open(CC_RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} rows before crash → {CC_RESULTS_PATH}")
            raise
        elapsed = time.time() - t0

        scored = score_cc(bfs_res, cc_res, rbc, start_node)
        results.append({
            "case_id": cid,
            "query_id": case["query_id"],
            "effective_size": case["effective_size"],
            "budget": args.budget,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "elapsed": round(elapsed, 2),
            **scored,
        })
        done.add(cid)

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(CC_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no sources or no BFS CF)")
    print(f"\nDone. {len(results)} results → {CC_RESULTS_PATH}")
    print_summary(results, args.budget)


if __name__ == "__main__":
    main()
