"""CC evaluation on the MSMARCO test set (GPU).

Loads 8 test queries from data/msmarco_dataset.json using test qids from
data/msmarco_subsample/split.json. All cases are full 10-criteria top-down (k=10).

Default budget: pct_10 = max(1, int(2^10 * 10 / 100)) = 102 ablations.
Override with --budget N (fixed) or --pct N (percent of 2^10).

Writes data/msmarco_test/cc_results_pct10.json (or cc_results_pct<N>.json /
cc_results_budget<N>.json). Resume on query_id.

CLI:
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \\
    python3 run_cc_msmarco_test.py --vllm [--pct 10] [--model PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASET_PATH = "data/msmarco_dataset.json"
SPLIT_PATH = "data/msmarco_subsample/split.json"
OUT_DIR = "data/msmarco_test"

EFFECTIVE_SIZE = 10
DEFAULT_PCT = 10


def _pct_budget(pct: int) -> int:
    return max(1, int(2**EFFECTIVE_SIZE * pct / 100))


def load_test_cases(dataset_path: str, split_path: str) -> list[dict]:
    with open(dataset_path) as f:
        data = json.load(f)
    with open(split_path) as f:
        test_qids = set(json.load(f)["test"])
    cases = []
    for qid, rec in data["lattices"].items():
        if qid not in test_qids:
            continue
        cases.append({
            "query_id": qid,
            "query": rec["question"],
            "sources": rec.get("sources", ""),
            "criteria": rec["criteria"],
            "plausible_answers": rec["plausible_answers"],
            "rankings_by_combo": rec["rankings"],
            "start_node": rec["criteria"],
        })
    return cases


def print_summary(results: list[dict], budget: int) -> None:
    n = len(results)
    if not n:
        print("No results.")
        return
    outcomes = Counter(r["outcome"] for r in results)
    correct = outcomes.get("correct", 0)
    print(f"\nCC MSMARCO test (top-down, budget={budget}): {n} cases")
    print(f"  accuracy:  {correct/n:.1%} ({correct}/{n})")
    print(f"  outcomes:  {dict(outcomes)}")
    print(f"  mean_calls: {sum(r['calls'] for r in results)/n:.1f}")
    print(f"  mean_cost:  {sum(r['calls'] + r['budget'] for r in results)/n:.1f}")


def main() -> None:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer
    from search.contextcite import contextcite_search

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--budget", type=int, default=None,
                        help="Fixed num_ablations (overrides --pct)")
    parser.add_argument("--pct", type=int, default=DEFAULT_PCT,
                        help=f"Percent-of-lattice budget: max(1, int(2^10 * pct/100)) (default {DEFAULT_PCT})")
    args = parser.parse_args()

    budget = args.budget if args.budget is not None else _pct_budget(args.pct)
    if args.budget is not None:
        out_name = f"cc_results_budget{args.budget}.json"
    else:
        out_name = f"cc_results_pct{args.pct}.json"
    out_path = os.path.join(OUT_DIR, out_name)

    cases = load_test_cases(DATASET_PATH, SPLIT_PATH)
    print(f"Loaded {len(cases)} test cases  (budget={budget}, out={out_path})")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        with open(out_path) as f:
            results: list[dict] = json.load(f)
        done = {r["query_id"] for r in results}
        print(f"Resuming: {len(done)} cases already done")
    except FileNotFoundError:
        results = []
        done = set()

    skipped = 0
    for case in cases:
        qid = case["query_id"]
        if qid in done:
            continue
        if not case.get("sources"):
            print(f"  Skipping qid={qid}: no sources")
            skipped += 1
            continue

        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            print(f"  Skipping qid={qid}: BFS found no CF")
            skipped += 1
            continue

        print(f"  [qid={qid}] ...", end=" ", flush=True)
        t0 = time.time()
        try:
            cc_res = contextcite_search(
                criteria, rbc,
                case["query"], case["sources"], case["plausible_answers"],
                model, tokenizer,
                num_ablations=budget,
                batch_size=args.batch_size,
                start_node=start_node,
                direction="down",
            )
        except Exception as e:
            print(f"ERROR: {e}")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} rows before crash → {out_path}")
            raise
        elapsed = time.time() - t0

        scored = score_cc(bfs_res, cc_res, rbc, start_node)
        results.append({
            "query_id": qid,
            "effective_size": EFFECTIVE_SIZE,
            "budget": budget,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "elapsed": round(elapsed, 2),
            **scored,
        })
        done.add(qid)

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"\nSkipped {skipped} cases")
    print(f"\nDone. {len(results)} results → {out_path}")
    print_summary(results, budget)


if __name__ == "__main__":
    main()
