"""CC (LSM) on the MSMARCO sub-lattice cases at the selected budget (GPU).

Reads  logged_results/llama31-8b/msmarco_criteria/cc_cases.json (exported by run_subsample.py).
Writes logged_results/llama31-8b/msmarco_criteria/cc_results.json with resume on case_id.

Default budget = pct_10 (max(1, int(0.10 * 2**k)) per case). Override with --pct.

CLI: CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\
       conda run -n coral python3 -m scripts.criteria.msmarco.run_cc_full --vllm
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

CC_CASES_PATH = "logged_results/llama31-8b/msmarco_criteria/cc_cases.json"
RESULTS_DIR   = "logged_results/llama31-8b/msmarco_criteria"


def main() -> None:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--pct", type=int, default=10,
                        help="Percent budget per case (default 10 = pct_10).")
    parser.add_argument("--budget", type=int, default=None,
                        help="Fixed budget per case (overrides --pct).")
    args = parser.parse_args()

    if args.budget is not None:
        config_label = f"budget_{args.budget}"
    else:
        config_label = f"pct_{args.pct}"
    CC_RESULTS_PATH = f"{RESULTS_DIR}/cc_results_{config_label}.json"

    with open(CC_CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}")
    print(f"Config: {config_label}  →  {CC_RESULTS_PATH}")

    model, tokenizer = load_model_and_tokenizer(args)
    from search.contextcite import contextcite_search

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
        k = case["effective_size"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        if args.budget is not None:
            num_ablations = args.budget
        else:
            num_ablations = max(1, int(2**k * args.pct / 100))
        print(
            f"  [c{cid:>3} {config_label} n={num_ablations:>3}] qid={case['query_id']} k={k} ...",
            end=" ", flush=True,
        )
        t0 = time.time()
        try:
            cc_res = contextcite_search(
                criteria, rbc,
                case["query"], case["sources"], case["plausible_answers"],
                model, tokenizer,
                num_ablations=num_ablations,
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
            "effective_size": k,
            "config": config_label,
            "budget": num_ablations,
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "elapsed": round(elapsed, 2),
            **scored,
        })

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(CC_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no sources or no BFS CF)")
    print(f"\nDone. {len(results)} rows → {CC_RESULTS_PATH}")


if __name__ == "__main__":
    main()
