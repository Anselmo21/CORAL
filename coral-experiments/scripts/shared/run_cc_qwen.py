"""ContextCite (LSM) runner for all four Qwen3-32B subsampled datasets.

Reads  logged_results/<dataset-root>/<dataset>/cc_cases.json
Writes logged_results/<dataset-root>/<dataset>/cc_results_budget_<N>.json  (resumable)

Default --dataset-root is 'qwen3-32b-temp0' (matches scripts.qwen3_32b.run_subsample
after the temp=0 relattice + 16-qid allowlist).

Usage (GPU machine):
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\
      conda run --no-capture-output -n coral \\
      python3 -m scripts.shared.run_cc_qwen --dataset msmarco_criteria --budget 16
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASETS = [
    "msmarco_criteria",
    "msmarco_pragma",
    "curated_criteria",
    "curated_pragma",
]


def main() -> None:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.set_defaults(model="Qwen/Qwen3-32B")
    parser.add_argument(
        "--dataset", required=True, choices=DATASETS,
        help="Which Qwen3-32B dataset to run CC on.",
    )
    parser.add_argument(
        "--budget", type=int, required=True,
        help="Fixed number of CC ablations per case.",
    )
    parser.add_argument(
        "--dataset-root", default="qwen3-32b-temp0",
        help="logged_results/<dataset-root>/<dataset>/ prefix. "
             "Default 'qwen3-32b-temp0' matches scripts.qwen3_32b.run_subsample.",
    )
    args = parser.parse_args()

    out_dir         = f"logged_results/{args.dataset_root}/{args.dataset}"
    cc_cases_path   = f"{out_dir}/cc_cases.json"
    cc_results_path = f"{out_dir}/cc_results_budget_{args.budget}.json"

    with open(cc_cases_path) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {cc_cases_path}")
    print(f"Model:  {args.model}")
    print(f"Budget: {args.budget}  →  {cc_results_path}")

    model, tokenizer = load_model_and_tokenizer(args)
    from search.contextcite import contextcite_search

    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(cc_results_path) as f:
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

        criteria   = case["criteria"]
        rbc        = case["rankings_by_combo"]
        start_node = case["start_node"]
        k          = case["effective_size"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        print(
            f"  [c{cid:>3} b={args.budget} k={k}] qid={case['query_id']} ...",
            end=" ", flush=True,
        )
        t0 = time.time()
        try:
            cc_res = contextcite_search(
                criteria, rbc,
                case["query"], case["sources"], case["plausible_answers"],
                model, tokenizer,
                num_ablations=args.budget,
                batch_size=args.batch_size,
                start_node=start_node,
                direction="down",
            )
        except Exception as e:
            print(f"ERROR: {e}")
            with open(cc_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} rows before crash → {cc_results_path}")
            raise
        elapsed = time.time() - t0

        scored = score_cc(bfs_res, cc_res, rbc, start_node)
        results.append({
            "case_id":        cid,
            "query_id":       case["query_id"],
            "effective_size": k,
            "budget":         args.budget,
            "bfs_key":        bfs_res["key"],
            "bfs_layer":      bfs_res["size"],
            "bfs_calls":      bfs_res["calls"],
            "elapsed":        round(elapsed, 2),
            **scored,
        })

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(cc_results_path, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no sources or BFS not found)")
    print(f"\nDone. {len(results)} rows → {cc_results_path}")


if __name__ == "__main__":
    main()
