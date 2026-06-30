"""ContextCite (LSM) on the Qwen3-32B sub-lattice cases (GPU).

Mirrors scripts.criteria.msmarco.run_cc_full but reads and writes under
logged_results/qwen3-32b/<dataset>/. Loads the backbone once, runs contextcite_search
per case, scores with the shared score_cc, and writes cc_results_<config>.json with
resume on case_id. Reads cc_cases.json produced by scripts.qwen3_32b.run_subsample.

Run after the lattices exist and run_subsample has exported cc_cases. Example:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen3_32b.run_cc_full --vllm --model Qwen/Qwen3-32B --pct 10
    CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen3_32b.run_cc_full --vllm --model Qwen/Qwen3-32B --budget 16
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASETS = ["msmarco_criteria", "msmarco_pragma", "sonnet_criteria", "sonnet_pragma"]


def run_dataset(dataset, args, model, tokenizer, contextcite_search):
    cc_cases_path = f"logged_results/qwen3-32b/{dataset}/cc_cases.json"
    if not os.path.exists(cc_cases_path):
        print(f"skip {dataset}: no {cc_cases_path}")
        return

    config_label = f"budget_{args.budget}" if args.budget is not None else f"pct_{args.pct}"
    results_path = f"logged_results/qwen3-32b/{dataset}/cc_results_{config_label}.json"
    with open(cc_cases_path) as f:
        cases = json.load(f)
    print(f"[{dataset}] {len(cases)} cases  config={config_label}  ->  {results_path}")

    try:
        with open(results_path) as f:
            results = json.load(f)
        done = {r["case_id"] for r in results}
    except FileNotFoundError:
        results, done = [], set()

    for case in cases:
        cid = case["case_id"]
        if cid in done or not case.get("sources"):
            continue
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        k = case["effective_size"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            continue

        num_ablations = args.budget if args.budget is not None else max(1, int(2 ** k * args.pct / 100))
        print(f"  [c{cid:>3} {config_label} n={num_ablations:>3}] qid={case['query_id']} k={k} ...",
              end=" ", flush=True)
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
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
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
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("OK" if scored["outcome"] == "correct" else scored["outcome"].upper())

    print(f"  done: {len(results)} rows -> {results_path}")


def main() -> None:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--pct", type=int, default=10)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="One Qwen dataset; default runs all four.")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    from search.contextcite import contextcite_search

    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        run_dataset(ds, args, model, tokenizer, contextcite_search)


if __name__ == "__main__":
    main()
