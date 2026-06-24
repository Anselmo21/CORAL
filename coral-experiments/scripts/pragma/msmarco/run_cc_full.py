"""CC (LSM) on the MSMARCO pragma sub-lattice cases at the selected budget (GPU).

Reads  data/msmarco_pragma/cc_cases.json (exported by run_cc_ablation.py --export).
Writes data/msmarco_pragma/cc_results.json with resume on case_id.

Default budget = pct_10 (max(1, int(0.10 * 2**k)) per case). Override with --pct.

CLI:
  python3 -m scripts.pragma.msmarco.run_cc_ablation --export   # one-time, builds cc_cases.json
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\
    python3 -m scripts.pragma.msmarco.run_cc_full --vllm
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

CC_CASES_PATH   = "data/msmarco_pragma/cc_cases.json"
CC_RESULTS_PATH = "data/msmarco_pragma/cc_results.json"


def main() -> None:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--pct", type=int, default=10,
                        help="Percent budget per case (default 10 = pct_10).")
    parser.add_argument("--shard",      type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    pct = args.pct
    with open(CC_CASES_PATH) as f:
        all_cases = json.load(f)

    if args.shard is not None:
        cases = [c for c in all_cases if c["case_id"] % args.num_shards == args.shard]
        out_path = CC_RESULTS_PATH.replace(".json", f"_shard{args.shard}.json")
        print(f"Shard {args.shard}/{args.num_shards}: {len(cases)} cases (of {len(all_cases)})")
    else:
        cases = all_cases
        out_path = CC_RESULTS_PATH
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}; output → {out_path}")

    model, tokenizer = load_model_and_tokenizer(args)
    from search.contextcite import contextcite_search

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path) as f:
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

        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        k = case["effective_size"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        num_ablations = max(1, int(2**k * pct / 100))
        print(
            f"  [c{cid:>3} pct_{pct}={num_ablations:>3}] qid={case['query_id']} k={k} ...",
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
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} rows before crash → {out_path}")
            raise
        elapsed = time.time() - t0

        scored = score_cc(bfs_res, cc_res, rbc, start_node)
        results.append({
            "case_id":        cid,
            "query_id":       case["query_id"],
            "effective_size": k,
            "budget_pct":     pct,
            "budget":         num_ablations,
            "bfs_key":        bfs_res["key"],
            "bfs_layer":      bfs_res["size"],
            "bfs_calls":      bfs_res["calls"],
            "elapsed":        round(elapsed, 2),
            **scored,
        })

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no BFS CF)")
    print(f"\nDone. {len(results)} rows → {out_path}")


if __name__ == "__main__":
    main()
