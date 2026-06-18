"""CC eval at pct_10 budget on the MSMARCO pragma sub-lattice test set (GPU).

Two phases:

  Phase 1 — Export (local, no GPU):
    python3 run_cc_msmarco_pragma_test_eval.py --export
    Reads  data/msmarco_pragma/cases_test.json  (143 test cases)
    Writes data/msmarco_pragma/cc_cases_test.json

  Phase 2 — Run (GPU):
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \\
      python3 run_cc_msmarco_pragma_test_eval.py --run --vllm
    Writes data/msmarco_pragma/cc_results_test_pct10.json
    Resume on case_id.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASET_PATH  = "data/msmarco-pragma-final8-with-lattices.json"
CASES_PATH    = "data/msmarco_pragma/cases_test.json"
CC_CASES_PATH = "data/msmarco_pragma/cc_cases_test.json"
OUT_PATH      = "data/msmarco_pragma/cc_results_test_pct10.json"

PCT: int = 10


def _pct_budget(k: int) -> int:
    return max(1, int(2**k * PCT / 100))


# ---------------------------------------------------------------------------
# Export (local)
# ---------------------------------------------------------------------------

def export_cases() -> None:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    lattice_by_qid: dict[str, dict] = {str(r["query_id"]): r for r in dataset}

    with open(CASES_PATH) as f:
        cases = json.load(f)

    cc_cases: list[dict] = []
    for cid, case in enumerate(cases):
        qid        = str(case["qid"])
        rec        = lattice_by_qid[qid]
        cfg        = rec["lattice"]["config"]
        start_node: list[str] = case["start_node"]

        pragma_set = set(start_node)
        full_rbc   = rec["lattice"]["answers"]
        sub_rbc    = {
            key: ranking for key, ranking in full_rbc.items()
            if all(p in pragma_set for p in (key.split(", ") if key else []))
        }

        cc_cases.append({
            "case_id":           cid,
            "query_id":          qid,
            "query":             cfg["question"],
            "sources":           cfg["sources"],
            "plausible_answers": cfg["plausible_answers"],
            "criteria":          start_node,
            "rankings_by_combo": sub_rbc,
            "start_node":        start_node,
            "effective_size":    case["size"],
        })

    os.makedirs(os.path.dirname(CC_CASES_PATH), exist_ok=True)
    with open(CC_CASES_PATH, "w") as f:
        json.dump(cc_cases, f, indent=2)
    print(f"Exported {len(cc_cases)} CC test cases → {CC_CASES_PATH}")
    sizes = defaultdict(int)
    for c in cc_cases:
        sizes[c["effective_size"]] += 1
    print("By size:", dict(sorted(sizes.items())))


# ---------------------------------------------------------------------------
# Run (GPU)
# ---------------------------------------------------------------------------

def run_eval() -> None:
    from scripts.shared.load_cc_model import add_model_args, load_model_and_tokenizer
    from search.contextcite import contextcite_search

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--run",  action="store_true")
    args = parser.parse_args()

    with open(CC_CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    try:
        with open(OUT_PATH) as f:
            results: list[dict] = json.load(f)
        done = {r["case_id"] for r in results}
        print(f"Resuming: {len(done)} cases already done")
    except FileNotFoundError:
        results = []
        done    = set()

    skipped = 0
    for case in cases:
        cid    = case["case_id"]
        k      = case["effective_size"]
        budget = _pct_budget(k)

        if cid in done:
            continue

        bfs_res = bfs_search(
            case["criteria"], case["rankings_by_combo"],
            start_node=case["start_node"], direction="down",
        )
        if not bfs_res["found"]:
            skipped += 1
            continue

        print(
            f"  [c{cid:>3} k={k} b={budget}] qid={case['query_id']} ...",
            end=" ", flush=True,
        )
        t0 = time.time()
        try:
            cc_res = contextcite_search(
                case["criteria"], case["rankings_by_combo"],
                case["query"], case["sources"], case["plausible_answers"],
                model, tokenizer,
                num_ablations=budget,
                batch_size=args.batch_size,
                start_node=case["start_node"],
                direction="down",
            )
        except Exception as e:
            print(f"ERROR: {e}")
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} rows before crash → {OUT_PATH}")
            raise
        elapsed = time.time() - t0

        scored = score_cc(bfs_res, cc_res, case["rankings_by_combo"], case["start_node"])
        results.append({
            "case_id":        cid,
            "query_id":       case["query_id"],
            "effective_size": k,
            "budget":         budget,
            "bfs_key":        bfs_res["key"],
            "bfs_layer":      bfs_res["size"],
            "bfs_calls":      bfs_res["calls"],
            "elapsed":        round(elapsed, 2),
            **scored,
        })
        done.add(cid)

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"\nSkipped {skipped} cases (no BFS CF)")

    correct = sum(1 for r in results if r["outcome"] == "correct")
    print(f"\nDone. {len(results)} cases  acc={correct/len(results):.1%} ({correct}/{len(results)})  → {OUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export",     action="store_true")
    parser.add_argument("--run",        action="store_true")
    parser.add_argument("--vllm",       action="store_true")
    parser.add_argument("--model",      default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.export:
        export_cases()
    elif args.run:
        run_eval()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
