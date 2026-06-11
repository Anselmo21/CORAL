"""CC budget ablation on the MSMARCO test sub-lattice cases (GPU).

Sweeps the same budget grid as the ERA ablation:
  Fixed:   [4, 8, 16, 32, 64, 128, 256]
  Percent: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]% of 2^k

Reads  data/msmarco_test_subsample/cc_cases.json
       (exported by run_msmarco_test_subsample.py).
Writes data/msmarco_test_subsample/cc_ablation_results.json
       with resume on (case_id, budget_label).

CLI: conda run -n coral python3 run_cc_msmarco_test_ablation.py [--model PATH] [--vllm] [--batch-size 1]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

CC_CASES_PATH = "data/msmarco_test_subsample/cc_cases.json"
CC_ABLATION_RESULTS_PATH = "data/msmarco_test_subsample/cc_ablation_results.json"

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _pct_budget(k: int, pct: int) -> int:
    return max(1, int(2**k * pct / 100))


def _all_budgets(k: int, pct_only: int | None = None) -> list[tuple[str, int]]:
    if pct_only is not None:
        return [(f"pct_{pct_only}", _pct_budget(k, pct_only))]
    out = [(f"fixed_{b}", b) for b in BUDGETS_FIXED]
    out += [(f"pct_{p}", _pct_budget(k, p)) for p in BUDGETS_PCT]
    return out


def print_summary(results: list[dict]) -> None:
    by_budget: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_budget[r["budget_label"]].append(r)

    print(f"\n{'='*90}")
    print(f"  CC Budget Ablation (MSMARCO test) — {len(results)} rows, {len(by_budget)} budget configs")
    print(f"{'='*90}")

    all_labels = [f"fixed_{b}" for b in BUDGETS_FIXED] + [f"pct_{p}" for p in BUDGETS_PCT]
    for label in all_labels:
        rows = by_budget.get(label, [])
        if not rows:
            continue
        correct = sum(1 for r in rows if r["outcome"] == "correct")
        avg_cost = sum(r["calls"] + r["budget"] for r in rows) / len(rows)
        print(f"  {label:<12} n={len(rows):>3}  acc={correct/len(rows):>5.0%} ({correct:>3}/{len(rows):<3})  avg_cost={avg_cost:>7.1f}")


def main() -> None:
    from load_cc_model import add_model_args, load_model_and_tokenizer
    from search.contextcite import contextcite_search

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--pct", type=int, default=None,
                        help="Run only this percent budget (e.g. 10 for pct_10). Omit for full sweep.")
    args = parser.parse_args()

    with open(CC_CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(CC_ABLATION_RESULTS_PATH), exist_ok=True)
    try:
        with open(CC_ABLATION_RESULTS_PATH) as f:
            results: list[dict] = json.load(f)
        done = {(r["case_id"], r["budget_label"]) for r in results}
        print(f"Resuming: {len(done)} (case, budget) pairs already done")
    except FileNotFoundError:
        results = []
        done = set()

    skipped = 0
    for case in cases:
        cid = case["case_id"]
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

        for label, num_ablations in _all_budgets(k, pct_only=args.pct):
            if (cid, label) in done:
                continue

            print(
                f"  [c{cid:>3} {label:<12}] qid={case['query_id']} k={k} ...",
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
                with open(CC_ABLATION_RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved {len(results)} rows before crash → {CC_ABLATION_RESULTS_PATH}")
                raise
            elapsed = time.time() - t0

            scored = score_cc(bfs_res, cc_res, rbc, start_node)
            results.append({
                "case_id": cid,
                "query_id": case["query_id"],
                "effective_size": k,
                "budget_label": label,
                "budget": num_ablations,
                "bfs_key": bfs_res["key"],
                "bfs_layer": bfs_res["size"],
                "bfs_calls": bfs_res["calls"],
                "elapsed": round(elapsed, 2),
                **scored,
            })
            done.add((cid, label))

            tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
            print(f"{tag} ({elapsed:.1f}s)")

            with open(CC_ABLATION_RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no sources or no BFS CF)")
    print(f"\nDone. {len(results)} rows → {CC_ABLATION_RESULTS_PATH}")
    print_summary(results)


if __name__ == "__main__":
    main()
