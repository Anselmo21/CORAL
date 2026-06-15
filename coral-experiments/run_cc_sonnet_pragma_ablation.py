"""CC budget ablation on the Sonnet pragma lattice dataset (GPU).

Loads data/sonnet-pragma-dataset.json — 16 records, each a full 10-pragma
top-down lattice with 1024 pre-computed combo rankings. Halted records are
skipped (id=45), leaving 15 usable cases.

The lattice dimensions are the `pragmas` (temporal / geographic / perspective /
trustworthiness conditions). `sources` is a single joined string ready for CC.
Answer keys are already in combo_key format (sorted, ", "-joined).

Sweeps the same budget grid as the other ablation studies:
  Fixed:   [4, 8, 16, 32, 64, 128, 256]
  Percent: [10, 20, ..., 100]% of 2^10 = 1024

Writes data/sonnet_pragma/cc_ablation_results.json with resume on
(query_id, budget_label).

CLI:
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \\
    python3 run_cc_sonnet_pragma_ablation.py --vllm [--model PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASET_PATH = "data/sonnet-pragma-dataset.json"
SPLIT_PATH = "data/sonnet_pragma/split.json"
OUT_PATH = "data/sonnet_pragma/cc_ablation_results.json"

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT: list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EFFECTIVE_SIZE = 10


def _pct_budget(pct: int) -> int:
    return max(1, int(2**EFFECTIVE_SIZE * pct / 100))


def _all_budgets() -> list[tuple[str, int]]:
    out = [(f"fixed_{b}", b) for b in BUDGETS_FIXED]
    out += [(f"pct_{p}", _pct_budget(p)) for p in BUDGETS_PCT]
    return out


def load_cases(dataset_path: str, split_path: str) -> list[dict]:
    with open(dataset_path) as f:
        data = json.load(f)
    with open(split_path) as f:
        dev_qids = set(json.load(f)["dev"])
    cases = []
    for rec in data:
        if str(rec["id"]) not in dev_qids:
            continue
        cfg = rec["lattice"]["config"]
        cases.append({
            "query_id": str(rec["id"]),
            "query": cfg["question"],
            "sources": cfg["sources"],
            "criteria": cfg["criteria"],
            "plausible_answers": cfg["plausible_answers"],
            "rankings_by_combo": rec["lattice"]["answers"],
            "start_node": cfg["criteria"],
        })
    return cases


def print_summary(results: list[dict]) -> None:
    by_budget: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_budget[r["budget_label"]].append(r)

    print(f"\n{'='*90}")
    print(f"  CC Budget Ablation — Sonnet pragma — {len(results)} rows, {len(by_budget)} budget configs")
    print(f"{'='*90}")

    all_labels = [f"fixed_{b}" for b in BUDGETS_FIXED] + [f"pct_{p}" for p in BUDGETS_PCT]
    for label in all_labels:
        rows = by_budget.get(label, [])
        if not rows:
            continue
        correct = sum(1 for r in rows if r["outcome"] == "correct")
        avg_cost = sum(r["calls"] + r["budget"] for r in rows) / len(rows)
        print(
            f"  {label:<12} n={len(rows):>3}  acc={correct/len(rows):>5.0%} "
            f"({correct:>3}/{len(rows):<3})  avg_cost={avg_cost:>7.1f}"
        )


def main() -> None:
    from load_cc_model import add_model_args, load_model_and_tokenizer
    from search.contextcite import contextcite_search

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    args = parser.parse_args()

    cases = load_cases(DATASET_PATH, SPLIT_PATH)
    print(f"Loaded {len(cases)} dev cases from {DATASET_PATH} (split: {SPLIT_PATH})")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    try:
        with open(OUT_PATH) as f:
            results: list[dict] = json.load(f)
        done = {(r["query_id"], r["budget_label"]) for r in results}
        print(f"Resuming: {len(done)} (query_id, budget) pairs already done")
    except FileNotFoundError:
        results = []
        done = set()

    skipped = 0
    for case in cases:
        qid = case["query_id"]
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            print(f"  Skipping qid={qid}: BFS found no CF")
            skipped += 1
            continue

        for label, num_ablations in _all_budgets():
            if (qid, label) in done:
                continue

            print(f"  [qid={qid} {label:<12}] ...", end=" ", flush=True)
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
                with open(OUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved {len(results)} rows before crash → {OUT_PATH}")
                raise
            elapsed = time.time() - t0

            scored = score_cc(bfs_res, cc_res, rbc, start_node)
            results.append({
                "query_id": qid,
                "effective_size": EFFECTIVE_SIZE,
                "budget_label": label,
                "budget": num_ablations,
                "bfs_key": bfs_res["key"],
                "bfs_layer": bfs_res["size"],
                "bfs_calls": bfs_res["calls"],
                "elapsed": round(elapsed, 2),
                **scored,
            })
            done.add((qid, label))

            tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
            print(f"{tag} ({elapsed:.1f}s)")

            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    if skipped:
        print(f"\nSkipped {skipped} cases")
    print(f"\nDone. {len(results)} rows → {OUT_PATH}")
    print_summary(results)


if __name__ == "__main__":
    main()
