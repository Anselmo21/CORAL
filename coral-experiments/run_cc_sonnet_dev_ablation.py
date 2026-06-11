"""CC budget ablation on the sonnet dev sub-lattice cases.

Mirrors the ERA budget ablation (run_sonnet_subsample.py) on the dev set.

Two phases:

  Phase 1 — Local export (run on this machine):
    conda run -n coral python3 run_cc_sonnet_dev_ablation.py --export
    Reads  data/sonnet_subsample/cases.json  and writes
           data/sonnet_subsample/cc_cases.json

  Phase 2 — GPU run:
    conda run -n coral python3 run_cc_sonnet_dev_ablation.py --run [--model PATH] [--batch-size 8]
    Sweeps fixed budgets [4,8,16,32,64,128,256] and pct budgets [10..100]% of 2^k.
    Writes data/sonnet_subsample/cc_ablation_results.json with resume on (case_id, budget_label).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from cc_budget_ablation_dev import score_cc
from run_cc_sonnet_test_ablation import BUDGETS_FIXED, BUDGETS_PCT, _all_budgets, print_summary
from run_sonnet_subsample import _load_lattices, _load_split, _parse_combo_key
from search.bfs import bfs_search
from search.lattice import combo_key

DATA_PATH = "data/sonnet_dataset.json"
DEV_CASES_PATH = "data/sonnet_subsample/cases.json"
CC_CASES_PATH = "data/sonnet_subsample/cc_cases.json"
CC_ABLATION_RESULTS_PATH = "data/sonnet_subsample/cc_ablation_results.json"


# ---------------------------------------------------------------------------
# Export (local)
# ---------------------------------------------------------------------------

def _sublattice_rbc(
    rbc: dict[str, list[str]],
    start_node: list[str],
    full_criteria: list[str],
) -> dict[str, list[str]]:
    start_set = set(start_node)
    sorted_full = sorted(full_criteria)
    out: dict[str, list[str]] = {}
    for k, v in rbc.items():
        if not k:
            out[k] = v
            continue
        parts = _parse_combo_key(k, sorted_full)
        if parts and set(parts) <= start_set:
            out[k] = v
    return out


def export_cases() -> None:
    with open(DEV_CASES_PATH) as f:
        cases = json.load(f)
    with open(DATA_PATH) as f:
        data = json.load(f)

    dev_qids, _ = _load_split()
    lattices = _load_lattices(dev_qids)

    cc_cases: list[dict] = []
    for case_id, case in enumerate(cases):
        qid = case["qid"]
        start_node = case["start_node"]
        criteria, rbc = lattices[qid]
        q = data[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            continue

        sub_rbc = _sublattice_rbc(rbc, start_node, criteria)
        cc_cases.append({
            "case_id": case_id,
            "query_id": qid,
            "query": q["question"],
            "sources": q.get("sources") or "",
            "plausible_answers": list(q.get("plausible_answers") or []),
            "criteria": start_node,
            "start_node": start_node,
            "direction": "down",
            "effective_size": len(start_node),
            "k": len(start_node),
            "rankings_by_combo": sub_rbc,
        })

    os.makedirs(os.path.dirname(CC_CASES_PATH), exist_ok=True)
    with open(CC_CASES_PATH, "w") as f:
        json.dump(cc_cases, f, indent=2)

    # Verify no broken cases
    bad = [c["case_id"] for c in cc_cases
           if combo_key(c["start_node"]) not in c["rankings_by_combo"]]
    print(f"Exported {len(cc_cases)} dev CC cases → {CC_CASES_PATH}")
    if bad:
        print(f"  WARNING: {len(bad)} cases missing start_node key: {bad}")
    else:
        print(f"  All start_node keys verified present.")


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------

def run_ablation(args: argparse.Namespace) -> None:
    from search.contextcite import contextcite_search
    from load_cc_model import load_model_and_tokenizer

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

        for label, num_ablations in _all_budgets(k):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true",
                      help="Local: build data/sonnet_subsample/cc_cases.json")
    mode.add_argument("--run", action="store_true",
                      help="GPU: run CC budget ablation sweep")
    from load_cc_model import add_model_args
    add_model_args(parser)
    args = parser.parse_args()

    if args.export:
        export_cases()
    else:
        run_ablation(args)


if __name__ == "__main__":
    main()
