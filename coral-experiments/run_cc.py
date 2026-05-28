"""
CC runner — run on GPU machine, reads cc_cases.json, writes cc_results.json.

Usage:
    conda run -n coral python3 run_cc.py [--model META_LLAMA_PATH] [--batch-size 1]

Requires:
    - cc_cases.json (exported by experiment.py)
    - baseline_lattice_runs_skeleton_llama31_8b_local.json (for rankings_by_combo)
    - base-q.json (for query/sources — already embedded in cc_cases.json)
    - HuggingFace model + tokenizer
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from search.bfs import bfs_search
from search.contextcite import contextcite_search
from search.lattice import combo_key

CC_CASES_PATH = "data/cc_cases.json"
CC_RESULTS_PATH = "data/cc_results.json"
DATA_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"


def load_lattice_data(path: str) -> dict[int, dict]:
    with open(path) as f:
        data = json.load(f)
    lookup: dict[int, dict] = {}
    for run in data["runs"]:
        qid = run["query_id"]
        if qid not in lookup:
            lookup[qid] = {}
        key = (
            len(run["criteria_used"]),
            tuple(sorted(run["criteria_used"])),
        )
        lookup.setdefault(qid, {})[key] = run
    return lookup


def find_run(
    lattice_lookup: dict, query_id: int, criteria: list[str], parent_size: int
) -> dict | None:
    qid_runs = lattice_lookup.get(query_id, {})
    for (n, _crit_key), run in qid_runs.items():
        if n == parent_size and set(run["criteria_used"]) >= set(criteria):
            return run
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ContextCite on experiment cases")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model path",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-ablations", type=int, default=64)
    args = parser.parse_args()

    with open(CC_CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}")

    lattice_lookup = load_lattice_data(DATA_PATH)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )
    print("Model loaded.")

    results: list[dict] = []
    try:
        with open(CC_RESULTS_PATH) as f:
            results = json.load(f)
        done_ids = {r["case_id"] for r in results}
        print(f"Resuming: {len(done_ids)} cases already done")
    except FileNotFoundError:
        done_ids = set()

    for case in cases:
        if case["case_id"] in done_ids:
            continue

        qid = case["query_id"]
        start_node = case["start_node"]
        direction = case["direction"]
        parent_size = case["parent_size"]
        query = case["query"]
        sources = case["sources"]

        run = find_run(lattice_lookup, qid, start_node, parent_size)
        if run is None:
            print(f"  [SKIP] case {case['case_id']}: no matching run found")
            continue

        criteria = run["criteria_used"]
        rbc = run["rankings_by_combo"]
        answers = run["plausible_answers"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        if not bfs_res["found"]:
            print(f"  [SKIP] case {case['case_id']}: BFS found no CF")
            continue

        print(f"  [{case['case_id']:>3}] Q{qid} size={case['effective_size']} {direction}...", end=" ", flush=True)
        t0 = time.time()

        cc_res = contextcite_search(
            criteria, rbc, query, sources, answers,
            model, tokenizer,
            num_ablations=args.num_ablations,
            batch_size=args.batch_size,
            start_node=start_node,
            direction=direction,
        )

        elapsed = time.time() - t0

        start_key = combo_key(start_node)
        baseline_top1 = (rbc.get(start_key) or [""])[0]

        proxy_key = cc_res.get("key")
        proxy_layer = len(proxy_key.split(", ")) if proxy_key else None

        if not cc_res.get("found") or proxy_key is None:
            outcome = "not_found"
            proxy_verified = False
        else:
            ranking = rbc.get(proxy_key, [])
            proxy_verified = bool(ranking) and ranking[0] != baseline_top1
            if not proxy_verified:
                outcome = "false_positive"
            elif proxy_layer != bfs_res["size"]:
                outcome = "wrong_layer"
            else:
                outcome = "correct"

        gap: int | None = None
        if outcome == "correct":
            gap = 0
        elif outcome == "wrong_layer" and proxy_layer is not None:
            gap = abs(bfs_res["size"] - proxy_layer)

        result = {
            "case_id": case["case_id"],
            "query_id": qid,
            "effective_size": case["effective_size"],
            "direction": direction,
            "key": proxy_key,
            "proxy_layer": proxy_layer,
            "proxy_verified": proxy_verified,
            "outcome": outcome,
            "gap": gap,
            "calls": cc_res["calls"],
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "elapsed": round(elapsed, 2),
        }
        results.append(result)
        done_ids.add(case["case_id"])

        tag = "OK" if outcome == "correct" else outcome.upper()
        print(f"{tag} ({elapsed:.1f}s, {cc_res['calls']} calls)")

        with open(CC_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} results saved to {CC_RESULTS_PATH}")

    correct = sum(1 for r in results if r["outcome"] == "correct")
    print(f"Accuracy: {correct}/{len(results)} ({correct / len(results):.1%})")


if __name__ == "__main__":
    main()
