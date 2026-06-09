"""ERA v2 evaluation on data/test.jsonl (top-down rows only).

Mirrors run_era_test_eval.py but calls era_search_v2 (margin-based pruning,
margin_threshold=0.25). Runs two configurations with weighted entropy:
  1. Phase-1 budget = 10% of lattice size (2^k)
  2. Phase-1 budget = 64 (fixed)

Usage:
  conda run -n coral python3 run_era_v2_test_eval.py
"""

import json
import os
from collections import Counter, defaultdict

from era_budget_ablation import (
    LATTICES_PATH,
    TEST_CASES_PATH,
    budget_from_pct,
    effective_size,
    load_manifest_cases,
)
from experiment import score_proxy
from search.bfs import bfs_search
from search.heuristics import weighted_entropy
from search.rank import era_search_v2

OUT_DIR = "data/era_ablation_study/test"
RESULTS_PATH = os.path.join(OUT_DIR, "era_v2_test_topdown_results.json")

CONFIGS: list[tuple[str, int | None]] = [
    ("pct_10", 10),
    ("budget_64", 64),
]


def run_eval(cases: list[dict]) -> list[dict]:
    results: list[dict] = []

    for i, case in enumerate(cases):
        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]
        eff = effective_size(case)

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)

        row: dict = {
            "case_index": i,
            "query_id": case["query_id"],
            "effective_size": eff,
            "k": case["k"],
            "direction": direction,
            "fixed_criteria": case["fixed_criteria"],
            "toggleable_criteria": case["toggleable_criteria"],
            "bfs_found": bfs_res["found"],
            "bfs_key": bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
        }

        for label, spec in CONFIGS:
            if label == "pct_10":
                budget = budget_from_pct(eff, spec)
                budget_pct = spec
            else:
                budget = spec
                budget_pct = None

            era_res = era_search_v2(
                criteria,
                rbc,
                budget=budget,
                entropy_fn=weighted_entropy,
                start_node=start_node,
                direction=direction,
            )
            scored = score_proxy(bfs_res, era_res, rbc, start_node)
            scored["calls"] = era_res["calls"]
            scored["budget_used"] = era_res.get("budget_used", 0)
            scored["budget_allocated"] = budget
            scored["budget_pct"] = budget_pct
            scored["pruned_nodes"] = era_res.get("pruned_nodes", 0)
            row[label] = scored

        results.append(row)
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(cases)} cases")

    return results


def print_summary(results: list[dict]) -> None:
    n = len(results)
    bfs_found = sum(1 for r in results if r["bfs_found"])
    print(f"\nCases: {n}  BFS found CF: {bfs_found}/{n}")

    for label, _ in CONFIGS:
        outcomes = Counter(r[label]["outcome"] for r in results)
        correct = outcomes.get("correct", 0)
        jaccs = [r[label]["jaccard"] for r in results if r[label].get("jaccard") is not None]
        mean_j = sum(jaccs) / len(jaccs) if jaccs else float("nan")
        mean_alloc = sum(r[label]["budget_allocated"] for r in results) / n
        mean_calls = sum(r[label]["calls"] for r in results) / n
        mean_used = sum(r[label]["budget_used"] for r in results) / n
        mean_pruned = sum(r[label].get("pruned_nodes", 0) for r in results) / n
        print(f"\n  [{label}]  accuracy={correct/n:.1%} ({correct}/{n})")
        print(f"    outcomes: {dict(outcomes)}")
        print(f"    mean_jaccard={mean_j:.3f}  mean_allocated={mean_alloc:.1f}  "
              f"mean_budget_used={mean_used:.1f}  mean_phase2_calls={mean_calls:.1f}  "
              f"mean_pruned={mean_pruned:.1f}")

    print("\n  By effective_size (accuracy):")
    by_k: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_k[r["effective_size"]].append(r)
    header = f"  {'k':>3} | {'N':>3} |"
    for label, _ in CONFIGS:
        header += f" {label:>10} |"
    print(header)
    for k in sorted(by_k):
        bucket = by_k[k]
        nk = len(bucket)
        line = f"  {k:>3} | {nk:>3} |"
        for label, _ in CONFIGS:
            c = sum(1 for r in bucket if r[label]["outcome"] == "correct")
            line += f" {c/nk:>9.0%} |"
        print(line)


def main() -> None:
    cases = load_manifest_cases(
        LATTICES_PATH,
        TEST_CASES_PATH,
        manifest_directions=["top-down"],
    )
    print(f"Loaded {len(cases)} top-down test cases from {TEST_CASES_PATH}")
    print("ERA v2: weighted_entropy, margin_threshold=0.25, configs: 10% lattice budget, fixed budget=64")

    results = run_eval(cases)
    print_summary(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
