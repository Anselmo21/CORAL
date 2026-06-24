"""ERA v2 vs ERA v1 on the MSMARCO + Sonnet test sub-lattice cases.

Default mode: runs `era_search_v2` (margin-based pruning, margin_threshold=0.25)
at budget=16 / softmax entropy / direction=down, head-to-head against the
existing v1 test results.

`--sweep` mode: runs era_search_v2 across the full fixed-budget grid
`[4, 8, 16, 32, 64, 128, 256]` per case, then prints an accuracy-by-size ×
budget table (the v2 analogue of the dev v1 sweep). This is the table to
inspect for "is v2's accuracy monotonic in budget".

Reads:
  data/{msmarco,sonnet}_test_subsample/cases.json   (case manifests)
  data/{msmarco,sonnet}_test_subsample/results.json (v1 era + bfs)

Writes (default):
  data/{msmarco,sonnet}_test_subsample/era_v2_results.json
Writes (--sweep):
  data/{msmarco,sonnet}_test_subsample/era_v2_budget_sweep.json

CLI:
  conda run -n coral python3 -m scripts.cross_dataset.run_era_v2_test_subsample
  conda run -n coral python3 -m scripts.cross_dataset.run_era_v2_test_subsample --sweep
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Callable

from scripts.msmarco.run_subsample import (
    _load_lattices as msmarco_load_lattices,
    _load_split as msmarco_load_split,
    score_proxy as msmarco_score,
)
from scripts.sonnet.run_subsample import (
    _load_lattices as sonnet_load_lattices,
    _load_split as sonnet_load_split,
    score_proxy as sonnet_score,
)
from search.rank import era_search, era_search_v2

ERA_BUDGET = 16
SWEEP_BUDGETS = [4, 8, 16, 32, 64, 128, 256]

DATASETS = [
    {
        "name": "msmarco",
        "cases_path": "data/msmarco_test_subsample/cases.json",
        "v1_results_path": "data/msmarco_test_subsample/results.json",
        "out_path": "data/msmarco_test_subsample/era_v2_results.json",
        "sweep_out_path": "data/msmarco_test_subsample/era_v2_budget_sweep.json",
        "load_split": msmarco_load_split,
        "load_lattices": msmarco_load_lattices,
        "score_proxy": msmarco_score,
    },
    {
        "name": "sonnet",
        "cases_path": "data/sonnet_test_subsample/cases.json",
        "v1_results_path": "data/sonnet_test_subsample/results.json",
        "out_path": "data/sonnet_test_subsample/era_v2_results.json",
        "sweep_out_path": "data/sonnet_test_subsample/era_v2_budget_sweep.json",
        "load_split": sonnet_load_split,
        "load_lattices": sonnet_load_lattices,
        "score_proxy": sonnet_score,
    },
]


def run_dataset(ds: dict) -> list[dict]:
    with open(ds["cases_path"]) as f:
        cases = json.load(f)
    with open(ds["v1_results_path"]) as f:
        v1_data = json.load(f)
    v1_by_case = {r["case_id"]: r for r in v1_data["results"]}

    _, test_qids = ds["load_split"]()
    lattices = ds["load_lattices"](test_qids)

    score_proxy: Callable = ds["score_proxy"]
    results: list[dict] = []
    for case_id, case in enumerate(cases):
        qid = case["qid"]
        if qid not in lattices:
            continue
        v1_row = v1_by_case.get(case_id)
        if v1_row is None or "bfs" not in v1_row:
            continue

        criteria, rbc = lattices[qid]
        start_node = case["start_node"]

        bfs_for_proxy = {
            "key": v1_row["bfs"]["key"],
            "size": v1_row["bfs"]["layer"],
        }
        era_v2_res = era_search_v2(
            criteria,
            rbc,
            budget=ERA_BUDGET,
            start_node=start_node,
            direction="down",
        )
        scored = score_proxy(bfs_for_proxy, era_v2_res, rbc, start_node)

        results.append({
            "case_id": case_id,
            "qid": qid,
            "size": case["size"],
            "start_node": start_node,
            "bfs": v1_row["bfs"],
            "era_v1": v1_row.get("era"),
            "era_v2": scored,
        })

        if (case_id + 1) % 25 == 0:
            print(f"  {ds['name']:>8} ... {case_id + 1}/{len(cases)}", flush=True)

    with open(ds["out_path"], "w") as f:
        json.dump({"era_budget": ERA_BUDGET, "results": results}, f, indent=2)
    print(f"  -> {ds['out_path']}")
    return results


def _cost(d: dict) -> float:
    return d["calls"] + d.get("budget_used", 0)


def print_summary(name: str, results: list[dict]) -> None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["size"]].append(r)

    print(f"\n{'=' * 72}")
    print(f"  {name.upper()} TEST  (budget={ERA_BUDGET}, softmax entropy, direction=down)")
    print(f"{'=' * 72}")
    print(f"  {'size':>4} {'n':>3} | {'v1 acc':>7} {'v1 cost':>8} | {'v2 acc':>7} {'v2 cost':>8} | {'Δ acc':>6}")
    print(f"  {'-' * 4} {'-' * 3}-+-{'-' * 7} {'-' * 8}-+-{'-' * 7} {'-' * 8}-+-{'-' * 6}")
    for size in sorted(by_size):
        bucket = by_size[size]
        n = len(bucket)
        v1_c = sum(1 for r in bucket if r["era_v1"]["outcome"] == "correct")
        v2_c = sum(1 for r in bucket if r["era_v2"]["outcome"] == "correct")
        v1_cost = sum(_cost(r["era_v1"]) for r in bucket) / n
        v2_cost = sum(_cost(r["era_v2"]) for r in bucket) / n
        delta = (v2_c - v1_c) / n
        print(f"  {size:>4} {n:>3} | {v1_c/n:>6.0%}  {v1_cost:>7.1f} | "
              f"{v2_c/n:>6.0%}  {v2_cost:>7.1f} | {delta:>+5.0%}")

    n = len(results)
    v1_c = sum(1 for r in results if r["era_v1"]["outcome"] == "correct")
    v2_c = sum(1 for r in results if r["era_v2"]["outcome"] == "correct")
    v1_cost = sum(_cost(r["era_v1"]) for r in results) / n
    v2_cost = sum(_cost(r["era_v2"]) for r in results) / n
    print(f"\n  overall n={n} | v1 acc={v1_c/n:.1%} cost={v1_cost:.1f} | "
          f"v2 acc={v2_c/n:.1%} cost={v2_cost:.1f}")

    v1_correct_v2_wrong = [r for r in results
                            if r["era_v1"]["outcome"] == "correct"
                            and r["era_v2"]["outcome"] != "correct"]
    v2_correct_v1_wrong = [r for r in results
                            if r["era_v2"]["outcome"] == "correct"
                            and r["era_v1"]["outcome"] != "correct"]
    print(f"  flip v1->v2: lost {len(v1_correct_v2_wrong)}, gained {len(v2_correct_v1_wrong)}")


def run_dataset_sweep(ds: dict) -> list[dict]:
    with open(ds["cases_path"]) as f:
        cases = json.load(f)
    with open(ds["v1_results_path"]) as f:
        v1_data = json.load(f)
    v1_by_case = {r["case_id"]: r for r in v1_data["results"]}

    _, test_qids = ds["load_split"]()
    lattices = ds["load_lattices"](test_qids)

    score_proxy: Callable = ds["score_proxy"]
    results: list[dict] = []
    for case_id, case in enumerate(cases):
        qid = case["qid"]
        if qid not in lattices:
            continue
        v1_row = v1_by_case.get(case_id)
        if v1_row is None or "bfs" not in v1_row:
            continue

        criteria, rbc = lattices[qid]
        start_node = case["start_node"]
        bfs_for_proxy = {
            "key": v1_row["bfs"]["key"],
            "size": v1_row["bfs"]["layer"],
        }

        era_v1_by_budget: dict[str, dict] = {}
        era_v2_by_budget: dict[str, dict] = {}
        for budget in SWEEP_BUDGETS:
            v1_res = era_search(
                criteria, rbc,
                budget=budget, start_node=start_node, direction="down",
            )
            v2_res = era_search_v2(
                criteria, rbc,
                budget=budget, start_node=start_node, direction="down",
            )
            label = f"era_fixed_{budget}"
            era_v1_by_budget[label] = score_proxy(bfs_for_proxy, v1_res, rbc, start_node)
            era_v2_by_budget[label] = score_proxy(bfs_for_proxy, v2_res, rbc, start_node)

        results.append({
            "case_id": case_id,
            "qid": qid,
            "size": case["size"],
            "start_node": start_node,
            "bfs": v1_row["bfs"],
            "era_v1": era_v1_by_budget,
            "era_v2": era_v2_by_budget,
        })

        if (case_id + 1) % 25 == 0:
            print(f"  {ds['name']:>8} sweep ... {case_id + 1}/{len(cases)}", flush=True)

    with open(ds["sweep_out_path"], "w") as f:
        json.dump({"budgets": SWEEP_BUDGETS, "results": results}, f, indent=2)
    print(f"  -> {ds['sweep_out_path']}")
    return results


def _accuracy_table(results: list[dict], version_key: str) -> dict[int, dict[int, tuple[int, int]]]:
    by_size_budget: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for r in results:
        size = r["size"]
        era = r.get(version_key, {})
        for budget in SWEEP_BUDGETS:
            scored = era.get(f"era_fixed_{budget}")
            if scored is None:
                continue
            by_size_budget[size][budget][1] += 1
            if scored["outcome"] == "correct":
                by_size_budget[size][budget][0] += 1
    return {s: {b: tuple(v) for b, v in bm.items()} for s, bm in by_size_budget.items()}


def _print_single_table(title: str, table: dict[int, dict[int, tuple[int, int]]]) -> None:
    print(f"\n  {title}")
    print(f"  {'size':>4} " + " ".join(f"{b:>7}" for b in SWEEP_BUDGETS))
    for size in sorted(table):
        row = [f"  {size:>4}"]
        for b in SWEEP_BUDGETS:
            c, t = table[size].get(b, (0, 0))
            row.append(f"{c/t*100:>6.1f}%" if t else "    -   ")
        print(" ".join(row))
    overall = {b: [0, 0] for b in SWEEP_BUDGETS}
    for bm in table.values():
        for b, (c, t) in bm.items():
            overall[b][0] += c
            overall[b][1] += t
    row = ["  all "]
    for b in SWEEP_BUDGETS:
        c, t = overall[b]
        row.append(f"{c/t*100:>6.1f}%" if t else "    -   ")
    print(" ".join(row))


def _non_monotonic_count(results: list[dict], version_key: str) -> int:
    n = 0
    for r in results:
        era = r.get(version_key, {})
        flags = [era.get(f"era_fixed_{b}", {}).get("outcome") == "correct"
                 for b in SWEEP_BUDGETS]
        first = next((i for i, f in enumerate(flags) if f), None)
        if first is None:
            continue
        if any(not f for f in flags[first:]):
            n += 1
    return n


def print_sweep_summary(name: str, results: list[dict]) -> None:
    v1_table = _accuracy_table(results, "era_v1")
    v2_table = _accuracy_table(results, "era_v2")

    print(f"\n{'=' * 78}")
    print(f"  {name.upper()} TEST  accuracy-by-size × budget  (softmax, top-down)")
    print(f"{'=' * 78}")
    _print_single_table("ERA v1", v1_table)
    _print_single_table("ERA v2", v2_table)

    # Aggregate v1 vs v2 deltas at each budget
    print(f"\n  v2 − v1 (overall accuracy at each budget)")
    print(f"  {'budget':>6} | {'v1':>6} {'v2':>6} {'Δ':>6}")
    for b in SWEEP_BUDGETS:
        v1c = sum(v1_table[s].get(b, (0, 0))[0] for s in v1_table)
        v1t = sum(v1_table[s].get(b, (0, 0))[1] for s in v1_table)
        v2c = sum(v2_table[s].get(b, (0, 0))[0] for s in v2_table)
        v2t = sum(v2_table[s].get(b, (0, 0))[1] for s in v2_table)
        v1a = v1c / v1t if v1t else 0.0
        v2a = v2c / v2t if v2t else 0.0
        print(f"  {b:>6} | {v1a:>5.1%} {v2a:>5.1%} {v2a - v1a:>+5.1%}")

    # Non-monotonicity on same cases
    v1_nm = _non_monotonic_count(results, "era_v1")
    v2_nm = _non_monotonic_count(results, "era_v2")
    print(f"\n  non-monotonic cases (correct at some smaller b, wrong at some larger b):")
    print(f"    v1: {v1_nm}/{len(results)}")
    print(f"    v2: {v2_nm}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="run full budget grid [4,8,16,32,64,128,256] per case")
    args = parser.parse_args()

    all_results: list[tuple[str, list[dict]]] = []
    for ds in DATASETS:
        print(f"\n>>> {ds['name']}")
        if args.sweep:
            results = run_dataset_sweep(ds)
        else:
            results = run_dataset(ds)
        all_results.append((ds["name"], results))

    for name, results in all_results:
        if args.sweep:
            print_sweep_summary(name, results)
        else:
            print_summary(name, results)


if __name__ == "__main__":
    main()
