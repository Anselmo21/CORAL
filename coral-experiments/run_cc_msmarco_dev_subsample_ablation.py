"""CC budget ablation on MSMARCO dev sub-lattice cases (GPU).

Modes:
  --export          Build data/msmarco_subsample/cc_cases.json locally first.
  --run             Run CC ablation on GPU.
  --pct N           Run only pct_N instead of full sweep.
  --shard N         Run only cases where case_id % num_shards == N (0-indexed).
  --num-shards N    Total number of shards (default 1).
  --merge           Merge shard files into cc_ablation_results.json.

Budget grid:
  Fixed:   [4, 8, 16, 32, 64, 128, 256]
  Percent: [10…100]% of 2^k  (budget = max(1, int(2^k * pct / 100)))

Writes:
  data/msmarco_subsample/cc_cases.json              — CC-ready export
  data/msmarco_subsample/cc_ablation_results.json   — merged results
  data/msmarco_subsample/cc_ablation_results_shard<N>.json — per-shard
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from cc_budget_ablation_dev import score_cc
from run_msmarco_subsample import _load_lattices, _load_split
from search.bfs import bfs_search

CASES_PATH            = "data/msmarco_subsample/cases.json"
CC_CASES_PATH         = "data/msmarco_subsample/cc_cases.json"
CC_ABLATION_RESULTS_PATH       = "data/msmarco_subsample/cc_ablation_results.json"
CC_ABLATION_SHARD_PATH_TMPL    = "data/msmarco_subsample/cc_ablation_results_shard{}.json"
DATA_PATH             = "data/msmarco_dataset.json"

BUDGETS_FIXED: list[int] = [4, 8, 16, 32, 64, 128, 256]
BUDGETS_PCT:   list[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _pct_budget(k: int, pct: int) -> int:
    return max(1, int(2**k * pct / 100))


def _all_budgets(k: int, pct_only: int | None = None) -> list[tuple[str, int]]:
    if pct_only is not None:
        return [(f"pct_{pct_only}", _pct_budget(k, pct_only))]
    out = [(f"fixed_{b}", b) for b in BUDGETS_FIXED]
    out += [(f"pct_{p}", _pct_budget(k, p)) for p in BUDGETS_PCT]
    return out


def _sublattice_rbc(
    rbc: dict[str, list[str]],
    start_node: list[str],
) -> dict[str, list[str]]:
    start_set = set(start_node)
    out: dict[str, list[str]] = {}
    for k, v in rbc.items():
        if not k:
            out[k] = v
            continue
        if set(k.split(", ")) <= start_set:
            out[k] = v
    return out


def export_cc_cases() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)
    raw_lats = data["lattices"]

    with open(CASES_PATH) as f:
        cases = json.load(f)

    dev_qids, _ = _load_split()
    lattices = _load_lattices(dev_qids)

    cc_cases: list[dict] = []
    skipped = 0
    for case_id, case in enumerate(cases):
        qid = case["qid"]
        start_node = case["start_node"]
        if qid not in lattices:
            skipped += 1
            continue
        criteria, rbc = lattices[qid]
        q = raw_lats[qid]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        if not bfs_res["found"]:
            skipped += 1
            continue

        sub_rbc = _sublattice_rbc(rbc, start_node)
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

    with open(CC_CASES_PATH, "w") as f:
        json.dump(cc_cases, f, indent=2)
    print(f"Exported {len(cc_cases)} CC cases → {CC_CASES_PATH}  (skipped {skipped})")


def print_summary(results: list[dict]) -> None:
    by_budget: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_budget[r["budget_label"]].append(r)

    print(f"\n{'='*90}")
    print(f"  CC Budget Ablation (MSMARCO dev sub-lattices) — {len(results)} rows")
    print(f"{'='*90}")

    all_labels = [f"fixed_{b}" for b in BUDGETS_FIXED] + [f"pct_{p}" for p in BUDGETS_PCT]
    for label in all_labels:
        rows = by_budget.get(label, [])
        if not rows:
            continue
        correct = sum(1 for r in rows if r["outcome"] == "correct")
        avg_cost = sum(r["calls"] + r["budget"] for r in rows) / len(rows)
        print(f"  {label:<12} n={len(rows):>3}  acc={correct/len(rows):>5.0%} ({correct:>3}/{len(rows):<3})  avg_cost={avg_cost:>7.1f}")


def merge_shards(num_shards: int) -> None:
    results: list[dict] = []
    for i in range(num_shards):
        path = CC_ABLATION_SHARD_PATH_TMPL.format(i)
        try:
            with open(path) as f:
                shard = json.load(f)
            results.extend(shard)
            print(f"  shard {i}: {len(shard)} rows")
        except FileNotFoundError:
            print(f"  shard {i}: MISSING ({path})")
    results.sort(key=lambda r: (r["case_id"], r["budget_label"]))
    with open(CC_ABLATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Merged {len(results)} rows → {CC_ABLATION_RESULTS_PATH}")
    print_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export CC cases locally (no GPU needed)")
    parser.add_argument("--run", action="store_true", help="Run CC ablation on GPU")
    parser.add_argument("--merge", action="store_true", help="Merge shard files into final results")
    parser.add_argument("--pct", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")

    try:
        from load_cc_model import add_model_args
        add_model_args(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    if args.export:
        export_cc_cases()
        return

    if args.merge:
        merge_shards(args.num_shards)
        return

    if not args.run:
        parser.print_help()
        return

    from load_cc_model import load_model_and_tokenizer

    with open(CC_CASES_PATH) as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if c["case_id"] % args.num_shards == args.shard]
    print(f"Shard {args.shard}/{args.num_shards}: {len(cases)}/{len(all_cases)} cases")

    model, tokenizer = load_model_and_tokenizer(args)
    from search.contextcite import contextcite_search

    out_path = (CC_ABLATION_SHARD_PATH_TMPL.format(args.shard)
                if args.num_shards > 1 else CC_ABLATION_RESULTS_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path) as f:
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
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved {len(results)} rows before crash → {out_path}")
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

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no sources or no BFS CF)")
    print(f"\nDone. {len(results)} rows → {out_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
