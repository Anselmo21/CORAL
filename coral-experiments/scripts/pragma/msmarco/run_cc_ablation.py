"""CC budget ablation on the MSMARCO pragma sub-lattice cases (GPU).

Two phases:

  Phase 1 — Export (local, no GPU):
    python3 -m scripts.pragma.msmarco.run_cc_ablation --export
    Reads  data/msmarco_pragma/cases.json
    Writes data/msmarco_pragma/cc_cases.json

  Phase 2 — Run (GPU), optionally sharded:
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \\
      python3 -m scripts.pragma.msmarco.run_cc_ablation --run --vllm

    # Multi-GPU shards (one terminal per GPU):
    python3 -m scripts.pragma.msmarco.run_cc_ablation --run --vllm --shard 0 --num-shards 3
    python3 -m scripts.pragma.msmarco.run_cc_ablation --run --vllm --shard 1 --num-shards 3
    python3 -m scripts.pragma.msmarco.run_cc_ablation --run --vllm --shard 2 --num-shards 3

    # After shards finish, merge:
    python3 -m scripts.pragma.msmarco.run_cc_ablation --merge

Sweeps fixed [4,8,16,32,64,128,256] and pct [10..100]% of 2^k budgets. Each shard
writes data/msmarco_pragma/cc_ablation_results_shard{N}.json; merged output is
data/msmarco_pragma/cc_ablation_results.json. Resume on (case_id, budget_label).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASET_PATH   = "data/msmarco-pragma-final8-with-lattices.json"
CASES_PATH     = "data/msmarco_pragma/cases.json"
CC_CASES_PATH  = "data/msmarco_pragma/cc_cases.json"
OUT_PATH       = "data/msmarco_pragma/cc_ablation_results.json"
OUT_PATH_SHARD = "data/msmarco_pragma/cc_ablation_results_shard{}.json"

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
    print(f"Exported {len(cc_cases)} CC cases → {CC_CASES_PATH}")
    sizes = defaultdict(int)
    for c in cc_cases:
        sizes[c["effective_size"]] += 1
    print("By size:", dict(sorted(sizes.items())))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    by_budget: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_budget[r["budget_label"]].append(r)

    print(f"\n{'='*90}")
    print(f"  CC Budget Ablation — MSMARCO pragma — {len(results)} rows")
    print(f"{'='*90}")
    for label in [f"fixed_{b}" for b in BUDGETS_FIXED] + [f"pct_{p}" for p in BUDGETS_PCT]:
        rows = by_budget.get(label, [])
        if not rows:
            continue
        correct  = sum(1 for r in rows if r["outcome"] == "correct")
        avg_cost = sum(r["calls"] + r["budget"] for r in rows) / len(rows)
        print(
            f"  {label:<12} n={len(rows):>3}  acc={correct/len(rows):>5.0%} "
            f"({correct:>3}/{len(rows):<3})  avg_cost={avg_cost:>7.1f}"
        )


# ---------------------------------------------------------------------------
# Run (GPU)
# ---------------------------------------------------------------------------

def merge_shards() -> None:
    import glob
    shard_files = sorted(glob.glob(OUT_PATH_SHARD.format("*")))
    if not shard_files:
        print("No shard files found.")
        return
    results: list[dict] = []
    for path in shard_files:
        with open(path) as f:
            shard = json.load(f)
        results.extend(shard)
        print(f"  {path}: {len(shard)} rows")
    results.sort(key=lambda r: (r["case_id"], r["budget_label"]))
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMerged {len(results)} rows → {OUT_PATH}")
    print_summary(results)


def run_ablation(args: argparse.Namespace) -> None:
    from scripts.shared.load_cc_model import load_model_and_tokenizer
    from search.contextcite import contextcite_search

    shard      = args.shard
    num_shards = args.num_shards

    with open(CC_CASES_PATH) as f:
        all_cases = json.load(f)

    if shard is not None:
        cases = [c for c in all_cases if c["case_id"] % num_shards == shard]
        out_path = OUT_PATH_SHARD.format(shard)
        print(f"Shard {shard}/{num_shards}: {len(cases)} cases (of {len(all_cases)})")
    else:
        cases    = all_cases
        out_path = OUT_PATH
    print(f"Output → {out_path}")

    model, tokenizer = load_model_and_tokenizer(args)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path) as f:
            results: list[dict] = json.load(f)
        done = {(r["case_id"], r["budget_label"]) for r in results}
        print(f"Resuming: {len(done)} (case_id, budget) pairs already done")
    except FileNotFoundError:
        results = []
        done    = set()

    skipped = 0
    for case in cases:
        cid = case["case_id"]
        k   = case["effective_size"]

        bfs_res = bfs_search(
            case["criteria"], case["rankings_by_combo"],
            start_node=case["start_node"], direction="down",
        )
        if not bfs_res["found"]:
            skipped += 1
            continue

        for label, num_ablations in _all_budgets(k, pct_only=args.pct):
            if (cid, label) in done:
                continue

            print(
                f"  [c{cid:>3} k={k} {label:<12}] qid={case['query_id']} ...",
                end=" ", flush=True,
            )
            t0 = time.time()
            try:
                cc_res = contextcite_search(
                    case["criteria"], case["rankings_by_combo"],
                    case["query"], case["sources"], case["plausible_answers"],
                    model, tokenizer,
                    num_ablations=num_ablations,
                    batch_size=args.batch_size,
                    start_node=case["start_node"],
                    direction="down",
                )
            except Exception as e:
                print(f"ERROR: {e}")
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved {len(results)} rows before crash → {out_path}")
                raise
            elapsed = time.time() - t0

            scored = score_cc(bfs_res, cc_res, case["rankings_by_combo"], case["start_node"])
            results.append({
                "case_id":        cid,
                "query_id":       case["query_id"],
                "effective_size": k,
                "budget_label":   label,
                "budget":         num_ablations,
                "bfs_key":        bfs_res["key"],
                "bfs_layer":      bfs_res["size"],
                "bfs_calls":      bfs_res["calls"],
                "elapsed":        round(elapsed, 2),
                **scored,
            })
            done.add((cid, label))

            tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
            print(f"{tag} ({elapsed:.1f}s)")

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    if skipped:
        print(f"\nSkipped {skipped} cases (no BFS CF)")
    print(f"\nDone. {len(results)} rows → {out_path}")
    print_summary(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from scripts.shared.load_cc_model import add_model_args

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--export",     action="store_true")
    parser.add_argument("--run",        action="store_true")
    parser.add_argument("--merge",      action="store_true")
    parser.add_argument("--pct",        type=int, default=None,
                        help="Run only this percent budget (e.g. 10 for pct_10). Omit for full sweep.")
    parser.add_argument("--shard",      type=int, default=None,
                        help="Shard index (0-based). Processes cases where case_id %% num-shards == shard.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards.")
    args = parser.parse_args()

    if args.export:
        export_cases()
    elif args.merge:
        merge_shards()
    elif args.run:
        run_ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
