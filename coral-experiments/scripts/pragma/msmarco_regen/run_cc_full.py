"""CC (LSM) on the 16-lattice MSMARCO pragma regen cases at one budget config.

Two modes:
    --export        builds  data/msmarco_pragma/cc_cases.json
    --run --vllm    runs CC on cc_cases.json at the chosen budget,
                    writes data/msmarco_pragma/cc_results_<config>.json

Budget config (mutually exclusive):
    --pct 10        per-case budget = max(1, int(2**k * 10/100))   (default)
    --budget 16     fixed budget = 16 ablations per case

CLI:
    python3 -m scripts.pragma.msmarco_regen.run_cc_full --export
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \\
      conda run -n coral python3 -m scripts.pragma.msmarco_regen.run_cc_full \\
      --run --vllm --pct 10
    (or --budget 16)
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.pragma.msmarco_regen.run_subsample import _load_dataset
from scripts.shared.cc_budget_ablation_dev import score_cc
from search.bfs import bfs_search

DATASET_PATH    = "data/sources/msmarco_pragma.json"
CASES_PATH      = "data/msmarco_pragma/cases.json"
CC_CASES_PATH   = "data/msmarco_pragma/cc_cases.json"


def _restrict_rbc(rbc: dict[str, list[str]], start_node: list[str]) -> dict[str, list[str]]:
    pragma_set = set(start_node)
    out: dict[str, list[str]] = {}
    for key, ranking in rbc.items():
        parts = [p for p in key.split(", ") if p] if key else []
        if all(p in pragma_set for p in parts):
            out[key] = ranking
    return out


def export_cases() -> None:
    data = _load_dataset()
    lattices = data["lattices"]
    with open(CASES_PATH) as f:
        cases = json.load(f)

    cc_cases: list[dict] = []
    for cid, case in enumerate(cases):
        qid = str(case["qid"])
        rec = lattices.get(qid)
        if rec is None:
            continue
        start_node: list[str] = case["start_node"]
        sub_rbc = _restrict_rbc(rec["rankings"], start_node)
        cc_cases.append({
            "case_id":           cid,
            "query_id":          qid,
            "query":             rec["question"],
            "sources":           rec["sources"],
            "plausible_answers": rec["plausible_answers"],
            "criteria":          start_node,
            "rankings_by_combo": sub_rbc,
            "start_node":        start_node,
            "effective_size":    case["size"],
        })

    os.makedirs(os.path.dirname(CC_CASES_PATH), exist_ok=True)
    with open(CC_CASES_PATH, "w") as f:
        json.dump(cc_cases, f, indent=2)
    print(f"Exported {len(cc_cases)} CC cases → {CC_CASES_PATH}")


def run(args: argparse.Namespace) -> None:
    from scripts.shared.load_cc_model import load_model_and_tokenizer
    from search.contextcite import contextcite_search

    if args.budget is not None:
        config_label = f"budget_{args.budget}"
    else:
        config_label = f"pct_{args.pct}"
    out_path = f"data/msmarco_pragma/cc_results_{config_label}.json"

    with open(CC_CASES_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_PATH}")
    print(f"Config: {config_label}  →  {out_path}")

    model, tokenizer = load_model_and_tokenizer(args)

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

        if args.budget is not None:
            num_ablations = args.budget
        else:
            num_ablations = max(1, int(2**k * args.pct / 100))
        print(f"  [c{cid:>3} {config_label} n={num_ablations:>3}] qid={case['query_id']} k={k} ...",
              end=" ", flush=True)
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
            "case_id":  cid,
            "query_id": case["query_id"],
            "effective_size": k,
            "config":   config_label,
            "budget":   num_ablations,
            "bfs_key":  bfs_res["key"],
            "bfs_layer": bfs_res["size"],
            "bfs_calls": bfs_res["calls"],
            "elapsed":  round(elapsed, 2),
            **scored,
        })

        tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
        print(f"{tag} ({elapsed:.1f}s)")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    if skipped:
        print(f"Skipped {skipped} cases (no sources or no BFS CF)")
    print(f"\nDone. {len(results)} rows → {out_path}")


def main() -> None:
    from scripts.shared.load_cc_model import add_model_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help="Build cc_cases.json from cases.json (no GPU).")
    parser.add_argument("--run",    action="store_true",
                        help="Run CC against cc_cases.json (GPU).")
    parser.add_argument("--pct",    type=int, default=10,
                        help="Percent budget per case (default 10). Ignored if --budget set.")
    parser.add_argument("--budget", type=int, default=None,
                        help="Fixed budget per case (overrides --pct).")
    add_model_args(parser)
    args = parser.parse_args()

    if args.export:
        export_cases()
        return
    if args.run:
        run(args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
