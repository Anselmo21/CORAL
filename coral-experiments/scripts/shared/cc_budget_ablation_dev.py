"""ContextCite budget ablation on dev.jsonl.

Two phases:

  Phase 1 — Local export (run on this machine):
      conda run -n coral python cc_budget_ablation_dev.py --export
    Produces  data/cc_cases_dev.json  containing per-case
    {case_id, query_id, query, sources, criteria, fixed_criteria, toggleable_criteria,
     start_node, direction, effective_size, effective_rbc} for every dev.jsonl row.
    Sources are read from the `sources` field of each dev.jsonl row. To override
    on a per-query basis, pass --sources path/to/override.json
    ({qid: {"sources": str}} or {qid: str}).

  Phase 2 — GPU run (on the remote box):
      conda run -n coral python cc_budget_ablation_dev.py --run
    Default: single eval budget `num_ablations=64` (matches ERA64 / contextcite default).
    Use `--sweep` to run the full ablation grid in BUDGETS (up to `--num-ablations-max`).
    Resumes by (case_id, budget); writes data/cc_results_dev.json incrementally.
    Cases with empty `sources` are skipped.
"""

import argparse
import json
import os
import time
from itertools import combinations

from search.lattice import combo_key


DEV_LATTICES_PATH = "data/all_lattices.json"
DEV_CASES_PATH = "data/dev.jsonl"
CC_CASES_DEV_PATH = "data/cc_cases_dev.json"
CC_RESULTS_DEV_PATH = "data/cc_results_dev.json"

# Default eval budget (aligned with ERA64 and search/contextcite.py default).
CC_EVAL_NUM_ABLATIONS: int = 64

# Full sweep grid (--sweep only); mirrors era_budget_ablation.BUDGETS.
BUDGETS: list[int] = [4, 8, 16, 32, 64, 128, 256]


# ─────────────────────────── Exporter (local) ────────────────────────────


def build_manifest_cc_cases(
    lattices_path: str,
    cases_path: str,
    sources_override: dict[int, str] | None = None,
    directions: set[str] | None = None,
) -> list[dict]:
    """Build CC cases from a jsonl manifest (dev.jsonl, test.jsonl, …). Sources come from the per-row `sources`
    field; `sources_override` (qid → str) can substitute on a per-qid basis.

    `directions` restricts which dev rows are exported (e.g. {"down"} for
    top-down only). `case_id` always reflects the dev.jsonl row index so
    filtered exports remain traceable to the source manifest.
    """
    with open(lattices_path) as f:
        lat_data = json.load(f)
    lattices: dict[str, dict] = lat_data["lattices"]

    cases: list[dict] = []
    missing_sources: set[int] = set()

    with open(cases_path) as f:
        for case_id, line in enumerate(f):
            rec = json.loads(line)
            qid = int(rec["query_id"])
            lat = lattices.get(str(qid))
            if lat is None:
                continue

            direction = "down" if rec["direction"] == "top-down" else "up"
            if directions is not None and direction not in directions:
                continue

            fixed = list(rec["fixed_criteria"])
            toggleable = list(rec["toggleable_criteria"])
            full_rankings: dict[str, list[str]] = lat["rankings"]

            # Project full lattice rankings down to keys over toggleable subsets
            effective_rbc: dict[str, list[str]] = {}
            for r in range(len(toggleable) + 1):
                for sub in combinations(toggleable, r):
                    sub_list = list(sub)
                    sub_key = combo_key(sub_list)
                    full_key = combo_key(sub_list + fixed)
                    ranking = full_rankings.get(full_key)
                    if ranking is not None:
                        effective_rbc[sub_key] = ranking

            start_node = sorted(toggleable) if direction == "down" else []
            effective_size = len(toggleable)

            sources = rec.get("sources", "") or ""
            if sources_override and qid in sources_override:
                sources = sources_override[qid]
            if not sources:
                missing_sources.add(qid)

            cases.append({
                "case_id": case_id,
                "query_id": qid,
                "query": lat.get("question", ""),
                "sources": sources,
                "plausible_answers": lat["plausible_answers"],
                "criteria": sorted(toggleable),
                "fixed_criteria": sorted(fixed),
                "toggleable_criteria": sorted(toggleable),
                "start_node": start_node,
                "direction": direction,
                "effective_size": effective_size,
                "k": rec["k"],
                "rankings_by_combo": effective_rbc,
            })

    if missing_sources:
        print(
            f"  WARN: {len(missing_sources)} query_ids missing sources "
            f"(will be skipped by --run). Example qids: "
            f"{sorted(missing_sources)[:5]}"
        )
    return cases


def export_cases(args: argparse.Namespace) -> None:
    sources_override: dict[int, str] | None = None
    if args.sources:
        with open(args.sources) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            sources_override = {int(k): (v["sources"] if isinstance(v, dict) else v)
                                for k, v in raw.items()}
        else:
            sources_override = {int(r["query_id"]): r["sources"] for r in raw}
        print(f"Loaded sources override for {len(sources_override)} queries from {args.sources}")

    directions: set[str] | None = None
    if args.down_only:
        directions = {"down"}

    cases = build_manifest_cc_cases(
        DEV_LATTICES_PATH, DEV_CASES_PATH, sources_override, directions=directions,
    )
    os.makedirs(os.path.dirname(CC_CASES_DEV_PATH) or ".", exist_ok=True)
    with open(CC_CASES_DEV_PATH, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Exported {len(cases)} cases to {CC_CASES_DEV_PATH}")


def build_dev_cc_cases(
    lattices_path: str,
    cases_path: str,
    sources_override: dict[int, str] | None = None,
    directions: set[str] | None = None,
) -> list[dict]:
    return build_manifest_cc_cases(
        lattices_path, cases_path, sources_override, directions=directions,
    )


# ────────────────────────── Runner (GPU box) ─────────────────────────────


def score_cc(bfs_res: dict, cc_res: dict, rbc: dict, start_node: list[str]) -> dict:
    from search.lattice import jaccard_keys

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

    return {
        "proxy_key": proxy_key,
        "proxy_layer": proxy_layer,
        "proxy_verified": proxy_verified,
        "outcome": outcome,
        "gap": gap,
        "calls": cc_res["calls"],
        "jaccard": jaccard_keys(bfs_res.get("key"), proxy_key),
    }


def run_sweep(args: argparse.Namespace) -> None:
    from scripts.shared.load_cc_model import load_model_and_tokenizer
    from search.bfs import bfs_search
    from search.contextcite import contextcite_search

    with open(CC_CASES_DEV_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {CC_CASES_DEV_PATH}")

    if args.sweep:
        budgets = [b for b in BUDGETS if b <= args.num_ablations_max]
        print(f"Sweep budgets: {budgets}")
    else:
        budgets = [args.budget]
        print(f"Eval budget (num_ablations): {args.budget}")

    model, tokenizer = load_model_and_tokenizer(args)

    try:
        with open(CC_RESULTS_DEV_PATH) as f:
            results: list[dict] = json.load(f)
        done = {(r["case_id"], r["budget"]) for r in results}
        print(f"Resuming: {len(done)} (case, budget) pairs already done")
    except FileNotFoundError:
        results = []
        done = set()

    for case in cases:
        cid = case["case_id"]
        if not case.get("sources"):
            # Sources missing — can't fit a meaningful surrogate.
            continue

        criteria = case["criteria"]
        rbc = case["rankings_by_combo"]
        start_node = case["start_node"]
        direction = case["direction"]

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction=direction)
        if not bfs_res["found"]:
            # No CF reachable; nothing to evaluate.
            continue

        for budget in budgets:
            if (cid, budget) in done:
                continue

            print(
                f"  [c{cid:>3} b={budget:>3}] Q{case['query_id']} "
                f"size={case['effective_size']} {direction}...",
                end=" ", flush=True,
            )
            t0 = time.time()

            cc_res = contextcite_search(
                criteria, rbc,
                case["query"], case["sources"], case["plausible_answers"],
                model, tokenizer,
                num_ablations=budget,
                batch_size=args.batch_size,
                start_node=start_node,
                direction=direction,
            )
            elapsed = time.time() - t0

            scored = score_cc(bfs_res, cc_res, rbc, start_node)
            row = {
                "case_id": cid,
                "query_id": case["query_id"],
                "effective_size": case["effective_size"],
                "direction": direction,
                "budget": budget,
                "bfs_key": bfs_res["key"],
                "bfs_layer": bfs_res["size"],
                "bfs_calls": bfs_res["calls"],
                "elapsed": round(elapsed, 2),
                **scored,
            }
            results.append(row)
            done.add((cid, budget))

            tag = "OK" if scored["outcome"] == "correct" else scored["outcome"].upper()
            print(f"{tag} ({elapsed:.1f}s, {scored['calls']} calls)")

            with open(CC_RESULTS_DEV_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} rows in {CC_RESULTS_DEV_PATH}")


# ────────────────────────────── CLI ──────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true",
                      help="Local: build data/cc_cases_dev.json")
    mode.add_argument("--run", action="store_true",
                      help="GPU box: run ContextCite on each case")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all BUDGETS (default: single --budget only)")
    parser.add_argument("--budget", type=int, default=CC_EVAL_NUM_ABLATIONS,
                        help=f"num_ablations for contextcite_search (default {CC_EVAL_NUM_ABLATIONS})")

    parser.add_argument("--sources",
                        help="JSON of {query_id: {sources: str}} for the dev queries "
                             "(used by --export to embed sources in cases)")
    parser.add_argument("--down-only", action="store_true",
                        help="--export: include only top-down dev rows")
    from scripts.shared.load_cc_model import add_model_args
    add_model_args(parser)
    parser.add_argument("--num-ablations-max", type=int, default=256,
                        help="Cap the budget sweep (skip BUDGETS above this)")
    args = parser.parse_args()

    if args.export:
        export_cases(args)
    elif args.run:
        run_sweep(args)


if __name__ == "__main__":
    main()
