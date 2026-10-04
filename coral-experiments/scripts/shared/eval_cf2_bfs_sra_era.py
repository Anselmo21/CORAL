"""BFS / SRA / ERA on the cf2 case pool (CPU step, runnable while CC/LSM is on GPU).

Reads the self-contained cases exported by scripts.shared.export_cc_cf2_cases
(logged_results/<run>-cf2/<dataset>/cc_cases.json) and evaluates, per case:

  - BFS  — exact ground-truth traversal (down from the sub-lattice start node)
  - SRA  — native combsum-Borda sra_search (paper convention)
  - ERA  — mra_search(normalize_fn="softmax") at budgets 4/8/16, stored as
           mra_b{4,8,16} (paper-facing "ERA" label, see PRESENTATION_PLOTS.md)
  - ERA (elicited) — same mra_search config plus scores_by_combo fed from the
           run's elicited per-answer scores ("-100" runs) or probabilities
           ("-percentage" runs), voter_norm="max" (the plot_qwen3_scores_bfs_sra
           convention), stored as mra_scores_b{4,8,16} and labeled "ERA (elicited)"
           in the cf2 per-lattice artifact

Rows match the plot_qwen3_scores_bfs_sra results.json schema (score_proxy
cells) plus case_id/cf_removals, and are cached to
logged_results/<run>-cf2/<dataset>/results.json — delete to recompute.
CC/LSM columns land separately as cc_results_budget_<N>.json and are not
touched here.

Usage:
    conda run -n coral python3 -m scripts.shared.eval_cf2_bfs_sra_era [--runs <run> ...] [--datasets <ds> ...]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.criteria.msmarco.run_subsample import score_proxy
from scripts.shared.plot_qwen3_scores_bfs_sra import DATASET_SUBDIRS, RUNS as SCORE_RUNS
from scripts.shared.run_cra_eval import load_score_lattices
from search.bfs import bfs_search
from search.rank import mra_search, sra_search

# The cf2 pool covers exactly the four score/prob-elicited runs.
RUNS = list(SCORE_RUNS)
DATASETS = [ds for ds, _, _ in DATASET_SUBDIRS]
DATA_SUBDIR = {ds: sub for ds, sub, _ in DATASET_SUBDIRS}
# Must match the mra_b<N> columns in plot_qwen3_temp0_bfs_lsm.ERA_METHODS.
BUDGETS = [4, 8, 16]


def _compute_rows(cases: list[dict], sbc_by_qid: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        start = case["start_node"]
        rbc = case["rankings_by_combo"]
        sbc = sbc_by_qid[str(case["query_id"])]

        bfs_res = bfs_search(start, rbc, start_node=start, direction="down")
        if not bfs_res["found"]:
            # cf2 cases were selected so BFS finds a CF; flag if that breaks.
            print(f"    WARNING: BFS found no CF for case {case['case_id']} "
                  f"(qid {case['query_id']}, k={case['k']}) — skipped")
            continue
        row = {
            "case_id": case["case_id"],
            "qid": case["query_id"],
            "size": case["k"],
            "cf_removals": case["cf_removals"],
            "start_node": start,
            "bfs": {"found": True, "key": bfs_res["key"],
                    "layer": bfs_res["size"], "calls": bfs_res["calls"]},
        }
        bfs_proxy = {"key": bfs_res["key"], "size": bfs_res["size"], "found": True}

        sra_res = sra_search(start, rbc, start_node=start, direction="down")
        row["sra"] = score_proxy(bfs_proxy, sra_res, rbc, start)

        for budget in BUDGETS:
            res = mra_search(start, rbc, budget=budget, start_node=start,
                             direction="down", normalize_fn="softmax")
            row[f"mra_b{budget}"] = score_proxy(bfs_proxy, res, rbc, start)
            res = mra_search(start, rbc, budget=budget, start_node=start,
                             direction="down", normalize_fn="softmax",
                             scores_by_combo=sbc, voter_norm="max")
            row[f"mra_scores_b{budget}"] = score_proxy(bfs_proxy, res, rbc, start)
        rows.append(row)
    return rows


def _summarize(rows: list[dict]) -> str:
    parts = []
    cols = [f"mra{v}_b{b}" for b in BUDGETS for v in ("", "_scores")]
    for col in ["sra"] + cols:
        n = sum(1 for r in rows if col in r)
        outcomes = Counter(r[col]["outcome"] for r in rows if col in r)
        parts.append(f"{col} {100.0 * outcomes['correct'] / n:.0f}%")
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS, choices=RUNS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--force", action="store_true", help="recompute even if results.json exists")
    args = ap.parse_args()

    for run in args.runs:
        for ds in args.datasets:
            out_dir = Path(f"logged_results/{run}-cf2/{ds}")
            out_path = out_dir / "results.json"
            if out_path.exists() and not args.force:
                rows = json.load(open(out_path))
                print(f"  {out_path}: cached, {len(rows)} rows ({_summarize(rows)})")
                continue
            cases = json.load(open(out_dir / "cc_cases.json"))
            lattices = load_score_lattices(f"coral-runs-lattices/{run}/{DATA_SUBDIR[ds]}")
            sbc_by_qid = {qid: sbc for qid, (_, _, sbc) in lattices.items()}
            rows = _compute_rows(cases, sbc_by_qid)
            json.dump(rows, open(out_path, "w"))
            print(f"  {out_path}: {len(rows)} rows ({_summarize(rows)})")


if __name__ == "__main__":
    main()
