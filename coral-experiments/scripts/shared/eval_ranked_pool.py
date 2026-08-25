"""BFS / SRA / ERA(MRA) on a ranking-only case pool (see generate_ranked_pool).

Same evaluation as scripts.shared.eval_cf2_bfs_sra_era but sourced from
data/cases_16x20_cfmin<N>_<model>-ranked.json plus the raw ranking-only
lattices (answers.json already maps combo_key -> answer-string ranking, so no
loader indirection and no elicited scores; the mra_scores_b* columns do not
exist for these runs).

Writes logged_results/<model>-ranked-cf2/<dataset>/results.json (delete or
--force to recompute) and prints an accuracy / mean-calls summary table.

Usage:
    conda run -n coral python3 -m scripts.shared.eval_ranked_pool \
        [--model qwen35-122b] [--min-removals 2] \
        [--lattice-root /Users/kathling/coral-data-gen/lattices]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.criteria.msmarco.run_subsample import score_proxy
from scripts.shared.generate_ranked_pool import DATASET_SUBDIRS
from search.bfs import bfs_search
from search.rank import mra_search, sra_search

BUDGETS = [4, 8, 16]


def _compute_rows(cases: list[dict], base: str) -> list[dict]:
    rbc_by_qid = {}
    rows = []
    for i, case in enumerate(cases):
        qid, start = case["qid"], case["start_node"]
        if qid not in rbc_by_qid:
            rbc_by_qid[qid] = json.load(open(f"{base}/{qid}/answers.json"))
        rbc = rbc_by_qid[qid]

        bfs_res = bfs_search(start, rbc, start_node=start, direction="down")
        if not bfs_res["found"]:
            print(f"    WARNING: BFS found no CF (qid {qid}, k={case['size']}) — skipped")
            continue
        row = {
            "case_id": i,
            "qid": qid,
            "size": case["size"],
            "cf_removals": case["cf_removals"],
            "start_node": start,
            "bfs": {"found": True, "key": bfs_res["key"],
                    "layer": bfs_res["size"], "calls": bfs_res["calls"]},
        }
        bfs_proxy = {"key": bfs_res["key"], "size": bfs_res["size"], "found": True}

        row["sra"] = score_proxy(bfs_proxy,
                                 sra_search(start, rbc, start_node=start, direction="down"),
                                 rbc, start)
        for budget in BUDGETS:
            res = mra_search(start, rbc, budget=budget, start_node=start,
                             direction="down", normalize_fn="softmax")
            row[f"mra_b{budget}"] = score_proxy(bfs_proxy, res, rbc, start)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-122b")
    ap.add_argument("--min-removals", type=int, default=2)
    ap.add_argument("--lattice-root", default="/Users/kathling/coral-data-gen/lattices")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = f"{args.model}-ranked"
    pool = json.load(open(f"data/cases_16x20_cfmin{args.min_removals}_{run}.json"))["cases"][run]

    methods = ["bfs", "sra"] + [f"mra_b{b}" for b in BUDGETS]
    print(f"{'dataset':<18} " + " ".join(f"{m:>16}" for m in methods) + "   (acc% / mean calls)")
    for ds, subdir in DATASET_SUBDIRS.items():
        out_dir = Path(f"logged_results/{run}-cf2/{ds}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.json"
        if out_path.exists() and not args.force:
            rows = json.load(open(out_path))
        else:
            rows = _compute_rows(pool[ds], f"{args.lattice_root}/{subdir}-{args.model}")
            json.dump(rows, open(out_path, "w"))
        cells = []
        for m in methods:
            n = len(rows)
            if m == "bfs":
                acc, calls = 100.0, sum(r["bfs"]["calls"] for r in rows) / n
            else:
                acc = 100.0 * sum(r[m]["outcome"] == "correct" for r in rows) / n
                calls = sum(r[m].get("calls", 0) + r[m].get("budget_used", 0) for r in rows) / n
            cells.append(f"{acc:5.1f} / {calls:6.1f}")
        print(f"{ds:<18} " + " ".join(f"{c:>16}" for c in cells) + f"   n={len(rows)}")


if __name__ == "__main__":
    main()
