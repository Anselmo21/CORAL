"""Merge size-10 MS MARCO / Llama-3.1 lattices into data/msmarco_llama_size10.json.

Sources:
  - data/all_lattices.json  — 40 size-10 queries (has sources, question, split)
  - data/baseline_lattice_runs_skeleton_llama31_8b_local.json — 7 size-10 queries;
      6 overlap with all_lattices (identical rankings, prefer all_lattices version),
      1 unique (qid 1166119, no sources/question in repo)

Quality filter applied to ALL candidates (baseline and new alike):
  BFS counterfactual must appear at a node with ≤7 criteria (not trivially shallow).

Output schema matches all_lattices.json.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")
from search.bfs import bfs_search

ALL_LATTICES_PATH = "data/all_lattices.json"
BASELINE_PATH = "data/baseline_lattice_runs_skeleton_llama31_8b_local.json"
OUT_PATH = "data/msmarco_llama_size10.json"

CF_SIZE_THRESHOLD = 7


def _passes_filter(criteria: list[str], rankings: dict[str, list[str]]) -> bool:
    try:
        res = bfs_search(criteria, rankings, start_node=list(criteria), direction="down")
    except (IndexError, KeyError):
        return False
    if not res["found"] or not res["key"]:
        return False
    cf_size = len([x for x in res["key"].split(", ") if x])
    return cf_size <= CF_SIZE_THRESHOLD


def main() -> None:
    with open(ALL_LATTICES_PATH) as f:
        all_lats_raw = json.load(f)
    all_lats: dict[str, dict] = all_lats_raw["lattices"]

    with open(BASELINE_PATH) as f:
        baseline_raw = json.load(f)
    baseline_size10: list[dict] = [
        r for r in baseline_raw["runs"] if r["target_lattice_size"] == 10
    ]
    baseline_qids = {str(r["query_id"]) for r in baseline_size10}

    # Candidate pool: all size-10 qids from all_lattices + unique baseline qid
    candidates: dict[str, dict] = {}

    for qid, rec in all_lats.items():
        if rec["n_criteria"] == 10:
            candidates[qid] = {
                "query_id": rec["query_id"],
                "split": rec.get("split", "unassigned"),
                "question": rec.get("question"),
                "sources": rec.get("sources"),
                "criteria": rec["criteria"],
                "plausible_answers": rec["plausible_answers"],
                "n_criteria": rec["n_criteria"],
                "n_combos": rec["n_combos"],
                "rankings": rec["rankings"],
                "model": "meta-llama/Llama-3.1-8B-Instruct",
            }

    # Add qid 1166119 — only in baseline, not in all_lattices
    unique_qid = "1166119"
    if unique_qid not in candidates:
        r = next(x for x in baseline_size10 if str(x["query_id"]) == unique_qid)
        candidates[unique_qid] = {
            "query_id": r["query_id"],
            "split": "unassigned",
            "question": None,
            "sources": None,
            "criteria": r["criteria_used"],
            "plausible_answers": r["plausible_answers"],
            "n_criteria": len(r["criteria_used"]),
            "n_combos": r["num_lattice_cells"],
            "rankings": r["rankings_by_combo"],
            "model": baseline_raw["model"],
        }

    # Apply quality filter to all candidates
    lattices: dict[str, dict] = {}
    dropped: list[str] = []
    for qid, rec in candidates.items():
        if _passes_filter(rec["criteria"], rec["rankings"]):
            lattices[qid] = rec
        else:
            dropped.append(qid)

    out = {
        "n_lattices": len(lattices),
        "description": (
            "Size-10 MS MARCO lattices generated with Llama-3.1-8B-Instruct. "
            f"Quality-filtered: BFS counterfactual at node size ≤{CF_SIZE_THRESHOLD}."
        ),
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lattices": lattices,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(lattices)} lattices to {OUT_PATH}  (dropped {len(dropped)}: {sorted(dropped)})")
    for qid, rec in sorted(lattices.items()):
        src = "baseline+all" if qid in baseline_qids else ("baseline-only" if qid == unique_qid else "all_lattices")
        print(f"  qid {qid:>8}  sources={'yes' if rec['sources'] else 'no ':3}  ({src})")


if __name__ == "__main__":
    main()
