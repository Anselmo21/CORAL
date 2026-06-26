"""Sub-lattice BFS / SRA / ERA budget sweep on the 16-qid subset of the
MSMARCO pragma regen.

Wraps scripts.pragma.msmarco_v2.run_subsample but drops the same two qids
that scripts.pragma.msmarco.run_16_lattices drops, for a 1:1 qid overlap
with the original msmarco_pragma headline set:

  EXCLUDE_QIDS = {"480543", "51587"}

Outputs:
  logged_results/msmarco_pragma_v2/cases.json
  logged_results/msmarco_pragma_v2/per_case/case_<i>.json
  logged_results/msmarco_pragma_v2/results.json
"""
from __future__ import annotations

import argparse
import json
import os

from scripts.pragma.msmarco_v2.run_subsample import (
    DATASET_PATH,
    _load_lattices as _load_lattices_all,
    sample_cases,
    run_one_case,
)
import scripts.pragma.msmarco_v2.run_subsample as base

EXCLUDE_QIDS: set[str] = {"480543", "51587"}

OUT_DIR      = "logged_results/msmarco_pragma_v2"
CASES_PATH   = f"{OUT_DIR}/cases.json"
RESULTS_PATH = f"{OUT_DIR}/results.json"


def _patch_paths() -> None:
    base.OUT_DIR = OUT_DIR
    base.CASES_PATH = CASES_PATH
    base.RESULTS_PATH = RESULTS_PATH


def _load_qids() -> list[str]:
    with open(DATASET_PATH) as f:
        d = json.load(f)
    qids = [qid for qid in d["lattices"] if qid not in EXCLUDE_QIDS]
    return sorted(qids, key=lambda q: int(q) if q.isdigit() else q)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()

    _patch_paths()
    qids = _load_qids()
    lattices = _load_lattices_all(qids)
    print(f"Qids ({len(qids)}): {qids}")
    print(f"Lattices loaded: {len(lattices)}")
    print(f"Excluded: {sorted(EXCLUDE_QIDS)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.resample or not os.path.exists(CASES_PATH):
        print("\nSampling cases ...")
        cases = sample_cases(lattices, seed=42)
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Saved {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} existing cases")

    print(f"\nRunning BFS/SRA/ERA on {len(cases)} cases ...")
    results: list[dict] = []
    for i, case in enumerate(cases):
        qid = case["qid"]
        crit, rbc = lattices[qid]
        print(f"  [{i+1:>3}/{len(cases)}] qid={qid} k={case['size']} ...",
              end=" ", flush=True)
        result = run_one_case(i, case, crit, rbc)
        results.append(result)
        bfs_layer = result["bfs"]["layer"]
        sra_ok = "OK" if result["sra"]["outcome"] == "correct" else result["sra"]["outcome"]
        print(f"BFS_layer={bfs_layer} SRA={sra_ok}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
