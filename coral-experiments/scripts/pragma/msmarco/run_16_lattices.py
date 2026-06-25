"""Sub-lattice BFS / SRA / ERA budget sweep on the 16-lattice MSMARCO regen.

Same sampling logic as scripts.pragma.msmarco.run_subsample (uniform
shuffle + first-fit, MIN_PRAGMAS_REMOVED=2, TARGET_PER_SIZE=20, k=10 always
included one-per-qid), but drops two qids first:

  EXCLUDE_QIDS = {"480543", "51587"}

leaving 16 lattices. (52412 kept from the original 3-qid exclusion because it
has the highest k=10 BFS cost of the three.)

Outputs:
  data/msmarco_pragma/cases.json
  data/msmarco_pragma/per_case/case_<i>.json
  data/msmarco_pragma/results.json
"""
from __future__ import annotations

import argparse
import json
import os

from scripts.pragma.msmarco.run_subsample import (
    DATASET_PATH,
    _load_lattices as _load_lattices_all,
    sample_cases,
    run_one_case,
)
import scripts.pragma.msmarco.run_subsample as base

EXCLUDE_QIDS: set[str] = {"480543", "51587"}

OUT_DIR      = "data/msmarco_pragma"
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
