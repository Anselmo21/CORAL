"""Stratified sub-lattice sweep on the MSMARCO pragma dataset.

Identical to `run_subsample.py` except for *sub-lattice sampling*: for each qid,
pragmas are grouped by singleton ranking class, and sub-lattices are accepted
only when the plurality class's share inside the sub-lattice is ≤ MAX_PLURALITY
(falling back to ≤ FALLBACK_PLURALITY if no candidate at the strict limit). The
goal is to break the pragma-voter-cascade documented in
reports/era_pragma_pruning/REPORT.md.

All other logic (CF-must-remove-≥2-pragmas filter, BFS / SRA / ERA-budget-grid
sweep, per-case checkpointing) reuses scripts.pragma.msmarco.run_subsample.

Outputs:
  data/msmarco_pragma_stratified/cases.json
  data/msmarco_pragma_stratified/per_case/case_<i>.json
  data/msmarco_pragma_stratified/results.json
  data/msmarco_pragma_stratified/sampling_diag.json    — per-case plurality_share
                                                         + which classes appear

CLI:
  conda run -n coral python3 -m scripts.pragma.msmarco.run_stratified
  conda run -n coral python3 -m scripts.pragma.msmarco.run_stratified --resample
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key
from scripts.pragma.msmarco.run_subsample import (
    ERA_BUDGETS_FIXED,
    ERA_BUDGETS_PCT,
    MIN_PRAGMAS_REMOVED,
    SIZES,
    TARGET_PER_SIZE,
    _load_all_qids,
    _load_lattices,
    _qualifies,
    print_summary,
    run_one_case,
)

SEED: int = 42
MAX_PLURALITY: float = 0.50        # strict target: plurality ≤ half the sub-lattice
FALLBACK_PLURALITY: float = 0.70   # relax to 70% only if the strict cap is infeasible

OUT_DIR: str = "data/msmarco_pragma_stratified"
CASES_PATH: str = f"{OUT_DIR}/cases.json"
RESULTS_PATH: str = f"{OUT_DIR}/results.json"
SAMPLING_DIAG_PATH: str = f"{OUT_DIR}/sampling_diag.json"


# ---------------------------------------------------------------------------
# Patch run_one_case / checkpoint paths so this runner writes under OUT_DIR
# ---------------------------------------------------------------------------

import scripts.pragma.msmarco.run_subsample as base


def _patch_paths() -> None:
    base.OUT_DIR = OUT_DIR
    base.CASES_PATH = CASES_PATH
    base.RESULTS_PATH = RESULTS_PATH


# ---------------------------------------------------------------------------
# Classify singletons + helpers
# ---------------------------------------------------------------------------

def _ranking_class(rbc: dict[str, list[str]], pragma: str) -> tuple[str, ...]:
    """Class identity = the full tuple of the singleton ranking."""
    return tuple(rbc.get(pragma, []))


def _class_map(crit: list[str], rbc: dict[str, list[str]]) -> dict[str, str]:
    """Return {pragma -> class_id} where class_id is "C0", "C1", ... sorted by class size desc."""
    by_class: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for p in crit:
        by_class[_ranking_class(rbc, p)].append(p)
    # Order classes by descending size, then by smallest member name for determinism
    ordered = sorted(by_class.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    name_of: dict[str, str] = {}
    for idx, (_ranking, members) in enumerate(ordered):
        for p in members:
            name_of[p] = f"C{idx}"
    return name_of


def _plurality_share(start_node: list[str], class_of: dict[str, str]) -> float:
    classes = Counter(class_of[p] for p in start_node)
    return max(classes.values()) / len(start_node) if start_node else 0.0


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_stratified_cases(
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Return (cases, diag_rows). Each case is the same shape as run_subsample's
    cases: {qid, size, start_node}. diag_rows include per-case plurality_share +
    class_counts for inspection."""
    rng = random.Random(seed)
    cases: list[dict] = []
    diag_rows: list[dict] = []

    qid_class_maps = {qid: _class_map(crit, rbc) for qid, (crit, rbc) in lattices.items()}

    for size in SIZES:
        # Build all candidates per qid with their plurality share
        candidates_strict: list[tuple[float, dict]] = []
        candidates_relaxed: list[tuple[float, dict]] = []
        for qid, (crit, rbc) in lattices.items():
            class_of = qid_class_maps[qid]
            for combo in combinations(crit, size):
                share = _plurality_share(list(combo), class_of)
                cand = {"qid": qid, "size": size, "start_node": list(combo),
                        "plurality_share": share,
                        "class_counts": dict(Counter(class_of[p] for p in combo))}
                if share <= MAX_PLURALITY:
                    candidates_strict.append((share, cand))
                if share <= FALLBACK_PLURALITY:
                    candidates_relaxed.append((share, cand))

        # Prefer strict (≤MAX_PLURALITY share). If we can't fill the quota with
        # CF-qualifying strict candidates, fall through to relaxed (≤FALLBACK_PLURALITY).
        rng.shuffle(candidates_strict)
        rng.shuffle(candidates_relaxed)

        accepted: list[dict] = []
        seen_keys: set[tuple] = set()
        for cand_list, _label in [(candidates_strict, "strict"), (candidates_relaxed, "relaxed")]:
            if len(accepted) >= TARGET_PER_SIZE:
                break
            for _share, cand in cand_list:
                key = (cand["qid"], tuple(sorted(cand["start_node"])))
                if key in seen_keys:
                    continue
                crit, rbc = lattices[cand["qid"]]
                if _qualifies(rbc, crit, cand["start_node"]):
                    seen_keys.add(key)
                    accepted.append(cand)
                    if len(accepted) >= TARGET_PER_SIZE:
                        break

        strict_count = sum(1 for a in accepted if a["plurality_share"] <= MAX_PLURALITY)
        relaxed_count = len(accepted) - strict_count
        print(f"  size={size}: pool_strict={len(candidates_strict):>5}  "
              f"pool_relaxed={len(candidates_relaxed):>5}  "
              f"accepted={len(accepted):>3}/{TARGET_PER_SIZE}  "
              f"(strict={strict_count} relaxed={relaxed_count})")

        for a in accepted:
            diag_rows.append({
                "qid": a["qid"], "size": a["size"],
                "plurality_share": a["plurality_share"],
                "class_counts": a["class_counts"],
                "regime": "strict" if a["plurality_share"] <= MAX_PLURALITY else "relaxed",
            })
            cases.append({"qid": a["qid"], "size": a["size"], "start_node": a["start_node"]})

    # Full k=10 cases (every qid) — same as baseline. These can't be stratified
    # since the entire pragma set is fixed, but include them for parity.
    for qid, (crit, _) in sorted(lattices.items()):
        class_of = qid_class_maps[qid]
        cases.append({"qid": qid, "size": 10, "start_node": list(crit)})
        diag_rows.append({
            "qid": qid, "size": 10,
            "plurality_share": _plurality_share(list(crit), class_of),
            "class_counts": dict(Counter(class_of[p] for p in crit)),
            "regime": "full_lattice",
        })
    print(f"  size=10: added {len(lattices)} full lattices (cannot stratify)")
    return cases, diag_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()

    _patch_paths()
    qids = _load_all_qids()
    lattices = _load_lattices(qids)
    print(f"Qids ({len(qids)}): {qids}")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.resample or not os.path.exists(CASES_PATH):
        print("\nSampling stratified cases ...")
        cases, diag_rows = sample_stratified_cases(lattices, seed=SEED)
        with open(CASES_PATH, "w") as f:
            json.dump(cases, f, indent=2)
        with open(SAMPLING_DIAG_PATH, "w") as f:
            json.dump(diag_rows, f, indent=2)
        print(f"Saved {len(cases)} cases → {CASES_PATH}")
    else:
        with open(CASES_PATH) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} existing cases from {CASES_PATH}")

    print(f"\nRunning BFS/SRA/ERA on {len(cases)} stratified cases ...")
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
    print_summary(results)


if __name__ == "__main__":
    main()
