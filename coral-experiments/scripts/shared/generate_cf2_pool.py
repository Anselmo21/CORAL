"""Generate the cf2 case pool: sub-lattices with a depth floor on the shallowest CF.

Replaces the ad-hoc data/cases_16x20_cf2removals.json (which pinned CF depth to
exactly k-2 with a closest-depth fallback). Rule here is a floor, not a pin:
a size-k subset of a base lattice's 10 atoms is valid iff BFS (down from the
subset) finds a counterfactual AND its shallowest CF is >= 3 removals below the
start node for k >= 4 (>= 2 for k=3, where only depths 2-3 exist). Depth
distribution beyond the floor is natural — no per-depth quotas.

Per (run, dataset): 20 sub-lattices per size k=3..9, sampled round-robin over
the 16 base qids from data/lattice_selection_16_cf_layer2to8.json (each qid's
valid subsets shuffled with random.Random(0); shortfalls are logged, never
silently padded). k=10 cases are the 16 base lattices themselves
(cf_removals = 10 - base CF layer, not depth-controlled).

Writes data/cases_16x20_cfmin3.json (schema-compatible with
scripts.shared.export_cc_cf2_cases).

Usage:
    conda run -n coral python3 -m scripts.shared.generate_cf2_pool
"""
from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key

SELECTION_PATH = "data/lattice_selection_16_cf_layer2to8.json"
OUT_PATH = "data/cases_16x20_cfmin3.json"
QUOTA = 20
SIZES = list(range(3, 10))
DATASET_SUBDIRS = {
    "msmarco_criteria": "msmarco-criteria",
    "msmarco_pragma":   "msmarco-pragmas",
    "curated_criteria": "curated-criteria",
    "curated_pragma":   "curated-pragmas",
}


def _load_rbc(base: str, qid: str) -> tuple[list[str], dict[str, list[str]]]:
    cfg = json.load(open(f"{base}/{qid}/config.json"))
    ans = json.load(open(f"{base}/{qid}/answers.json"))
    pa = cfg["plausible_answers"]
    rbc = {k: [pa[j] for j in e["ranking"]] for k, e in ans.items()}
    return cfg["atoms"], rbc


def _min_depth(k: int) -> int:
    return 2 if k == 3 else 3


def _valid_subsets(atoms: list[str], rbc: dict, k: int) -> list[tuple[list[str], int]]:
    """All size-k subsets whose shallowest CF is >= _min_depth(k). Returns (start, depth)."""
    out = []
    for combo in combinations(sorted(atoms), k):
        start = list(combo)
        if not rbc.get(combo_key(start)):
            continue
        res = bfs_search(start, rbc, start_node=start, direction="down")
        if not res["found"]:
            continue
        depth = k - res["size"]
        if depth >= _min_depth(k):
            out.append((start, depth))
    return out


def main() -> None:
    selection = json.load(open(SELECTION_PATH))["selection"]
    pool: dict[str, dict[str, list[dict]]] = {}
    for run, by_ds in selection.items():
        pool[run] = {}
        for ds, info in by_ds.items():
            qids, base_cf_layer = info["qids"], info["cf_layer"]
            base = f"coral-runs-lattices/{run}/{DATASET_SUBDIRS[ds]}"
            lattices = {qid: _load_rbc(base, qid) for qid in qids}
            rng = random.Random(0)
            cases: list[dict] = []
            for k in SIZES:
                per_qid = {}
                for qid in qids:
                    atoms, rbc = lattices[qid]
                    subs = _valid_subsets(atoms, rbc, k)
                    rng.shuffle(subs)
                    per_qid[qid] = subs
                picked, i = [], 0
                while len(picked) < QUOTA and any(per_qid.values()):
                    qid = qids[i % len(qids)]
                    i += 1
                    if per_qid[qid]:
                        start, depth = per_qid[qid].pop()
                        picked.append({"qid": qid, "size": k,
                                       "start_node": start, "cf_removals": depth})
                if len(picked) < QUOTA:
                    print(f"    SHORTFALL {run}/{ds} k={k}: only {len(picked)}/{QUOTA} valid subsets")
                cases.extend(picked)
            for qid in qids:
                atoms, _ = lattices[qid]
                cases.append({"qid": qid, "size": 10, "start_node": sorted(atoms),
                              "cf_removals": 10 - base_cf_layer[qid]})
            pool[run][ds] = cases
            depths = Counter(c["cf_removals"] for c in cases if c["size"] < 10)
            print(f"  {run}/{ds}: {len(cases)} cases; k<10 depth dist "
                  f"{dict(sorted(depths.items()))}")

    out = {
        "description": (
            f"{QUOTA} sub-lattices per size k=3..9 per (run, dataset), sampled from the 16 base "
            "lattices in lattice_selection_16_cf_layer2to8.json. A subset is valid iff BFS (down "
            "from the subset) finds a CF and the shallowest CF is >= 3 removals below the start "
            "node for k >= 4 (>= 2 for k = 3, where only depths 2-3 exist); depth beyond the "
            "floor is natural, not quota'd. Sampling: "
            "random.Random(0) shuffle per qid, round-robin over base qids; shortfalls logged. "
            "cf_removals records actual depth. k=10 cases are the 16 base lattices themselves "
            "(cf_removals = 10 - base CF layer, i.e. 2-8, not depth-controlled). "
            "Generated by scripts.shared.generate_cf2_pool."
        ),
        "target": "cf_removals >= 3 for k >= 4; >= 2 for k = 3",
        "quota_per_size": QUOTA,
        "sizes": SIZES + [10],
        "cases": pool,
    }
    json.dump(out, open(OUT_PATH, "w"))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
