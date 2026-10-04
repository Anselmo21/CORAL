"""Select 16 base lattices + generate the sub-lattice case pool for a
ranking-only lattice run (original prompt: rankings only, no score/percentage
elicitation), e.g. /Users/kathling/coral-data-gen/lattices/<dataset>-qwen35-122b.

Differences from scripts.shared.generate_cf2_pool:
  - source format: config.json has question/criteria (not query/atoms) and
    answers.json maps combo_key -> ranking as answer strings directly
    (no index lists, no scores);
  - selection is done here too (the cf2 selection script no longer exists):
    filter = non-halted AND BFS CF layer in [2, 8]; when a dataset has more
    than 16 eligible qids, prefer qids already used in
    data/lattice_selection_16_cf_layer2to8.json for the same model's
    score-elicited runs (comparability), then ascending qid; when fewer,
    fall back to layer-9 (L1) lattices, recorded in fallback_l1;
  - depth floor is >= MIN_REMOVALS (default 2) for every size k.

Writes:
  data/lattice_selection_16_<run>.json
  data/cases_16x20_cfmin<MIN_REMOVALS>_<run>.json   (schema-compatible with
                                                     generate_cf2_pool output)

Usage:
    conda run -n coral python3 -m scripts.shared.generate_ranked_pool \
        [--model qwen35-122b] [--lattice-root /Users/kathling/coral-data-gen/lattices] \
        [--min-removals 2]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import combinations

from search.bfs import bfs_search
from search.lattice import combo_key

# dataset key used in results pipelines -> source dir prefix in coral-data-gen
DATASET_SUBDIRS = {
    "msmarco_criteria": "msmarco-criteria",
    "msmarco_pragma":   "msmarco-pragmas",
    "curated_criteria": "sonnet-criteria",
    "curated_pragma":   "sonnet-pragmas",
}
N_SELECT = 16
QUOTA = 20
SIZES = list(range(3, 10))
PRIOR_SELECTION = "data/lattice_selection_16_cf_layer2to8.json"


def _qid_sort_key(qid: str):
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def _load_lattice(base: str, qid: str):
    cfg = json.load(open(f"{base}/{qid}/config.json"))
    rbc = json.load(open(f"{base}/{qid}/answers.json"))
    return cfg, rbc


def _prior_qids(model: str) -> dict[str, set[str]]:
    """qids per dataset used by the same model's score-elicited runs, if any."""
    try:
        sel = json.load(open(PRIOR_SELECTION))["selection"]
    except FileNotFoundError:
        return {}
    out: dict[str, set[str]] = {}
    for run, by_ds in sel.items():
        if model not in run:
            continue
        for ds, info in by_ds.items():
            out.setdefault(ds, set()).update(info["qids"])
    return out


def _select(base: str, prior: set[str]):
    """Return (qids, cf_layer, fallback_l1) for one dataset dir."""
    import os
    eligible: dict[str, int] = {}
    layer9: dict[str, int] = {}
    for qid in sorted(os.listdir(base), key=_qid_sort_key):
        cfg, rbc = _load_lattice(base, qid)
        if cfg.get("halted"):
            continue
        crit = cfg["criteria"]
        res = bfs_search(crit, rbc, start_node=crit, direction="down")
        if not res["found"]:
            continue
        if 2 <= res["size"] <= 8:
            eligible[qid] = res["size"]
        elif res["size"] == 9:
            layer9[qid] = res["size"]
    rank = lambda q: (0 if q in prior else 1, _qid_sort_key(q))
    qids = sorted(eligible, key=rank)[:N_SELECT]
    fallback = []
    if len(qids) < N_SELECT:
        fallback = sorted(layer9, key=rank)[: N_SELECT - len(qids)]
        qids += fallback
    cf_layer = {q: eligible.get(q, layer9.get(q)) for q in qids}
    return qids, cf_layer, fallback


def _valid_subsets(atoms, rbc, k, min_removals):
    out = []
    for combo in combinations(sorted(atoms), k):
        start = list(combo)
        if not rbc.get(combo_key(start)):
            continue
        res = bfs_search(start, rbc, start_node=start, direction="down")
        if res["found"] and k - res["size"] >= min_removals:
            out.append((start, k - res["size"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-122b")
    ap.add_argument("--lattice-root", default="/Users/kathling/coral-data-gen/lattices")
    ap.add_argument("--min-removals", type=int, default=2)
    args = ap.parse_args()

    run = f"{args.model}-ranked"
    prior = _prior_qids(args.model)
    selection, fallback_l1, pool = {}, {}, {run: {}}

    for ds, subdir in DATASET_SUBDIRS.items():
        base = f"{args.lattice_root}/{subdir}-{args.model}"
        qids, cf_layer, fallback = _select(base, prior.get(ds, set()))
        selection[ds] = {"qids": qids, "cf_layer": cf_layer}
        if fallback:
            fallback_l1[ds] = fallback
        if len(qids) < N_SELECT:
            print(f"  SHORTFALL {ds}: only {len(qids)}/{N_SELECT} base lattices even with L1 fallback")

        lattices = {qid: _load_lattice(base, qid) for qid in qids}
        rng = random.Random(0)
        cases = []
        for k in SIZES:
            per_qid = {}
            for qid in qids:
                cfg, rbc = lattices[qid]
                subs = _valid_subsets(cfg["criteria"], rbc, k, args.min_removals)
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
                print(f"    SHORTFALL {ds} k={k}: only {len(picked)}/{QUOTA} valid subsets")
            cases.extend(picked)
        for qid in qids:
            cfg, _ = lattices[qid]
            cases.append({"qid": qid, "size": 10, "start_node": sorted(cfg["criteria"]),
                          "cf_removals": 10 - selection[ds]["cf_layer"][qid]})
        pool[run][ds] = cases
        depths = Counter(c["cf_removals"] for c in cases if c["size"] < 10)
        print(f"  {ds}: {len(cases)} cases; k<10 depth dist {dict(sorted(depths.items()))}")

    sel_path = f"data/lattice_selection_16_{run}.json"
    json.dump({
        "description": (
            f"{N_SELECT} lattices per dataset from {args.lattice_root}/<dataset>-{args.model} "
            "(ranking-only prompt, no score elicitation). Filter: non-halted AND BFS "
            "(down from full set) CF layer in [2, 8]; where short, layer-9 (L1) lattices "
            "fill the quota (see fallback_l1). Eligible qids ranked by presence in "
            f"{PRIOR_SELECTION} for the same model (comparability), then ascending qid. "
            "Generated by scripts.shared.generate_ranked_pool."
        ),
        "filter": "non-halted AND bfs_cf_layer in [2, 8] (L1 fallback where short)",
        "n_per_config": N_SELECT,
        "runs": [run],
        "datasets": list(DATASET_SUBDIRS),
        "fallback_l1": fallback_l1,
        "selection": {run: selection},
    }, open(sel_path, "w"))
    print(f"wrote {sel_path}")

    cases_path = f"data/cases_16x20_cfmin{args.min_removals}_{run}.json"
    json.dump({
        "description": (
            f"{QUOTA} sub-lattices per size k=3..9, sampled from the {N_SELECT} base lattices "
            f"in {sel_path}. A subset is valid iff BFS (down from the subset) finds a CF and "
            f"the shallowest CF is >= {args.min_removals} removals below the start node; depth "
            "beyond the floor is natural, not quota'd. Sampling: random.Random(0) shuffle per "
            "qid, round-robin over base qids; shortfalls logged. k=10 cases are the base "
            "lattices themselves (cf_removals = 10 - base CF layer; 1 for L1 fallbacks). "
            "Generated by scripts.shared.generate_ranked_pool."
        ),
        "target": f"cf_removals >= {args.min_removals} for all k",
        "quota_per_size": QUOTA,
        "sizes": SIZES + [10],
        "cases": pool,
    }, open(cases_path, "w"))
    print(f"wrote {cases_path}")


if __name__ == "__main__":
    main()
