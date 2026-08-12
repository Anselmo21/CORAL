"""Sub-lattice BFS / SRA / ERA sweep on the Llama-3.3-70B lattices.

Mirror of scripts.llama31_70b.run_subsample: reuses the base per-case runner and
sampling utilities, points outputs at logged_results/llama33-70b/<dataset>/,
and picks 16 random qids per dataset from those whose full-set BFS finds a
counterfactual at layer <= 8 (i.e. at least 2 concepts must be removed).

Each dataset's raw file lives under data/llama33-70b/. Format matches the
llama31-70b flat-qid layout (rankings under raw_answers).

Usage:
    conda run -n coral python3 -m scripts.llama33_70b.run_subsample
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import scripts.criteria.msmarco.run_subsample as base
import scripts.llama33_70b.run_subsample as lmod
import scripts.shared.paper_results as pr
import scripts.shared.plot_paper_results as pp
from scripts.criteria.msmarco.run_subsample import run_one_case
from scripts.shared.sampling import sample_cases, SEED
from scripts.shared.plot_style import apply_plot_style
from search.bfs import bfs_search
from search.lattice import combo_key

# Extend the shared pipeline to include ERA(b=16) for these plots. Restricted to
# this script so other datasets keep their existing paper_results columns.
pr.ERA_BUDGETS = (4, 8, 16)
if not any(m[0] == "era_b16" for m in pp.METHODS):
    pp.METHODS.append(("era_b16", "ERA (b=16)", "D", "-", "#6a3d9a", 1.8, False))

# Restrict the hop-distance bars to cases where the minimal CF was NOT found
# (i.e. outcome != "correct"). Correct cases have hop=0 by definition and just
# dilute the mean.
import numpy as _np


def _hop_stats_missed_only(rows_by_size, col):
    xs, means, stds = [], [], []
    for size, bucket in rows_by_size.items():
        hops = []
        for r in bucket:
            cell = r.get(col)
            if not cell or cell.get("outcome") == "correct":
                continue
            h = cell.get("hop_to_bfs")
            if h is not None:
                hops.append(h)
        if hops:
            xs.append(size)
            means.append(float(_np.mean(hops)))
            stds.append(float(_np.std(hops)))
    return xs, means, stds


pp._hop_stats = _hop_stats_missed_only

DATA_DIR = Path("data/llama33-70b")
OUT_PREFIX = "llama33-70b"
N_QIDS = 16
MAX_CF_LAYER = 8            # full-set BFS CF must lie at layer <= 8

DATASETS = [
    ("msmarco_criteria", "msmarco-criteria-llama33-70b.json", "Llama-3.3-70B MSMARCO criteria"),
    ("msmarco_pragma",   "msmarco-pragmas-llama33-70b.json",  "Llama-3.3-70B MSMARCO pragma"),
    ("curated_criteria",  "curated-criteria-llama33-70b.json",  "Llama-3.3-70B Curated criteria"),
    ("curated_pragma",    "curated-pragmas-llama33-70b.json",   "Llama-3.3-70B Curated pragma"),
]

# Set per dataset before loaders run (paper_results.rerun_dataset re-imports
# this module and calls _load_lattices, which reads DATASET_FILE).
DATASET_FILE: str = ""


def _read_all() -> dict:
    path = DATA_DIR / DATASET_FILE
    return {qid: lat for qid, lat in json.loads(path.read_text()).items()
            if qid != "_meta"}


def _searchable(lat: dict) -> bool:
    if lat.get("halted"):
        return False
    crit = lat.get("criteria") or []
    return bool(crit) and combo_key(crit) in (lat.get("raw_answers") or {})


def _cf_le_max(lat: dict) -> bool:
    crit = list(lat["criteria"])
    res = bfs_search(crit, lat["raw_answers"], start_node=crit, direction="down")
    return res["found"] and res["size"] <= MAX_CF_LAYER


def _pick_qids() -> list[str]:
    raw = _read_all()
    eligible = [q for q, lat in raw.items() if _searchable(lat) and _cf_le_max(lat)]
    eligible.sort(key=lambda q: int(q) if q.isdigit() else q)
    rng = random.Random(SEED)
    n = min(N_QIDS, len(eligible))
    return sorted(rng.sample(eligible, n),
                  key=lambda q: int(q) if q.isdigit() else q)


def _load_lattices(qids):
    raw = _read_all()
    out = {}
    for qid in qids:
        lat = raw.get(qid)
        if lat is not None and _searchable(lat):
            out[qid] = (list(lat["criteria"]), lat["raw_answers"])
    return out


# paper_results.rerun_dataset expects the module to expose score_proxy.
score_proxy = base.score_proxy


def _patch_paths(out_dir: str) -> None:
    base.OUT_DIR = out_dir
    base.CASES_PATH = f"{out_dir}/cases.json"
    base.RESULTS_PATH = f"{out_dir}/results.json"


def _run_one_dataset(name: str, fname: str) -> None:
    global DATASET_FILE
    DATASET_FILE = fname
    out_dir = f"logged_results/{OUT_PREFIX}/{name}"
    os.makedirs(f"{out_dir}/per_case", exist_ok=True)
    _patch_paths(out_dir)

    qids = _pick_qids()
    print(f"  picked {len(qids)} qids: {qids}")
    lattices = _load_lattices(qids)

    cases = sample_cases(lattices, seed=SEED)
    with open(f"{out_dir}/cases.json", "w") as f:
        json.dump(cases, f, indent=2)

    results = [run_one_case(i, c, *lattices[c["qid"]]) for i, c in enumerate(cases)]
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {out_dir}/  ({len(results)} cases)")


def main() -> None:
    for name, fname, nice in DATASETS:
        print(f"\n=== {nice} ({name}) ===")
        _run_one_dataset(name, fname)

    for name, fname, nice in DATASETS:
        lmod.DATASET_FILE = fname
        rows = pr.rerun_dataset(f"{OUT_PREFIX}/{name}", "scripts.llama33_70b.run_subsample")
        with open(f"logged_results/{OUT_PREFIX}/{name}/paper_results.json", "w") as f:
            json.dump(rows, f, indent=2)

    apply_plot_style()
    for name, fname, nice in DATASETS:
        pp.plot_dataset(f"{OUT_PREFIX}/{name}", f"{OUT_PREFIX}/{name}", nice)


if __name__ == "__main__":
    main()
