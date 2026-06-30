"""Sub-lattice BFS / SRA / ERA sweep on the Qwen3-32B lattices.

Mirrors scripts.pragma.msmarco.run_16_lattices: imports the base run_subsample, points
its output at logged_results/qwen3-32b/<dataset>/, and runs the existing run_one_case
over the Qwen datasets from coral-data-gen. Adds a loader for the Qwen format (flat qid
keys, rankings under raw_answers) and a cc_cases export for the GPU CC step. CC/LSM
itself needs a GPU and runs separately via scripts.qwen3_32b.run_cc_full.

A dataset can read several consolidated files; their qids merge, and files that do not
exist yet are skipped, so this works before and after the missing-qid run is generated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import scripts.criteria.msmarco.run_subsample as base
import scripts.qwen3_32b.run_subsample as qmod
from scripts.criteria.msmarco.run_subsample import score_proxy, run_one_case
from scripts.shared.sampling import sample_cases, SEED
from scripts.shared.paper_results import rerun_dataset
from scripts.shared.merge_cc_paper import merge_dataset
from scripts.shared.plot_paper_results import plot_dataset
from scripts.shared.plot_style import apply_plot_style
from search.bfs import bfs_search
from search.lattice import combo_key, restrict_rbc

WORKSPACE = Path(__file__).resolve().parents[3]
QWEN_DIR = WORKSPACE / "coral-data-gen" / "datasets" / "qwen3-32b"

DATASETS = [
    ("msmarco_criteria", ["msmarco-criteria-qwen3-32.json",
                          "msmarco-criteria-qwen3-32b-missing.json"], "Qwen MSMARCO criteria"),
    ("msmarco_pragma",   ["msmarco-pragmas-qwen3-32b.json"], "Qwen MSMARCO pragma"),
    ("sonnet_criteria",  ["sonnet-criteria-qwen3-32b.json"], "Qwen Sonnet criteria"),
    ("sonnet_pragma",    ["sonnet-pragmas-qwen3-32b.json"], "Qwen Sonnet pragma"),
]

# Set per dataset before the loaders run.
DATASET_FILES: list[str] = []


def _read_all() -> dict:
    merged: dict = {}
    for fn in DATASET_FILES:
        path = QWEN_DIR / fn
        if not path.exists():
            continue
        for qid, lat in json.loads(path.read_text()).items():
            if qid != "_meta":
                merged[qid] = lat
    return merged


def _searchable(lat: dict) -> bool:
    if lat.get("halted"):
        return False
    crit = lat.get("criteria") or []
    return bool(crit) and combo_key(crit) in (lat.get("raw_answers") or {})


def _load_all_qids() -> list[str]:
    qids = [q for q, lat in _read_all().items() if _searchable(lat)]
    return sorted(qids, key=lambda q: int(q) if q.isdigit() else q)


def _load_lattices(qids):
    raw = _read_all()
    out = {}
    for qid in qids:
        lat = raw.get(qid)
        if lat is not None and _searchable(lat):
            out[qid] = (list(lat["criteria"]), lat["raw_answers"])
    return out


def _export_cc_cases(cases, raw):
    cc = []
    for case_id, case in enumerate(cases):
        qid, start = case["qid"], case["start_node"]
        lat = raw[qid]
        crit, rbc = list(lat["criteria"]), lat["raw_answers"]
        if not bfs_search(crit, rbc, start_node=start, direction="down")["found"]:
            continue
        cc.append({
            "case_id": case_id,
            "query_id": qid,
            "query": lat.get("question", ""),
            "sources": lat.get("sources") or "",
            "plausible_answers": list(lat.get("plausible_answers") or []),
            "criteria": start,
            "start_node": start,
            "direction": "down",
            "effective_size": len(start),
            "k": len(start),
            "rankings_by_combo": restrict_rbc(rbc, start, crit),
        })
    return cc


def _patch_paths(out_dir: str) -> None:
    base.OUT_DIR = out_dir
    base.CASES_PATH = f"{out_dir}/cases.json"
    base.RESULTS_PATH = f"{out_dir}/results.json"


def _run_one_dataset(name: str, files: list[str]) -> None:
    global DATASET_FILES
    DATASET_FILES = files
    out_dir = f"logged_results/qwen3-32b/{name}"
    os.makedirs(f"{out_dir}/per_case", exist_ok=True)
    _patch_paths(out_dir)

    raw = _read_all()
    lattices = _load_lattices(_load_all_qids())
    print(f"  searchable lattices: {len(lattices)}")

    cases = sample_cases(lattices, seed=SEED)
    with open(f"{out_dir}/cases.json", "w") as f:
        json.dump(cases, f, indent=2)
    with open(f"{out_dir}/cc_cases.json", "w") as f:
        json.dump(_export_cc_cases(cases, raw), f, indent=2)

    results = [run_one_case(i, c, *lattices[c["qid"]]) for i, c in enumerate(cases)]
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {out_dir}/  ({len(results)} cases)")


def main() -> None:
    for name, files, nice in DATASETS:
        print(f"\n=== {nice} ({name}) ===")
        _run_one_dataset(name, files)

    for name, files, nice in DATASETS:
        qmod.DATASET_FILES = files
        rows = rerun_dataset(f"qwen3-32b/{name}", "scripts.qwen3_32b.run_subsample")
        with open(f"logged_results/qwen3-32b/{name}/paper_results.json", "w") as f:
            json.dump(rows, f, indent=2)
        merge_dataset(f"qwen3-32b/{name}")  # folds in CC columns once cc_results exist

    apply_plot_style()
    for name, files, nice in DATASETS:
        plot_dataset(f"qwen3-32b/{name}", f"qwen3-32b/{name}", nice)


if __name__ == "__main__":
    main()
