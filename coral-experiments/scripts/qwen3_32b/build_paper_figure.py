"""Build the Qwen3-32B paper figure from the committed 160-case flat dirs.

BFS/SRA/ERA come from results.json when it matches the current cases.json sample.
If results.json is stale (a different sample than cases.json / cc_cases.json), they
are recomputed on the current lattices with era_search, reproducing the logged
method. LSM comes from cc_results_budget_4/8, joined by case_id via merge_dataset.
"""
from __future__ import annotations

import json

import scripts.qwen3_32b.run_subsample as q
from search.bfs import bfs_search
from search.lattice import hop_distance
from search.rank import era_search, sra_search
from scripts.shared.merge_cc_paper import merge_dataset
from scripts.shared.plot_paper_results import plot_dataset
from scripts.shared.plot_style import apply_plot_style

DATASETS = [
    ("qwen3_msmarco_criteria", ["msmarco-criteria-qwen3-32.json",
                                "msmarco-criteria-qwen3-32b-missing.json"], "Qwen MSMARCO criteria"),
    ("qwen3_curated_criteria",  ["curated-criteria-qwen3-32b.json"], "Qwen Curated criteria"),
    ("qwen3_msmarco_pragma",   ["msmarco-pragmas-qwen3-32b.json"], "Qwen MSMARCO pragma"),
    ("qwen3_curated_pragma",    ["curated-pragmas-qwen3-32b.json"], "Qwen Curated pragma"),
]

ERA_BUDGETS = (4, 8)


QWEN3_PRELIM_ROOT = "logged_results/qwen3-32b-prelim"


def _ds_dir(name: str) -> str:
    return f"{QWEN3_PRELIM_ROOT}/{name.removeprefix('qwen3_')}"


def _load_results(name: str) -> list[dict]:
    d = json.loads(open(f"{_ds_dir(name)}/results.json").read())
    return d["results"] if isinstance(d, dict) and "results" in d else d


def _proxy_cell(cell: dict, bfs_key: str | None, concepts: list[str]) -> dict:
    proxy_key = cell.get("proxy_key")
    return {
        "outcome":     cell.get("outcome"),
        "calls":       cell.get("calls", 0),
        "budget_used": cell.get("budget_used", 0),
        "proxy_key":   proxy_key,
        "proxy_layer": cell.get("proxy_layer"),
        "hop_to_bfs":  hop_distance(proxy_key, bfs_key, concepts),
    }


def _is_stale(name: str, cases: list[dict]) -> bool:
    ids = {(str(c["qid"]), tuple(sorted(c["start_node"]))) for c in cases}
    res = {(str(r["qid"]), tuple(sorted(r["start_node"]))) for r in _load_results(name)}
    return len(ids & res) < len(ids)


def _rows_from_results(name: str) -> list[dict]:
    rows = []
    for r in _load_results(name):
        start = r["start_node"]
        bfs_key = r["bfs"].get("key")
        row = {
            "case_id": r["case_id"], "qid": r["qid"], "size": r["size"], "start_node": start,
            "bfs": {**r["bfs"], "hop_to_bfs": 0},
            "sra": _proxy_cell(r["sra"], bfs_key, start),
        }
        for b in ERA_BUDGETS:
            row[f"era_b{b}"] = _proxy_cell(r["era"][f"era_fixed_{b}"], bfs_key, start)
        rows.append(row)
    return rows


def _rows_recomputed(files: list[str], cases: list[dict]) -> list[dict]:
    q.DATASET_FILES = files
    lat = q._load_lattices(sorted({c["qid"] for c in cases}))
    rows = []
    for i, c in enumerate(cases):
        qid, start = c["qid"], c["start_node"]
        if qid not in lat:
            continue
        crit, rbc = lat[qid]
        bfs = bfs_search(crit, rbc, start_node=start, direction="down")
        bfs_for_proxy = {"key": bfs["key"], "size": bfs["size"]}
        row = {
            "case_id": c.get("case_id", i), "qid": qid, "size": c["size"], "start_node": start,
            "bfs": {"found": bfs["found"], "key": bfs["key"], "layer": bfs["size"],
                    "calls": bfs["calls"], "hop_to_bfs": 0},
            "sra": _score(bfs_for_proxy, sra_search(crit, rbc, start_node=start, direction="down"), rbc, start),
        }
        for b in ERA_BUDGETS:
            er = era_search(crit, rbc, budget=b, start_node=start, direction="down", propagate_scores=True)
            row[f"era_b{b}"] = _score(bfs_for_proxy, er, rbc, start)
        rows.append(row)
    return rows


def _score(bfs_for_proxy: dict, proxy: dict, rbc: dict, start: list[str]) -> dict:
    cell = q.score_proxy(bfs_for_proxy, proxy, rbc, start)
    cell["hop_to_bfs"] = hop_distance(cell.get("proxy_key"), bfs_for_proxy["key"], start)
    return cell


def main() -> None:
    apply_plot_style()
    for name, files, nice in DATASETS:
        cases = json.loads(open(f"{_ds_dir(name)}/cases.json").read())
        stale = _is_stale(name, cases)
        rows = _rows_recomputed(files, cases) if stale else _rows_from_results(name)
        with open(f"{_ds_dir(name)}/paper_results.json", "w") as f:
            json.dump(rows, f, indent=2)
        merge_dataset(name.removeprefix("qwen3_"), root=QWEN3_PRELIM_ROOT)
        plot_dataset(name.removeprefix("qwen3_"), name, nice, root=QWEN3_PRELIM_ROOT)
        print(f"  {nice}: {len(rows)} rows{' [recomputed]' if stale else ''} -> plots/{name}/combined.png")


if __name__ == "__main__":
    main()
