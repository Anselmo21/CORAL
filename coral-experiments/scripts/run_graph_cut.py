"""Load the qwen3-32b graphs and lattices the graph_cut search runs on."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from search.bfs import bfs_search
from search.lattice import combo_key, restrict_rbc


REPO = Path(__file__).resolve().parents[1]
DATA_GEN = Path(os.environ.get("CORAL_DATA_GEN", REPO.parent / "coral-data-gen"))
LOGGED = REPO / "logged_results" / "qwen3-32b-temp0"
RESULTS = REPO / "results"

DATASETS = [
    ("msmarco criteria", "msmarco_criteria",
     "datasets/qwen3-32b-temp0/msmarco-criteria-qwen3-32b-temp0.json",
     "graphs/qwen32b-msmarco-criteria-cf-removal"),
    ("msmarco pragma", "msmarco_pragma",
     "datasets/qwen3-32b-temp0/msmarco-pragmas-qwen3-32b-temp0.json",
     "graphs/qwen32b-msmarco-pragma-cf-removal"),
    ("sonnet criteria", "sonnet_criteria",
     "datasets/qwen3-32b-temp0/sonnet-criteria-qwen3-32b-temp0.json",
     "graphs/qwen32b-sonnet-criteria-cf-removal"),
]


def read_json(rel):
    return json.loads((DATA_GEN / rel).read_text(encoding="utf-8"))


def graph_supports(graphs_dir, qid, concepts, answers):
    """{answer -> supporting concepts} from the graph's Layer-3 to Layer-4 edges."""
    def match(lab, pool):
        """Some graphs number their labels ('1. trustworthiness: ...'); strip it."""
        if lab in pool:
            return lab
        stripped = re.sub(r"^\s*\d+[.)]\s*", "", lab)
        return stripped if stripped in pool else None

    try:
        data = read_json(f"{graphs_dir}/{qid}/graph.json")
    except FileNotFoundError:
        return None
    graph = data.get("graph")
    if not graph:
        return None
    concepts = set(concepts)
    label = {n["id"]: n.get("label", n["id"]) for n in graph["nodes"]}
    l3 = {n["id"] for n in graph["nodes"] if str(n.get("layer")) == "3"}
    l4 = {n["id"]: match(label[n["id"]], set(answers))
          for n in graph["nodes"] if str(n.get("layer")) == "4"}
    sup = {a: set() for a in answers}
    for e in graph["edges"]:
        s, t = e.get("source"), e.get("target")
        if s not in l3 or t not in l4 or l4[t] is None:
            continue
        c = match(label[s], concepts)
        if c is not None:
            sup[l4[t]].add(c)
    return sup


def load_cases(dataset_dir, lattice_rel, graphs_rel):
    """Cases with both a BFS counterfactual and a graph."""
    lattice = read_json(lattice_rel)
    cases = json.loads((LOGGED / dataset_dir / "cases.json").read_text())
    rows = []
    for case in cases:
        qid = str(case["qid"])
        row = lattice.get(qid)
        if row is None:
            continue
        criteria = row["criteria"]
        start = list(case["start_node"])
        rbc = restrict_rbc(row["raw_answers"], start, criteria)
        bfs = bfs_search(criteria, rbc, start_node=start, direction="down")
        sup = graph_supports(graphs_rel, qid, criteria,
                             rbc.get(combo_key(start), []))
        if not bfs["found"] or sup is None:
            continue
        rows.append({"qid": qid, "criteria": criteria, "start": start,
                     "rbc": rbc, "bfs": bfs,
                     "supports": {a: (s & set(start)) for a, s in sup.items()}})
    return rows


def summarize(cells, n):
    calls = sum(c.get("calls", 0) + c.get("budget_used", 0) for c in cells)
    correct = sum(1 for c in cells if c.get("outcome") == "correct")
    outcomes = {}
    for c in cells:
        outcomes[c["outcome"]] = outcomes.get(c["outcome"], 0) + 1
    return {"accuracy_pct": round(100 * correct / n, 1),
            "mean_calls": round(calls / n, 2), "outcomes": outcomes}
