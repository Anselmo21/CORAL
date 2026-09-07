"""Sweep graph_cut_search against ERA over the Phase-1 budget, on qwen3-32b."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from scripts.criteria.msmarco.run_subsample import score_proxy
from search.bfs import bfs_search
from search.graph_cut import graph_cut_search
from search.lattice import combo_key, restrict_rbc
from search.rank import era_search

REPO = Path(__file__).resolve().parents[2]
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
    ("sonnet pragma", "sonnet_pragma",
     "datasets/qwen3-32b-temp0/sonnet-pragmas-qwen3-32b-temp0.json",
     "graphs/qwen32b-sonnet-pragma-cf-removal"),
]
BUDGETS = (0, 2, 4, 8, 16)


def read_json(rel):
    return json.loads((DATA_GEN / rel).read_text(encoding="utf-8"))


def graph_supports(graphs_dir, qid, concepts, answers):
    """{answer -> supporting concepts} from the graph's Layer-3 to Layer-4 edges."""
    def match(lab, pool):
        """Some graphs number their labels ('1. trustworthiness: ...'); others keep
        only the answer's title ('Answer A (Rent and Invest Default)'), others
        just the title's parenthetical ('Rent and Invest Default'), and the
        llama31-8b graphs shorten answers to a bare name ('Pseudoephedrine
        (Sudafed)') or drift on case ('Eltamd' for 'EltaMD'). Strip the
        number, then take a prefix match, then a parenthetical match, then a
        containment match (exact case, then case-folded), each only when
        exactly one candidate qualifies."""
        if lab in pool:
            return lab
        stripped = re.sub(r"^\s*\d+[.)]\s*", "", lab)
        if stripped in pool:
            return stripped
        hits = [p for p in pool if p.startswith(stripped)]
        if len(hits) == 1:
            return hits[0]
        hits = [p for p in pool if f"({stripped})" in p]
        if len(hits) == 1:
            return hits[0]
        hits = [p for p in pool if stripped in p]
        if len(hits) == 1:
            return hits[0]
        hits = [p for p in pool if stripped.casefold() in p.casefold()]
        return hits[0] if len(hits) == 1 else None

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
        if s in l4 and t in l3:
            # a few llama31-8b graphs write the support edge answer-first
            s, t = t, s
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


def main():
    out = {}
    for label, dataset_dir, lattice_rel, graphs_rel in DATASETS:
        rows = load_cases(dataset_dir, lattice_rel, graphs_rel)
        n = len(rows)
        out[label] = {"n": n, "sweep": {}}
        print(f"\n{label} (n={n})")
        for b in BUDGETS:
            cells = {}
            for name, run in (
                ("era", lambda r: era_search(r["criteria"], r["rbc"], budget=b,
                                             start_node=r["start"],
                                             direction="down")),
                ("graph_cut", lambda r: graph_cut_search(
                    r["criteria"], r["rbc"], r["supports"], budget=b,
                    start_node=r["start"])),
            ):
                cells[name] = summarize(
                    [score_proxy(r["bfs"], run(r), r["rbc"], r["start"])
                     for r in rows], n)
            out[label]["sweep"][b] = cells
            e, g = cells["era"], cells["graph_cut"]
            print(f"  b={b:<2} ERA {e['accuracy_pct']:>5}% @ {e['mean_calls']:>6}"
                  f"   graph_cut {g['accuracy_pct']:>5}% @ {g['mean_calls']:>6}")

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "graph_cut.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
