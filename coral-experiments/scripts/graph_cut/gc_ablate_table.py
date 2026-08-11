"""Ablation for the Graph Cut Approach.

Writes results/graph_cut_ablate.json.
"""

from __future__ import annotations

import json

from scripts.criteria.msmarco.run_subsample import score_proxy
from scripts.graph_cut.run_graph_cut import (DATASETS, RESULTS, load_cases,
                                             summarize)
from search.bfs import bfs_search
from search.graph_cut import graph_cut_search
from search.rank import era_search, sra_search

BUDGETS = (0, 2, 4, 8, 16)
VARIANTS = (("full", {}),
            ("no tier", {"tier": False}),
            ("no rescue", {"pool": False}),
            ("no flip", {"flip_boost": False}),
            ("graph 1st", {"graph_first": True}),
            ("no entropy", {"entropy_cut": False}),
            ("entropy only", {"tier": False, "pool": False}))


def main():
    out = {}
    for label, ddir, lat, gr in DATASETS:
        rows = load_cases(ddir, lat, gr)
        n = len(rows)
        block = {"n": n, "queries": len({r["qid"] for r in rows}),
                 "bfs_calls": round(sum(r["bfs"]["calls"] for r in rows) / n, 2),
                 "sra": summarize(
                     [score_proxy(r["bfs"],
                                  sra_search(r["criteria"], r["rbc"],
                                             start_node=r["start"],
                                             direction="down"),
                                  r["rbc"], r["start"]) for r in rows], n),
                 "sweep": {}}
        print(f"\n{label} (n={n}, {block['queries']} queries)")
        print(f"  BFS oracle       {block['bfs_calls']:>6} calls")
        s = block["sra"]
        print(f"  SRA (no budget)  {s['accuracy_pct']:>5}% @ {s['mean_calls']:>6}")
        for b in BUDGETS:
            cells = {"era": summarize(
                [score_proxy(r["bfs"],
                             era_search(r["criteria"], r["rbc"], budget=b,
                                        start_node=r["start"], direction="down"),
                             r["rbc"], r["start"]) for r in rows], n)}
            for name, kw in VARIANTS:
                cells[name] = summarize(
                    [score_proxy(r["bfs"],
                                 graph_cut_search(r["criteria"], r["rbc"],
                                                  r["supports"], budget=b,
                                                  start_node=r["start"], **kw),
                                 r["rbc"], r["start"]) for r in rows], n)
            block["sweep"][b] = cells
            row = "  ".join(f"{k} {cells[k]['accuracy_pct']:>5}% @ "
                            f"{cells[k]['mean_calls']:>6}"
                            for k in ("era", "full", "no tier", "no rescue",
                                      "no flip", "graph 1st", "no entropy",
                                      "entropy only"))
            print(f"  b={b:<2} {row}")
        out[label] = block

    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "graph_cut_ablate.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
