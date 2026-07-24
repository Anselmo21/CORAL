"""SRA and ERA sweeps over rank aggregation methods.

Runs the searches with native combsum and the methods in search/flagr.py on
the 156-case subsample of each dataset. Results are scored against BFS and
written to logged_results/<dataset>/rank_agg_sra.json with a summary table.
Pass --era to sweep era_search at budgets 4 and 8 instead. That mode writes
rank_agg_era.json and leaves the SRA results alone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.criteria.sonnet.run_subsample import score_proxy
from search.bfs import bfs_search
from search.flagr import make_era_aggregator, make_sra_aggregator
from search.lattice import combo_key, hop_distance, parse_combo_key
from search.rank import era_search, sra_search

DATASETS = {
    "msmarco_criteria":       "data/llama31-8b/msmarco-criteria-lattices.json",
    "msmarco_pragma":         "data/llama31-8b/msmarco-pragmas-lattices.json",
    "sonnet_criteria":        "data/llama31-8b/sonnet-criteria-lattices.json",
    "sonnet_pragma":          "data/llama31-8b/sonnet-pragmas-lattices.json",
    "qwen3_msmarco_criteria": "data/qwen3-32b/msmarco-criteria-lattices.json",
    "qwen3_msmarco_pragma":   "data/qwen3-32b/msmarco-pragmas-lattices.json",
    "qwen3_sonnet_criteria":  "data/qwen3-32b/sonnet-criteria-lattices.json",
    "qwen3_sonnet_pragma":    "data/qwen3-32b/sonnet-pragmas-lattices.json",
}

SRA_METHODS = ["borda", "combsum", "combmnz", "condorcet", "copeland",
               "outranking", "kemeny", "rra", "mc2", "mc4", "mct",
               "prefrel", "agglomerative", "dibra"]

# era only sweeps the methods that separated from borda in the sra sweep
ERA_METHODS = ["mct", "outranking", "kemeny", "mc4"]
ERA_BUDGETS = (4, 8)


def run_dataset(name: str, data_path: str, runners, out_name: str, verbose: bool = True) -> list[dict]:
    cases = json.loads(Path(f"logged_results/{name}/cases.json").read_text())
    qids = {str(c["qid"]) for c in cases}

    # load the non-halted lattices from the consolidated data file
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    lat = {}
    for qid, q in data.items():
        if qid == "_meta" or qid not in qids or q.get("halted"):
            continue
        criteria = list(q["criteria"])
        sorted_crit = sorted(criteria)
        rbc = {combo_key(parse_combo_key(k, sorted_crit)): v
               for k, v in (q.get("raw_answers") or {}).items()}
        if combo_key(criteria) in rbc:
            lat[qid] = (criteria, rbc)

    rows = []
    t0 = time.time()
    for i, c in enumerate(cases):
        qid, start = str(c["qid"]), c["start_node"]
        if qid not in lat:
            continue
        crit, rbc = lat[qid]

        bfs = bfs_search(crit, rbc, start_node=start, direction="down")
        row = {"case_id": c.get("case_id", i), "qid": qid, "size": c["size"],
               "bfs": {"key": bfs["key"], "size": bfs["size"], "calls": bfs["calls"]}}

        for col, fn in runners:
            cell = score_proxy(bfs, fn(crit, rbc, start), rbc, start)
            cell["hop_to_bfs"] = hop_distance(cell["proxy_key"], bfs["key"], start)
            row[col] = cell
        rows.append(row)

        if verbose and (i + 1) % 20 == 0:
            print(f"  {name}: {i + 1}/{len(cases)} cases ({time.time() - t0:.0f}s)", flush=True)

    out = Path(f"logged_results/{name}/{out_name}")
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return rows


def summarize(name: str, rows: list[dict], cols: list[str]) -> str:
    lines = [f"\n=== {name} (n={len(rows)}) ===",
             f"  {'method':<16} | {'acc':>6} | {'wrong_l':>7} | {'not_f':>5} | {'false_p':>7} | {'cost μ':>7}"]
    for col in cols:
        cells = [r[col] for r in rows if col in r]
        n = len(cells)
        if not n:
            continue
        acc = 100 * sum(x["outcome"] == "correct" for x in cells) / n
        wl = sum(x["outcome"] == "wrong_layer" for x in cells)
        nf = sum(x["outcome"] == "not_found" for x in cells)
        fp = sum(x["outcome"] == "false_positive" for x in cells)
        cost = sum(x["calls"] + x.get("budget_used", 0) for x in cells) / n
        lines.append(f"  {col:<16} | {acc:5.1f}% | {wl:>7} | {nf:>5} | {fp:>7} | {cost:7.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--era", action="store_true")
    args = ap.parse_args()

    if args.era:
        runners = []
        for b in ERA_BUDGETS:
            runners.append((f"native_b{b}",
                lambda c, r, s, b=b: era_search(c, r, budget=b, start_node=s,
                                                direction="down", propagate_scores=True)))
            for m in ERA_METHODS:
                runners.append((f"{m}_b{b}",
                    lambda c, r, s, b=b, fn=make_era_aggregator(m): era_search(
                        c, r, budget=b, aggregator_fn=fn, start_node=s,
                        direction="down", propagate_scores=True)))
        out_name = "rank_agg_era.json"
    else:
        runners = [("native",
            lambda c, r, s: sra_search(c, r, aggregator="combsum",
                                       start_node=s, direction="down"))]
        for m in SRA_METHODS:
            runners.append((m,
                lambda c, r, s, fn=make_sra_aggregator(m): sra_search(
                    c, r, aggregator=fn, start_node=s, direction="down")))
        out_name = "rank_agg_sra.json"

    cols = [col for col, _ in runners]
    for name in args.datasets:
        rows = run_dataset(name, DATASETS[name], runners, out_name)
        print(summarize(name, rows, cols))


if __name__ == "__main__":
    main()
