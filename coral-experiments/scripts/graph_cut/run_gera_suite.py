"""GERA against the logged pools of every answering model.

Replays GERA (graph_cut_search), ERA (mra_search, equal to GERA with the
graph hooks off) and GRA (GERA without the entropy cut) on each model's
logged case pool, reusing the logged BFS, SRA and LSM cells so every method
is scored over the same cases. All three are swept over the ablation
budgets; GoT only (the graph's own proposals, no lattice search) rounds out
the pruning ablations. Dumps per-case cells for the strip plots.

Writes results/gera_suite.json and results/gera_suite_cells/<model>_<ds>.json.

Usage:
    python3 -m scripts.graph_cut.run_gera_suite --verify   # coverage only
    python3 -m scripts.graph_cut.run_gera_suite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.criteria.msmarco.run_subsample import score_proxy
from scripts.graph_cut.run_graph_cut import graph_supports
from search.graph_cut import graph_cut_search
from search.lattice import combo_key, restrict_rbc
from search.rank import mra_search

REPO = Path(__file__).resolve().parents[2]
LOGGED = REPO / "logged_results"
RESULTS = REPO / "results"
RUNS = REPO / "coral-runs-lattices"
DATA_GEN = REPO.parent / "coral-data-gen"

HEADLINE_BUDGET = 4
SWEEP_BUDGETS = (0, 2, 4, 8, 16, 32, 64, 128)

DATASETS = ("msmarco_criteria", "msmarco_pragma", "curated_criteria", "curated_pragma")

# lattice: (kind, path template); pool: which logged file carries bfs+sra
MODELS = {
    "llama31-8b": {
        "lattice": ("consolidated", "data/llama31-8b/{dsf}-lattices.json"),
        "pool": "per_case",
        "graphs": "llama31-8b-{gds}-cf-removal",
    },
    "llama31-70b": {
        "lattice": ("consolidated", "data/llama31-70b/{dsf}-llama31-70b.json"),
        "pool": "paper_results",
        "graphs": "llama31-70b-{gds}-cf-removal",
    },
    "qwen3-32b": {
        # the logged pool replays 156/156 only against the temp0 lattices;
        # data/qwen3-32b/*-lattices.json is a different elicitation
        "lattice": ("consolidated", "../coral-data-gen/datasets/qwen3-32b-temp0/{dst}-qwen3-32b-temp0.json"),
        "pool": "paper_results",
        "graphs": "qwen32b-{gds}-cf-removal",
    },
    "qwen35-122b-100": {
        "lattice": ("rundir", "coral-runs-lattices/qwen35-122b-100/{dsd}"),
        "pool": "results",
        "graphs": "qwen35-122b-{gds}-cf-removal",
    },
    "qwen35-122b-ranked-cf2": {
        # ranked-elicitation lattices (answer texts, coral-data-gen layout);
        # pool and LSM live under the logged -ranked-cf2 root
        "lattice": ("rundir", "../coral-data-gen/lattices/{dst}-qwen35-122b"),
        "pool": "results",
        "graphs": "qwen35-122b-{gds}-cf-removal",
    },
}

# dataset name spellings: dsf = lattice file stem, dsd = run dir, gds = graphs dir
DS_FILE = {"msmarco_criteria": "msmarco-criteria", "msmarco_pragma": "msmarco-pragmas",
           "curated_criteria": "curated-criteria", "curated_pragma": "curated-pragmas"}
DS_TEMP0 = {"msmarco_criteria": "msmarco-criteria", "msmarco_pragma": "msmarco-pragmas",
            "curated_criteria": "sonnet-criteria", "curated_pragma": "sonnet-pragmas"}
DS_GRAPH = {"msmarco_criteria": "msmarco-criteria", "msmarco_pragma": "msmarco-pragma",
            "curated_criteria": "sonnet-criteria", "curated_pragma": "sonnet-pragma"}


def load_lattices(model: str, ds: str) -> dict[str, tuple[list[str], dict]]:
    kind, tmpl = MODELS[model]["lattice"]
    path = REPO / tmpl.format(dsf=DS_FILE[ds], dsd=DS_FILE[ds], dst=DS_TEMP0[ds])
    out: dict[str, tuple[list[str], dict]] = {}
    if kind == "consolidated":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for qid, q in raw.items():
            if qid == "_meta" or not isinstance(q, dict):
                continue
            crit = list(q.get("criteria") or [])
            rbc = q.get("raw_answers") or {}
            if crit and combo_key(crit) in rbc:
                out[qid] = (crit, rbc)
        return out
    for d in sorted(path.iterdir()):
        answers = d / "answers.json"
        config = d / "config.json"
        if not answers.exists() or not config.exists():
            continue
        cfg = json.loads(config.read_text(encoding="utf-8"))
        # coral-runs configs say "atoms"; coral-data-gen configs say "criteria"
        crit = list(cfg.get("atoms") or cfg.get("criteria") or [])
        plausible = cfg.get("plausible_answers") or []
        rbc = {}
        for key, cell in json.loads(answers.read_text(encoding="utf-8")).items():
            ranking = cell.get("ranking") if isinstance(cell, dict) else cell
            if ranking and isinstance(ranking[0], int):
                ranking = [plausible[i] for i in ranking]
            rbc[key] = ranking
        if crit and combo_key(crit) in rbc:
            out[d.name] = (crit, rbc)
    return out


def load_pool(model: str, ds: str) -> list[dict]:
    root = LOGGED / model / ds
    kind = MODELS[model]["pool"]
    if kind == "per_case":
        return [json.loads(f.read_text(encoding="utf-8"))
                for f in sorted((root / "per_case").glob("case_*.json"))]
    name = "paper_results.json" if kind == "paper_results" else "results.json"
    return json.loads((root / name).read_text(encoding="utf-8"))


def load_lsm(model: str, ds: str) -> dict[int, list[dict]]:
    """Logged LSM rows per budget. Older runs carry no start_node and are
    joined by case_id instead; case_id and (qid, start node) agree row for row
    wherever both exist."""
    out: dict[int, list[dict]] = {}
    for path in sorted((LOGGED / model / ds).glob("cc_results_budget_*.json")):
        b = int(path.stem.rsplit("_", 1)[1])
        out[b] = json.loads(path.read_text(encoding="utf-8"))
    return out


def match_lsm(lsm, case, qid, start):
    """The case's LSM row per budget, by start node or by case_id."""
    out = {}
    for b, rows_b in lsm.items():
        for r in rows_b:
            if str(r["query_id"]) != qid:
                continue
            if "start_node" in r:
                if tuple(r["start_node"]) == tuple(start):
                    out[b] = r
                    break
            elif r.get("case_id") == case.get("case_id"):
                out[b] = r
                break
    return out


def got_only_search(rbc, supports, start_node, greedy=False):
    """The graph's own proposal, evaluate_graphs.py's way.

    Each alternative answer proposes retaining its support; the graph's
    counterfactual is the largest proposal, and one call verifies it - the
    got-min convention behind the recoverability numbers. greedy=True keeps
    probing down the proposal list instead, as an upper bound on what the
    graph alone could deliver.
    """
    start_key = combo_key(start_node)
    baseline_ranking = rbc.get(start_key, [])
    if not baseline_ranking:
        return {"found": False, "key": None, "size": None, "ranking": None,
                "calls": 0, "budget_used": 0, "pruned_nodes": 0}
    baseline = baseline_ranking[0]
    start = set(start_node)
    proposals, seen = [], set()
    for answer, sup in supports.items():
        if answer == baseline:
            continue
        s = frozenset(sup & start)
        if s == frozenset(start) or s in seen:
            continue
        seen.add(s)
        proposals.append(s)
    proposals.sort(key=lambda s: (-len(s), combo_key(sorted(s))))
    if not greedy:
        proposals = proposals[:1]
    calls = 0
    for s in proposals:
        key = combo_key(sorted(s))
        ranking = rbc.get(key)
        calls += 1
        if ranking and ranking[0] != baseline:
            return {"found": True, "key": key, "size": len(s), "ranking": ranking,
                    "calls": calls, "budget_used": 0, "pruned_nodes": 0}
    return {"found": False, "key": None, "size": None, "ranking": None,
            "calls": calls, "budget_used": 0, "pruned_nodes": 0}


def load_cases(model: str, ds: str, verbose=False):
    """Pool rows joined with their lattice, graph supports and LSM cell."""
    lattices = load_lattices(model, ds)
    lsm = load_lsm(model, ds)
    graphs_rel = "graphs/" + MODELS[model]["graphs"].format(gds=DS_GRAPH[ds])
    rows, dropped = [], {"no_lattice": 0, "no_graph": 0, "no_bfs_cf": 0, "bfs_layer_moved": 0}
    for case in load_pool(model, ds):
        qid = str(case["qid"])
        start = list(case["start_node"])
        logged_bfs = case["bfs"]
        if qid not in lattices:
            dropped["no_lattice"] += 1
            continue
        crit, rbc_full = lattices[qid]
        rbc = restrict_rbc(rbc_full, start, crit)
        if not logged_bfs.get("found"):
            dropped["no_bfs_cf"] += 1
            continue
        bfs_key = logged_bfs.get("key")
        ranking = rbc.get(bfs_key)
        baseline = (rbc.get(combo_key(start)) or [""])[0]
        if not ranking or ranking[0] == baseline:
            dropped["bfs_layer_moved"] += 1
            continue
        sup = graph_supports(graphs_rel, qid, crit, rbc.get(combo_key(start), []))
        if sup is None:
            dropped["no_graph"] += 1
            continue
        rows.append({
            "qid": qid, "start": start, "size": case["size"], "criteria": crit,
            "rbc": rbc,
            "bfs": {"found": True, "key": bfs_key, "size": logged_bfs.get("layer"),
                    "calls": logged_bfs.get("calls")},
            "sra": case.get("sra"),
            "lsm": match_lsm(lsm, case, qid, start),
            "supports": {a: (set(s) & set(start)) for a, s in sup.items()},
        })
    if verbose:
        budgets = sorted({b for r in rows for b in r["lsm"]})
        n_lsm = {b: sum(1 for r in rows if b in r["lsm"]) for b in budgets}
        print(f"  {model}/{ds}: {len(rows)} cases kept, {dropped} dropped, lsm={n_lsm}")
    return rows


def agg(cells, n=None):
    n = n or len(cells)
    if not cells:
        return {"accuracy_pct": None, "mean_calls": None, "n": 0}
    acc = 100.0 * sum(1 for c in cells if c.get("outcome") == "correct") / n
    cost = float(np.mean([c.get("calls", 0) + c.get("budget_used", 0) for c in cells]))
    return {"accuracy_pct": round(acc, 1), "mean_calls": round(cost, 2), "n": n}


def lsm_cell(r, budget):
    """Logged LSM row reshaped to the score_proxy cell schema."""
    lsm = r["lsm"].get(budget)
    if lsm is None:
        return None
    return {"outcome": lsm["outcome"], "calls": lsm["calls"],
            "budget_used": lsm.get("budget", 0), "proxy_layer": lsm.get("proxy_layer")}


def run_model_dataset(model: str, ds: str, verbose=False):
    rows = load_cases(model, ds, verbose=verbose)
    cells_dump = []
    for r in rows:
        kw = dict(start_node=r["start"])
        entry = {"qid": r["qid"], "start_node": r["start"], "size": r["size"],
                 "bfs": r["bfs"], "sra": r["sra"]}
        for b in sorted(r["lsm"]):
            entry[f"lsm_b{b}"] = lsm_cell(r, b)
        for b in SWEEP_BUDGETS:
            gera = graph_cut_search(r["criteria"], r["rbc"], r["supports"],
                                    budget=b, **kw)
            entry[f"gera_b{b}"] = score_proxy(r["bfs"], gera, r["rbc"], r["start"])
            gra = graph_cut_search(r["criteria"], r["rbc"], r["supports"],
                                   budget=b, entropy_cut=False, **kw)
            entry[f"gra_b{b}"] = score_proxy(r["bfs"], gra, r["rbc"], r["start"])
            era = mra_search(r["start"], r["rbc"], budget=b, start_node=r["start"],
                             direction="down", normalize_fn="softmax")
            entry[f"era_b{b}"] = score_proxy(r["bfs"], era, r["rbc"], r["start"])
        got = got_only_search(r["rbc"], r["supports"], r["start"])
        entry["got_only"] = score_proxy(r["bfs"], got, r["rbc"], r["start"])
        greedy = got_only_search(r["rbc"], r["supports"], r["start"], greedy=True)
        entry["got_greedy"] = score_proxy(r["bfs"], greedy, r["rbc"], r["start"])
        cells_dump.append(entry)
    return rows, cells_dump


def summarize(cells_dump):
    n = len(cells_dump)
    out = {"n": n,
           "bfs_calls": round(float(np.mean([c["bfs"]["calls"] for c in cells_dump])), 2) if n else None,
           "sra": agg([c["sra"] for c in cells_dump if c["sra"]], n),
           "sweep": {}, "ablations": {}, "lsm": {}}
    lsm_budgets = sorted({int(k.rsplit("b", 1)[1]) for c in cells_dump
                          for k in c if k.startswith("lsm_b")})
    for b in lsm_budgets:
        cells = [c[f"lsm_b{b}"] for c in cells_dump if c.get(f"lsm_b{b}")]
        # every method is scored over the same cases; a partially-run LSM
        # budget must fail loudly, not shrink its own denominator
        if len(cells) != n:
            raise ValueError(f"lsm b={b}: {len(cells)} of {n} cases covered")
        out["lsm"][b] = agg(cells, n)
    for b in SWEEP_BUDGETS:
        out["sweep"][b] = {m: agg([c[f"{m}_b{b}"] for c in cells_dump], n)
                           for m in ("gera", "gra", "era")}
    out["ablations"] = {"gera": out["sweep"][HEADLINE_BUDGET]["gera"],
                        "no_entropy": out["sweep"][HEADLINE_BUDGET]["gra"],
                        "no_graph": out["sweep"][HEADLINE_BUDGET]["era"],
                        "got_only": agg([c["got_only"] for c in cells_dump], n),
                        "got_greedy": agg([c["got_greedy"] for c in cells_dump], n)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="report coverage and exit")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    if args.verify:
        for model in args.models:
            for ds in DATASETS:
                load_cases(model, ds, verbose=True)
        return

    out = {}
    cells_dir = RESULTS / "gera_suite_cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        out[model] = {}
        for ds in DATASETS:
            rows, cells_dump = run_model_dataset(model, ds, verbose=True)
            out[model][ds] = summarize(cells_dump)
            slim = [{k: v for k, v in c.items() if k != "rbc"} for c in cells_dump]
            (cells_dir / f"{model}_{ds}.json").write_text(json.dumps(slim), encoding="utf-8")
            s = out[model][ds]
            g4 = s["sweep"][4]["gera"]
            print(f"{model}/{ds}: n={s['n']} bfs={s['bfs_calls']}"
                  f" gera_b4 {g4['accuracy_pct']}%@{g4['mean_calls']}")
    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "gera_suite.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
