"""Compare named ERA configurations against BFS across the four datasets.

Edit the CONFIGS list to define what to run; everything else (BFS reference,
case-loading, accuracy/cost reporting) is shared. Cost = era_search.calls +
era_search.budget_used (apples-to-apples vs BFS.calls).

Examples:
    python -m scripts.shared.compare_configs
    python -m scripts.shared.compare_configs --sizes 9 10
    python -m scripts.shared.compare_configs --dataset curated_pragma
    python -m scripts.shared.compare_configs --sizes 10 --dataset msmarco_criteria
"""
import argparse
import functools
import json
from importlib import import_module

from search.bfs import bfs_search
from search.rank import era_search
from metric import evaluate

DATASETS = [
    ("curated_pragma",    "scripts.pragma.curated.run_subsample"),
    ("msmarco_pragma",   "scripts.pragma.msmarco.run_subsample"),
    ("msmarco_criteria", "scripts.criteria.msmarco.run_subsample"),
    ("curated_criteria",  "scripts.criteria.curated.run_subsample"),
]

# Edit this list to compare different ERA configs. Each entry is
# (label, kwargs) where kwargs is passed straight to era_search.
CONFIGS = [
    ("ERA(b=16, prop=T)", {"budget": 16, "propagate_scores": True}),
    ("ERA(b=16, prop=F)", {"budget": 16, "propagate_scores": False}),
    ("ERA(b=8,  prop=T)", {"budget":  8, "propagate_scores": True}),
    ("ERA(b=4,  prop=T)", {"budget":  4, "propagate_scores": True}),
]


def make_eval_lattices(name, modpath, sizes):
    with open(f"logged_results/llama31-8b/{name}/cases.json") as f:
        cases = json.load(f)
    mod = import_module(modpath)
    qids = sorted({c["qid"] for c in cases})
    loader = getattr(mod, "_load_lattices", None) or getattr(mod, "_load_pragma_lattices")
    lattices = loader(qids)
    out = []
    for c in cases:
        if sizes and c["size"] not in sizes:
            continue
        if c["qid"] not in lattices:
            continue
        criteria, rbc = lattices[c["qid"]]
        out.append({
            "criteria": criteria,
            "rankings_by_combo": rbc,
            "start_node": c["start_node"],
        })
    return out


def run_search(eval_lats, search_callable, cost_fn):
    total_cost = 0

    def fn(criteria, rbc, *, start_node):
        nonlocal total_cost
        r = search_callable(criteria, rbc, start_node=start_node)
        total_cost += cost_fn(r)
        return r

    res = evaluate(eval_lats, fn)
    n = res["total"]
    return {
        "accuracy": res["score"],
        "correct": res["correct"],
        "total": n,
        "avg_cost": total_cost / n if n else 0.0,
    }


def run_bfs(eval_lats):
    return run_search(eval_lats, bfs_search, lambda r: r["calls"])


def run_era(eval_lats, kwargs):
    callable_ = functools.partial(era_search, direction="down", **kwargs)
    return run_search(eval_lats, callable_, lambda r: r["calls"] + r.get("budget_used", 0))


def fmt_row(label, stats, bfs_cost):
    acc = stats["accuracy"]
    cost = stats["avg_cost"]
    if bfs_cost is None:
        gap = "—"
    else:
        gap = f"-{(1 - cost/bfs_cost)*100:>5.1f}%" if cost < bfs_cost else f"+{(cost/bfs_cost - 1)*100:>5.1f}%"
    return f"  {label:<22} {acc:>6.1%} ({stats['correct']:>3}/{stats['total']:<3})   {cost:>7.2f}   {gap}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=None,
                    help="Filter cases to these sub-lattice sizes (e.g. --sizes 9 10).")
    ap.add_argument("--dataset", choices=[d[0] for d in DATASETS], default=None,
                    help="Run only one dataset.")
    args = ap.parse_args()

    sizes = set(args.sizes) if args.sizes else None
    size_tag = f"size ∈ {sorted(sizes)}" if sizes else "all sizes"
    datasets = [d for d in DATASETS if (args.dataset is None or d[0] == args.dataset)]

    print(f"\nComparing configs on {size_tag}\n")

    for name, modpath in datasets:
        eval_lats = make_eval_lattices(name, modpath, sizes)
        if not eval_lats:
            print(f"  [{name}] no cases match filter, skipping\n")
            continue
        print(f"=== {name}  ({len(eval_lats)} cases) ===")
        print(f"  {'config':<22} {'accuracy':<16}   {'cost':<7}   {'vs BFS'}")
        print("  " + "-" * 64)

        bfs_stats = run_bfs(eval_lats)
        print(fmt_row("BFS", bfs_stats, None))
        for label, kwargs in CONFIGS:
            era_stats = run_era(eval_lats, kwargs)
            print(fmt_row(label, era_stats, bfs_stats["avg_cost"]))
        print()


if __name__ == "__main__":
    main()
