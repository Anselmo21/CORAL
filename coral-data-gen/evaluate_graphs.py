"""Evaluate Graph-of-Thoughts counterfactuals against ground-truth lattices.

Naming convention: S denotes a RETAINED concept subset (concepts sent to the LLM).
A counterfactual = the LARGEST S ⊂ atoms whose LLM top-1 answer differs from the
baseline (full-set) answer. Bigger |S| = fewer concepts removed = more minimal
counterfactual.

For each query in graphs/qwen32b-msmarco-criteria-cf-removal/:
  - GoT-predicted counterfactual per alternative answer a_i:
      S_i = concepts with an edge (c -> a_i) in the graph
  - Ground-truth via BFS over the lattice, walking |atoms|, |atoms|-1, ...
    downward until we hit an S whose lattice top-1 differs from baseline.
    That S is S_true (the maximal retained subset that flips).
  - Distance = Jaccard(S_i, S_true).
  - Size gap = |S_i| - |S_true|. Positive => GoT retains more than the true
    maximal flipping subset (i.e. GoT claims a "tighter" counterfactual than
    actually works). Negative => GoT retains fewer (over-removes).
"""

from __future__ import annotations

import json
import os
import re
from itertools import combinations
from pathlib import Path
from statistics import mean, median

_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _normalize_label(label: str, atom_set: set) -> str | None:
    """Map a concept-node label to an atom string, tolerating a leading 'N. ' prefix."""
    if label in atom_set:
        return label
    stripped = _PREFIX_RE.sub("", label)
    if stripped in atom_set:
        return stripped
    return None

REPO = Path(__file__).resolve().parent

VARIANTS = {
    "criteria": {
        "graphs_dir": REPO / "graphs" / "qwen32b-msmarco-criteria-cf-removal",
        "lattice_dir": REPO / "lattices" / "msmarco-criteria-qwen3-32b",
        "out_path": REPO / "graph_eval_criteria_cf_removal.json",
    },
    "pragma": {
        "graphs_dir": REPO / "graphs" / "qwen32b-msmarco-pragma-cf-removal",
        "lattice_dir": REPO / "lattices" / "msmarco-pragmas-qwen3-32b",
        "out_path": REPO / "graph_eval_pragma_cf_removal.json",
    },
}


def load_lattice(lattice_dir: Path, qid: str) -> dict[frozenset[str], list[str]]:
    """Return {frozenset(retained concepts) -> ranked answer list}."""
    raw = json.loads((lattice_dir / qid / "answers.json").read_text())
    out: dict[frozenset[str], list[str]] = {}
    for k, v in raw.items():
        concepts = frozenset() if k == "" else frozenset(s.strip() for s in k.split(","))
        out[concepts] = v
    return out


def extract_got_subsets(graph: dict, atoms: list[str]):
    """For each non-baseline answer node, return S_i (retained subset the GoT proposes)."""
    nodes = graph["graph"]["nodes"]
    edges = graph["graph"]["edges"]
    atom_set = set(atoms)

    concept_label_by_id = {
        n["id"]: _normalize_label(n["label"], atom_set)
        for n in nodes if n["type"] in ("concept", "directive")
    }
    answer_nodes = [n for n in nodes if n["type"] == "answer"]
    baseline_label = graph["baseline_answer"]

    supports: dict[str, set[str]] = {a["id"]: set() for a in answer_nodes}
    for e in edges:
        if e["source"] in concept_label_by_id and e["target"] in supports:
            label = concept_label_by_id[e["source"]]
            if label is not None:
                supports[e["target"]].add(label)

    out = []
    for a in answer_nodes:
        if a["label"] == baseline_label:
            continue
        out.append({
            "answer_id": a["id"],
            "answer_label": a["label"],
            "S": supports[a["id"]],
        })
    return out


def bfs_max_flipping_subset(lattice, atoms, baseline_label):
    """Walk |S| = |atoms|-1, |atoms|-2, ... down. Return all subsets at the LARGEST |S|
    whose lattice top-1 != baseline.

    Only considers subsets present as lattice keys.
    """
    full = frozenset(atoms)
    full_top1 = lattice[full][0] if full in lattice else None
    lattice_complete = len(lattice) == 2 ** len(atoms)
    for retain_size in range(len(atoms) - 1, -1, -1):
        flips = []
        checked = 0
        for retained_tuple in combinations(sorted(atoms), retain_size):
            S = frozenset(retained_tuple)
            if S not in lattice:
                continue
            checked += 1
            if lattice[S][0] != baseline_label:
                flips.append({"S": S, "new_top1": lattice[S][0]})
        if flips:
            return {
                "max_retain_size": retain_size,  # |S_true|
                "num_max_flips": len(flips),
                "flips": flips,
                "full_top1": full_top1,
                "lattice_complete": lattice_complete,
                "lattice_coverage_at_max_size": checked,
            }
    return {
        "max_retain_size": None,
        "num_max_flips": 0,
        "flips": [],
        "full_top1": full_top1,
        "lattice_complete": lattice_complete,
        "lattice_coverage_at_max_size": 0,
    }


def jaccard(a: set, b: set) -> float:
    """Jaccard distance = |A △ B| / |A ∪ B|. 0 = identical, 1 = disjoint."""
    if not a and not b:
        return 0.0
    return len(a ^ b) / len(a | b)


def analyze_containment(pred_S, lattice, baseline, truth_flips):
    """Search subsets of pred_S (obtained by removing concepts from S_got).

    Returns:
      contains_any_true_max: True if any max-flipping S_true ⊆ pred_S.
      largest_flipping_subset_size: max |T| with T ⊆ pred_S, lattice[T][0] != baseline, T in lattice.
      levels_to_flipping_subset: |pred_S| - largest_flipping_subset_size, or None.
    """
    pred_S = frozenset(pred_S)
    contains_true_max = any(f["S"].issubset(pred_S) for f in truth_flips)

    pred_sorted = sorted(pred_S)
    n = len(pred_sorted)
    largest_flip_size = None
    for k in range(n, -1, -1):
        found = False
        for combo in combinations(pred_sorted, k):
            T = frozenset(combo)
            if T in lattice and lattice[T][0] != baseline:
                largest_flip_size = k
                found = True
                break
        if found:
            break

    return {
        "contains_true_max_cf": contains_true_max,
        "largest_flipping_subset_size": largest_flip_size,
        "levels_to_flipping_subset": (n - largest_flip_size) if largest_flip_size is not None else None,
    }


def evaluate_query(qid: str, graphs_dir: Path, lattice_dir: Path):
    graph = json.loads((graphs_dir / qid / "graph.json").read_text())
    atoms = graph["atoms"]
    n_atoms = len(atoms)
    baseline = graph["baseline_answer"]
    lattice = load_lattice(lattice_dir, qid)

    got_cfs = extract_got_subsets(graph, atoms)
    truth = bfs_max_flipping_subset(lattice, atoms, baseline)

    for cf in got_cfs:
        S = frozenset(cf["S"])
        actual_top1 = lattice[S][0] if S in lattice else None
        cf["retain_size"] = len(S)          # |S_i|
        cf["lattice_top1"] = actual_top1
        cf["flip_success"] = actual_top1 is not None and actual_top1 != baseline
        cf["target_match"] = actual_top1 is not None and actual_top1 == cf["answer_label"]
        cf["lattice_has_key"] = actual_top1 is not None

    # GoT's "min counterfactual" = largest S_i (fewest concepts removed)
    got_sorted = sorted(got_cfs, key=lambda c: -c["retain_size"])
    got_min = got_sorted[0] if got_sorted else None
    got_valid = [c for c in got_sorted if c["flip_success"]]
    got_best_valid = got_valid[0] if got_valid else None

    # Best-of-ties: among alts tied at the max |S_i|, prefer one that flips top-1,
    # then one whose flip matches its own predicted answer label.
    got_min_bot = None
    if got_sorted:
        max_size = got_sorted[0]["retain_size"]
        tied = [c for c in got_sorted if c["retain_size"] == max_size]
        # rank: flip_success, then target_match, then original order
        tied_ranked = sorted(tied,
                             key=lambda c: (not c["flip_success"], not c["target_match"]))
        got_min_bot = tied_ranked[0]

    def compare(pred_S):
        if truth["max_retain_size"] is None or pred_S is None:
            return {
                "jaccard": None,
                "size_gap": None,
                "closest_true_S": None,
                "size_match": None,
                "levels_below_true": None,
                "contains_true_max_cf": None,
                "largest_flipping_subset_size": None,
                "levels_to_flipping_subset": None,
            }
        best = min(truth["flips"], key=lambda f: jaccard(pred_S, f["S"]))
        size_gap = len(pred_S) - truth["max_retain_size"]
        containment = analyze_containment(pred_S, lattice, baseline, truth["flips"])
        return {
            "jaccard": jaccard(pred_S, best["S"]),
            "size_gap": size_gap,                           # (+) => S_got retains more (true CF is lower/smaller)
            "size_match": size_gap == 0,
            "levels_below_true": max(size_gap, 0),          # steps down the lattice from S_got to S_true's level
            "closest_true_S": sorted(best["S"]),
            **containment,
        }

    return {
        "qid": qid,
        "query": graph["query"],
        "baseline_answer": baseline,
        "num_atoms": n_atoms,
        "num_alternatives": len(got_cfs),
        "truth": {
            "lattice_complete": truth["lattice_complete"],
            "max_retain_size": truth["max_retain_size"],       # |S_true|
            "num_max_flips": truth["num_max_flips"],
            "full_top1_matches_baseline": (truth["full_top1"] == baseline) if truth["full_top1"] else None,
            "example_S_true": sorted(truth["flips"][0]["S"]) if truth["flips"] else None,
            "example_new_top1": truth["flips"][0]["new_top1"] if truth["flips"] else None,
        },
        "tied_at_max": sum(1 for c in got_sorted if got_min and c["retain_size"] == got_min["retain_size"]),
        "got_min_best_of_ties": None if got_min_bot is None else {
            "answer_label": got_min_bot["answer_label"],
            "S": sorted(got_min_bot["S"]),
            "retain_size": got_min_bot["retain_size"],
            "flip_success": got_min_bot["flip_success"],
            "target_match": got_min_bot["target_match"],
            "lattice_top1": got_min_bot["lattice_top1"],
            **{f"vs_true_{k}": v for k, v in compare(got_min_bot["S"]).items()},
        },
        "got_min": None if got_min is None else {
            "answer_label": got_min["answer_label"],
            "S": sorted(got_min["S"]),
            "retain_size": got_min["retain_size"],
            "flip_success": got_min["flip_success"],
            "target_match": got_min["target_match"],
            "lattice_top1": got_min["lattice_top1"],
            **{f"vs_true_{k}": v for k, v in compare(got_min["S"]).items()},
        },
        "got_best_valid": None if got_best_valid is None else {
            "answer_label": got_best_valid["answer_label"],
            "S": sorted(got_best_valid["S"]),
            "retain_size": got_best_valid["retain_size"],
            "target_match": got_best_valid["target_match"],
            "lattice_top1": got_best_valid["lattice_top1"],
            **{f"vs_true_{k}": v for k, v in compare(got_best_valid["S"]).items()},
        },
        "per_alternative": [
            {
                "answer_label": c["answer_label"],
                "retain_size": c["retain_size"],
                "flip_success": c["flip_success"],
                "target_match": c["target_match"],
                "lattice_top1": c["lattice_top1"],
                **{f"vs_true_{k}": v for k, v in compare(c["S"]).items()},
            }
            for c in got_sorted
        ],
    }


def summarize(results):
    incomplete = [r for r in results if not r["truth"]["lattice_complete"]]
    if incomplete:
        print(f"NOTE: {len(incomplete)} queries have incomplete lattices (singleton-only) — "
              f"excluded from BFS-based aggregate stats. qids: {[r['qid'] for r in incomplete]}")
    results = [r for r in results if r["truth"]["lattice_complete"]]
    baseline_mismatch = [r for r in results if r["truth"]["full_top1_matches_baseline"] is False]
    if baseline_mismatch:
        print(f"NOTE: {len(baseline_mismatch)} queries have baseline != lattice full-set top-1 — "
              f"excluded. qids: {[r['qid'] for r in baseline_mismatch]}")
    results = [r for r in results if r["truth"]["full_top1_matches_baseline"] is True]
    n = len(results)
    got_min_valid = [r for r in results if r["got_min"] and r["got_min"]["flip_success"]]
    got_min_target_match = [r for r in results if r["got_min"] and r["got_min"]["target_match"]]
    any_valid = [r for r in results if r["got_best_valid"]]

    def col(rs, path):
        out = []
        for r in rs:
            cur = r
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                out.append(cur)
        return out

    def stats(xs):
        if not xs:
            return "n=0"
        return f"n={len(xs)} mean={mean(xs):.3f} median={median(xs):.3f} min={min(xs):.3f} max={max(xs):.3f}"

    print(f"\n===== Aggregate over {n} queries =====")
    print(f"\nGoT best-S_i validity (flips top-1 in lattice): {len(got_min_valid)}/{n} = {len(got_min_valid)/n:.1%}")
    print(f"GoT best-S_i target match (lattice top-1 == GoT's predicted a_i): {len(got_min_target_match)}/{n} = {len(got_min_target_match)/n:.1%}")
    print(f"At least one S_i flips in lattice: {len(any_valid)}/{n} = {len(any_valid)/n:.1%}")

    print("\n-- GoT largest-S_i (its proposed minimal CF) vs true S_true --")
    print(f"  Jaccard   : {stats(col(results, ['got_min','vs_true_jaccard']))}")
    print(f"  Size gap  : {stats(col(results, ['got_min','vs_true_size_gap']))}   (positive => GoT retains more than S_true)")
    print(f"  |S_got|   : {stats(col(results, ['got_min','retain_size']))}")
    print(f"  |S_true|  : {stats([r['truth']['max_retain_size'] for r in results if r['truth']['max_retain_size'] is not None])}")

    # ---- Best-of-ties comparison ----
    n_tied = sum(1 for r in results if r.get("tied_at_max", 1) > 1)
    print(f"\n-- Ties at max |S_i|: {n_tied}/{n} queries have >1 alternative sharing max size --")
    bot_valid = sum(1 for r in results if r["got_min_best_of_ties"] and r["got_min_best_of_ties"]["flip_success"])
    bot_target = sum(1 for r in results if r["got_min_best_of_ties"] and r["got_min_best_of_ties"]["target_match"])
    print(f"Best-of-ties validity (flips top-1): {bot_valid}/{n} = {bot_valid/n:.1%}")
    print(f"Best-of-ties target match          : {bot_target}/{n} = {bot_target/n:.1%}")
    changed = [r for r in results
               if r["got_min"] and r["got_min_best_of_ties"]
               and r["got_min"]["answer_label"] != r["got_min_best_of_ties"]["answer_label"]]
    print(f"Best-of-ties picked a DIFFERENT alt than arbitrary got_min: {len(changed)} queries")
    for r in changed:
        print(f"  {r['qid']}: got_min flip={r['got_min']['flip_success']} tgt={r['got_min']['target_match']}"
              f"  -> bot flip={r['got_min_best_of_ties']['flip_success']} tgt={r['got_min_best_of_ties']['target_match']}")

    print("\n-- Best-valid S_i (largest S_i that actually flips) vs S_true --")
    print(f"  Jaccard   : {stats(col(results, ['got_best_valid','vs_true_jaccard']))}")
    print(f"  Size gap  : {stats(col(results, ['got_best_valid','vs_true_size_gap']))}")
    print(f"  |S_valid| : {stats(col(results, ['got_best_valid','retain_size']))}")

    # -------- Q1/Q2/Q3: size match, level distance, containment --------
    got_min_rs = [r for r in results if r["got_min"]]
    size_matches = col(got_min_rs, ['got_min', 'vs_true_size_match'])
    levels_below = col(got_min_rs, ['got_min', 'vs_true_levels_below_true'])
    contains_true = col(got_min_rs, ['got_min', 'vs_true_contains_true_max_cf'])
    largest_flip_subset = col(got_min_rs, ['got_min', 'vs_true_largest_flipping_subset_size'])
    levels_to_flip = col(got_min_rs, ['got_min', 'vs_true_levels_to_flipping_subset'])

    n_got = len(got_min_rs)
    n_true_smaller = sum(1 for r in got_min_rs if (r['got_min']['vs_true_size_gap'] or 0) > 0)
    n_true_bigger = sum(1 for r in got_min_rs if (r['got_min']['vs_true_size_gap'] or 0) < 0)
    n_size_match = sum(1 for m in size_matches if m)

    print("\n===== Q1: Does |S_got| match |S_true|? (GoT largest-S) =====")
    print(f"  size match : {n_size_match}/{n_got} = {n_size_match/n_got:.1%}")
    print(f"  |S_true| < |S_got| (true CF is smaller/lower): {n_true_smaller}/{n_got} = {n_true_smaller/n_got:.1%}")
    print(f"  |S_true| > |S_got| (true CF is larger — GoT over-removed): {n_true_bigger}/{n_got} = {n_true_bigger/n_got:.1%}")

    print("\n===== Q2: How many lattice levels below S_got is S_true? =====")
    lb_positive = [x for x in levels_below if x > 0]
    print(f"  |S_got| - |S_true| over cases where true CF is lower: {stats(lb_positive)}")
    print(f"  |S_got| - |S_true| over all cases (0 when S_true not lower): {stats(levels_below)}")

    print("\n===== Q3: Can we reach a true CF just by removing concepts from S_got? =====")
    # Restrict to cases where true CF is smaller (i.e. removing from S_got is even possible).
    smaller_rs = [r for r in got_min_rs if (r['got_min']['vs_true_size_gap'] or 0) > 0]
    n_smaller = len(smaller_rs)
    if n_smaller:
        n_super_max = sum(1 for r in smaller_rs if r['got_min']['vs_true_contains_true_max_cf'])
        n_contains_any = sum(1 for r in smaller_rs
                             if r['got_min']['vs_true_largest_flipping_subset_size'] is not None)
        print(f"  Cases where true CF is smaller than |S_got|: {n_smaller}")
        print(f"  S_got is a superset of some MAX-flipping S_true : {n_super_max}/{n_smaller} = {n_super_max/n_smaller:.1%}")
        print(f"  S_got contains SOME flipping subset (any size)  : {n_contains_any}/{n_smaller} = {n_contains_any/n_smaller:.1%}")
        # levels from S_got down to the largest flipping subset ⊂ S_got
        levels_to_reachable = [r['got_min']['vs_true_levels_to_flipping_subset']
                               for r in smaller_rs
                               if r['got_min']['vs_true_levels_to_flipping_subset'] is not None]
        print(f"  Levels to remove from S_got to reach ANY flipping subset: {stats(levels_to_reachable)}")
    else:
        print("  No cases where |S_true| < |S_got|.")


def run_variant(name: str, cfg: dict):
    graphs_dir = cfg["graphs_dir"]
    lattice_dir = cfg["lattice_dir"]
    out_path = cfg["out_path"]
    qids = sorted(os.listdir(graphs_dir))
    results = [evaluate_query(q, graphs_dir, lattice_dir) for q in qids]
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n########## Variant: {name} ##########")
    print(f"graphs:  {graphs_dir}")
    print(f"lattice: {lattice_dir}")
    print(f"Wrote per-query results to {out_path}")
    summarize(results)


def main():
    for name, cfg in VARIANTS.items():
        run_variant(name, cfg)


if __name__ == "__main__":
    main()
