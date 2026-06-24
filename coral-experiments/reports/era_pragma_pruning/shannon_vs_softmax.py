"""Diagnose where Shannon and softmax entropy diverge on the 15-lattice regen.

For each k=8 and k=10 case, re-runs ERA at pct_10 with both entropy functions,
instrumenting Phase 1's actual budget allocation (which (layer, key) pairs got
ground-truth lookups) and Phase 2's outcome (which layer it returned at).

Compares them on cases where they disagree on `correct` vs not-correct.

Output: reports/era_pragma_pruning/data/shannon_vs_softmax.jsonl + console
breakdown of the disagreement cases.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations

from scripts.pragma.msmarco_regen.run_subsample import _load_lattices, score_proxy
from scripts.pragma.msmarco_regen.run_15_lattices import _load_qids
from search.bfs import bfs_search
from search.heuristics import (
    max_jump_cutoff,
    shannon_entropy,
    softmax_entropy,
)
from search.lattice import combo_key, iter_search_layers
from search.rank import _aggregate_with_scores, _get_child_rankings

CASES_PATH = "data/15_lattices_cf_layer2to8/cases.json"
OUT_PATH = "reports/era_pragma_pruning/data/shannon_vs_softmax.jsonl"

K_VALUES = {8, 10}


def instrumented_era(
    criteria: list[str],
    rankings_by_combo: dict[str, list[str]],
    start_node: list[str],
    budget: int,
    entropy_fn,
):
    """ERA down-search mirroring search.rank.era_search exactly, but capturing
    which (layer_size, key) got budget-spent ground-truth lookups, and which
    layer Phase 2 returned at."""
    aggregator = "combsum"
    cutoff_fn = max_jump_cutoff
    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)
    s = len(start_node)

    known: dict[str, list[str]] = {}
    estimated_rankings: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}
    phase1_picks: list[tuple[int, str]] = []   # (layer_size, key)

    for c in criteria:
        known[c] = rankings_by_combo[c]

    budget_remaining = budget
    for layer_size in range(2, s):
        layer_nodes = []
        for combo in combinations(range(s), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)
            voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
            if not voter_rankings:
                continue
            ranking, scores = _aggregate_with_scores(voter_rankings, answers, aggregator)
            if not scores:
                continue
            estimated_rankings[key] = ranking
            ent = entropy_fn(scores)
            entropies[key] = ent
            layer_nodes.append((key, ent))
        selected = cutoff_fn(layer_nodes, budget_remaining)
        for key in selected:
            known[key] = rankings_by_combo[key]
            budget_remaining -= 1
            phase1_picks.append((layer_size, key))

    budget_used = budget - budget_remaining

    # Phase 2
    calls = 0
    pruned_nodes = 0
    phase2_trace: list[dict] = []
    for layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        unknown_nodes = []
        free_found = None
        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                if known[key] and known[key][0] != baseline_answer:
                    free_found = (layer_size, key)
                    break
                continue
            if key not in estimated_rankings:
                unknown_nodes.append((subset, key, 0, float("inf")))
                continue
            ranking = estimated_rankings[key]
            ent = entropies.get(key, 0.0)
            displacement = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            unknown_nodes.append((subset, key, displacement, ent))

        if free_found is not None:
            phase2_trace.append({"layer_size": layer_size, "outcome": "free_cf",
                                 "key": free_found[1]})
            return {
                "found": True, "key": free_found[1], "size": free_found[0],
                "calls": calls, "budget_used": budget_used,
                "pruned_nodes": pruned_nodes,
                "phase1_picks": phase1_picks, "phase2_trace": phase2_trace,
            }

        if not unknown_nodes:
            phase2_trace.append({"layer_size": layer_size, "outcome": "no_unknowns"})
            continue

        entropy_inputs = [(k, e) for _, k, _, e in unknown_nodes]
        high_entropy_keys = set(cutoff_fn(entropy_inputs, len(unknown_nodes)))
        candidates = [(s_, k_, d_, e_) for s_, k_, d_, e_ in unknown_nodes
                      if not (d_ < 1 and k_ not in high_entropy_keys)]
        pruned_nodes += len(unknown_nodes) - len(candidates)
        candidates.sort(key=lambda x: (-x[2], x[3]))

        layer_outcome = {"layer_size": layer_size,
                         "n_unknown": len(unknown_nodes),
                         "n_pruned": len(unknown_nodes) - len(candidates),
                         "n_checked": 0}

        for subset, key, _, _ in candidates:
            calls += 1
            layer_outcome["n_checked"] += 1
            gt = rankings_by_combo.get(key)
            if gt:
                known[key] = gt
            if gt and gt[0] != baseline_answer:
                layer_outcome["outcome"] = "cf_found"
                layer_outcome["key"] = key
                phase2_trace.append(layer_outcome)
                return {
                    "found": True, "key": key, "size": len(subset),
                    "calls": calls, "budget_used": budget_used,
                    "pruned_nodes": pruned_nodes,
                    "phase1_picks": phase1_picks, "phase2_trace": phase2_trace,
                }
        layer_outcome["outcome"] = "no_cf_in_layer"
        phase2_trace.append(layer_outcome)

    return {
        "found": False, "key": None, "size": None,
        "calls": calls, "budget_used": budget_used,
        "pruned_nodes": pruned_nodes,
        "phase1_picks": phase1_picks, "phase2_trace": phase2_trace,
    }


def main() -> None:
    with open(CASES_PATH) as f:
        all_cases = json.load(f)
    cases = [c for c in all_cases if c["size"] in K_VALUES]
    print(f"Inspecting {len(cases)} cases (k∈{sorted(K_VALUES)})")

    qids = _load_qids()
    lattices = _load_lattices(qids)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rows: list[dict] = []
    disagreements: list[dict] = []
    for case in cases:
        qid = str(case["qid"])
        start_node = case["start_node"]
        size = case["size"]
        criteria, rbc = lattices[qid]
        budget = max(1, int(2**size * 0.10))

        bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
        bfs_layer = bfs_res["size"]

        sm = instrumented_era(criteria, rbc, start_node, budget, softmax_entropy)
        sh = instrumented_era(criteria, rbc, start_node, budget, shannon_entropy)

        sm_scored = score_proxy(bfs_res, sm, rbc, start_node)
        sh_scored = score_proxy(bfs_res, sh, rbc, start_node)

        row = {
            "qid": qid, "size": size, "bfs_layer": bfs_layer,
            "softmax": {"outcome": sm_scored["outcome"], "layer": sm.get("size"),
                        "calls": sm["calls"], "budget_used": sm["budget_used"],
                        "phase1_layer_dist": dict(sorted(
                            {l: sum(1 for ll, _ in sm["phase1_picks"] if ll == l)
                             for l, _ in sm["phase1_picks"]}.items())),
                        "phase2_layers_visited": [t["layer_size"] for t in sm["phase2_trace"]]},
            "shannon": {"outcome": sh_scored["outcome"], "layer": sh.get("size"),
                        "calls": sh["calls"], "budget_used": sh["budget_used"],
                        "phase1_layer_dist": dict(sorted(
                            {l: sum(1 for ll, _ in sh["phase1_picks"] if ll == l)
                             for l, _ in sh["phase1_picks"]}.items())),
                        "phase2_layers_visited": [t["layer_size"] for t in sh["phase2_trace"]]},
        }
        rows.append(row)
        if (sm_scored["outcome"] == "correct") != (sh_scored["outcome"] == "correct"):
            disagreements.append(row)

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows → {OUT_PATH}")

    # Console summary
    print(f"\nDisagreements (one method correct, the other wrong): {len(disagreements)}")
    for r in disagreements:
        sm = r["softmax"]
        sh = r["shannon"]
        print(f"\n  qid={r['qid']} k={r['size']} bfs_layer={r['bfs_layer']}")
        print(f"    softmax: outcome={sm['outcome']}  returned_layer={sm['layer']}  "
              f"calls={sm['calls']} budget_used={sm['budget_used']}")
        print(f"      Phase 1 budget by layer: {sm['phase1_layer_dist']}")
        print(f"      Phase 2 layers visited:  {sm['phase2_layers_visited']}")
        print(f"    shannon: outcome={sh['outcome']}  returned_layer={sh['layer']}  "
              f"calls={sh['calls']} budget_used={sh['budget_used']}")
        print(f"      Phase 1 budget by layer: {sh['phase1_layer_dist']}")
        print(f"      Phase 2 layers visited:  {sh['phase2_layers_visited']}")


if __name__ == "__main__":
    main()
