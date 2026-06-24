"""Inspect qid 51843 at k=10: ERA(b=16) succeeds where ERA(pct_10) fails.

Re-runs both ERA configs with full Phase 1 + Phase 2 instrumentation and
shows what each did differently.
"""
from __future__ import annotations

from itertools import combinations

from scripts.pragma.msmarco_regen.run_subsample import _load_lattices, score_proxy
from search.bfs import bfs_search
from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key, iter_search_layers
from search.rank import _aggregate_with_scores, _get_child_rankings


def instrumented_era(criteria, rbc, start_node, budget, label):
    aggregator = "combsum"
    entropy_fn = softmax_entropy
    cutoff_fn = max_jump_cutoff
    start_key = combo_key(start_node)
    baseline_ranking = rbc.get(start_key, [])
    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)
    s = len(start_node)

    known: dict[str, list[str]] = {}
    estimated: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}
    phase1_picks: list[tuple[int, str]] = []

    for c in criteria:
        known[c] = rbc[c]

    budget_rem = budget
    for layer_size in range(2, s):
        layer_nodes = []
        for combo in combinations(range(s), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)
            voters = _get_child_rankings(subset, known, estimated)
            if not voters:
                continue
            ranking, scores = _aggregate_with_scores(voters, answers, aggregator)
            if not scores:
                continue
            estimated[key] = ranking
            entropies[key] = entropy_fn(scores)
            layer_nodes.append((key, entropies[key]))
        selected = cutoff_fn(layer_nodes, budget_rem)
        for key in selected:
            known[key] = rbc[key]
            budget_rem -= 1
            phase1_picks.append((layer_size, key))

    budget_used = budget - budget_rem

    print(f"\n{'─'*70}")
    print(f"  {label}: budget={budget} → spent {budget_used}")
    print(f"{'─'*70}")
    layer_dist: dict[int, int] = {}
    for l, _ in phase1_picks:
        layer_dist[l] = layer_dist.get(l, 0) + 1
    print(f"  Phase 1 budget by layer: {dict(sorted(layer_dist.items()))}")

    # Was the *actual* CF node lookuped up by Phase 1?
    # We'll discover during Phase 2

    # Phase 2
    calls = 0
    print(f"\n  Phase 2 trace:")
    for layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        # Check known cells for free CFs
        known_cfs = []
        unknown_nodes = []
        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                if known[key] and known[key][0] != baseline_answer:
                    known_cfs.append(key)
                continue
            if key not in estimated:
                unknown_nodes.append((subset, key, 0, float("inf")))
                continue
            ranking = estimated[key]
            ent = entropies.get(key, 0.0)
            disp = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            unknown_nodes.append((subset, key, disp, ent))

        if known_cfs:
            print(f"    layer {layer_size:>2}: found FREE CF among {len(subsets)} nodes → "
                  f"key='{known_cfs[0][:40]}...'  → RETURN (free, calls so far={calls})")
            return {"found": True, "key": known_cfs[0], "size": layer_size,
                    "calls": calls, "budget_used": budget_used}

        if not unknown_nodes:
            print(f"    layer {layer_size:>2}: all {len(subsets)} nodes known, none CF → skip")
            continue

        # Cutoff for pruning
        entropy_inputs = [(k, e) for _, k, _, e in unknown_nodes]
        high_ent = set(cutoff_fn(entropy_inputs, len(unknown_nodes)))
        candidates = [(s_, k_, d_, e_) for s_, k_, d_, e_ in unknown_nodes
                      if not (d_ < 1 and k_ not in high_ent)]
        pruned = len(unknown_nodes) - len(candidates)
        candidates.sort(key=lambda x: (-x[2], x[3]))

        n_disp0 = sum(1 for _, _, d, _ in unknown_nodes if d < 1)
        n_disp_pos = sum(1 for _, _, d, _ in unknown_nodes if d >= 1)
        n_known_here = sum(1 for s_ in subsets if combo_key(s_) in known)
        print(f"    layer {layer_size:>2}: total={len(subsets)} known={n_known_here} unknown={len(unknown_nodes)}  "
              f"disp≥1={n_disp_pos} disp=0={n_disp0}  pruned={pruned}  → check {len(candidates)}")

        # Try candidates
        found_idx = None
        for idx, (subset, key, _, _) in enumerate(candidates):
            calls += 1
            gt = rbc.get(key)
            if gt:
                known[key] = gt
            if gt and gt[0] != baseline_answer:
                found_idx = idx
                break

        if found_idx is not None:
            print(f"             → candidate #{found_idx+1} of {len(candidates)} was CF → RETURN at layer {layer_size}  (total calls={calls})")
            return {"found": True, "key": candidates[found_idx][1], "size": layer_size,
                    "calls": calls, "budget_used": budget_used}
        else:
            print(f"             → none of {len(candidates)} candidates was a CF → continue down  (calls={calls})")

    return {"found": False, "size": None, "calls": calls, "budget_used": budget_used}


def main() -> None:
    qid = "51843"
    lats = _load_lattices([qid])
    criteria, rbc = lats[qid]
    start_node = list(criteria)
    print(f"qid={qid}  k={len(criteria)}  baseline top-1: {rbc[combo_key(start_node)][0][:80]}")

    bfs_res = bfs_search(criteria, rbc, start_node=start_node, direction="down")
    print(f"\nBFS:  found CF at layer {bfs_res['size']} with {bfs_res['calls']} calls")
    print(f"      CF key: '{bfs_res['key'][:80]}...'")
    print(f"      CF top-1: {rbc[bfs_res['key']][0][:80]}")

    # ERA pct_10
    budget_pct10 = max(1, int(2**10 * 0.10))
    res_pct = instrumented_era(criteria, rbc, start_node, budget_pct10, "ERA pct_10")
    scored_pct = score_proxy(bfs_res, res_pct, rbc, start_node)
    print(f"\n  Final: outcome={scored_pct['outcome']}  layer={res_pct.get('size')}  "
          f"calls+budget={res_pct['calls']}+{res_pct['budget_used']}={res_pct['calls']+res_pct['budget_used']}")

    # ERA b=16
    res_16 = instrumented_era(criteria, rbc, start_node, 16, "ERA b=16")
    scored_16 = score_proxy(bfs_res, res_16, rbc, start_node)
    print(f"\n  Final: outcome={scored_16['outcome']}  layer={res_16.get('size')}  "
          f"calls+budget={res_16['calls']}+{res_16['budget_used']}={res_16['calls']+res_16['budget_used']}")


if __name__ == "__main__":
    main()
