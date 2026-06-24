"""Instrumented ERA Phase 2: capture per-layer entropy values + cutoff decisions.

Re-runs ERA on a representative slice of each dataset (k=8 cases), exposing
metrics that don't show up in the regular results:

  - For each Phase 2 layer:
      * entropy_values: list of (key, entropy, displacement)
      * gap_max: largest entropy gap (sorted desc)
      * gap_max_frac: gap_max / (max_entropy - min_entropy)
      * max_gap_idx: index where the cutoff lands (Python max() tie-break = last index)
      * n_high_entropy: |high_entropy_keys| (nodes on the high-entropy side of cutoff)
      * n_displacement_0: nodes where estimated top-1 == baseline
      * n_pruned: displacement-0 AND NOT in high_entropy_keys

Per-layer rows go to reports/era_pragma_pruning/data/layers.jsonl
"""
from __future__ import annotations

import json
import os
import random
from itertools import combinations

from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key, iter_search_layers
from search.rank import _aggregate_with_scores, _get_child_rankings

K_TARGET = 8        # focus on k=8 sub-lattices (rich enough but cheap)
N_CASES_PER_DATASET = 30
SEED = 42
OUT_PATH = "reports/era_pragma_pruning/data/layers.jsonl"


def instrumented_era(
    criteria: list[str],
    rankings_by_combo: dict[str, list[str]],
    start_node: list[str],
    budget: int,
    case_meta: dict,
    out_records: list[dict],
) -> dict:
    """ERA down-search with Phase-2 layer logging. Mirrors era_search exactly."""
    aggregator = "combsum"
    entropy_fn = softmax_entropy
    cutoff_fn = max_jump_cutoff

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)
    s = len(start_node)

    known: dict[str, list[str]] = {}
    estimated_rankings: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}

    for c in criteria:
        known[c] = rankings_by_combo[c]

    # Phase 1
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

    # Phase 2
    for layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        unknown_nodes: list[tuple[list[str], str, int, float]] = []
        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                continue
            if key not in estimated_rankings:
                unknown_nodes.append((subset, key, 0, float("inf")))
                continue
            ranking = estimated_rankings[key]
            ent = entropies.get(key, 0.0)
            displacement = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            unknown_nodes.append((subset, key, displacement, ent))

        if not unknown_nodes:
            continue

        entropy_inputs = [(key, ent) for _, key, _, ent in unknown_nodes]
        high_entropy_keys = set(cutoff_fn(entropy_inputs, len(unknown_nodes)))

        # Reproduce the cutoff decision details
        sorted_nodes = sorted(entropy_inputs, key=lambda x: x[1], reverse=True)
        entropy_vals = [e for _, e in sorted_nodes]
        gaps = [entropy_vals[i] - entropy_vals[i + 1] for i in range(len(entropy_vals) - 1)]
        if gaps:
            max_gap = max(gaps)
            ent_range = entropy_vals[0] - entropy_vals[-1]
            max_gap_frac = (max_gap / ent_range) if ent_range > 0 else float("nan")
            max_gap_idx = max(range(len(gaps)), key=lambda i: (gaps[i], i))
        else:
            max_gap = 0.0
            max_gap_frac = float("nan")
            max_gap_idx = 0

        n_disp_0 = sum(1 for _, _, d, _ in unknown_nodes if d < 1)
        n_pruned = sum(1 for _, key, d, _ in unknown_nodes
                       if d < 1 and key not in high_entropy_keys)

        out_records.append({
            **case_meta,
            "layer_size": layer_size,
            "n_unknown": len(unknown_nodes),
            "entropy_values": entropy_vals,
            "max_gap": max_gap,
            "max_gap_frac": max_gap_frac,
            "max_gap_idx": max_gap_idx,
            "n_high_entropy": len(high_entropy_keys),
            "n_disp_0": n_disp_0,
            "n_pruned": n_pruned,
        })

        # Continue the search (so calls and outcome are accurate but we don't need them here)
        candidates = [
            (subset, key, d, e)
            for subset, key, d, e in unknown_nodes
            if not (d < 1 and key not in high_entropy_keys)
        ]
        candidates.sort(key=lambda x: (-x[2], x[3]))
        for subset, key, _, _ in candidates:
            gt = rankings_by_combo.get(key)
            if gt:
                known[key] = gt
            if gt and gt[0] != baseline_answer:
                return {"found": True}
    return {"found": False}


def _sample_cases(cases_path: str, k: int, n: int, rng: random.Random) -> list[dict]:
    with open(cases_path) as f:
        cases = json.load(f)
    pool = [c for c in cases if c["size"] == k]
    rng.shuffle(pool)
    return pool[:n]


def main() -> None:
    rng = random.Random(SEED)
    out_records: list[dict] = []

    # --- MSMARCO criteria ---
    from scripts.msmarco.run_subsample import _load_all_qids as m_q, _load_lattices as m_l
    lats = m_l(m_q())
    cases = _sample_cases("data/msmarco_subsample/cases.json", K_TARGET, N_CASES_PER_DATASET, rng)
    print(f"MSMARCO criteria: {len(cases)} cases @ k={K_TARGET}")
    for i, c in enumerate(cases):
        crit, rbc = lats[c["qid"]]
        budget = max(1, int(2**K_TARGET * 0.10))
        instrumented_era(crit, rbc, c["start_node"], budget,
                         {"dataset": "msmarco_criteria", "qid": c["qid"], "case_idx": i},
                         out_records)

    # --- SONNET criteria ---
    from scripts.sonnet.run_subsample import _load_all_qids as s_q, _load_lattices as s_l
    lats = s_l(s_q())
    cases = _sample_cases("data/sonnet_subsample/cases.json", K_TARGET, N_CASES_PER_DATASET, rng)
    print(f"SONNET  criteria: {len(cases)} cases @ k={K_TARGET}")
    for i, c in enumerate(cases):
        crit, rbc = lats[c["qid"]]
        budget = max(1, int(2**K_TARGET * 0.10))
        instrumented_era(crit, rbc, c["start_node"], budget,
                         {"dataset": "sonnet_criteria", "qid": c["qid"], "case_idx": i},
                         out_records)

    # --- MSMARCO pragma ---
    from scripts.pragma.msmarco.run_subsample import _load_all_qids as pm_q, _load_lattices as pm_l
    lats = pm_l(pm_q())
    cases = _sample_cases("data/msmarco_pragma/cases.json", K_TARGET, N_CASES_PER_DATASET, rng)
    print(f"MSMARCO pragma:   {len(cases)} cases @ k={K_TARGET}")
    for i, c in enumerate(cases):
        crit, rbc = lats[c["qid"]]
        budget = max(1, int(2**K_TARGET * 0.10))
        instrumented_era(crit, rbc, c["start_node"], budget,
                         {"dataset": "msmarco_pragma", "qid": c["qid"], "case_idx": i},
                         out_records)

    # --- SONNET pragma ---
    from scripts.pragma.sonnet.run_subsample import _load_all_qids as ps_q, _load_pragma_lattices as ps_l
    lats = ps_l(ps_q())
    cases = _sample_cases("data/sonnet_pragma/cases.json", K_TARGET, N_CASES_PER_DATASET, rng)
    print(f"SONNET  pragma:   {len(cases)} cases @ k={K_TARGET}")
    for i, c in enumerate(cases):
        crit, rbc = lats[c["qid"]]
        budget = max(1, int(2**K_TARGET * 0.10))
        instrumented_era(crit, rbc, c["start_node"], budget,
                         {"dataset": "sonnet_pragma", "qid": c["qid"], "case_idx": i},
                         out_records)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWrote {len(out_records)} layer rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
