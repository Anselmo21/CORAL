"""ERA's top-down search, with the GoT graph deciding where the probes go.

The graph says which concepts back each candidate answer. A node's contrast is
how much more of the baseline's backing it strips than its best rival's, so a
positive contrast is the model's own explanation calling for a flip there.

A node ERA prunes still gets a second chance at the end of the layer when the
graph vouches for it, taken in contrast order, and that pool closes as soon as
one of the stronger ones comes back holding the baseline. The nodes ERA is
unsure about get led by the top max-jump tier of contrast, and if every node in
that tier comes back holding the baseline, the rest of the layer is skipped.
When contrast is flat across a layer there is no tier and nothing gets skipped,
so a graph with nothing to say falls back to plain ERA on its own.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key, iter_search_layers
from search.rank import _get_child_voters, _normalize_max, _store_estimate


def graph_contrast(subset, supports, answers, baseline):
    """max over rivals of cov(rival) - cov(baseline), 0.0 when there are no rivals."""
    present = set(subset)
    cov = lambda a: (len(present & supports.get(a, set()))
                     / len(supports[a]) if supports.get(a) else 0.0)
    base = cov(baseline)
    return max((cov(a) - base for a in answers if a != baseline), default=0.0)


def _aggregate(voters, answers, weights):
    """aggregator_default with per-voter weights; all-ones weights reproduce it."""
    agg = {a: 0.0 for a in answers}
    for vk, sv in voters.items():
        sv = _normalize_max(sv)
        for a in answers:
            agg[a] += weights.get(vk, 1.0) * sv.get(a, 0.0)
    agg = _normalize_max(agg)
    return sorted(agg, key=lambda a: (-agg[a], a)), agg


def graph_cut_search(criteria, rbc, supports, *, budget, start_node,
                     tier=True, pool=True, flip_boost=True, graph_first=False,
                     entropy_cut=True, perm_rng=None):
    """Top-down counterfactual search with the graph guiding Phase 2.

    tier=False, pool=False, flip_boost=False reduces to era_search(direction=
    'down', propagate_scores=True). flip_boost weights a measured voter that
    already flipped by the node's voter count; it is an ERA-native change kept
    separable from the graph. graph_first probes the graph's tier ahead of the
    displaced nodes instead of behind them. entropy_cut=False keeps every node
    in the layer as a candidate, so nothing is pruned for low entropy and the
    layer stop is the only thing left skipping work. perm_rng permutes the
    graph's concept labels for attribution controls.
    """
    baseline_ranking = rbc.get(combo_key(start_node), [])
    if not baseline_ranking:
        return {"found": False, "key": None, "size": None, "ranking": None,
                "calls": 0, "budget_used": 0, "pruned_nodes": 0}
    baseline = baseline_ranking[0]
    answers = list(baseline_ranking)

    if perm_rng is not None:
        elems = sorted(start_node)
        shuffled = list(elems)
        perm_rng.shuffle(shuffled)
        relabel = dict(zip(elems, shuffled))
        supports = {a: {relabel[c] for c in s} for a, s in supports.items()}

    known = {c: rbc[c] for c in criteria if c in rbc}
    if "" in rbc:
        known[""] = rbc[""]
    estimates, entropies = {}, {}

    budget_remaining = budget
    for layer_size in range(2, len(start_node)):
        layer = []
        for combo in combinations(range(len(start_node)), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)
            voters = _get_child_voters(subset, known, estimates, answers)
            if not voters:
                continue
            weights = {}
            if flip_boost:
                w = float(len(voters))
                weights = {vk: w for vk in voters if known.get(vk)
                           and known[vk][0] != baseline}
            ranking, scores = _aggregate(voters, answers, weights)
            if not scores:
                continue
            estimates[key] = _store_estimate(ranking, scores, answers, True)
            entropies[key] = softmax_entropy(scores)
            layer.append((key, entropies[key]))
        for sel in max_jump_cutoff(layer, budget_remaining):
            if sel in rbc:
                known[sel] = rbc[sel]
                budget_remaining -= 1
    budget_used = budget - budget_remaining

    calls = pruned = 0
    skipped = []

    def probe(key):
        nonlocal calls
        calls += 1
        return rbc.get(key)

    for _size, subsets in iter_search_layers(criteria, start_node, "down"):
        unknown = []
        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                if known[key] and known[key][0] != baseline:
                    return {"found": True, "key": key, "size": len(subset),
                            "ranking": known[key], "calls": calls,
                            "budget_used": budget_used, "pruned_nodes": pruned}
                continue
            if key not in estimates:
                continue
            scores = estimates[key]
            ranking = sorted(scores, key=lambda a: (-scores[a], a))
            disp = ranking.index(baseline) if baseline in ranking else len(ranking)
            g = graph_contrast(subset, supports, answers, baseline)
            rival = max((v for a, v in scores.items() if a != baseline), default=0.0)
            unknown.append((subset, key, disp, entropies.get(key, 0.0), g,
                            scores.get(baseline, 0.0) - rival))
        if not unknown:
            continue

        high = (set(max_jump_cutoff([(k, e) for _, k, _, e, _, _ in unknown],
                                    len(unknown)))
                if entropy_cut else {u[1] for u in unknown})
        cands, rescued = [], []
        for u in unknown:
            if u[2] >= 1 or u[1] in high:
                cands.append(u)
            elif pool and u[4] > 0:
                rescued.append(u)
            else:
                pruned += 1
                skipped.append({"key": u[1], "subset": u[0],
                                "size": len(u[0]), "ent": u[3]})
        pool_bar = float(np.median([u[4] for u in rescued])) if rescued else 0.0
        rescued.sort(key=lambda u: -u[4])

        claims, claims_left = set(), None
        if not tier:
            ordered = sorted(cands, key=lambda u: (-u[2], u[3]))
        else:
            displaced = sorted((u for u in cands if u[2] >= 1),
                               key=lambda u: (u[5], u[3]))
            uncertain = [u for u in cands if u[2] < 1]
            if len({u[4] for u in uncertain}) > 1:
                claims = set(max_jump_cutoff([(u[1], u[4]) for u in uncertain],
                                             len(uncertain)))
                claims_left = len(claims)
            uncertain.sort(key=lambda u: (u[1] not in claims, -u[4], u[5], u[3]))
            ordered = uncertain + displaced if graph_first else displaced + uncertain

        cut = pool_done = False
        for subset, key, disp, _ent, g, _margin in ordered + rescued:
            is_pool = disp < 1 and key not in high
            if cut and not is_pool and disp < 1 and key not in claims:
                pruned += 1
                skipped.append({"key": key, "subset": subset,
                                "size": len(subset), "ent": _ent})
                continue
            if pool_done and is_pool:
                pruned += 1
                skipped.append({"key": key, "subset": subset,
                                "size": len(subset), "ent": _ent})
                continue
            truth = probe(key)
            if truth:
                known[key] = truth
                if truth[0] != baseline:
                    return {"found": True, "key": key, "size": len(subset),
                            "ranking": truth, "calls": calls,
                            "budget_used": budget_used, "pruned_nodes": pruned}
                if key in claims and claims_left is not None:
                    claims_left -= 1
                    cut = claims_left == 0
                if is_pool and g >= pool_bar:
                    pool_done = True
    return {"found": False, "key": None, "size": None, "ranking": None,
            "calls": calls, "budget_used": budget_used, "pruned_nodes": pruned}
