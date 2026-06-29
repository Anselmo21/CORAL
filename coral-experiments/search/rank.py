from itertools import combinations

from search.heuristics import (
    softmax_entropy,
    max_jump_cutoff,
    score_margin,
    norm_minmax,
    norm_zscore,
    norm_rank,
    norm_softmax,
    shannon_entropy,
)
from search.lattice import combo_key, iter_search_layers


# ─── Rank aggregators (used by sra_search) ─────────────────────────────────

def _combsum(voter_rankings: dict[str, list[str]], answers: list[str]) -> dict[str, float]:
    """Sum Borda scores (n - rank) per answer across voters."""
    n = len(answers)
    scores: dict[str, float] = {a: 0.0 for a in answers}
    for ranking in voter_rankings.values():
        for rank, ans in enumerate(ranking):
            if ans in scores:
                scores[ans] += float(n - rank)
    return scores


def _combmnz(voter_rankings: dict[str, list[str]], answers: list[str]) -> dict[str, float]:
    """CombMNZ: CombSUM × number of voters that listed each answer."""
    n = len(answers)
    sums: dict[str, float] = {a: 0.0 for a in answers}
    counts: dict[str, int] = {a: 0 for a in answers}
    for ranking in voter_rankings.values():
        for rank, ans in enumerate(ranking):
            if ans in sums:
                sums[ans] += float(n - rank)
                counts[ans] += 1
    return {a: sums[a] * counts[a] for a in answers}


def _borda(voter_rankings: dict[str, list[str]], answers: list[str]) -> dict[str, float]:
    """Borda count. Equivalent to CombSUM when every voter ranks every answer
    (true for all callers here), kept separate for clarity."""
    return _combsum(voter_rankings, answers)


_AGGREGATORS = {
    "combsum": _combsum,
    "combmnz": _combmnz,
    "borda": _borda,
}


def _aggregate_with_scores(
    voter_rankings: dict[str, list[str]],
    answers: list[str],
    aggregator: str | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Rank-aggregate voter rankings into a single ranking + per-answer scores."""
    if not voter_rankings:
        return [], {}
    if aggregator is None:
        fn = _combsum
    elif isinstance(aggregator, str):
        fn = _AGGREGATORS[aggregator.lower()]
    else:
        fn = aggregator
    scores = fn(voter_rankings, answers)
    ordered = sorted(scores, key=lambda a: (-scores[a], a))
    return ordered, scores


# ─── Score-vector aggregation (unified ERA path) ───────────────────────────

def _ranking_to_score_vector(
    ranking: list[str],
    answers: list[str],
) -> dict[str, float]:
    """Borda score vector from a ranking: answer at rank r gets n - r points."""
    n = len(answers)
    sv: dict[str, float] = {a: 0.0 for a in answers}
    for rank, ans in enumerate(ranking):
        if ans in sv:
            sv[ans] = float(n - rank)
    return sv


def _normalize_max(scores: dict[str, float]) -> dict[str, float]:
    """Max-normalize to [0, 1]. No-op on empty / all-zero input."""
    max_abs = max((abs(v) for v in scores.values()), default=0.0)
    if max_abs == 0:
        return scores
    return {a: v / max_abs for a, v in scores.items()}


def make_aggregator(norm_fn=None):
    """Factory for ERA aggregators with pluggable post-sum normalization.

    norm_fn: callable(dict[str, float]) -> dict[str, float]
             Applied to the entry-wise sum of voter score vectors before storing.
             Defaults to max-normalize if None.

    Custom aggregators must match the signature:
      (voter_scores, answers) -> (ranking, scores)
    """
    _norm = norm_fn if norm_fn is not None else _normalize_max

    def _agg(
        voter_scores: dict[str, dict[str, float]],
        answers: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        if not voter_scores:
            return [], {}
        agg: dict[str, float] = {a: 0.0 for a in answers}
        for sv in voter_scores.values():
            sv = _normalize_max(sv)
            for a in answers:
                agg[a] += sv.get(a, 0.0)
        agg = _norm(agg)
        ordered = sorted(agg, key=lambda a: (-agg[a], a))
        return ordered, agg

    return _agg


aggregator_default = make_aggregator()


def _get_child_voters(
    subset: list[str],
    known: dict[str, list[str]],
    estimated_scores: dict[str, dict[str, float]],
    answers: list[str],
) -> dict[str, dict[str, float]]:
    """Voter score vectors for size-(|subset|-1) children of ``subset``.

    Known children → Borda from ranking, max-normalized to [0, 1].
    Estimated children → previously-stored score vector (already normalized).
    """
    out: dict[str, dict[str, float]] = {}
    for i in range(len(subset)):
        child = subset[:i] + subset[i + 1:]
        if not child:
            continue
        child_key = combo_key(child)
        if child_key in known:
            out[child_key] = _normalize_max(_ranking_to_score_vector(known[child_key], answers))
        elif child_key in estimated_scores:
            out[child_key] = estimated_scores[child_key]
    return out


def _store_estimate(
    ranking: list[str],
    scores: dict[str, float],
    answers: list[str],
    propagate_scores: bool,
) -> dict[str, float]:
    """Pick what to store as a voter for upstream layers.

    propagate_scores=True  → keep full magnitudes from this layer's aggregation.
    propagate_scores=False → discard magnitudes; re-derive Borda from the ranking,
                             which makes upstream layers re-aggregate on pure ranks.
    """
    if propagate_scores:
        # return scores
        return _normalize_max(scores)
    return _normalize_max(_ranking_to_score_vector(ranking, answers))


def _era_phase1_down(
    start_node: list[str],
    rankings_by_combo: dict[str, list[str]],
    answers: list[str],
    known: dict[str, list[str]],
    estimated_scores: dict[str, dict[str, float]],
    entropies: dict[str, float],
    budget_remaining: int,
    aggregator_fn,
    entropy_fn,
    cutoff_fn,
    propagate_scores: bool,
    norm_fn=None,
) -> tuple[int, list[str]]:
    """
    ERA downward Phase-1 budget allocation.

    Estimates rankings bottom-up through subsets of ``start_node`` (layers
    2 → s-1), scores entropy per node, and spends ``budget_remaining`` units
    looking up the highest-uncertainty nodes selected by ``cutoff_fn``.

    Mutates ``known`` / ``estimated_scores`` / ``entropies`` in place.
    """
    s = len(start_node)
    looked_up: list[str] = []

    for layer_size in range(2, s):
        layer_nodes: list[tuple[str, float]] = []
        for combo in combinations(range(s), layer_size):
            subset = [start_node[i] for i in combo]
            key = combo_key(subset)

            voter_scores = _get_child_voters(subset, known, estimated_scores, answers)
            if not voter_scores:
                continue
            ranking, scores = aggregator_fn(voter_scores, answers)
            if not scores:
                continue

            estimated_scores[key] = _store_estimate(ranking, scores, answers, propagate_scores)
            _scores_for_entropy = norm_fn(scores) if norm_fn else scores
            entropy_score = entropy_fn(_scores_for_entropy, num_voters=len(voter_scores))
            entropies[key] = entropy_score
            layer_nodes.append((key, entropy_score))

        for sel_key in cutoff_fn(layer_nodes, budget_remaining):
            if sel_key in rankings_by_combo:
                known[sel_key] = rankings_by_combo[sel_key]
                budget_remaining -= 1
                looked_up.append(sel_key)

    return budget_remaining, looked_up


# ─── Public search functions ───────────────────────────────────────────────

def sra_search(
    criteria, rankings_by_combo,
    aggregator=None,
    start_node=None,
    direction="down",
    rank_displacement_threshold=1,
    verbose=False,
):
    """Simple Rank Aggregation (SRA) search.

    Estimates all node rankings by aggregating singleton rankings (CombSUM by
    default). Within each layer, nodes with higher baseline displacement are
    tried first. Returns the first node whose true top-1 ≠ baseline top-1.
    """
    if aggregator is None:
        aggregator = "combsum"
    if start_node is None:
        start_node = list(criteria)

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline_ranking}")

    if not baseline_ranking:
        return {"found": False, "key": None, "size": None, "ranking": None, "calls": 0}

    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)
    singleton_rankings = {c: rankings_by_combo[c] for c in criteria if c in rankings_by_combo}

    calls = 0
    for layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        layer = []
        if verbose:
            print(f"\nLayer size: {layer_size}")

        for subset in subsets:
            key = combo_key(subset)

            if key in singleton_rankings:
                estimated_ranking = singleton_rankings[key]
            elif not subset:
                estimated_ranking = []
            else:
                voter_rankings = {c: singleton_rankings[c] for c in subset if c in singleton_rankings}
                estimated_ranking, _ = _aggregate_with_scores(voter_rankings, answers, aggregator)

            if not subset:
                baseline_rank = len(answers)
            elif baseline_answer in estimated_ranking:
                baseline_rank = estimated_ranking.index(baseline_answer)
            else:
                baseline_rank = len(estimated_ranking)
            layer.append((key, estimated_ranking, baseline_rank))

        layer.sort(key=lambda x: x[2], reverse=True)
        layer = [node for node in layer if node[2] >= rank_displacement_threshold]

        for key, estimated_ranking, baseline_rank in layer:
            if key not in singleton_rankings:
                calls += 1
            if verbose:
                print(f"\t{key}: {baseline_rank}")
            llm_ranking = rankings_by_combo.get(key)
            if not llm_ranking:
                continue
            if llm_ranking[0] != baseline_answer:
                return {
                    "found": True,
                    "key": key,
                    "size": layer_size,
                    "ranking": llm_ranking,
                    "calls": calls,
                }

    return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls}


def era_search(
    criteria,
    rankings_by_combo,
    budget=None,
    aggregator_fn=None,
    entropy_fn=None,
    cutoff_fn=None,
    norm_fn=None,
    start_node=None,
    direction="down",
    propagate_scores: bool = True,
    verbose=False,
):
    """
    Entropy-Rank Aggregation (ERA) search algorithm.

    Goal: find a counterfactual (CF) node — a subset of criteria where the
    LLM's top-1 answer changes from baseline (the full-criteria ranking).

    --- SETUP ---

    Free preloads: all singletons and the empty set are added to `known`
    before any budget is spent. They serve as bottom-level voters.

    State:
      known            {combo_key -> ranking}         ground-truth rankings in hand
      estimated_scores {combo_key -> score_vector}    aggregation-derived estimates
      entropies        {combo_key -> float}            uncertainty per node

    --- TOP-DOWN SEARCH (direction="down") ---

    Phase 1 (budget allocation):
      Traverses bottom-up through layers 2 -> s-1 (s = |start_node|).
      For each node at layer k:
        1. Collect voter score vectors from its size-(k-1) children:
             known child    -> Borda scores (n - rank), max-normalized to [0, 1]
             estimated child -> stored score vector (already normalized)
        2. Aggregate: entry-wise sum of voter vectors, max-normalize -> scores + ranking.
        3. Store estimate (controlled by propagate_scores):
             True  -> store full aggregated magnitudes; upstream layers see confidence.
             False -> re-derive integer Borda from ranking; discard magnitudes.
        4. Compute softmax entropy of scores: shift by max, apply softmax, Shannon
           entropy. High entropy = answers nearly tied = uncertain who is top-1.
      After each layer: cutoff_fn selects high-entropy nodes to look up
      (1 budget unit each). Default max_jump_cutoff sorts by entropy desc,
      finds the largest consecutive gap, selects everything above it.

    Phase 2 (search, top-down layers s-1 -> 0):
      Known nodes: free CF check — if ranking[0] != baseline_answer, return found.
      Unknown nodes:
        - Retrieve Phase-1 scores/ranking.
        - Compute displacement: position of baseline_answer in estimated ranking.
          (0 = estimate says no change; higher = estimate confident this is a CF)
        - Prune: drop node if displacement < 1 AND not high-entropy.
        - Sort remaining by (-displacement, +entropy): try most-likely CFs first,
          prefer lower-entropy (more confident) estimates among ties.
        - Look up ground truth; count as call. Return on first CF found.

    --- BOTTOM-UP SEARCH (direction="up") ---

    Phase 1: pre-loads the s+1 layer for free (they become ground-truth voters
    for s+2 estimates). budget_used is always 0 for upward search.
    Phase 2: re-estimates fresh per layer against the growing `known` dict —
    voter quality compounds upward as more nodes are looked up.

    --- RETURN VALUE ---

    {
      "found":        bool,
      "key":          combo_key of the CF node (or None),
      "size":         layer size of the CF,
      "ranking":      ground-truth ranking at CF node,
      "calls":        Phase-2 LLM lookups,
      "budget_used":  Phase-1 LLM lookups,
      "pruned_nodes": nodes skipped in Phase 2,
    }

    Total cost = budget_used + calls.
    """
    if aggregator_fn is None:
        aggregator_fn = aggregator_default
    if entropy_fn is None:
        entropy_fn = softmax_entropy
    if cutoff_fn is None:
        cutoff_fn = max_jump_cutoff
    if start_node is None:
        start_node = list(criteria)
    if budget is None:
        budget = len(rankings_by_combo)

    budget_remaining = budget

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline_ranking}")

    if not baseline_ranking:
        return {"found": False, "key": None, "size": None, "ranking": None,
                "calls": 0, "budget_used": 0, "pruned_nodes": 0}

    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)

    known: dict[str, list[str]] = {}
    estimated_scores: dict[str, dict[str, float]] = {}
    entropies: dict[str, float] = {}

    # Preload singletons + empty set as free voters / free CF checks.
    for c in criteria:
        if c in rankings_by_combo:
            known[c] = rankings_by_combo[c]
    if "" in rankings_by_combo:
        known[""] = rankings_by_combo[""]

    # Phase 1: spend budget
    if direction == "down":
        budget_remaining, _ = _era_phase1_down(
            start_node, rankings_by_combo, answers, known,
            estimated_scores, entropies, budget_remaining,
            aggregator_fn, entropy_fn, cutoff_fn, propagate_scores,
            norm_fn,
        )
    elif direction == "up":
        extra = [c for c in criteria if c not in start_node]
        known[start_key] = baseline_ranking
        for c in extra:
            key = combo_key(list(start_node) + [c])
            if key in rankings_by_combo:
                known[key] = rankings_by_combo[key]

    budget_used = budget - budget_remaining

    # Phase 2: search
    calls = 0
    pruned_nodes = 0
    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        if verbose:
            print(f"\nLayer size: {_layer_size}")

        unknown_nodes: list[tuple[list[str], str, float, float]] = []

        for subset in subsets:
            key = combo_key(subset)

            # Free CF check on cached nodes.
            if key in known:
                if known[key] and known[key][0] != baseline_answer:
                    return {
                        "found": True,
                        "key": key,
                        "size": len(subset),
                        "ranking": known[key],
                        "calls": calls,
                        "budget_used": budget_used,
                        "pruned_nodes": pruned_nodes,
                    }
                continue

            if direction == "up":
                # Re-estimate fresh each layer against the growing known dict.
                voter_scores = _get_child_voters(subset, known, estimated_scores, answers)
                if not voter_scores:
                    continue
                ranking, scores = aggregator_fn(voter_scores, answers)
                if not scores:
                    continue
                estimated_scores[key] = _store_estimate(ranking, scores, answers, propagate_scores)
                _scores_for_entropy = norm_fn(scores) if norm_fn else scores
                entropy_score = entropy_fn(_scores_for_entropy, num_voters=len(voter_scores))
                entropies[key] = entropy_score
            else:
                # Use Phase-1 estimate.
                if key not in estimated_scores:
                    continue  # empty set / no Phase-1 voter pool
                scores = estimated_scores[key]
                ranking = sorted(scores, key=lambda a: (-scores[a], a))
                entropy_score = entropies.get(key, 0.0)

            displacement = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            unknown_nodes.append((subset, key, displacement, entropy_score))

        if not unknown_nodes:
            continue

        entropy_inputs = [(key, ent) for _, key, _, ent in unknown_nodes]
        high_entropy_keys = set(cutoff_fn(entropy_inputs, len(unknown_nodes)))

        candidates = [
            (subset, key, displacement, ent)
            for subset, key, displacement, ent in unknown_nodes
            if not (displacement < 1 and key not in high_entropy_keys)
        ]
        pruned_nodes += len(unknown_nodes) - len(candidates)

        if verbose:
            for subset, key, displacement, entropy_score in candidates:
                print(f"\t{key}: displacement={displacement} entropy={entropy_score:.3f}")

        candidates.sort(key=lambda x: (-x[2], x[3]))

        for subset, key, _, _ in candidates:
            calls += 1
            ground_truth = rankings_by_combo.get(key)
            if ground_truth:
                known[key] = ground_truth
            if ground_truth and ground_truth[0] != baseline_answer:
                return {
                    "found": True,
                    "key": key,
                    "size": len(subset),
                    "ranking": ground_truth,
                    "calls": calls,
                    "budget_used": budget_used,
                    "pruned_nodes": pruned_nodes,
                }

    return {"found": False, "key": None, "size": None, "ranking": None,
            "calls": calls, "budget_used": budget_used, "pruned_nodes": pruned_nodes}


# ─── MRA (minimal ERA reimplementation) ────────────────────────────────────
#
# Per node:
#   1. _mra_get_child_voters  — size-(n-1) children as raw Borda score vectors
#   2. _mra_aggregate_borda   — sum those vectors answer-by-answer
#   3. softmax_entropy        — uncertainty on aggregated scores
#   4. store scores + entropy for upstream layers
#
# No normalization yet; hook that into _mra_aggregate_borda later.

def _mra_get_child_voters(
    subset: list[str],
    known: dict[str, dict[str, float]],
    estimated_scores: dict[str, dict[str, float]],
    answers: list[str],
) -> dict[str, dict[str, float]]:
    """Raw Borda score vectors for size-(|subset|-1) children of ``subset``."""
    out: dict[str, dict[str, float]] = {}
    for combo in combinations(range(len(subset)), len(subset) - 1):
        child = [subset[i] for i in combo]
        child_key = combo_key(child)
        if child_key in known:
            out[child_key] = _normalize_max(known[child_key])
        elif child_key in estimated_scores:
            out[child_key] = estimated_scores[child_key]
    return out


def _mra_aggregate_borda(
    voter_scores: dict[str, dict[str, float]],
    answers: list[str],
) -> tuple[list[str], dict[str, float]]:
    """Sum Borda score vectors across voters. Returns (ranking, scores)."""
    if not voter_scores:
        return [], {}
    agg: dict[str, float] = {a: 0.0 for a in answers}
    for sv in voter_scores.values():
        for a in answers:
            agg[a] += sv.get(a, 0.0)
    ranking = sorted(agg, key=lambda a: (-agg[a], a))
    return ranking, agg


def _mra_ranking_from_scores(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda a: (-scores[a], a))


def _mra_displacement(ranking: list[str], baseline_answer: str) -> int:
    return ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)


def _mra_top1(scores: dict[str, float]) -> str:
    return _mra_ranking_from_scores(scores)[0]


# Normalizers for MRA entropy (applied to aggregated scores before softmax_entropy).
# Pass a callable or a string key to mra_search(norm_fn=...).
MRA_NORM_FNS = {
    "none": None,
    "max": _normalize_max,
    "minmax": norm_minmax,
    "zscore": norm_zscore,
    "rank": norm_rank,
    "softmax": norm_softmax,
}


def _resolve_norm_fn(norm_fn):
    if norm_fn is None:
        return None
    if isinstance(norm_fn, str):
        key = norm_fn.lower()
        if key not in MRA_NORM_FNS:
            raise ValueError(f"unknown norm_fn {norm_fn!r}; choose from {list(MRA_NORM_FNS)}")
        return MRA_NORM_FNS[key]
    return norm_fn


def _mra_max_jump_stats(
    layer_nodes: list[tuple],
    budget_remaining: int,
) -> dict:
    """Per-layer max-jump tiering: counts above/below the largest entropy gap.

    Accepts (key, entropy) or (key, entropy, ...) tuples.
    """
    nodes = [(n[0], n[1]) for n in layer_nodes]
    sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    n = len(sorted_nodes)
    if n == 0:
        return {"n_above": 0, "n_below": 0, "selected": []}

    if n == 1:
        selected = max_jump_cutoff(nodes, budget_remaining)
        return {
            "n_above": 1,
            "n_below": 0,
            "cutoff_high": sorted_nodes[0][1],
            "cutoff_low": None,
            "gap": None,
            "selected": selected,
        }

    gaps = [sorted_nodes[i][1] - sorted_nodes[i + 1][1] for i in range(n - 1)]
    max_gap_idx = max(range(len(gaps)), key=lambda i: (gaps[i], i))
    selected = max_jump_cutoff(nodes, budget_remaining)

    return {
        "n_above": max_gap_idx + 1,
        "n_below": n - (max_gap_idx + 1),
        "cutoff_high": sorted_nodes[max_gap_idx][1],
        "cutoff_low": sorted_nodes[max_gap_idx + 1][1],
        "gap": gaps[max_gap_idx],
        "max_gap_idx": max_gap_idx,
        "selected": selected,
    }


def mra_search(
    concepts,
    rankings_by_combo, # oracle (LLM) rankings of all concept subsets,
    plausible_answers=None,
    start_node=None,
    fixed_set=None,
    budget=8,
    aggregator_fn=None,
    normalize_fn=None,
    entropy_fn=None,
    direction="down",
    verbose=False,
):
    normalize_fn = _resolve_norm_fn(normalize_fn)
    if entropy_fn is None:
        entropy_fn = shannon_entropy
    # TODO: add support for "bottom-up" search (direction="up")
    # if no start node is provided, use the full set of concepts
    if start_node is None:
        start_node = list(concepts)
    if fixed_set is None:
        fixed_set = []

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    if not baseline_ranking:
        return {"found": False, "key": None, "size": None, "ranking": None,
                "calls": 0, "budget_used": 0, "pruned_nodes": 0}

    baseline_answer = baseline_ranking[0]
    if plausible_answers is None:
        plausible_answers = list(baseline_ranking)

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline answer: {baseline_answer}")

    known: dict[str, dict[str, float]] = {}
    estimated_scores: dict[str, dict[str, float]] = {}
    entropies: dict[str, float] = {}

    for c in concepts:
        ranking = rankings_by_combo.get(c, [])
        if not ranking:
            raise ValueError(f"Concept {c} has no rankings")
        known[c] = _ranking_to_score_vector(ranking, plausible_answers)

    if "" in rankings_by_combo:
        known[""] = _ranking_to_score_vector(rankings_by_combo[""], plausible_answers)

    if verbose:
        print(f"Phase 1: bottom-up Borda estimation (budget={budget})")
        if normalize_fn is None:
            norm_label = "none"
        else:
            norm_label = next(
                (k for k, v in MRA_NORM_FNS.items() if v is normalize_fn),
                "custom",
            )
        print(f"entropy norm: {norm_label}")

    budget_remaining = budget

    for layer_size, subsets in iter_search_layers(concepts, fixed_set, "up"):
        # if budget_remaining <= 0:
        #     break
        if layer_size >= len(concepts):
        # if layer_size >= 4:
            continue  # skip full set (start node)
        if verbose:
            print(f"\nLayer {layer_size}: {len(subsets)} subsets")

        layer_nodes: list[tuple[str, float]] = []
        entropies_layer: dict[str, float] = {}
        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                continue

            voter_scores = _mra_get_child_voters(
                subset, known, estimated_scores, plausible_answers,
            )
            if not voter_scores:
                continue

            ranking, scores = _mra_aggregate_borda(voter_scores, plausible_answers)

            # normalize aggregated scores before softmax entropy
            scores_for_entropy = normalize_fn(scores) if normalize_fn else scores
            
            # estimated_scores[key] = scores
            estimated_scores[key] = scores_for_entropy
            # print(f"scores: {scores}")
            # print(f"scores_for_entropy: {scores_for_entropy}")

            entropy = entropy_fn(scores_for_entropy)
            entropies_layer[key] = entropy
            entropies[key] = entropy
            layer_nodes.append((key, entropy, scores_for_entropy))

        mj = _mra_max_jump_stats(layer_nodes, budget_remaining)
        if verbose:
            print(f"  max-jump: gap={mj['gap']:.4f}" if mj.get("gap") is not None else "  max-jump: n/a")
            print(f"  above={mj['n_above']}  below={mj['n_below']}  selected={len(mj['selected'])}")

        for key in mj["selected"]:
            if budget_remaining <= 0:
                break
            gt = rankings_by_combo.get(key, [])
            if gt:
                known[key] = _ranking_to_score_vector(gt, plausible_answers)
            budget_remaining -= 1

    budget_used = budget - budget_remaining

    # Phase 2: top-down search
    calls = 0
    pruned_nodes = 0
    for layer_size, subsets in iter_search_layers(concepts, start_node, "down"):
        if verbose:
            print(f"\nPhase 2 layer {layer_size}: {len(subsets)} subsets")

        unknown_nodes: list[tuple[str, int, float, dict[str, float]]] = []

        for subset in subsets:
            key = combo_key(subset)

            # free CF check on nodes already in known (singletons + budget lookups)
            if key in known:
                if _mra_top1(known[key]) != baseline_answer:
                    return {
                        "found": True,
                        "key": key,
                        "size": len(subset),
                        "ranking": rankings_by_combo.get(key),
                        "calls": calls,
                        "budget_used": budget_used,
                        "pruned_nodes": pruned_nodes,
                    }
                continue

            if key not in estimated_scores:
                continue

            scores = estimated_scores[key]
            ranking = _mra_ranking_from_scores(scores)
            displacement = _mra_displacement(ranking, baseline_answer)
            entropy_score = entropies.get(key, 0.0)
            unknown_nodes.append((key, displacement, entropy_score, scores))

        if not unknown_nodes:
            continue

        entropy_inputs = [(k, e) for k, _, e, _ in unknown_nodes]
        high_entropy_keys = set(
            max_jump_cutoff(entropy_inputs, len(unknown_nodes))
        )

        # prune: low entropy AND baseline still in top 2 (displacement <= 1)
        candidates = [
            (key, disp, ent, scores)
            for key, disp, ent, scores in unknown_nodes
            if not (disp <= 0 and key not in high_entropy_keys)
        ]
        pruned_nodes += len(unknown_nodes) - len(candidates)

        if verbose:
            mj = _mra_max_jump_stats(
                [(k, e) for k, _, e, _ in unknown_nodes], len(unknown_nodes)
            )
            if mj.get("cutoff_low") is not None:
                print(f"  cutoff band: [{mj['cutoff_low']:.4f}, {mj['cutoff_high']:.4f}]")
            print(f"  candidates after prune: {len(candidates)} (pruned {len(unknown_nodes) - len(candidates)})")

        candidates.sort(key=lambda x: (-x[1], x[2]))  # displacement desc, then entropy

        for key, _, _, _ in candidates:
            calls += 1
            ground_truth = rankings_by_combo.get(key, [])
            if ground_truth:
                known[key] = _ranking_to_score_vector(ground_truth, plausible_answers)
            if ground_truth and ground_truth[0] != baseline_answer:
                return {
                    "found": True,
                    "key": key,
                    "size": layer_size,
                    "ranking": ground_truth,
                    "calls": calls,
                    "budget_used": budget_used,
                    "pruned_nodes": pruned_nodes,
                }

    return {
        "found": False,
        "key": None,
        "size": None,
        "ranking": None,
        "calls": calls,
        "budget_used": budget_used,
        "pruned_nodes": pruned_nodes,
    }