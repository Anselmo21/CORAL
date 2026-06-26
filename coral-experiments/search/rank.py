from itertools import combinations

from search.heuristics import softmax_entropy, max_jump_cutoff, score_margin
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


def aggregator_default(
    voter_scores: dict[str, dict[str, float]],
    answers: list[str],
) -> tuple[list[str], dict[str, float]]:
    """Default ERA aggregator: entry-wise sum + max-normalize.

    Custom aggregators must match this signature:
      (voter_scores, answers) -> (ranking, scores)
    where scores is a dict[answer, float] and ranking is its sort order.
    """
    if not voter_scores:
        return [], {}
    agg: dict[str, float] = {a: 0.0 for a in answers}
    for sv in voter_scores.values():
        for a in answers:
            agg[a] += sv.get(a, 0.0)
    agg = _normalize_max(agg)
    ordered = sorted(agg, key=lambda a: (-agg[a], a))
    return ordered, agg


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
        return scores
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
) -> tuple[int, list[str]]:
    """ERA downward Phase-1 budget allocation.

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
            entropy_score = entropy_fn(scores, num_voters=len(voter_scores))
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
    start_node=None,
    direction="down",
    propagate_scores: bool = True,
    verbose=False,
):
    """
    Entropy-Rank Aggregation (ERA) search algorithm.

    The default configuration of ERA aggregated rankings by retaining the scores
    from the previous layer. 

    Single unified aggregation path across all datasets. Voters are score
    vectors (max-normalized Borda for known children; previously-aggregated
    scores for estimated children). Aggregation is pluggable via
    ``aggregator_fn`` — default sums voter score vectors and max-normalizes
    the result.

    Top-Down Search
        Phase 1 (budget allocation): estimates rankings bottom-up (layers 2 → s-1).
        Uses entropy + max-jump cutoff to select high-uncertainty nodes for
        ground-truth lookup (1 LLM call each).

        Phase 2: traverses top-down. Known nodes are free CF checks. Unknown
        nodes are pruned if displacement < 1 AND not high-entropy, then sorted
        by displacement desc, entropy asc.

    Bottom-Up Search
        Phase 1: pre-loads the s+1 layer for free (ground-truth voters for s+2).
        Phase 2: traverses bottom-up; estimates computed fresh per layer against
        the growing ``known`` dict — voter quality compounds upward. All Phase-2
        costs counted as calls; budget_used always 0 for upward search.

    ``propagate_scores`` controls what gets stored between layers:
      - True  → full aggregated magnitudes (preserves voter confidence signal).
      - False → re-derived integer Borda from the ranking (discards magnitudes).
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
                entropy_score = entropy_fn(scores, num_voters=len(voter_scores))
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
