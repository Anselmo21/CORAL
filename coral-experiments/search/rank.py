from itertools import combinations

import pandas as pd
import pyflagr.Linear as Linear

from search.heuristics import shannon_entropy, softmax_entropy, max_jump_cutoff, score_margin
from search.lattice import combo_key, iter_search_layers

import os
from contextlib import contextmanager


@contextmanager
def _silence_stdout():
    """Suppress C-level stdout (e.g. pyflagr's progress lines) via fd-level redirect."""
    fflush = None
    try:
        import ctypes
        fflush = ctypes.cdll.LoadLibrary(None).fflush
    except Exception:
        pass
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        if fflush:
            fflush(None)
        os.dup2(devnull_fd, 1)
        yield
    finally:
        if fflush:
            fflush(None)
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)



def _answer_id_maps(answers: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map answers to comma-free item codes for pyflagr CSV I/O."""
    answer_to_code = {ans: f"i{idx}" for idx, ans in enumerate(answers)}
    code_to_answer = {code: ans for ans, code in answer_to_code.items()}
    return answer_to_code, code_to_answer


def _build_input_df(voter_rankings: dict[str, list[str]], answers: list[str]) -> pd.DataFrame:
    """
    Builds a DataFrame for the PyFlagr input.
    """
    answer_to_code, _ = _answer_id_maps(answers)
    n = len(answers)
    rows = []
    for idx, (_voter_label, ranking) in enumerate(voter_rankings.items()):
        safe_label = f"v{idx}"
        for rank, answer in enumerate(ranking):
            rows.append({
                "Query": "q",
                "Voter Name": safe_label,
                "Item Code": answer_to_code[answer],
                "Item Rank": rank + 1,
                "Item Score": n - rank,
                "Algorithm/Dataset": "sra",
            })
    return pd.DataFrame(rows)


def _aggregate_with_scores(
    voter_rankings: dict[str, list[str]],
    answers: list[str],
    aggregator,
) -> tuple[list[str], dict[str, float]]:
    _, code_to_answer = _answer_id_maps(answers)
    df = _build_input_df(voter_rankings, answers)
    with _silence_stdout():
        agg, _ = aggregator.aggregate(input_df=df)
    ranking = [code_to_answer[item_id] for item_id in agg["ItemID"].tolist()]
    if "Score" in agg.columns:
        scores = {
            code_to_answer[item_id]: float(score)
            for item_id, score in zip(agg["ItemID"], agg["Score"])
        }
    else:
        scores = {ans: float(len(ranking) - 1 - i) for i, ans in enumerate(ranking)}
    return ranking, scores


def _get_child_rankings(subset, known, estimated):
    voter_rankings = {}
    for i in range(len(subset)):
        child = subset[:i] + subset[i + 1:]
        if len(child) == 0:
            continue
        child_key = combo_key(child)
        if child_key in known:
            voter_rankings[child_key] = known[child_key]
        elif child_key in estimated:
            voter_rankings[child_key] = estimated[child_key]
    return voter_rankings



def sra_search(
    criteria, rankings_by_combo,
    aggregator=None,
    start_node=None,
    direction="down",
    top_k=1,
    verbose=False,
):
    """
    Simple Rank Aggregation (SRA) search algorithm.

    The baseline or "original" answer is the top answer for the start_node.
    Estimates all node rankings by aggregating singleton rankings via PyFlagr (CombSUM by default).
    
    Within each layer, we calculate the "displacement_score" for each node which
    measures the rank of the baseline answer in the estimated ranking.
    Nodes are sorted by the displacement_score in descending order.
    This means that the nodes with rankings where the baseline answer is ranked lower
    are considered to be more likely to be counterfactual, and thus prioritized for evaluation.
    
    Nodes with a displacement_score less than top_k are skipped.
    The first node where the baseline answer is not equal to the LLM's true answer is 
    returned
    .
    Returns {"found", "key", "size", "estimated_ranking", "calls"}.
    """
    if aggregator is None:
        aggregator = Linear.CombSUM()
    if start_node is None:
        start_node = list(criteria)

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)

    # 1. retrieve the rankings for each criterion from the LLM
    singleton_rankings = {c: rankings_by_combo[c] for c in criteria if c in rankings_by_combo}

    calls = 0

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline_ranking}")

    # 2. begin the search
    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        layer = []
        if verbose:
            print(f"\nLayer size: {_layer_size}")

        # 3. for each node in the layer, estimate the rankings using rank aggregation
        for subset in subsets:
            key = combo_key(subset)

            if key in singleton_rankings:
                # we already called the LLM for nodes with one criterion
                pass
            else:
                # get the rankings for the singleton criteria
                voter_rankings = {c: singleton_rankings[c] for c in subset}
                # aggregate the rankings for this criteria subset
                estimated_ranking, _ = _aggregate_with_scores(voter_rankings, answers, aggregator)

            baseline_rank = estimated_ranking.index(baseline_answer) if baseline_answer in estimated_ranking else len(estimated_ranking)
            layer.append((key, estimated_ranking, baseline_rank))

        # sort the layer by the baseline rank in descending order
        layer.sort(key=lambda x: x[2], reverse=True)
        
        # only keep the nodes who's scores are above the top_k threshold
        layer = [node for node in layer if node[2] >= top_k]

        # 4. for each node in the layer above the top_k threshold, check if it
        #    is a counterfactual by comparing the baseline answer to the LLM's answer
        for key, estimated_ranking, baseline_rank in layer:
            if key not in singleton_rankings:
                calls += 1
            if verbose:
                print(f"\t{key}: {baseline_rank}")
            # basically calling the LLM
            llm_ranking = rankings_by_combo.get(key)
            llm_answer = llm_ranking[0]
            if llm_answer != baseline_answer:
                return {
                    "found": True,
                    "key": key,
                    "size": len(subset),
                    "ranking": llm_ranking,
                    "calls": calls,
                }

    return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls}


def sra_search_v2(
    criteria, rankings_by_combo,
    aggregator=None,
    start_node=None,
    direction="down",
    top_k=1,
    verbose=False,
) -> dict:
    """
    SRA v2: symmetric seed layer for both directions.

    Downward: seeds with individual-criterion (singleton) rankings — same as sra_search.
    Upward: seeds with the s+1 layer (start_node + one extra criterion each). These nodes
    are pre-loaded for free and used as voters to estimate deeper supersets, mirroring the
    role singletons play in the downward direction.

    Returns {"found", "key", "size", "ranking", "calls"}.
    """
    if aggregator is None:
        aggregator = Linear.CombSUM()
    if start_node is None:
        start_node = list(criteria)

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    baseline_answer = baseline_ranking[0]
    answers = list(baseline_ranking)

    # Build the seed layer: singletons for down, s+1 nodes for up.
    if direction == "down":
        seed_rankings: dict[str, list[str]] = {
            c: rankings_by_combo[c] for c in criteria if c in rankings_by_combo
        }
    else:
        extra = [c for c in criteria if c not in start_node]
        seed_rankings = {}
        for c in extra:
            key = combo_key(list(start_node) + [c])
            if key in rankings_by_combo:
                seed_rankings[key] = rankings_by_combo[key]

    calls = 0

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline_ranking}")

    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        layer = []
        if verbose:
            print(f"\nLayer size: {_layer_size}")

        for subset in subsets:
            key = combo_key(subset)

            if key in seed_rankings:
                estimated_ranking = seed_rankings[key]
            elif direction == "down":
                voter_rankings = {c: seed_rankings[c] for c in subset if c in seed_rankings}
                estimated_ranking, _ = _aggregate_with_scores(voter_rankings, answers, aggregator)
            else:
                # Voters are the s+1 sub-supersets: start_node + each individual extra criterion
                extra_in_subset = [c for c in subset if c not in start_node]
                voter_rankings = {
                    combo_key(list(start_node) + [c]): seed_rankings[combo_key(list(start_node) + [c])]
                    for c in extra_in_subset
                    if combo_key(list(start_node) + [c]) in seed_rankings
                }
                estimated_ranking, _ = _aggregate_with_scores(voter_rankings, answers, aggregator)

            baseline_rank = (
                estimated_ranking.index(baseline_answer)
                if baseline_answer in estimated_ranking
                else len(estimated_ranking)
            )
            layer.append((key, estimated_ranking, baseline_rank))

        layer.sort(key=lambda x: x[2], reverse=True)
        layer = [node for node in layer if node[2] >= top_k]

        for key, estimated_ranking, baseline_rank in layer:
            if key not in seed_rankings:
                calls += 1
            if verbose:
                print(f"\t{key}: {baseline_rank}")
            llm_ranking = rankings_by_combo.get(key)
            llm_answer = llm_ranking[0]
            if llm_answer != baseline_answer:
                return {
                    "found": True,
                    "key": key,
                    "size": len(subset),
                    "ranking": llm_ranking,
                    "calls": calls,
                }

    return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls}


def era_search(
    criteria,
    rankings_by_combo,
    budget=None,
    aggregator=None,
    entropy_fn=None,
    cutoff_fn=None,
    start_node=None,
    direction="down",
    verbose=False,
):
    """
    Entropy-Rank Aggregation (ERA) search algorithm.

    ERA is given a budget, a number of extra LLM calls that can be made to form better estimates.

    The baseline or "original" answer is the top answer for the start_node.
    Estimates all node rankings by aggregating (k-1)-subsets via PyFlagr rank aggregation algorithms.

    Phase 1 (budget allocation): estimates rankings bottom-up. Uses entropy + max-jump cutoff
    to select high-uncertainty nodes for ground-truth lookup (1 budget unit each).

    Phase 2 (search): traverses layers top-down. Known nodes are checked for free CFs.
    Unknown nodes with low displacement AND low entropy are skipped (estimate is confident
    baseline stays on top). Remaining candidates are sorted by displacement descending,
    then entropy ascending (least certain first).

    The budget only applies to phase 1 where we estimate the rankings.
    The "calls" variable tracks the number of LLM calls made in phase 2.

    Returns {"found", "key", "size", "ranking", "calls", "budget_used", "pruned_nodes"}.
    """
    if aggregator is None:
        aggregator = Linear.CombSUM()
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
    baseline_answer = baseline_ranking[0]

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline_ranking}")

    answers = list(baseline_ranking)
    n = len(criteria)
    s = len(start_node)

    known = {}
    estimated_rankings = {}
    entropies: dict[str, float] = {}

    # get the rankings for the singleton criteria
    for c in criteria:
        known[c] = rankings_by_combo[c]

    # phase 1: spending the budget
    if direction == "down":
        # Bottom-up through subsets of start_node (layers 2 → s-1)
        for layer_size in range(2, s):
            layer_nodes = []
            for combo in combinations(range(s), layer_size):
                subset = [start_node[i] for i in combo]
                key = combo_key(subset)
                voter_rankings = _get_child_rankings(subset, known, estimated_rankings)

                with _silence_stdout():
                    ranking, scores = _aggregate_with_scores(voter_rankings, answers, aggregator)

                estimated_rankings[key] = ranking
                entropy_score = entropy_fn(scores)
                entropies[key] = entropy_score
                layer_nodes.append((key, entropy_score))

            selected_keys = cutoff_fn(layer_nodes, budget_remaining)
            for key in selected_keys:
                known[key] = rankings_by_combo[key]
                budget_remaining -= 1

    elif direction == "up":
        # Bottom-up through supersets (layers s+1 → n)
        extra = [c for c in criteria if c not in start_node]
        known[start_key] = rankings_by_combo.get(start_key, [])

        # get the rankings for the nodes in layer s+1 from the LLM
        # this is to mirror the bottom-up approach where we get the rankings for individual criteria
        # this is bounded by the number of extra criteria (c)
        for combo in combinations(range(len(extra)), 1):
            subset = list(start_node) + [extra[i] for i in combo]
            key = combo_key(subset)
            if key in rankings_by_combo:
                known[key] = rankings_by_combo[key]

        for layer_size in range(s + 2, n + 1):
            num_extra = layer_size - s
            layer_nodes = []
            for combo in combinations(range(len(extra)), num_extra):
                subset = list(start_node) + [extra[i] for i in combo]
                key = combo_key(subset)

                voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
                if not voter_rankings:
                    continue
                with _silence_stdout():
                    ranking, scores = _aggregate_with_scores(voter_rankings, answers, aggregator)
                estimated_rankings[key] = ranking
                entropy_score = entropy_fn(scores)
                entropies[key] = entropy_score
                layer_nodes.append((key, entropy_score))

            selected_keys = cutoff_fn(layer_nodes, budget_remaining)
            for key in selected_keys:
                if key in rankings_by_combo:
                    known[key] = rankings_by_combo[key]
                    budget_remaining -= 1

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

            # Check cached (known) nodes for free CFs
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

            ranking = estimated_rankings[key]
            displacement = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            entropy_score = entropies.get(key, 0.0)
            unknown_nodes.append((subset, key, displacement, entropy_score))

        if not unknown_nodes:
            continue

        # Use max-jump cutoff on entropy to identify high-uncertainty nodes
        entropy_inputs = [(key, ent) for _, key, _, ent in unknown_nodes]
        high_entropy_keys = set(cutoff_fn(entropy_inputs, len(unknown_nodes)))

        # Skip nodes where baseline is still top-1 (low displacement) AND estimate is confident (low entropy)
        candidates = [
            (subset, key, displacement, ent)
            for subset, key, displacement, ent in unknown_nodes
            if not (displacement < 1 and key not in high_entropy_keys)
        ]

        pruned_nodes += len(unknown_nodes) - len(candidates)

        if verbose:
            for subset, key, displacement, entropy_score in candidates:
                print(f"\t{key}: displacement={displacement} entropy={entropy_score:.3f}")

        # Sort by displacement desc, then entropy asc (least certain first)
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

    return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls, "budget_used": budget_used, "pruned_nodes": pruned_nodes}

def era_search_v2(
    criteria,
    rankings_by_combo,
    budget=None,
    aggregator=None,
    entropy_fn=None,
    cutoff_fn=None,
    margin_fn=None,
    margin_threshold=0.25,
    start_node=None,
    direction="down",
    verbose=False,
):
    """
    Entropy-Rank Aggregation (ERA) v2 search algorithm.

    ERA is given a budget, a number of extra LLM calls that can be made to form better estimates.

    The baseline or "original" answer is the top answer for the start_node.
    Estimates all node rankings by aggregating (k-1)-subsets via PyFlagr rank aggregation algorithms.

    Phase 1 (budget allocation): estimates rankings bottom-up. Uses entropy + max-jump cutoff
    to select high-uncertainty nodes for ground-truth lookup (1 budget unit each).

    Phase 2 (search): traverses layers top-down. Known nodes are checked for free.
    Unknown nodes are pruned and prioritized using score margin (gap between top-1 and
    top-2 aggregated scores). Nodes with displacement == 0 and a high margin are skipped
    (the estimate is confident the baseline stays on top). Remaining candidates are sorted
    by displacement descending, then margin ascending (least confident first).

    The budget only applies to phase 1 where we estimate the rankings.
    The "calls" variable tracks the number of LLM calls made in phase 2.

    Returns {"found", "key", "size", "ranking", "calls", "budget_used", "pruned_nodes"}.
    """
    if aggregator is None:
        aggregator = Linear.CombSUM()
    if entropy_fn is None:
        entropy_fn = softmax_entropy
    if cutoff_fn is None:
        cutoff_fn = max_jump_cutoff
    if margin_fn is None:
        margin_fn = score_margin
    if start_node is None:
        start_node = list(criteria)
    if budget is None:
        budget = len(rankings_by_combo)
    
    budget_remaining = budget

    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    baseline_answer = baseline_ranking[0]

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline_ranking}")

    answers = list(baseline_ranking)
    n = len(criteria)
    s = len(start_node)

    known = {}
    estimated_rankings = {}
    entropies: dict[str, float] = {}
    margins: dict[str, float] = {}

    # get the rankings for the singleton criteria
    for c in criteria:
        known[c] = rankings_by_combo[c]

    # phase 1: spending the budget
    if direction == "down":
        # Bottom-up through subsets of start_node (layers 2 → s-1)
        for layer_size in range(2, s):
            layer_nodes = []
            for combo in combinations(range(s), layer_size):
                subset = [start_node[i] for i in combo]
                key = combo_key(subset)
                voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
                
                with _silence_stdout():
                    ranking, scores = _aggregate_with_scores(voter_rankings, answers, aggregator)
                
                estimated_rankings[key] = ranking
                entropy_score = entropy_fn(scores)
                entropies[key] = entropy_score
                margins[key] = margin_fn(scores, baseline_answer)
                layer_nodes.append((key, entropy_score))

            selected_keys = cutoff_fn(layer_nodes, budget_remaining)
            for key in selected_keys:
                known[key] = rankings_by_combo[key]
                budget_remaining -= 1

    elif direction == "up":
        # Bottom-up through supersets (layers s+1 → n)
        extra = [c for c in criteria if c not in start_node]
        known[start_key] = rankings_by_combo.get(start_key, [])

        # get the rankings for the nodes in layer s+1 from the LLM
        # this is to mirror the bottom-up approach where we get the rankings for individual criteria
        # this is bounded by the number of extra criteria (c)
        for combo in combinations(range(len(extra)), 1):
            subset = list(start_node) + [extra[i] for i in combo]
            key = combo_key(subset)
            if key in rankings_by_combo:
                known[key] = rankings_by_combo[key]

        for layer_size in range(s + 2, n + 1):
            num_extra = layer_size - s
            layer_nodes = []
            for combo in combinations(range(len(extra)), num_extra):
                subset = list(start_node) + [extra[i] for i in combo]
                key = combo_key(subset)

                voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
                if not voter_rankings:
                    continue
                with _silence_stdout():
                    ranking, scores = _aggregate_with_scores(voter_rankings, answers, aggregator)
                estimated_rankings[key] = ranking
                entropy_score = entropy_fn(scores)
                entropies[key] = entropy_score
                margins[key] = margin_fn(scores, baseline_answer)
                layer_nodes.append((key, entropy_score))

            selected_keys = cutoff_fn(layer_nodes, budget_remaining)
            for key in selected_keys:
                if key in rankings_by_combo:
                    known[key] = rankings_by_combo[key]
                    budget_remaining -= 1

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

            # Check cached (known) nodes for free CFs
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

            ranking = estimated_rankings[key]

            # displacement: position of the baseline answer in the estimated ranking.
            # 0 = baseline is still top-1 (no estimated flip). Higher = more displaced.
            displacement = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            margin_val = margins.get(key, float('inf'))
            unknown_nodes.append((subset, key, displacement, margin_val))

        if not unknown_nodes:
            continue

        # Pruning displacement-0 nodes (where the estimate ranks the baseline answer first).
        #
        # Margin = baseline_answer_score - best_other_score from the aggregated Borda scores.
        # Positive margin means the baseline leads; larger = more confident it stays on top.
        # Normalized margin = margin / n_voters, so values are comparable across layers
        # (a layer-7 node sums 6 voters, producing larger raw margins than a layer-3 node).
        #
        # Two filters applied together:
        # 1. max-jump cutoff identifies the distinctly high-margin tier within the layer
        #    (finds the largest gap in sorted margins, selects nodes above it). This is
        #    the selective filter — it prevents pruning nodes whose margins are close to
        #    the layer average.
        # 2. margin_threshold is an absolute floor — even if max-jump selects a node,
        #    it's only pruned if its normalized margin exceeds this value. This prevents
        #    pruning when all margins in the layer are low (where the "largest gap" might
        #    still separate two small, unreliable values).
        n_voters = max(_layer_size - 1, 1)
        zero_disp = [(key, m / n_voters) for _, key, d, m in unknown_nodes if d == 0]
        safe_to_skip: set[str] = set()
        if zero_disp:
            high_margin_keys = set(cutoff_fn(zero_disp, len(zero_disp)))
            safe_to_skip = {k for k, norm_m in zero_disp if k in high_margin_keys and norm_m > margin_threshold}

        candidates = [
            (subset, key, displacement, margin_val)
            for subset, key, displacement, margin_val in unknown_nodes
            if not (displacement == 0 and key in safe_to_skip)
        ]

        pruned_nodes += len(safe_to_skip)

        if verbose:
            print(f"  Unknown: {len(unknown_nodes)}  Safe-to-skip: {len(safe_to_skip)}  Candidates: {len(candidates)}  Skipped: {len(unknown_nodes) - len(candidates)}")
            for subset, key, displacement, m in unknown_nodes:
                tag = "SKIP" if (displacement == 0 and key in safe_to_skip) else ""
                print(f"    {key}: disp={displacement} margin={m:.4f} {tag}")

        # Sort by displacement desc, then margin asc (least confident first)
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

    return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls, "budget_used": budget_used, "pruned_nodes": pruned_nodes}
