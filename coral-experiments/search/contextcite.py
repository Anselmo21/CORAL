import numpy as np
from search.heuristics import softmax_entropy, max_jump_cutoff
from context_cite import (
    ContextCiter,
    ConceptCriteriaPartitioner,
    CONCEPT_CRITERIA_PROMPT_TEMPLATE,
)

from search.lattice import combo_key, iter_search_layers


def contextcite_search(
    criteria,
    rankings_by_combo,
    query,
    sources,
    answers,
    model,
    tokenizer,
    num_ablations=64,
    ablation_keep_prob=0.5,
    batch_size=1,
    criteria_baseline="empty",
    start_node=None,
    direction="down",
    prune_threshold=0.0,
):
    """
    ContextCite-guided lattice search.

    Fits a Lasso surrogate on LLM logit-probs under random criteria ablations.
    For each lattice node, builds a binary presence/absence mask over all criteria
    and calls `solver.predict(mask)` to get the surrogate's predicted logit prob
    for the baseline answer under that subset. Nodes are checked in ascending order
    of predicted logit (least supported baseline first = most likely CFs first).

    Pruning (optional): if `prune_threshold` is set, nodes whose predicted logit
    exceeds the threshold are skipped without an LLM call. The predicted logit is
    in the solver's normalised space (logit_prob / num_output_tokens), so a sensible
    threshold can be derived from `solver.predict(start_node_mask)` at call time.

    Returns {"found", "key", "ranking", "calls", "pruned_nodes", "attributions",
             "num_ablations", "response", "solver"}.
    """
    if start_node is None:
        start_node = list(criteria)

    start_key = combo_key(start_node)
    baseline = rankings_by_combo.get(start_key, [])
    baseline_top1 = baseline[0] if baseline else None

    partitioner = ConceptCriteriaPartitioner(
        question=query,
        concepts=list(criteria),
        sources_block=sources,
        answers=list(answers),
    )

    cc = ContextCiter(
        model,
        tokenizer,
        context=partitioner.context,
        query="",
        partitioner=partitioner,
        prompt_template=CONCEPT_CRITERIA_PROMPT_TEMPLATE,
        criteria_baseline=criteria_baseline,
        num_ablations=num_ablations,
        batch_size=batch_size,
        ablation_keep_prob=ablation_keep_prob,
    )

    attr = cc.get_attributions(as_dataframe=False, verbose=False)
    attr_scores: dict[str, float] = {criteria[i]: float(attr[i]) for i in range(len(criteria))}

    solver = cc.last_fitted_solver  # LassoRegression with .predict(masks)
    criteria_to_idx = {c: i for i, c in enumerate(criteria)}
    n_criteria = len(criteria)

    def surrogate_score(subset: list[str]) -> float:
        mask = np.zeros(n_criteria, dtype=np.float32)
        for c in subset:
            if c in criteria_to_idx:
                mask[criteria_to_idx[c]] = 1.0
        return float(solver.predict(mask.reshape(1, -1))[0])

    calls = 0
    pruned_nodes = 0

    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        layer = []
        for subset in subsets:
            key = combo_key(subset)
            score = surrogate_score(subset)
            layer.append((key, score, subset))

        # Ascending: lowest predicted logit first = baseline least supported = most likely CF
        layer.sort(key=lambda x: (x[1], x[0]))

        for key, score, subset in layer:
            # unlikely to be a CF, logit probability is still positive
            if score > prune_threshold:
                pruned_nodes += 1
                continue

            calls += 1
            ranking = rankings_by_combo.get(key)
            if ranking is not None and ranking[0] != baseline_top1:
                return {
                    "found": True,
                    "key": key,
                    "ranking": ranking,
                    "calls": calls,
                    "pruned_nodes": pruned_nodes,
                    "attributions": attr_scores,
                    "num_ablations": num_ablations,
                    "response": cc.response,
                    "solver": solver,
                }

    return {
        "found": False,
        "key": None,
        "ranking": None,
        "calls": calls,
        "pruned_nodes": pruned_nodes,
        "attributions": attr_scores,
        "num_ablations": num_ablations,
        "response": cc.response,
        "solver": solver,
    }

def contextcite_search_entropy(
    criteria,
    rankings_by_combo,
    query,
    sources,
    answers,
    model,
    tokenizer,
    num_ablations=64,
    ablation_keep_prob=0.5,
    batch_size=1,
    criteria_baseline="empty",
    start_node=None,
    direction="down",
    prune_threshold=0.0,
    entropy_fn=None,
    pruning_fn=None,
):
    """
    ContextCite-guided lattice search.

    Fits a Lasso surrogate on LLM logit-probs under random criteria ablations.
    For each lattice node, builds a binary presence/absence mask over all criteria
    and calls `solver.predict(mask)` to get the surrogate's predicted logit prob
    for the baseline answer under that subset. Nodes are checked in ascending order
    of predicted logit (least supported baseline first = most likely CFs first).

    Pruning (optional): if `prune_threshold` is set, nodes whose predicted logit
    exceeds the threshold are skipped without an LLM call. The predicted logit is
    in the solver's normalised space (logit_prob / num_output_tokens), so a sensible
    threshold can be derived from `solver.predict(start_node_mask)` at call time.

    Returns {"found", "key", "ranking", "calls", "pruned_nodes", "attributions",
             "num_ablations", "response", "solver"}.
    """
    if start_node is None:
        start_node = list(criteria)
    if entropy_fn is None:
        entropy_fn = softmax_entropy
    if pruning_fn is None:
        pruning_fn = max_jump_cutoff

    start_key = combo_key(start_node)
    baseline = rankings_by_combo.get(start_key, [])
    baseline_top1 = baseline[0] if baseline else None

    partitioner = ConceptCriteriaPartitioner(
        question=query,
        concepts=list(criteria),
        sources_block=sources,
        answers=list(answers),
    )

    cc = ContextCiter(
        model,
        tokenizer,
        context=partitioner.context,
        query="",
        partitioner=partitioner,
        prompt_template=CONCEPT_CRITERIA_PROMPT_TEMPLATE,
        criteria_baseline=criteria_baseline,
        num_ablations=num_ablations,
        batch_size=batch_size,
        ablation_keep_prob=ablation_keep_prob,
    )

    attr = cc.get_attributions(as_dataframe=False, verbose=False)
    attr_scores: dict[str, float] = {criteria[i]: float(attr[i]) for i in range(len(criteria))}

    solver = cc.last_fitted_solver  # LassoRegression with .predict(masks)
    criteria_to_idx = {c: i for i, c in enumerate(criteria)}
    n_criteria = len(criteria)

    def surrogate_score(subset: list[str]) -> float:
        mask = np.zeros(n_criteria, dtype=np.float32)
        for c in subset:
            if c in criteria_to_idx:
                mask[criteria_to_idx[c]] = 1.0
        return float(solver.predict(mask.reshape(1, -1))[0])

    def entropy_score(surr_logit: float) -> float:
        # Convert surrogate logit-prob -> Bernoulli prob over {baseline, other},
        # then call the entropy_fn (which expects dict[str, float]).
        probability = 1.0 / (1.0 + np.exp(-surr_logit))
        return entropy_fn({"baseline": probability, "other": 1.0 - probability})

    calls = 0
    pruned_nodes = 0

    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        layer = []
        for subset in subsets:
            key = combo_key(subset)
            surr_score = surrogate_score(subset)
            score = entropy_score(surr_score)
            layer.append((key, score, subset))

        # Ascending: lowest entropy first = surrogate most confident
        layer.sort(key=lambda x: (x[1], x[0]))
        # max_jump_cutoff expects [(key, score)] 2-tuples
        high_entropy_keys = set(pruning_fn([(k, s) for k, s, _ in layer], len(layer)))
        pruned_nodes += len(high_entropy_keys)
        layer = [node for node in layer if node[0] not in high_entropy_keys]

        for key, score, subset in layer:
            # # unlikely to be a CF, logit probability is still positive
            # if score > prune_threshold:
            #     pruned_nodes += 1
            #     continue

            calls += 1
            ranking = rankings_by_combo.get(key)
            if ranking is not None and ranking[0] != baseline_top1:
                return {
                    "found": True,
                    "key": key,
                    "ranking": ranking,
                    "calls": calls,
                    "pruned_nodes": pruned_nodes,
                    "attributions": attr_scores,
                    "num_ablations": num_ablations,
                    "response": cc.response,
                    "solver": solver,
                }

    return {
        "found": False,
        "key": None,
        "ranking": None,
        "calls": calls,
        "pruned_nodes": pruned_nodes,
        "attributions": attr_scores,
        "num_ablations": num_ablations,
        "response": cc.response,
        "solver": solver,
    }
