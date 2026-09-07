import numpy as np
from context_cite import (
    ContextCiter,
    VLLMContextCiter,
    ConceptCriteriaPartitioner,
    CONCEPT_CRITERIA_PROMPT_TEMPLATE,
)

from search.lattice import combo_key, iter_search_layers


def _make_citer(model, tokenizer, **kwargs):
    """Return VLLMContextCiter if model is a vllm LLM, else ContextCiter."""
    if VLLMContextCiter is not None:
        from vllm import LLM
        if isinstance(model, LLM):
            return VLLMContextCiter(model, tokenizer, **kwargs)
    return ContextCiter(model, tokenizer, **kwargs)


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

    Fits a Lasso linear model on LLM logit-probs under random criteria ablations.
    For each lattice node, builds a binary presence/absence mask over all criteria
    and calls `solver.predict(mask)` to get the surrogate's predicted logit prob
    for the baseline answer under that subset. Nodes are checked in ascending order
    of predicted logit (least supported baseline first = most likely CFs first).

    Pruning: nodes whose predicted logit exceeds the threshold are skipped without 
    an LLM call. By default, the threshold is set to 0, which is equivalent to
    a probability of 50% for generating the baseline answer. 
    This means that by pruning, we are only evaluating nodes that are likely to be CFs
    (below 50% prob that we'll generate the baseline answer).
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

    cc = _make_citer(
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

    result = _surrogate_guided_search(
        criteria, rankings_by_combo, baseline_top1,
        start_node, direction, surrogate_score, prune_threshold,
    )
    result.update({
        "attributions": attr_scores,
        "num_ablations": num_ablations,
        "response": cc.response,
        "solver": solver,
        "usage": dict(cc.usage),
    })
    return result


def _surrogate_guided_search(
    criteria,
    rankings_by_combo,
    baseline_top1,
    start_node,
    direction,
    surrogate_score,
    prune_threshold,
):
    """Layer-by-layer traversal ordered by a surrogate score.

    Within each layer, nodes are checked in ascending predicted score
    (baseline least supported first); nodes with score > prune_threshold are
    skipped without verification. Returns the first verified CF.
    """
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
                }

    return {
        "found": False,
        "key": None,
        "ranking": None,
        "calls": calls,
        "pruned_nodes": pruned_nodes,
    }


FORCED_RANKING_PREFIX = '{"ranking": ['


def contextcite_rank_search(
    criteria,
    rankings_by_combo,
    query,
    sources,
    answers,
    model,
    tokenizer,
    mode="criteria",
    num_ablations=64,
    ablation_keep_prob=0.5,
    batch_size=1,
    criteria_baseline=None,
    start_node=None,
    direction="down",
    prune_threshold=0.0,
):
    """Rank-target ContextCite search ("LSM-rank").

    Renders the exact ground-truth scoring prompt (coral-runs/qa.py, vendored
    in context_cite.ScoringPromptPartitioner), teacher-forces the response
    prefix '{"ranking": [<baseline_index>', and fits the Lasso surrogate on
    logit(p) of the forced index token — the same per-token quantity the
    ContextCite paper uses, restricted to the position that decides the
    ground-truth top-1. Predicted score > 0 then means predicted
    P(index token) > 50% over the vocabulary, which guarantees the index is
    argmax, so prune_threshold=0.0 has sound non-CF semantics.

    criteria_baseline defaults to None (NOT "empty" as in contextcite_search):
    reference subtraction would turn the target into lift-over-empty-prompt
    and destroy the threshold-0 argmax semantics. Keep it None unless
    deliberately A/B-testing.
    """
    if start_node is None:
        start_node = list(criteria)

    from context_cite import ScoringPromptPartitioner, SCORING_PROMPT_TEMPLATE

    start_key = combo_key(start_node)
    baseline = rankings_by_combo.get(start_key, [])
    baseline_top1 = baseline[0] if baseline else None
    if baseline_top1 is None:
        raise ValueError(f"no baseline ranking for start node {start_key!r}")
    try:
        baseline_index = list(answers).index(baseline_top1)
    except ValueError:
        raise ValueError(
            f"baseline top-1 {baseline_top1!r} not found in answers "
            f"{list(answers)!r} — stale or mismatched cases"
        )
    # Single-digit index assumed by the one-token target expectation below.
    assert len(answers) <= 10, "multi-digit answer indices not supported"

    partitioner = ScoringPromptPartitioner(
        question=query,
        atoms=list(criteria),
        sources_block=sources,
        answers=list(answers),
        mode=mode,
    )
    forced = FORCED_RANKING_PREFIX + str(baseline_index)

    cc = _make_citer(
        model,
        tokenizer,
        context=partitioner.context,
        query="",
        partitioner=partitioner,
        prompt_template=SCORING_PROMPT_TEMPLATE,
        response_override=forced,
        criteria_baseline=criteria_baseline,
        num_ablations=num_ablations,
        batch_size=batch_size,
        ablation_keep_prob=ablation_keep_prob,
    )

    # Fit on the char range covering just the index digit -> its token(s).
    prefix_len = len(FORCED_RANKING_PREFIX)
    attr = cc.get_attributions(
        start_idx=prefix_len, end_idx=len(forced),
        as_dataframe=False, verbose=False,
    )
    attr_scores = {c: float(attr[i]) for i, c in enumerate(criteria)}

    tok_start, tok_end = cc._indices_to_token_indices(prefix_len, len(forced))
    index_token_span = int(tok_end - tok_start)
    if index_token_span != 1:
        print(
            f"WARNING: forced index digit spans {index_token_span} tokens "
            f"(BPE merge?); target is P(merged span), semantics still hold"
        )

    solver = cc.last_fitted_solver
    criteria_to_idx = {c: i for i, c in enumerate(criteria)}
    n_criteria = len(criteria)

    def surrogate_score(subset):
        mask = np.zeros(n_criteria, dtype=np.float32)
        for c in subset:
            if c in criteria_to_idx:
                mask[criteria_to_idx[c]] = 1.0
        return float(solver.predict(mask.reshape(1, -1))[0])

    result = _surrogate_guided_search(
        criteria, rankings_by_combo, baseline_top1,
        start_node, direction, surrogate_score, prune_threshold,
    )
    result.update({
        "attributions": attr_scores,
        "num_ablations": num_ablations,
        "response": cc.response,
        "solver": solver,
        "usage": dict(cc.usage),
        "baseline_index": baseline_index,
        "forced_response": forced,
        "index_token_span": index_token_span,
    })
    return result
