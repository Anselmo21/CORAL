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
):
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
    attr_scores = {criteria[i]: float(attr[i]) for i in range(len(criteria))}

    calls = 0

    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        layer = []
        for subset in subsets:
            agg_score = sum(attr_scores.get(c, 0.0) for c in subset)
            key = combo_key(subset)
            layer.append((key, agg_score))

        layer.sort(key=lambda x: (x[1], x[0]))

        for key, _score in layer:
            calls += 1
            ranking = rankings_by_combo.get(key)
            if ranking is not None and ranking[0] != baseline_top1:
                return {
                    "found": True,
                    "key": key,
                    "ranking": ranking,
                    "calls": calls,
                    "attributions": attr_scores,
                    "num_ablations": num_ablations,
                    "response": cc.response,
                }

    return {
        "found": False,
        "key": None,
        "ranking": None,
        "calls": calls,
        "attributions": attr_scores,
        "num_ablations": num_ablations,
        "response": cc.response,
    }
