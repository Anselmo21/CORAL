from search.lattice import combo_key, iter_search_layers


def bfs_search(criteria, rankings_by_combo, start_node=None, direction="down",
               verbose=False):
    if start_node is None:
        start_node = list(criteria)

    start_key = combo_key(start_node)
    baseline = rankings_by_combo.get(start_key, [])
    calls = 0

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline}")

    if not baseline:
        return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls}

    baseline_top1 = baseline[0]

    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        if verbose:
            print(f"Layer size: {_layer_size}")
        for subset in subsets:
            key = combo_key(subset)
            calls += 1

            ranking = rankings_by_combo.get(key)
            if verbose:
                print(f"\t{key}: {ranking}")
            if ranking and ranking[0] != baseline_top1:
                if verbose:
                    print(f"\n\nFound counterfactual: {key} : {ranking}")
                return {
                    "found": True,
                    "key": key,
                    "size": len(subset),
                    "ranking": ranking,
                    "calls": calls,
                }

    return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls}
