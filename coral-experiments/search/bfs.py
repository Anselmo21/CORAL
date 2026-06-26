from search.lattice import combo_key, iter_search_layers

'''
Breadth-first search algorithm for finding counterfactuals in a 
brute-force manner.
Exhaustively searches all subsets of the start_node until a
counterfactual is found.

Only supports finding a single counterfactual.
TODO: add support for finding multiple counterfactuals.
'''
def bfs_search(criteria, oracle_ranking, 
               start_node=None, direction="down",
               verbose=False):
    
    # set start_node to the full set of concepts if not provided
    if start_node is None:
        start_node = list(criteria)

    start_key = combo_key(start_node)
    baseline = oracle_ranking.get(start_key, [])
    calls = 0

    if verbose:
        print(f"Start key: {start_key}")
        print(f"Baseline: {baseline}")

    if not baseline:
        return {"found": False, "key": None, "size": None, "ranking": None, "calls": calls}

    original_answer = baseline[0]       # what we want to flip

    for _layer_size, subsets in iter_search_layers(criteria, start_node, direction):
        if verbose:
            print(f"Layer size: {_layer_size}")
        
        for subset in subsets:
            key = combo_key(subset)
            calls += 1

            ranking = oracle_ranking.get(key)
            if verbose:
                print(f"\t{key}: {ranking}")
            if ranking and ranking[0] != original_answer:
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
