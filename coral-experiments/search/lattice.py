from itertools import combinations


def combo_key(criteria_subset):
    """
    Helper function to convert a subset of criteria to a key.
    Kind of like a hash function.
    """
    return ", ".join(sorted(criteria_subset))


def iter_search_layers(criteria, start_node, direction):
    """
    Yield (layer_size, list_of_subsets) for each layer in search order.

    Down: layers |start|-1, |start|-2, ..., 1 — subsets of start_node.
    Up:   layers |start|+1, |start|+2, ..., |criteria| — supersets of start_node.
    """
    s = len(start_node)

    if direction == "down":
        for layer_size in range(s - 1, 0, -1):
            subsets = []
            for combo in combinations(range(s), layer_size):
                subsets.append([start_node[i] for i in combo])
            yield layer_size, subsets

    elif direction == "up":
        extra = [c for c in criteria if c not in start_node]
        n = len(criteria)
        for layer_size in range(s + 1, n + 1):
            num_extra = layer_size - s
            subsets = []
            for combo in combinations(range(len(extra)), num_extra):
                subset = list(start_node) + [extra[i] for i in combo]
                subsets.append(subset)
            yield layer_size, subsets
