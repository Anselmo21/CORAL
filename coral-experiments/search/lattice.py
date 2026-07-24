from itertools import combinations


def combo_key(criteria_subset):
    """
    Helper function to convert a subset of criteria to a key.
    Kind of like a hash function.
    """
    return ", ".join(sorted(criteria_subset))


def parse_combo_key(raw_key: str, sorted_criteria: list[str]) -> list[str]:
    """Parse a raw combo key back to a list of criteria.

    Handles names that contain ', ' (e.g. 'side, back, stomach') by scanning
    sorted criteria greedily left-to-right.
    """
    if not raw_key:
        return []
    result: list[str] = []
    remaining = raw_key
    for c in sorted_criteria:
        if remaining == c:
            result.append(c)
            remaining = ""
            break
        if remaining.startswith(c + ", "):
            result.append(c)
            remaining = remaining[len(c) + 2 :]
    return result if not remaining else []


def restrict_rbc(
    rbc: dict[str, list[str]],
    start_node: list[str],
    full_criteria: list[str],
) -> dict[str, list[str]]:
    """Return rbc restricted to subsets of start_node."""
    start_set = set(start_node)
    sorted_full = sorted(full_criteria)
    out: dict[str, list[str]] = {}
    for k, v in rbc.items():
        if not k:
            out[k] = v
            continue
        parts = parse_combo_key(k, sorted_full)
        if parts and set(parts) <= start_set:
            out[k] = v
    return out


def _key_to_set(key: str | None) -> set[str]:
    if not key:
        return set()
    return {c.strip() for c in key.split(",") if c.strip()}


def _key_sets(key_a: str, key_b: str, concepts) -> tuple[set[str], set[str]]:
    """Concept sets for two combo keys.

    Pass the known concept list to keep comma-bearing concepts intact.
    """
    if concepts is not None:
        cs = sorted(concepts)
        a = set(parse_combo_key(key_a, cs)) if key_a else set()
        b = set(parse_combo_key(key_b, cs)) if key_b else set()
        return a, b
    return _key_to_set(key_a), _key_to_set(key_b)


def hop_distance(key_a: str | None, key_b: str | None, concepts=None) -> int | None:
    """Symmetric-difference size |A △ B| between two combo keys (Hasse hops).

    Returns None when either key is missing.
    """
    if key_a is None or key_b is None:
        return None
    a, b = _key_sets(key_a, key_b, concepts)
    return len(a ^ b)


def jaccard_keys(key_a: str | None, key_b: str | None, concepts=None) -> float | None:
    """Jaccard similarity between the criteria sets encoded in two combo keys.

    Returns None when either key is missing/empty — those cases have no
    meaningful CF to compare and should be excluded from aggregate stats.
    """
    a, b = _key_sets(key_a or "", key_b or "", concepts)
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def iter_search_layers(criteria, start_node, direction):
    """
    Yield (layer_size, list_of_subsets) for each layer in search order.

    Down: layers |start|-1, |start|-2, ..., 1 — subsets of start_node.
    Up:   layers |start|+1, |start|+2, ..., |criteria| — supersets of start_node.
    """
    s = len(start_node)

    if direction == "down":
        for layer_size in range(s - 1, -1, -1):
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
