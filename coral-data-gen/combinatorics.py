from typing import List
from itertools import combinations

def generate_combinations(items: List[str], size: int) -> List[List[str]]:
    """
    Generate all combinations of a given size from the list of items.
    
    Args:
        items (List[str]): The list of items to generate combinations from.
        size (int): The size of each combination.
    
    Returns:
        List[List[str]]: A list of combinations, where each combination is a list of items.
    """
    if size == 0:
        return [[]]
    if not items:
        return None
    if size == 1:
        return [[item] for item in items]
    return [list(comb) for comb in combinations(items, size)]
