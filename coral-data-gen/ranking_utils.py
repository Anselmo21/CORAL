"""Canonical combo-key helpers used by generate_dataset.py."""
from typing import List


def normalize_combo(items: List[str]) -> List[str]:
    """Return a sorted copy of the combo so keys stay canonical."""
    return sorted(items)


def combo_key_from_items(items: List[str]) -> str:
    """Return the canonical key for a set of criteria: a comma-joined sorted string."""
    return ", ".join(normalize_combo(items))
