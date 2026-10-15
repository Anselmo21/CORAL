"""Keeps the items above the largest drop in a sorted list of scores (the Max Jump cutoff)"""

from __future__ import annotations

from collections.abc import Sequence


def max_jump_cutoff(scored_nodes: Sequence[tuple[str, float]], limit: int) -> list[str]:
    """Return up to limit nodes scoring above the largest gap between consecutive scores

    Ties between gaps go to the latest one so fewer nodes are cut

    Args:
        scored_nodes: Each node's key and score
        limit: The most nodes to return

    Returns:
        The keys of the kept nodes, highest score first
    """
    if limit <= 0:
        return []
    ordered = sorted(scored_nodes, key=lambda node: node[1], reverse=True)
    if len(ordered) <= 1:
        return [key for key, _score in ordered[:limit]]
    gaps = [ordered[index][1] - ordered[index + 1][1] for index in range(len(ordered) - 1)]
    cutoff = max(range(len(gaps)), key=lambda index: (gaps[index], index))
    return [key for key, _score in ordered[: cutoff + 1][:limit]]
