import math

import numpy as np


def score_margin(scores: dict[str, float], baseline_answer: str) -> float:
    baseline_score = None
    for k, v in scores.items():
        if k == baseline_answer or k.strip() == baseline_answer:
            baseline_score = v
            break
    if baseline_score is None:
        return float('-inf')
    best_other = max(
        (v for k, v in scores.items() if k != baseline_answer and k.strip() != baseline_answer),
        default=float('-inf'),
    )
    return baseline_score - best_other


def softmax_entropy(scores: dict[str, float], T: float = 1.0) -> float:
    values = list(scores.values())
    if not values:
        return 0.0
    scores = np.array(values, dtype=float)

    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores / T)
    probabilities = exp_scores / np.sum(exp_scores)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    return entropy


def shannon_entropy(scores: dict[str, float]) -> float:
    values = [v for v in scores.values() if v > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values]
    return -sum(p * math.log2(p) for p in probs)


def renyi_entropy(scores: dict[str, float], alpha: float = 2.0, T: float = 1.0) -> float:
    values = list(scores.values())
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    shifted = arr - np.max(arr)
    exp_scores = np.exp(shifted / T)
    probs = exp_scores / np.sum(exp_scores)
    if alpha == 1.0:
        return -float(np.sum(probs * np.log2(probs + 1e-12)))
    return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(probs ** alpha) + 1e-12))


def weighted_entropy(scores: dict[str, float], T: float = 1.0) -> float:
    values = list(scores.values())
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    shifted = arr - np.max(arr)
    exp_scores = np.exp(shifted / T)
    probs = exp_scores / np.sum(exp_scores)
    sorted_probs = np.sort(probs)[::-1]
    weights = 1.0 / np.arange(1, len(sorted_probs) + 1)
    return -float(np.sum(weights * sorted_probs * np.log2(sorted_probs + 1e-12)))


def max_jump_cutoff(nodes_with_entropy: list[tuple[str, float]], budget_remaining: int) -> list[str]:
    """
    Determines a cutoff point for entropy scores.
    Selects nodes at or above the largest gap (highest-entropy side).
    When multiple gaps tie for largest, uses the latest gap so fewer nodes
    are excluded from the high-entropy tier.
    """
    if budget_remaining <= 0:
        return []

    sorted_nodes = sorted(nodes_with_entropy, key=lambda x: x[1], reverse=True)

    if len(sorted_nodes) <= 1:
        return [k for k, _ in sorted_nodes[:budget_remaining]]
    
    gaps = [sorted_nodes[i][1] - sorted_nodes[i + 1][1]
            for i in range(len(sorted_nodes) - 1)]
    
    max_gap_idx = max(range(len(gaps)), key=lambda i: (gaps[i], i))
    selected = sorted_nodes[:max_gap_idx + 1]
    return [k for k, _ in selected[:budget_remaining]]
