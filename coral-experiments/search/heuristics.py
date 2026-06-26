import math
import numpy as np
from kneed import KneeLocator
'''
Heuristics for scoring and pruning nodes in the ERA search algorithm.
'''

# Finds the score gap between the original answer and the next best answer.
def score_margin(scores: dict[str, float], original_answer: str) -> float:
    original_score = None
    for k, v in scores.items():
        if k == original_answer or k.strip() == original_answer:
            original_score = v
            break
    if original_score is None:
        return float('-inf')
    best_other = max(
        (v for k, v in scores.items() if k != original_answer and k.strip() != original_answer),
        default=float('-inf'),
    )
    return original_score - best_other


def softmax_entropy(
    scores: dict[str, float],
    T: float = 1.0,
    num_voters: int | None = None,
    normalize_spread: bool = True,
) -> float:
    '''
    Computes the softmax entropy of the scores.
    First applies a shift to the scores to ensure they are all positive.
    Then applies a softmax function to the scores to get the probabilities.
    Then computes the entropy of the probabilities using the Shannon entropy formula.

    The temperature parameter T controls the sharpness of the softmax function.
    A higher temperature results in a more uniform distribution of probabilities.
    A lower temperature results in a more peaked distribution of probabilities.
    '''
    values = list(scores.values())
    if not values:
        return 0.0

    if normalize_spread:
        v_min, v_max = min(values), max(values)
        if v_max>v_min:
            values = [v / (v_max - v_min) for v in values]
        shifted_scores = values - np.max(values)
    else:
        arr = np.array(values, dtype=float)
        if num_voters and num_voters > 0:
            arr = arr / num_voters
        shifted_scores = arr - np.max(arr)

    exp_scores = np.exp(shifted_scores / T)
    probabilities = exp_scores / np.sum(exp_scores)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    return entropy


def shannon_entropy(
    scores: dict[str, float],
    num_voters: int | None = None,
) -> float:
    values = [v for v in scores.values() if v > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values]
    return -sum(p * math.log2(p) for p in probs)

def voter_disagreement_entropy(voter_rankings, answers):
    n = len(answers)
    rank_matrix = []
    for ranking in voter_rankings.values():
        rank_vec = [ranking.index(a) if a in ranking else n for a in answers]
        rank_matrix.append(rank_vec)
    arr = np.array(rank_matrix, dtype=float)  # shape: (n_voters, n_answers)
    variances = arr.var(axis=0)               # per-answer rank variance
    if variances.sum() == 0:
        return 0.0
    probs = variances / variances.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

def renyi_entropy(
    scores: dict[str, float],
    alpha: float = 2.0,
    T: float = 1.0,
    num_voters: int | None = None,
) -> float:
    values = list(scores.values())
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if num_voters and num_voters > 0:
        arr = arr / num_voters
    shifted = arr - np.max(arr)
    exp_scores = np.exp(shifted / T)
    probs = exp_scores / np.sum(exp_scores)
    if alpha == 1.0:
        return -float(np.sum(probs * np.log2(probs + 1e-12)))
    return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(probs ** alpha) + 1e-12))


def weighted_entropy(
    scores: dict[str, float],
    T: float = 1.0,
    num_voters: int | None = None,
) -> float:
    '''
    Scales the entropy by the weights of the probabilities.
    The weights are 1/rank, so the most likely answer has the highest weight.
    '''
    values = list(scores.values())
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if num_voters and num_voters > 0:
        arr = arr / num_voters
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

def kneedle_cutoff(nodes_with_entropy: list[tuple[str, float]], 
                   budget_remaining: int,
                   curve: str = "convex",
                   direction: str = "decreasing",
                   S: int = 1) -> list[str]:
    '''
    Finds the knee of the entropy curve.
    The knee is the point where the entropy curve changes from concave to convex.
    '''
    if budget_remaining <= 0 or not nodes_with_entropy:
        return []
    sorted_nodes = sorted(nodes_with_entropy, key=lambda x: x[1], reverse=True)
    
    if len(sorted_nodes) <= 1:
        return [k for k, _ in sorted_nodes[:budget_remaining]]
    ys = [x[1] for x in sorted_nodes]
    xs = range(len(sorted_nodes))
    kl = KneeLocator(xs, ys, S=S, curve=curve, direction=direction)
    cut = kl.elbow if kl.elbow is not None else len(sorted_nodes) - 1

    return [k for k, _ in sorted_nodes[:cut + 1][:budget_remaining]]