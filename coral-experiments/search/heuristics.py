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


def softmax_entropy(scores: dict[str, float], T: float = 1.0, **_) -> float:
    """Softmax entropy of an aggregated score vector.

    Shifts by the max before softmax (numerically stable, scale-invariant to
    the top answer). Does NOT range-normalize: the gap between the top answer
    and runners-up is preserved, so near-ties produce high entropy and clear
    winners produce low entropy.
    """
    values = list(scores.values())
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    shifted = arr - arr.max()
    exp_scores = np.exp(shifted / T)
    probabilities = exp_scores / np.sum(exp_scores)
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))


def shannon_entropy(
    scores: dict[str, float],
    num_voters: int | None = None,
) -> float:
    values = [v for v in scores.values() if v > 0]
    # values = [scores.get(k, 0) for k in scores.keys()]
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values]
    # print(f"probs: {probs}")
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
    # Fallback to 0 (top node only) when no knee found; selecting all nodes would
    # over-spend the budget on a flat curve.
    cut = kl.elbow if kl.elbow is not None else 0

    return [k for k, _ in sorted_nodes[:cut + 1][:budget_remaining]]


# ---------------------------------------------------------------------------
# Score normalization functions (applied to aggregated scores before entropy)
# ---------------------------------------------------------------------------

def norm_minmax(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def norm_zscore(scores: dict[str, float]) -> dict[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    mu, sigma = values.mean(), values.std()
    if sigma == 0:
        return {k: 0.0 for k in scores}
    keys = list(scores)
    return {keys[i]: float((values[i] - mu) / sigma) for i in range(len(keys))}


def norm_rank(scores: dict[str, float]) -> dict[str, float]:
    """Discard magnitudes; assign 1/rank weights (rank-1 gets 1.0)."""
    ordered = sorted(scores, key=lambda k: -scores[k])
    return {k: 1.0 / (rank + 1) for rank, k in enumerate(ordered)}


def voter_disagreement(scores: dict[str, float], voter_scores=None, **_) -> float:
    """Per-answer variance across voter score vectors, then Shannon H.
    Falls back to shannon_entropy on aggregated scores if no voters passed."""
    if voter_scores is None or len(voter_scores) < 2:
        return shannon_entropy(scores)
    answers = list(scores.keys())
    n = len(answers)
    mat = np.array([[vs.get(a, 0.0) for a in answers] for vs in voter_scores.values()])
    variances = mat.var(axis=0)
    total = variances.sum()
    if total == 0:
        return 0.0
    probs = variances / total
    return -np.sum(probs * np.log2(probs + 1e-12))


def top1_vote_fraction(scores: dict[str, float], voter_scores=None, **_) -> float:
    """Fraction of child voters that disagree on who is top-1.
    0 = all agree (low uncertainty), 1 = max disagreement."""
    if voter_scores is None or len(voter_scores) < 2:
        vals = sorted(scores.values(), reverse=True)
        if len(vals) < 2 or vals[0] == 0:
            return 0.0
        return 1.0 - (vals[0] - vals[1]) / vals[0]
    top1s = []
    for vs in voter_scores.values():
        best = max(vs, key=lambda a: (vs.get(a, 0.0), a))
        top1s.append(best)
    from collections import Counter
    counts = Counter(top1s)
    majority = counts.most_common(1)[0][1]
    return 1.0 - majority / len(top1s)


def score_gap_inv(scores: dict[str, float], **_) -> float:
    """Inverted gap between top-1 and top-2 aggregated scores.
    Large return = small gap = high uncertainty."""
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return 0.0
    gap = vals[0] - vals[1]
    return 1.0 / (gap + 1e-8)


def norm_softmax(scores: dict[str, float]) -> dict[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    shifted = values - values.max()
    exp_v = np.exp(shifted)
    probs = exp_v / exp_v.sum()
    return dict(zip(scores.keys(), probs.tolist()))


# ---------------------------------------------------------------------------
# CRA informativeness functions (voter-disagreement family)
#
# All accept (scores, voter_scores=None, **_) and return a float where
# higher = more informative = more worth spending budget on.
# ---------------------------------------------------------------------------

def cra_top1_disagreement(
    scores: dict[str, float],
    voter_scores: dict[str, dict[str, float]] | None = None,
    **_,
) -> float:
    """Fraction of child voters whose top-1 differs from the aggregate top-1.

    0 = all voters agree with aggregate → low information value.
    1 = every voter disagrees → maximum uncertainty about the decision.
    """
    if voter_scores is None or len(voter_scores) < 2:
        vals = sorted(scores.values(), reverse=True)
        if len(vals) < 2 or vals[0] == 0:
            return 0.0
        return 1.0 - (vals[0] - vals[1]) / (vals[0] + 1e-12)

    agg_top1 = max(scores, key=lambda a: (scores[a], a))
    n_disagree = 0
    for vs in voter_scores.values():
        voter_top1 = max(vs, key=lambda a: (vs.get(a, 0.0), a))
        if voter_top1 != agg_top1:
            n_disagree += 1
    return n_disagree / len(voter_scores)


def cra_score_variance(
    scores: dict[str, float],
    voter_scores: dict[str, dict[str, float]] | None = None,
    **_,
) -> float:
    """Mean per-answer score variance across voters.

    High variance = voters strongly disagree about individual answer quality.
    """
    if voter_scores is None or len(voter_scores) < 2:
        return 0.0
    answers = list(scores.keys())
    mat = np.array([[vs.get(a, 0.0) for a in answers] for vs in voter_scores.values()])
    return float(mat.var(axis=0).mean())


def cra_kendall_tau(
    scores: dict[str, float],
    voter_scores: dict[str, dict[str, float]] | None = None,
    **_,
) -> float:
    """Mean pairwise Kendall tau distance across voter rankings.

    Normalized to [0, 1]. Higher = more disagreement in pairwise ordering.
    """
    if voter_scores is None or len(voter_scores) < 2:
        return 0.0
    answers = list(scores.keys())
    n_ans = len(answers)
    if n_ans < 2:
        return 0.0
    n_pairs = n_ans * (n_ans - 1) // 2

    rankings: list[list[int]] = []
    for vs in voter_scores.values():
        ordered = sorted(range(n_ans), key=lambda i: (-vs.get(answers[i], 0.0), answers[i]))
        rank_vec = [0] * n_ans
        for r, idx in enumerate(ordered):
            rank_vec[idx] = r
        rankings.append(rank_vec)

    total_tau = 0.0
    count = 0
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            discordant = 0
            for a in range(n_ans):
                for b in range(a + 1, n_ans):
                    if (rankings[i][a] - rankings[i][b]) * (rankings[j][a] - rankings[j][b]) < 0:
                        discordant += 1
            total_tau += discordant / n_pairs
            count += 1
    return total_tau / count if count > 0 else 0.0


def cra_distance_to_aggregate(
    scores: dict[str, float],
    voter_scores: dict[str, dict[str, float]] | None = None,
    **_,
) -> float:
    """Mean L2 distance from each voter's score vector to the aggregate.

    High distance = voters are far from consensus.
    """
    if voter_scores is None or len(voter_scores) < 2:
        return 0.0
    answers = list(scores.keys())
    agg = np.array([scores.get(a, 0.0) for a in answers])
    if np.linalg.norm(agg) > 0:
        agg = agg / np.linalg.norm(agg)
    total = 0.0
    for vs in voter_scores.values():
        v = np.array([vs.get(a, 0.0) for a in answers])
        if np.linalg.norm(v) > 0:
            v = v / np.linalg.norm(v)
        total += float(np.linalg.norm(v - agg))
    return total / len(voter_scores)