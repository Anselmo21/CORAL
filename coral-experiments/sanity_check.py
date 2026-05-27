"""
sanity_check.py — Diagnostic: how well do ERA's pre-call score estimates
predict the actual ground-truth score margins?

For every estimated node across all lattices, computes:
  delta_orig_estimated = estimated_score[baseline] - max(estimated_score[others])
  delta_orig_actual    = actual_borda_score[baseline] - max(actual_borda_score[others])

Then plots, correlates, and interprets.

Usage:
    conda run -n coral python3 sanity_check.py
"""

from __future__ import annotations

import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from search.lattice import combo_key
from search.rank import _get_child_rankings


def _borda_scores(ranking: list[str]) -> dict[str, float]:
    n = len(ranking)
    return {answer: float(n - 1 - i) for i, answer in enumerate(ranking)}


def _combsum_scores(
    voter_rankings: dict[str, list[str]], answers: list[str],
) -> tuple[list[str], dict[str, float]]:
    n = len(answers)
    totals: dict[str, float] = {a: 0.0 for a in answers}
    for ranking in voter_rankings.values():
        for rank_idx, answer in enumerate(ranking):
            if answer in totals:
                totals[answer] += n - 1 - rank_idx
    sorted_answers = sorted(totals, key=lambda a: totals[a], reverse=True)
    return sorted_answers, totals


def _delta_orig(scores: dict[str, float], baseline_answer: str) -> float:
    baseline_score = scores.get(baseline_answer, float("-inf"))
    best_other = max(
        (v for k, v in scores.items() if k != baseline_answer),
        default=float("-inf"),
    )
    return baseline_score - best_other


def collect_deltas(
    lattices: list[dict],
    direction: str = "down",
) -> tuple[list[float], list[float]]:
    deltas_estimated: list[float] = []
    deltas_actual: list[float] = []

    for lattice in lattices:
        criteria = lattice["criteria"]
        rbc = lattice["rankings_by_combo"]
        start_node = lattice.get("start_node", list(criteria))
        start_key = combo_key(start_node)
        baseline_ranking = rbc.get(start_key, [])
        if not baseline_ranking:
            continue
        baseline_answer = baseline_ranking[0]
        answers = list(baseline_ranking)
        s = len(start_node)
        n = len(criteria)

        known: dict[str, list[str]] = {}
        estimated_rankings: dict[str, list[str]] = {}

        if direction == "down":
            for c in criteria:
                if c in rbc:
                    known[c] = rbc[c]

            for layer_size in range(2, s):
                for combo in combinations(range(s), layer_size):
                    subset = [start_node[i] for i in combo]
                    key = combo_key(subset)
                    voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
                    if not voter_rankings:
                        continue
                    ranking, scores_est = _combsum_scores(voter_rankings, answers)
                    estimated_rankings[key] = ranking

                    actual_ranking = rbc.get(key)
                    if actual_ranking is None:
                        continue
                    scores_act = _borda_scores(actual_ranking)

                    deltas_estimated.append(_delta_orig(scores_est, baseline_answer))
                    deltas_actual.append(_delta_orig(scores_act, baseline_answer))

        elif direction == "up":
            known[start_key] = rbc.get(start_key, [])
            extra = [c for c in criteria if c not in start_node]

            for combo in combinations(range(len(extra)), 1):
                subset = list(start_node) + [extra[i] for i in combo]
                key = combo_key(subset)
                if key in rbc:
                    known[key] = rbc[key]

            for layer_size in range(s + 2, n + 1):
                num_extra = layer_size - s
                for combo in combinations(range(len(extra)), num_extra):
                    subset = list(start_node) + [extra[i] for i in combo]
                    key = combo_key(subset)
                    voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
                    if not voter_rankings:
                        continue
                    ranking, scores_est = _combsum_scores(voter_rankings, answers)
                    estimated_rankings[key] = ranking

                    actual_ranking = rbc.get(key)
                    if actual_ranking is None:
                        continue
                    scores_act = _borda_scores(actual_ranking)

                    deltas_estimated.append(_delta_orig(scores_est, baseline_answer))
                    deltas_actual.append(_delta_orig(scores_act, baseline_answer))

    return deltas_estimated, deltas_actual


def sanity_check(
    lattices: list[dict],
    direction: str = "down",
) -> None:
    deltas_est, deltas_act = collect_deltas(lattices, direction)

    if len(deltas_est) < 2:
        print(f"Only {len(deltas_est)} data points — not enough for analysis.")
        return

    x = np.array(deltas_est)
    y = np.array(deltas_act)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        print(f"Dropped {n_dropped} non-finite data points")
    x = x[finite_mask]
    y = y[finite_mask]
    n = len(x)
    if n < 2:
        print(f"Only {n} finite data points — not enough for analysis.")
        return

    # --- Correlations ---
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    print(f"Data points: {n}")
    print(f"Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")
    print(f"Spearman r = {r_spearman:.4f}  (p = {p_spearman:.2e})")

    # --- Estimation error ---
    error = x - y
    print(f"\nEstimation error (estimated - actual):")
    print(f"  Mean = {error.mean():.4f}")
    print(f"  Std  = {error.std():.4f}")

    # --- Linear regression ---
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_sq = r_value ** 2
    print(f"\nLinear regression (actual ~ estimated):")
    print(f"  Slope     = {slope:.4f}")
    print(f"  Intercept = {intercept:.4f}")
    print(f"  R²        = {r_sq:.4f}")

    # --- Interpretation ---
    print(f"\n{'─' * 52}")
    if r_pearson > 0.7 and error.std() < np.std(y):
        print("The estimated deltas are strongly correlated with actual deltas")
        print("and the error std is smaller than the signal std — the estimates")
        print("carry meaningful information for pruning.")
    elif r_pearson > 0.4:
        print("Moderate correlation — the estimates capture the direction of")
        print("score margins but are noisy. Pruning will help on average but")
        print("will misclassify a non-trivial fraction of nodes.")
    else:
        print("Weak or no correlation — the estimated deltas do not reliably")
        print("predict actual score margins. Using them for pruning risks")
        print("discarding true counterfactuals.")

    if abs(error.mean()) > 0.5 * error.std():
        direction_word = "optimistic" if error.mean() > 0 else "pessimistic"
        print(f"There is a systematic {direction_word} bias (mean error {error.mean():.2f}).")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.3, s=12, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="actual flip boundary (y=0)")
    ax.axvline(0, color="blue", linestyle="--", linewidth=0.8, label="estimated flip boundary (x=0)")

    x_range = np.array([x.min(), x.max()])
    ax.plot(x_range, slope * x_range + intercept, color="orange", linewidth=1.5,
            label=f"regression (R²={r_sq:.2f})")

    ax.set_xlabel("delta_orig_estimated (baseline score - best other, estimated)")
    ax.set_ylabel("delta_orig_actual (baseline score - best other, actual)")
    ax.set_title("ERA estimate calibration")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150)
    print(f"\nPlot saved to sanity_check.png")
    plt.show()


if __name__ == "__main__":
    with open("data/base.json") as f:
        data = json.load(f)

    lattices = []
    for row in data["rows"]:
        for run in row["baseline_lattice_runs"]:
            lattices.append({
                "criteria": run["criteria_used"],
                "rankings_by_combo": run["rankings_by_combo"],
                "start_node": run["criteria_used"],
            })

    sanity_check(lattices)
