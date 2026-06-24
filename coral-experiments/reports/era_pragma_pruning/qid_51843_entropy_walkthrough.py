"""Walk through softmax_entropy on a real layer-6 node from qid 51843."""
from itertools import combinations
import numpy as np

from scripts.pragma.msmarco_regen.run_subsample import _load_lattices
from search.heuristics import softmax_entropy
from search.lattice import combo_key
from search.rank import _aggregate_with_scores, _get_child_rankings

lats = _load_lattices(["51843"])
criteria, rbc = lats["51843"]
start_node = list(criteria)
baseline = rbc[combo_key(start_node)]
answers = list(baseline)

print(f"Answer space (n = {len(answers)}):")
for i, a in enumerate(answers):
    print(f"  [{i}] {a[:75]}")

# Prime known with singletons
known = {c: rbc[c] for c in criteria}
estimated = {}

# Build layer-2 .. layer-5 estimates so layer-6 voters exist
for layer_size in range(2, 6):
    for combo in combinations(range(len(start_node)), layer_size):
        subset = [start_node[i] for i in combo]
        key = combo_key(subset)
        voters = _get_child_rankings(subset, known, estimated)
        ranking, _ = _aggregate_with_scores(voters, answers, "combsum")
        estimated[key] = ranking

# Pick the FIRST layer-6 node
first_combo = next(combinations(range(len(start_node)), 6))
subset = [start_node[i] for i in first_combo]
voters = _get_child_rankings(subset, known, estimated)
ranking, scores = _aggregate_with_scores(voters, answers, "combsum")

print(f"\nNode: layer 6, sub-lattice of size 6 (first such combo)")
print(f"  Voters: {len(voters)} (each a size-5 sub-lattice)")
print(f"\n  Borda scores per answer (sorted desc):")
for ans, score in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"    {score:>7.1f}   {ans[:70]}")

print(f"\n  softmax_entropy step-by-step:")
values = np.array(list(scores.values()), dtype=float)
print(f"    1) values            = {values.tolist()}")
shifted = values - np.max(values)
print(f"    2) shifted (- max)   = {shifted.tolist()}")
exp_scores = np.exp(shifted)
print(f"    3) exp(shifted)      = {[f'{v:.4g}' for v in exp_scores]}")
probs = exp_scores / np.sum(exp_scores)
print(f"    4) probs (normalize) = {[f'{p:.4g}' for p in probs]}")
log_terms = probs * np.log2(probs + 1e-12)
print(f"    5) p*log2(p)         = {[f'{t:.5f}' for t in log_terms]}")
H = -np.sum(log_terms)
print(f"    6) H = -sum(p log p) = {H:.5f} bits")
print(f"\n  (Library returns: {softmax_entropy(scores):.5f} bits — matches)")

# Sanity: max possible entropy
max_H = np.log2(len(answers))
print(f"\n  Theoretical max entropy for {len(answers)} answers = log2({len(answers)}) = {max_H:.3f} bits")
print(f"  This node uses {100*H/max_H:.1f}% of the available entropy range.")

# What if T were larger?
print(f"\n  Effect of temperature on this same Borda dict:")
for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
    H_T = softmax_entropy(scores, T=T)
    print(f"    T={T}: H = {H_T:.4f} bits   (probs concentrate when T low, spread when T high)")
