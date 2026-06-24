"""Find a layer where entropy range = 0 and explain why.

Pattern: only a handful of distinct singleton rankings exist. As layer size
grows, voter sets of size (m-1) almost always include voters from EVERY distinct
ranking class — so different parent nodes aggregate the SAME multiset of
voter-rankings → SAME Borda score dict → SAME softmax entropy.
"""
import json
from collections import Counter

# Pull zero-range layers from the instrumented data
with open("reports/era_pragma_pruning/data/layers.jsonl") as f:
    rows = [json.loads(line) for line in f]

zero_range = [
    r for r in rows
    if r["dataset"] == "sonnet_pragma"
    and len(r["entropy_values"]) > 5
    and all(e == r["entropy_values"][0] for e in r["entropy_values"])
]
print(f"sonnet_pragma layers with bit-identical entropy across all nodes (n_unknown>5): {len(zero_range)}")

if zero_range:
    sample = zero_range[0]
    print(f"\nExample: qid={sample['qid']} layer_size={sample['layer_size']} "
          f"n_unknown={sample['n_unknown']} entropy={sample['entropy_values'][0]:.6f}")

# Now reproduce one and inspect WHY voter sets collapse to the same multiset.
from itertools import combinations
from scripts.pragma.sonnet.run_subsample import _load_pragma_lattices
from search.lattice import combo_key
from search.rank import _aggregate_with_scores, _get_child_rankings
from search.heuristics import softmax_entropy

qid = zero_range[0]["qid"] if zero_range else "6"
target_layer = zero_range[0]["layer_size"] if zero_range else 4

lats = _load_pragma_lattices([qid])
crit, rbc = lats[qid]
s = len(crit)
print(f"\nqid={qid}, n_criteria={s}, target_layer={target_layer}")

# Count distinct singleton rankings, group pragmas by their ranking
ranking_classes: dict[tuple, list[str]] = {}
for c in crit:
    key = tuple(rbc[c])
    ranking_classes.setdefault(key, []).append(c)
print(f"\nDistinct singleton rankings: {len(ranking_classes)}")
for ranking, members in ranking_classes.items():
    print(f"  class top-3 = {ranking[:3]}")
    print(f"    pragmas ({len(members)}): {members}")

# Build estimates up to target_layer the same way Phase 1 does (bottom-up)
known = {c: rbc[c] for c in crit}
estimated_rankings = {}
answers = list(rbc[combo_key(crit)])

for layer_size in range(2, target_layer + 1):
    for combo in combinations(range(s), layer_size):
        subset = [crit[i] for i in combo]
        key = combo_key(subset)
        voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
        ranking, scores = _aggregate_with_scores(voter_rankings, answers, "combsum")
        estimated_rankings[key] = ranking

# Now look at the target layer's score dicts
score_tuples = []
ents = []
voter_multisets = []
for combo in combinations(range(s), target_layer):
    subset = [crit[i] for i in combo]
    key = combo_key(subset)
    voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
    ranking, scores = _aggregate_with_scores(voter_rankings, answers, "combsum")
    ents.append(softmax_entropy(scores))
    score_tuples.append(tuple(sorted(scores.items())))
    # voter signature: the multiset of voter RANKINGS (not voter keys)
    sig = tuple(sorted(tuple(vr) for vr in voter_rankings.values()))
    voter_multisets.append(sig)

print(f"\nLayer {target_layer} ({len(ents)} nodes):")
print(f"  distinct entropies   = {len(set(ents))}")
print(f"  distinct score dicts = {len(set(score_tuples))}")
print(f"  distinct voter-ranking multisets = {len(set(voter_multisets))}")
print(f"  entropy range = {max(ents) - min(ents):.6e}")

# Show the multiset of voter classes for a few nodes
print("\nFor 5 nodes, show the multiset of voter ranking-classes:")
class_ids = {ranking: i for i, ranking in enumerate(ranking_classes)}
for combo in list(combinations(range(s), target_layer))[:5]:
    subset = [crit[i] for i in combo]
    key = combo_key(subset)
    voter_rankings = _get_child_rankings(subset, known, estimated_rankings)
    voter_classes = Counter(class_ids[tuple(vr)] for vr in voter_rankings.values())
    print(f"  pragma_indices={list(combo)}  voter_classes={dict(voter_classes)}  ent={softmax_entropy(_aggregate_with_scores(voter_rankings, answers, 'combsum')[1]):.4f}")
