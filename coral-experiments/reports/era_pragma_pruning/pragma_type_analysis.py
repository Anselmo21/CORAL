"""Per-pragma-type informativeness on msmarco-pragma.

For each pragma in each qid:
  - extract the type prefix ("(Temporal):", "(Geographic):", etc.)
  - measure whether its singleton ranking exactly equals the baseline (full-set) ranking
  - measure whether its top-1 equals the baseline top-1

Aggregate by type to see if Temporal / Geographic are systematically less informative
than Perspective / Trustworthiness.
"""
import json
import re
from collections import defaultdict
from statistics import mean

from search.lattice import combo_key
from scripts.pragma.msmarco.run_subsample import _load_all_qids, _load_lattices

TYPE_RE = re.compile(r"^\(([^)]+)\):")


def pragma_type(p: str) -> str:
    m = TYPE_RE.match(p)
    return m.group(1) if m else "unknown"


lats = _load_lattices(_load_all_qids())

# {type -> list of (matches_baseline_full, matches_baseline_top1, qid, pragma_text)}
by_type: dict[str, list[dict]] = defaultdict(list)

for qid, (crit, rbc) in lats.items():
    baseline = tuple(rbc[combo_key(crit)])
    baseline_top1 = baseline[0]
    for p in crit:
        single = tuple(rbc[p])
        by_type[pragma_type(p)].append({
            "qid": qid,
            "pragma": p,
            "matches_full":   single == baseline,
            "matches_top1":   single[0] == baseline_top1,
        })

print(f"{'Type':<20} {'n':>4} {'full==baseline':>15} {'top1==baseline':>15}")
print(f"{'-'*20} {'-'*4} {'-'*15} {'-'*15}")
for t in sorted(by_type, key=lambda k: -mean(r["matches_full"] for r in by_type[k])):
    rows = by_type[t]
    n = len(rows)
    full = mean(r["matches_full"] for r in rows)
    top1 = mean(r["matches_top1"] for r in rows)
    print(f"{t:<20} {n:>4} {full*100:>14.1f}% {top1*100:>14.1f}%")

# Drill: per qid, which type contributes the most agreement
print("\nPer-qid breakdown (fraction matching baseline FULL ranking):")
all_types = sorted(by_type)
header = "  qid       | " + " | ".join(f"{t[:6]:>6}" for t in all_types)
print(header)
print("  " + "-" * (len(header) - 2))
for qid in sorted(lats):
    row = f"  {qid:<9} |"
    for t in all_types:
        recs = [r for r in by_type[t] if r["qid"] == qid]
        if recs:
            row += f" {mean(r['matches_full'] for r in recs)*100:>5.0f}%"
        else:
            row += f" {'—':>6}"
    print(row)
