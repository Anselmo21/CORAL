"""Generate a synthetic pragma-style dataset with forced 2-way and 3-way ties.

Each query has 10 criteria and 3 candidate answers {A, B, C}. Each criterion
carries a hand-picked strict preference over the answers. For every subset S
of the 10 criteria, raw_answers[combo_key(S)] is the Borda aggregation of
those singleton preferences, tie-broken alphabetically.

Query archetypes (all with CLEAR full-lattice baseline but mid-lattice ties
that force the search to reason about ambiguity, and CFs requiring >=2
removals so Phase-1 budgeting materially affects accuracy):

  SHIFT (10 queries): 6 (A>B>C), 4 (B>A>C). Baseline: A (16 vs 14). Sub-
    lattices at layer 2 (e.g. drop 2 A-supporters) hit A=B ties; CF at
    layer 3.
  RAZOR (10 queries): 7 (A>B>C), 3 (B>A>C). Baseline: A (17 vs 13). CF at
    layer 4; near-ties at layer 3.
  CASCADE (10 queries): 4 (A>B>C), 3 (B>C>A), 3 (C>A>B). Baseline: A wins
    on Borda 11 vs 9 vs 10. Ties among B and C emerge as A-supporters get
    removed, then CF cascades.

Output: data/synthetic/tied-lattices.json in the qwen3-style loader format
(fields: name, question, criteria, plausible_answers, raw_answers, halted).

Usage:
    conda run -n coral python3 -m scripts.synthetic.gen_tied_lattices
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

from search.lattice import combo_key

ANSWERS = ["A", "B", "C"]
N_CRITERIA = 10

# Six strict rankings over 3 answers.
PREFS = {
    "ABC": ["A", "B", "C"],
    "ACB": ["A", "C", "B"],
    "BAC": ["B", "A", "C"],
    "BCA": ["B", "C", "A"],
    "CAB": ["C", "A", "B"],
    "CBA": ["C", "B", "A"],
}


def _borda_ranking(criterion_prefs: list[list[str]]) -> list[str]:
    """Aggregate strict rankings via Borda, tie-break alphabetically."""
    scores = {a: 0 for a in ANSWERS}
    n = len(ANSWERS)
    for ranking in criterion_prefs:
        for rank, ans in enumerate(ranking):
            scores[ans] += n - 1 - rank
    return sorted(ANSWERS, key=lambda a: (-scores[a], a))


def _build_query(qid: str, criterion_prefs: list[list[str]], question: str) -> dict:
    criteria = [f"c{i+1}" for i in range(N_CRITERIA)]
    raw_answers: dict[str, list[str]] = {}
    for size in range(1, N_CRITERIA + 1):
        for combo in itertools.combinations(range(N_CRITERIA), size):
            key = combo_key([criteria[i] for i in combo])
            prefs = [criterion_prefs[i] for i in combo]
            raw_answers[key] = _borda_ranking(prefs)
    return {
        "name": qid,
        "question": question,
        "mode": "synthetic",
        "criteria": criteria,
        "plausible_answers": list(ANSWERS),
        "sources": [],
        "halted": False,
        "halted_reason": None,
        "combos_total": len(raw_answers),
        "combos_run": len(raw_answers),
        "raw_answers": raw_answers,
    }


def _shuffle_seed(prefs: list[list[str]], seed: int) -> list[list[str]]:
    """Positional shuffle so equivalent splits appear at different criterion
    positions across queries."""
    order = [(seed * 5 + i * 7) % N_CRITERIA for i in range(N_CRITERIA)]
    seen, ord2 = set(), []
    for p in order:
        if p not in seen:
            seen.add(p); ord2.append(p)
    for p in range(N_CRITERIA):
        if p not in seen:
            ord2.append(p)
    return [prefs[i] for i in ord2]


def _shift_prefs(seed: int) -> list[list[str]]:
    """6 (A>B>C), 4 (B>A>C). Full: A=16, B=14, C=0. CF at layer 3 (need to
    remove 3 A-supporters to flip). Mid-lattice ties at layer 2 (some
    sub-lattices of 4 A + 4 B tie A=B)."""
    prefs = [PREFS["ABC"]] * 6 + [PREFS["BAC"]] * 4
    return _shuffle_seed(prefs, seed)


def _razor_prefs(seed: int) -> list[list[str]]:
    """7 (A>B>C), 3 (B>A>C). Full: A=17, B=13, C=0. CF at layer 4 (drop 4
    A-supporters). Near-ties emerge at layer 3."""
    prefs = [PREFS["ABC"]] * 7 + [PREFS["BAC"]] * 3
    return _shuffle_seed(prefs, seed)


def _cascade_prefs(seed: int) -> list[list[str]]:
    """4 (A>B>C), 3 (B>C>A), 3 (C>A>B). Full: A=4*2+3*0+3*1=11, B=4*1+3*2+
    3*0=10, C=4*0+3*1+3*2=9. A wins narrowly. Removing A-supporters
    scrambles the ranking through B and C ties. Deep, layered CFs."""
    prefs = [PREFS["ABC"]] * 4 + [PREFS["BCA"]] * 3 + [PREFS["CAB"]] * 3
    return _shuffle_seed(prefs, seed)


def build_dataset() -> dict:
    ds: dict[str, dict] = {}
    for i in range(10):
        qid = f"shift_{i:02d}"
        ds[qid] = _build_query(qid, _shift_prefs(i), f"synthetic shift query {i}")
    for i in range(10):
        qid = f"razor_{i:02d}"
        ds[qid] = _build_query(qid, _razor_prefs(i), f"synthetic razor query {i}")
    for i in range(10):
        qid = f"cascade_{i:02d}"
        ds[qid] = _build_query(qid, _cascade_prefs(i), f"synthetic cascade query {i}")
    return ds


def main() -> None:
    ds = build_dataset()
    out_path = Path("data/synthetic/tied-lattices.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ds, f, indent=2)
    print(f"Wrote {len(ds)} queries → {out_path}")

    # Quick sanity summary.
    from collections import Counter
    full_top1 = Counter()
    for qid, q in ds.items():
        crit = q["criteria"]
        key = combo_key(crit)
        full_top1[q["raw_answers"][key][0]] += 1
    print(f"Full-lattice top-1 distribution: {dict(full_top1)}")


if __name__ == "__main__":
    main()
