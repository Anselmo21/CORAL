"""
test_lattice.py — Visual test suite for BFS and SRA lattice search.

Lattice: 3 criteria → 2^3 = 8 nodes (power-set of {price, quality, speed}).
Answers: ["A", "B", "C"]  (A is the original answer in all test cases)

The "" key holds the LLM result for the FULL SET call — its top-1 IS the original
answer we're trying to flip. "price, quality, speed" always equals "" (same call),
so the full-set node can never be a counterfactual. The search explores strict subsets.

Test cases:
  TC1 – CF at layer 2, first BFS node  : "price, quality" flips immediately
  TC2 – CF at layer 2, last BFS node   : "quality, speed" — SRA prioritises it first
  TC3 – CF at layer 1                  : "speed" singleton — SRA finds it first
  TC4 – No CF                          : all nodes exhausted
  TC5 – SRA false positive             : proxy estimate wrong; BFS finds real CF

Run:
  python test_lattice.py
  conda run -n coral python test_lattice.py   # to use real pyflagr
"""

import sys
import os
from contextlib import contextmanager
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@contextmanager
def _silence_stdout():
    """Suppress C-level stdout (e.g. pyflagr's progress lines) via fd-level redirect."""
    fflush = None
    try:
        import ctypes
        fflush = ctypes.cdll.LoadLibrary(None).fflush
    except Exception:
        pass
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        if fflush:
            fflush(None)
        os.dup2(devnull_fd, 1)
        yield
    finally:
        if fflush:
            fflush(None)
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)

import pandas as pd
from search.bfs import bfs_search as _bfs_reference
from search.lattice import combo_key


# ─── Mock CombSUM (no pyflagr required) ──────────────────────────────────────

class MockCombSUM:
    """Borda-count CombSUM. Ties broken alphabetically for determinism."""
    def aggregate(self, input_df):
        scores = input_df.groupby("Item Code")["Item Score"].sum()
        df = pd.DataFrame({"ItemID": scores.index, "Score": scores.values})
        df = df.sort_values(["Score", "ItemID"], ascending=[False, True]).reset_index(drop=True)
        return df, None


# ─── Try real pyflagr; fall back to mock ──────────────────────────────────────

try:
    import pyflagr.Linear as Linear
    _aggregator = Linear.CombSUM()
    _agg_label = "pyflagr CombSUM"
except ImportError:
    _aggregator = MockCombSUM()
    _agg_label = "MockCombSUM (no pyflagr)"


# ─── Tracked BFS ──────────────────────────────────────────────────────────────

def bfs_tracked(criteria, rankings_by_combo):
    """BFS with per-step logging. Logic is identical to search/bfs.py."""
    baseline = rankings_by_combo.get("", [])
    n = len(criteria)
    calls, steps = 0, []
    for layer_size in range(n, 0, -1):
        for idx in combinations(range(n), layer_size):
            subset = [criteria[i] for i in idx]
            key = combo_key(subset)
            calls += 1
            ranking = rankings_by_combo.get(key, [])
            is_cf = bool(ranking and ranking[0] != baseline[0])
            steps.append({
                "call": calls, "layer": layer_size,
                "key": key, "ranking": ranking, "is_cf": is_cf,
            })
            if is_cf:
                return {"found": True, "key": key, "ranking": ranking,
                        "calls": calls, "steps": steps}
    return {"found": False, "key": None, "ranking": None, "calls": calls, "steps": steps}


# ─── Tracked SRA ──────────────────────────────────────────────────────────────

def _borda_df(subset, singleton_rankings, answers):
    n = len(answers)
    rows = []
    for c in subset:
        for rank, ans in enumerate(singleton_rankings[c]):
            rows.append({
                "Query": "q", "Voter Name": c, "Item Code": ans,
                "Item Rank": rank + 1, "Item Score": n - 1 - rank,
                "Algorithm/Dataset": "sra",
            })
    return pd.DataFrame(rows)


def sra_tracked(criteria, rankings_by_combo, aggregator=None):
    """SRA with per-layer logging. Logic is identical to search/rank.py."""
    if aggregator is None:
        aggregator = _aggregator
    baseline = rankings_by_combo.get("", [])
    baseline_top1 = baseline[0]
    answers = list(baseline)
    singletons = {c: rankings_by_combo[c] for c in criteria if c in rankings_by_combo}
    n = len(criteria)
    calls, steps, layer_logs = 0, [], []

    for layer_size in range(n, 0, -1):
        raw = []
        for idx in combinations(range(n), layer_size):
            subset = [criteria[i] for i in idx]
            calls += 1
            df = _borda_df(subset, singletons, answers)
            with _silence_stdout():
                agg, _ = aggregator.aggregate(input_df=df)
            estimated = agg["ItemID"].tolist()
            brank = estimated.index(baseline_top1) if baseline_top1 in estimated else len(estimated)
            raw.append({
                "key": combo_key(subset),
                "estimated": estimated,
                "baseline_rank": brank,
                "layer": layer_size,
            })

        raw.sort(key=lambda x: x["baseline_rank"], reverse=True)
        # Store references (not copies) so is_cf/checked_pos set below are visible here too
        layer_logs.append({"layer_size": layer_size, "nodes": raw})

        for pos, item in enumerate(raw):
            is_cf = item["estimated"][0] != baseline_top1
            item.update({"is_cf": is_cf, "checked_pos": pos + 1})
            steps.append(item)
            if is_cf:
                return {"found": True, "key": item["key"],
                        "estimated_ranking": item["estimated"],
                        "calls": calls, "steps": steps, "layer_logs": layer_logs}

    return {"found": False, "key": None, "estimated_ranking": None,
            "calls": calls, "steps": steps, "layer_logs": layer_logs}


# ─── Display helpers ──────────────────────────────────────────────────────────

ABBREV = {"price": "P", "quality": "Q", "speed": "S"}

def abbrev(key):
    if not key:
        return "(baseline)"
    parts = [ABBREV.get(p.strip(), p.strip()) for p in key.split(",")]
    return ",".join(parts)

def fmt_ranking(ranking):
    return "[" + ", ".join(ranking) + "]"

def print_lattice(criteria, rankings_by_combo):
    """Print a Hasse diagram of the 3-node lattice with rankings and CF markers."""
    baseline = rankings_by_combo.get("", [])
    top1 = baseline[0]

    def node(key):
        r = rankings_by_combo.get(key, ["?"])
        mark = "★" if r[0] != top1 else "✓"
        return f"{mark}{fmt_ranking(r)}"

    n = len(criteria)
    # Layer keys in combinations order
    L3 = [combo_key([criteria[i] for i in idx])
          for idx in combinations(range(n), 3)]
    L2 = [combo_key([criteria[i] for i in idx])
          for idx in combinations(range(n), 2)]
    L1 = [combo_key([criteria[i] for i in idx])
          for idx in combinations(range(n), 1)]

    a3 = [abbrev(k) for k in L3]  # ["P,Q,S"]
    a2 = [abbrev(k) for k in L2]  # ["P,Q", "P,S", "Q,S"]
    a1 = [abbrev(k) for k in L1]  # ["P", "Q", "S"]

    n3 = [node(k) for k in L3]
    n2 = [node(k) for k in L2]
    n1 = [node(k) for k in L1]

    print("  LATTICE  (P=price  Q=quality  S=speed │ ✓=same top-1  ★=counterfactual)")
    print()
    print(f"  Layer 3     {a3[0]:<12}  {n3[0]}")
    print(f"  (k=3)      /     |     \\")
    print(f"  Layer 2  {a2[0]:<8}  {a2[1]:<8}  {a2[2]}")
    print(f"  (k=2)    {n2[0]:<12}  {n2[1]:<12}  {n2[2]}")
    print(f"           /  \\      /  \\      /  \\")
    print(f"  Layer 1  {a1[0]:<8}   {a1[1]:<8}   {a1[2]}")
    print(f"  (k=1)    {n1[0]:<12}   {n1[1]:<12}   {n1[2]}")
    print()


def print_bfs_steps(bfs_result, total_nodes=7):
    print(f"  BFS  (ground-truth evaluations, lexicographic order within layers):")
    for s in bfs_result["steps"]:
        mark = "★ FOUND" if s["is_cf"] else "✓"
        print(f"    call {s['call']:>2}  Layer {s['layer']}  {s['key']:<28}  "
              f"{fmt_ranking(s['ranking']):<16}  {mark}")
    if not bfs_result["found"]:
        remaining = total_nodes - bfs_result["calls"]
        print(f"    (exhausted all {bfs_result['calls']} nodes — no counterfactual)")
    print(f"    {'─'*68}")
    if bfs_result["found"]:
        print(f"    found={bfs_result['found']}  key=\"{bfs_result['key']}\"  calls={bfs_result['calls']}")
    else:
        print(f"    found=False  calls={bfs_result['calls']}")
    print()


def print_sra_steps(sra_result, criteria, rankings_by_combo, agg_label):
    baseline_top1 = rankings_by_combo[""][0]
    print(f"  SRA  ({agg_label}, sorts by estimated CF likelihood within layers):")
    singletons = {c: rankings_by_combo[c] for c in criteria if c in rankings_by_combo}
    print(f"    Singletons used as proxy voters:")
    for c in criteria:
        print(f"      {c:<10}  {fmt_ranking(singletons.get(c, ['?']))}")
    print()

    for ll in sra_result["layer_logs"]:
        ls = ll["layer_size"]
        nodes = ll["nodes"]
        print(f"    Layer {ls} — estimates ({len(nodes)} node{'s' if len(nodes)>1 else ''}, "
              f"unsorted → then sorted by baseline top-1 rank desc):")
        for nd in nodes:
            is_cf = nd["estimated"][0] != baseline_top1
            mark = "★ CF" if is_cf else "  "
            print(f"      {nd['key']:<28}  estimated={fmt_ranking(nd['estimated']):<16}  "
                  f"{baseline_top1!r}@rank{nd['baseline_rank']}  {mark}")
        print(f"    Layer {ls} — check order after sort:")
        for pos, nd in enumerate(nodes):
            is_cf = nd["estimated"][0] != baseline_top1
            mark = "★ FOUND" if is_cf else "✓"
            stop = "  ← stop" if is_cf else ""
            print(f"      #{pos+1}  {nd['key']:<28}  estimated[0]={nd['estimated'][0]}  {mark}{stop}")
            if is_cf:
                break
        print()

    print(f"    {'─'*68}")
    if sra_result["found"]:
        print(f"    found={sra_result['found']}  key=\"{sra_result['key']}\"  calls={sra_result['calls']}")
    else:
        print(f"    found=False  calls={sra_result['calls']}")
    print()


def print_comparison(bfs_r, sra_r):
    bfs_found = bfs_r["key"] if bfs_r["found"] else "—"
    sra_found = sra_r["key"] if sra_r["found"] else "—"
    sra_pos_note = ""
    if sra_r["found"] and sra_r["steps"]:
        last = sra_r["steps"][-1]
        sra_pos_note = f"  (CF was #{last['checked_pos']} checked in layer {last['layer']})"

    print(f"  COMPARISON:")
    print(f"    {'Algorithm':<12}  {'Calls':>6}  {'CF found':<30}  Call type")
    print(f"    {'─'*80}")
    print(f"    {'BFS':<12}  {bfs_r['calls']:>6}  {bfs_found:<30}  "
          f"ground-truth (LLM) evaluations, lex order")
    print(f"    {'SRA':<12}  {sra_r['calls']:>6}  {sra_found:<30}  "
          f"cheap proxy estimates, sorted order{sra_pos_note}")
    print(f"    NOTE: SRA never calls the LLM — it returns the ESTIMATED CF.")
    print(f"          BFS calls the LLM at every node; SRA calls it zero times.")
    if bfs_r["key"] and sra_r["key"] and bfs_r["key"] == sra_r["key"]:
        print(f"    → Agree: proxy estimate matched ground truth.")
    elif bfs_r["key"] and sra_r["key"]:
        print(f"    → DISAGREE: SRA returned '{sra_r['key']}' (estimated CF, unverified).")
        print(f"                BFS returned '{bfs_r['key']}' (verified ground truth).")
        print(f"                SRA produced a FALSE POSITIVE — proxy estimate was wrong.")
    print()


def header(title):
    line = "═" * 62
    print(f"\n  {line}")
    print(f"  {title}")
    print(f"  {line}")


# ─── Test cases ───────────────────────────────────────────────────────────────

CRITERIA = ["price", "quality", "speed"]

# All test cases use 4 answers so Borda scores have enough range to avoid ties in
# CombSUM estimates.  With 3 answers the full-set estimate always ties when 2 of 3
# singletons favour B, making pyflagr's tie-break non-deterministic.
#
# Key invariant: "" == "price, quality, speed" (same LLM call).  The full-set node
# can never be a counterfactual of itself; the search only finds CFs in strict subsets.

TEST_CASES = [
    {
        "name": 'TC1  —  CF at Layer 2, first BFS node ("price, quality")',
        "desc": ("The LLM's first call on the full set gives [A,B,C,D] → A is the original answer.\n"
                 "  Dropping 'speed' (→ price,quality) immediately flips to B.\n"
                 "  BFS finds it on call 2 (after 1 wasted full-set check).\n"
                 "  SRA's proxy agrees and checks it first within layer 2."),
        "rankings_by_combo": {
            "":                      ["A", "B", "C", "D"],  # full-set result = original answer A
            "price, quality, speed": ["A", "B", "C", "D"],  # same call as "" — never a CF
            "price, quality":        ["B", "A", "C", "D"],  # ★ CF  (first node BFS checks in L2)
            "price, speed":          ["A", "B", "C", "D"],
            "quality, speed":        ["A", "B", "C", "D"],
            "price":                 ["B", "A", "C", "D"],
            "quality":               ["B", "A", "C", "D"],
            "speed":                 ["A", "D", "C", "B"],
        },
        # Singletons: price=[B,A,C,D] B=3,A=2  quality=[B,A,C,D] B=3,A=2  speed=[A,D,C,B] A=3,B=0
        # CombSUM(P,Q,S): A=2+2+3=7, B=3+3+0=6 → A wins clearly → no CF est at L3  ✓
        # CombSUM(P,Q):   A=2+2=4,   B=3+3=6   → B wins clearly → CF est  ✓
        # CombSUM(P,S):   A=2+3=5,   B=3+0=3   → A wins → no CF est
        # CombSUM(Q,S):   A=2+3=5,   B=3+0=3   → A wins → no CF est
        "expected_bfs": {"found": True,  "key": "price, quality"},
        "expected_sra": {"found": True,  "key": "price, quality"},
    },
    {
        "name": 'TC2  —  CF at Layer 2, last BFS node ("quality, speed")',
        "desc": ("Dropping 'price' (→ quality,speed) flips to B, but BFS reaches it last in L2.\n"
                 "  SRA's proxy predicts this flip and checks it first — same call count\n"
                 "  but avoids two fruitless ground-truth lookups."),
        "rankings_by_combo": {
            "":                      ["A", "B", "C", "D"],  # full-set = original answer A
            "price, quality, speed": ["A", "B", "C", "D"],  # same as "" — never a CF
            "price, quality":        ["A", "B", "C", "D"],
            "price, speed":          ["A", "B", "C", "D"],
            "quality, speed":        ["B", "A", "C", "D"],  # ★ CF  (last node BFS checks in L2)
            "price":                 ["A", "C", "D", "B"],
            "quality":               ["B", "A", "C", "D"],
            "speed":                 ["B", "A", "C", "D"],
        },
        # Singletons: price=[A,C,D,B] A=3,B=0  quality=[B,A,C,D] B=3,A=2  speed=[B,A,C,D] B=3,A=2
        # CombSUM(P,Q,S): A=3+2+2=7, B=0+3+3=6 → A wins clearly → no CF est at L3  ✓
        # CombSUM(P,Q):   A=3+2=5,   B=0+3=3   → A wins → no CF est
        # CombSUM(P,S):   A=3+2=5,   B=0+3=3   → A wins → no CF est
        # CombSUM(Q,S):   A=2+2=4,   B=3+3=6   → B wins → CF est ✓  (A@rank1 = highest in layer)
        "expected_bfs": {"found": True,  "key": "quality, speed"},
        "expected_sra": {"found": True,  "key": "quality, speed"},
    },
    {
        "name": 'TC3  —  CF at Layer 1, last BFS node ("speed" singleton)',
        "desc": ("Only the speed singleton flips the answer. BFS exhausts layers 3 and 2\n"
                 "  before reaching it last in layer 1.\n"
                 "  SRA correctly ranks speed as most-likely CF and checks it first in L1."),
        "rankings_by_combo": {
            "":                      ["A", "B", "C", "D"],  # full-set = original answer A
            "price, quality, speed": ["A", "B", "C", "D"],  # same as "" — never a CF
            "price, quality":        ["A", "B", "C", "D"],
            "price, speed":          ["A", "B", "C", "D"],
            "quality, speed":        ["A", "B", "C", "D"],
            "price":                 ["A", "C", "B", "D"],  # A=3,B=1 (not 2) — avoids ties in pairs
            "quality":               ["A", "C", "B", "D"],
            "speed":                 ["B", "A", "C", "D"],  # ★ CF
        },
        # Singletons: price=[A,C,B,D] A=3,B=1  quality=[A,C,B,D] A=3,B=1  speed=[B,A,C,D] B=3,A=2
        # CombSUM(P,Q,S): A=3+3+2=8, B=1+1+3=5 → A clearly wins → no CF est at L3  ✓
        # CombSUM(P,Q):   A=3+3=6,   B=1+1=2   → A wins → no CF est  ✓
        # CombSUM(P,S):   A=3+2=5,   B=1+3=4   → A wins → no CF est  ✓
        # CombSUM(Q,S):   A=3+2=5,   B=1+3=4   → A wins → no CF est  ✓
        # CombSUM(P):     A=3 wins,  CombSUM(Q): A=3 wins,  CombSUM(S): B=3 wins → CF (A@rank1)
        "expected_bfs": {"found": True,  "key": "speed"},
        "expected_sra": {"found": True,  "key": "speed"},
    },
    {
        "name": "TC4  —  No Counterfactual",
        "desc": ("Every subset agrees with the original answer A.\n"
                 "  Both algorithms exhaust all 7 nodes and return found=False."),
        "rankings_by_combo": {
            "":                      ["A", "B", "C", "D"],
            "price, quality, speed": ["A", "B", "C", "D"],
            "price, quality":        ["A", "B", "C", "D"],
            "price, speed":          ["A", "B", "C", "D"],
            "quality, speed":        ["A", "B", "C", "D"],
            "price":                 ["A", "B", "C", "D"],
            "quality":               ["A", "B", "C", "D"],
            "speed":                 ["A", "B", "C", "D"],
        },
        "expected_bfs": {"found": False, "key": None},
        "expected_sra": {"found": False, "key": None},
    },
    {
        "name": 'TC5  —  SRA false positive (proxy disagrees with ground truth)',
        "desc": ("SRA's CombSUM estimate predicts 'price, quality' flips to B,\n"
                 "  but the ground-truth LLM call still returns A on top there.\n"
                 "  SRA returns the wrong node; BFS finds the real CF at 'quality, speed'."),
        "rankings_by_combo": {
            "":                      ["A", "B", "C", "D"],  # full-set = original answer A
            "price, quality, speed": ["A", "B", "C", "D"],  # same as "" — never a CF
            "price, quality":        ["A", "B", "C", "D"],  # SRA estimates CF here — WRONG
            "price, speed":          ["A", "B", "C", "D"],
            "quality, speed":        ["B", "A", "C", "D"],  # ★ actual CF — SRA misses it
            "price":                 ["B", "A", "C", "D"],
            "quality":               ["B", "A", "C", "D"],
            "speed":                 ["A", "D", "C", "B"],
        },
        # Same singletons as TC1: price=[B,A,C,D] B=3,A=2  quality=[B,A,C,D] B=3,A=2
        #                         speed=[A,D,C,B]  A=3,B=0
        # CombSUM(P,Q):   B=6, A=4  → SRA estimates B on top for "price, quality"
        #   but ground truth["price, quality"] = [A,B,C,D] → A on top → FALSE POSITIVE
        # CombSUM(Q,S):   A=2+3=5,  B=3+0=3  → SRA estimates A on top → misses real CF
        # BFS evaluates ground truth: skips full set + "price,quality" (A), "price,speed" (A),
        #   then finds "quality, speed" (B) on call 4.
        "expected_bfs": {"found": True,  "key": "quality, speed"},
        "expected_sra": {"found": True,  "key": "price, quality"},  # wrong node — false positive
    },
]


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_test(tc):
    criteria = CRITERIA
    rbc = tc["rankings_by_combo"]
    baseline = rbc[""]
    total_nodes = sum(1 for k in rbc if k != "")

    header(tc["name"])
    print(f"\n  {tc['desc']}")
    print(f"\n  Criteria : {', '.join(criteria)}")
    print(f"  Baseline : {fmt_ranking(baseline)}  (top-1 = {baseline[0]})")
    print()

    print_lattice(criteria, rbc)

    bfs_r = bfs_tracked(criteria, rbc)
    sra_r = sra_tracked(criteria, rbc)

    # Verify bfs_tracked matches the reference implementation
    ref = _bfs_reference(criteria, rbc)
    assert ref["found"] == bfs_r["found"], "bfs_tracked diverges from search/bfs.py"
    assert ref["key"]   == bfs_r["key"],   "bfs_tracked diverges from search/bfs.py"

    # Check expectations
    exp_bfs = tc["expected_bfs"]
    exp_sra = tc["expected_sra"]
    assert bfs_r["found"] == exp_bfs["found"], f"BFS found mismatch: {bfs_r['found']} != {exp_bfs['found']}"
    assert bfs_r["key"]   == exp_bfs["key"],   f"BFS key mismatch: {bfs_r['key']!r} != {exp_bfs['key']!r}"
    assert sra_r["found"] == exp_sra["found"], f"SRA found mismatch: {sra_r['found']} != {exp_sra['found']}"
    assert sra_r["key"]   == exp_sra["key"],   f"SRA key mismatch: {sra_r['key']!r} != {exp_sra['key']!r}"

    print_bfs_steps(bfs_r, total_nodes)
    print_sra_steps(sra_r, criteria, rbc, _agg_label)
    print_comparison(bfs_r, sra_r)
    print(f"  [PASS]  All assertions satisfied.\n")


if __name__ == "__main__":
    print(f"\n  Lattice Search — Test Suite")
    print(f"  Aggregator: {_agg_label}")
    print(f"  Criteria  : {', '.join(CRITERIA)}  →  2^3 = 8 nodes")

    for tc in TEST_CASES:
        run_test(tc)

    print("  All test cases passed.\n")
