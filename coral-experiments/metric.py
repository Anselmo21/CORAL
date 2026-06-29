"""
metric.py — Accuracy metric for lattice search methods.

Measures how often a search method finds the MINIMAL counterfactual — defined as
a CF at the same layer size as BFS (ground truth).

Scoring rules per lattice:
  - Correct  : proxy returns a key at the same layer size as BFS, AND that key is
                a genuine CF (verified against rankings_by_combo).
  - Wrong    : proxy returned a CF at a different layer size (too shallow or too deep).
  - FP       : proxy returned a key that is not actually a CF in rankings_by_combo.
  - Not found: proxy returned found=False.

Usage:
    from metric import evaluate, report
    import functools
    from search.rank import sra_search

    results = evaluate(lattices, functools.partial(sra_search, rank_displacement=1))
    report(results)

Each entry in `lattices` must have:
    "criteria"          : list[str]
    "rankings_by_combo" : dict[str, list]
    "start_node"        : list[str]  (optional; defaults to full criteria set)
"""

from collections import Counter, defaultdict

from search.bfs import bfs_search
from search.lattice import jaccard_keys


def _layer_size(key):
    """Criteria count encoded in a rankings_by_combo key ('a, b, c' → 3)."""
    if key is None:
        return 0
    return len(key.split(", ")) if key else 0


def _is_real_cf(key, rankings_by_combo, baseline_top1):
    ranking = rankings_by_combo.get(key, [])
    return bool(ranking) and ranking[0] != baseline_top1


def evaluate(lattices, search_fn):
    """
    Run search_fn on each lattice and score against BFS ground truth.

    Args:
        lattices  : iterable of dicts (see module docstring for required keys).
        search_fn : callable(criteria, rankings_by_combo, *, start_node) → result dict.
                    Any extra parameters should be bound with functools.partial before
                    passing in.  Must return at minimum {"found": bool, "key": str|None}.

    Returns:
        {
            "score"  : float   — correct / total,
            "correct": int,
            "total"  : int,
            "results": list[dict]  — one entry per lattice (see below),
        }

    Per-lattice result keys:
        index, start_node, bfs_key, bfs_layer,
        proxy_key, proxy_layer, proxy_verified, correct, outcome
    """
    correct = 0
    detail = []

    for i, lattice in enumerate(lattices):
        criteria   = lattice["criteria"]
        rbc        = lattice["rankings_by_combo"]
        start_node = lattice.get("start_node", list(criteria))
        start_key  = ", ".join(sorted(start_node))
        baseline_top1 = (rbc.get(start_key) or [""])[0]

        # ── Ground truth ──────────────────────────────────────────────────────
        bfs_result = bfs_search(criteria, rbc, start_node=start_node)
        bfs_key    = bfs_result["key"]          # caller guarantees found=True
        bfs_sz     = _layer_size(bfs_key)

        # ── Proxy method ──────────────────────────────────────────────────────
        proxy_result   = search_fn(criteria, rbc, start_node=start_node)
        proxy_key      = proxy_result.get("key")
        proxy_sz       = _layer_size(proxy_key) if proxy_key is not None else 0
        proxy_verified = _is_real_cf(proxy_key, rbc, baseline_top1) if proxy_key is not None else False

        # ── Scoring ───────────────────────────────────────────────────────────
        if not proxy_result.get("found") or proxy_key is None:
            outcome    = "not_found"
            is_correct = False
        elif not proxy_verified:
            outcome    = "false_positive"
            is_correct = False
        elif proxy_sz != bfs_sz:
            outcome    = "wrong_layer"
            is_correct = False
        else:
            outcome    = "correct"
            is_correct = True

        if is_correct:
            correct += 1

        gap: int | None = None
        if outcome == "correct":
            gap = 0
        elif outcome == "wrong_layer":
            gap = abs(bfs_sz - proxy_sz)

        detail.append({
            "index"          : i,
            "start_node"     : start_node,
            "bfs_key"        : bfs_key,
            "bfs_layer"      : bfs_sz,
            "proxy_key"      : proxy_key,
            "proxy_layer"    : proxy_sz,
            "proxy_verified" : proxy_verified,
            "correct"        : is_correct,
            "outcome"        : outcome,
            "gap"            : gap,
            "jaccard"        : jaccard_keys(bfs_key, proxy_key),
        })

    total = len(detail)
    return {
        "score"  : correct / total if total else 0.0,
        "correct": correct,
        "total"  : total,
        "results": detail,
    }


def report(eval_result, title="Search method"):
    """Print a human-readable summary of evaluate() output."""
    correct = eval_result["correct"]
    total   = eval_result["total"]
    score   = eval_result["score"]
    results = eval_result["results"]

    # ── Outcome breakdown ─────────────────────────────────────────────────────
    outcomes = Counter(r["outcome"] for r in results)

    print(f"\n{'═' * 52}")
    print(f"  {title}")
    print(f"{'═' * 52}")
    print(f"  Accuracy : {correct}/{total}  ({score:.1%})")
    print(f"\n  Outcome breakdown:")
    for outcome in ("correct", "wrong_layer", "false_positive", "not_found"):
        n = outcomes.get(outcome, 0)
        bar = "█" * n + "░" * (total - n)
        print(f"    {outcome:<16}  {n:>3}  {bar[:30]}")

    # ── Per-BFS-layer breakdown ───────────────────────────────────────────────
    by_layer = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_layer[r["bfs_layer"]]["total"] += 1
        if r["correct"]:
            by_layer[r["bfs_layer"]]["correct"] += 1

    if len(by_layer) > 1:
        print(f"\n  Accuracy by BFS layer (minimal CF size):")
        for sz in sorted(by_layer, reverse=True):
            c = by_layer[sz]["correct"]
            t = by_layer[sz]["total"]
            print(f"    size {sz}  {c}/{t}  ({c/t:.0%})")

    # ── Jaccard analysis (proxy CF criteria vs BFS CF criteria) ─────────────
    verified = [r for r in results if r["proxy_verified"]]
    jaccs_v = [r["jaccard"] for r in verified if r.get("jaccard") is not None]
    jaccs_all = [r["jaccard"] for r in results if r.get("jaccard") is not None]
    if jaccs_v or jaccs_all:
        print(f"\n  Jaccard(proxy CF, BFS CF):")
        if jaccs_v:
            avg = sum(jaccs_v) / len(jaccs_v)
            print(f"    verified mean: {avg:.3f}   min: {min(jaccs_v):.3f}   max: {max(jaccs_v):.3f}   n={len(jaccs_v)}")
        if jaccs_all:
            avg = sum(jaccs_all) / len(jaccs_all)
            print(f"    overall  mean: {avg:.3f}   min: {min(jaccs_all):.3f}   max: {max(jaccs_all):.3f}   n={len(jaccs_all)}")

    # ── Layer gap analysis ─────────────────────────────────────────────────
    wrong_layer = [r for r in results if r["outcome"] == "wrong_layer"]
    if wrong_layer:
        gaps = [r["gap"] for r in wrong_layer]
        avg_gap = sum(gaps) / len(gaps)
        print(f"\n  Layer gap ({len(wrong_layer)} wrong-layer cases):")
        print(f"    mean: {avg_gap:.2f}   min: {min(gaps)}   max: {max(gaps)}")
        deeper   = sum(1 for r in wrong_layer if r["proxy_layer"] < r["bfs_layer"])
        shallower = sum(1 for r in wrong_layer if r["proxy_layer"] > r["bfs_layer"])
        print(f"    proxy deeper (skipped true CF): {deeper}")
        print(f"    proxy shallower:                {shallower}")
        gap_counts = Counter(gaps)
        print(f"    distribution:")
        for g in sorted(gap_counts):
            print(f"      gap={g}: {'█' * gap_counts[g]} ({gap_counts[g]})")
    else:
        print(f"\n  Layer gap: no wrong-layer cases")

    # ── Wrong / false-positive details ───────────────────────────────────────
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            tag = f"[{r['outcome']}]"
            bfs = f"BFS→{r['bfs_key']} (size {r['bfs_layer']})"
            if r["proxy_key"]:
                prx = f"proxy→{r['proxy_key']} (size {r['proxy_layer']})"
            else:
                prx = "proxy→(none)"
            print(f"    #{r['index']:>3}  {tag:<18}  {bfs}  |  {prx}")
    print()
