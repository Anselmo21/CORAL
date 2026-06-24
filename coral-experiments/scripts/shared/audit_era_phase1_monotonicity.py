"""Verify ERA Phase-1 lookup monotonicity across the four main datasets.

Run: conda run -n coral python3 -m scripts.shared.audit_era_phase1_monotonicity
"""
from __future__ import annotations

import json
from typing import Any

from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key
from search.rank import _era_phase1_down, era_search
from scripts.msmarco.run_subsample import _load_all_qids as msmarco_qids, _load_lattices as msmarco_lattices
from scripts.sonnet.run_subsample import _load_all_qids as sonnet_qids, _load_lattices as sonnet_lattices
from scripts.pragma.msmarco.run_subsample import (
    _load_all_qids as msmarco_pragma_qids,
    _load_lattices as msmarco_pragma_lattices,
)
from scripts.pragma.sonnet.run_subsample import (
    _load_all_qids as sonnet_pragma_qids,
    _load_pragma_lattices as sonnet_pragma_lattices,
)

BUDGETS: list[int] = [4, 8, 16, 32, 64, 128, 256]


def phase1_lookups(
    criteria: list[str],
    rankings_by_combo: dict[str, list[str]],
    start_node: list[str],
    budget: int,
) -> list[str]:
    """Return the ordered Phase-1 lookup keys ERA charges against ``budget``.

    Calls the *exact* production helper (``search.rank._era_phase1_down``) that
    ``era_search`` uses, with ``era_search``'s downward defaults (combsum
    aggregation, softmax entropy, max-jump cutoff). This guarantees the audit
    measures the real ERA implementation rather than a re-implementation.
    """
    start_key = combo_key(start_node)
    baseline_ranking = rankings_by_combo.get(start_key, [])
    if not baseline_ranking:
        return []

    answers = list(baseline_ranking)
    known: dict[str, list[str]] = {}
    estimated_rankings: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}

    # Mirror era_search's pre-Phase-1 setup: singletons are pre-loaded as known.
    for c in criteria:
        if c in rankings_by_combo:
            known[c] = rankings_by_combo[c]

    _, looked_up = _era_phase1_down(
        start_node,
        rankings_by_combo,
        answers,
        known,
        estimated_rankings,
        entropies,
        budget,
        "combsum",
        softmax_entropy,
        max_jump_cutoff,
    )
    return looked_up


def check_case(
    criteria: list[str],
    rbc: dict[str, list[str]],
    start_node: list[str],
    budgets: list[int],
) -> dict[str, Any]:
    selections = {b: phase1_lookups(criteria, rbc, start_node, b) for b in budgets}
    sets = {b: set(selections[b]) for b in budgets}
    violations: list[dict[str, Any]] = []
    prefix_violations: list[dict[str, Any]] = []

    for i in range(len(budgets) - 1):
        b_small, b_large = budgets[i], budgets[i + 1]
        if not sets[b_small].issubset(sets[b_large]):
            violations.append({
                "b_small": b_small,
                "b_large": b_large,
                "only_in_small": sorted(sets[b_small] - sets[b_large]),
                "counts": (len(selections[b_small]), len(selections[b_large])),
            })

        small_list, large_list = selections[b_small], selections[b_large]
        if len(small_list) > len(large_list) or large_list[:len(small_list)] != small_list:
            prefix_violations.append({
                "b_small": b_small,
                "b_large": b_large,
                "small_len": len(small_list),
                "large_len": len(large_list),
            })

    # All-pairs set monotonicity (stronger than consecutive-only).
    all_pairs_ok = True
    for i, b_small in enumerate(budgets):
        for b_large in budgets[i + 1:]:
            if not sets[b_small].issubset(sets[b_large]):
                all_pairs_ok = False

    return {
        "monotonic_set": len(violations) == 0,
        "monotonic_prefix": len(prefix_violations) == 0,
        "all_pairs_set_monotonic": all_pairs_ok,
        "violations": violations,
        "prefix_violations": prefix_violations,
        "counts": {b: len(selections[b]) for b in budgets},
    }


def audit_dataset(
    name: str,
    cases_path: str,
    lattices: dict[str, tuple[list[str], dict[str, list[str]]]],
) -> dict[str, Any]:
    with open(cases_path) as f:
        cases = json.load(f)

    total = 0
    skipped = 0
    set_mono = 0
    prefix_mono = 0
    all_pairs_mono = 0
    impl_mismatches = 0
    violation_cases: list[dict[str, Any]] = []
    prefix_cases: list[dict[str, Any]] = []
    count_by_budget: dict[int, list[int]] = {b: [] for b in BUDGETS}

    for case_id, case in enumerate(cases):
        qid = str(case["qid"])
        if qid not in lattices:
            skipped += 1
            continue

        criteria, rbc = lattices[qid]
        start_node = case["start_node"]
        total += 1
        result = check_case(criteria, rbc, start_node, BUDGETS)

        # Self-validation: the number of Phase-1 lookups we measured must equal
        # what the real era_search actually spends (budget_used). This fails
        # loudly if era_search's downward defaults ever drift from the audit.
        for b in BUDGETS:
            era_res = era_search(criteria, rbc, budget=b, start_node=start_node, direction="down")
            if era_res["budget_used"] != result["counts"][b]:
                impl_mismatches += 1

        for b, count in result["counts"].items():
            count_by_budget[b].append(count)

        if result["all_pairs_set_monotonic"]:
            all_pairs_mono += 1

        if result["monotonic_set"]:
            set_mono += 1
        else:
            violation_cases.append({
                "case_id": case_id,
                "qid": qid,
                "size": case.get("size"),
                **result,
            })

        if result["monotonic_prefix"]:
            prefix_mono += 1
        else:
            prefix_cases.append({
                "case_id": case_id,
                "qid": qid,
                "size": case.get("size"),
                **result,
            })

    avg_counts = {
        b: sum(count_by_budget[b]) / len(count_by_budget[b]) if count_by_budget[b] else 0.0
        for b in BUDGETS
    }

    return {
        "name": name,
        "cases_path": cases_path,
        "total": total,
        "skipped": skipped,
        "set_monotonic": set_mono,
        "prefix_monotonic": prefix_mono,
        "all_pairs_monotonic": all_pairs_mono,
        "impl_mismatches": impl_mismatches,
        "violation_cases": violation_cases,
        "prefix_cases": prefix_cases,
        "avg_lookup_counts": avg_counts,
    }


def main() -> None:
    datasets = [
        ("MSMARCO criteria", "data/msmarco_subsample/cases.json", msmarco_lattices(msmarco_qids())),
        ("Sonnet criteria", "data/sonnet_subsample/cases.json", sonnet_lattices(sonnet_qids())),
        ("MSMARCO pragmas", "data/msmarco_pragma/cases.json", msmarco_pragma_lattices(msmarco_pragma_qids())),
        ("Sonnet pragmas", "data/sonnet_pragma/cases.json", sonnet_pragma_lattices(sonnet_pragma_qids())),
    ]

    print("ERA Phase-1 monotonicity audit")
    print(f"Budget grid: {BUDGETS}")
    print("=" * 72)

    summaries: list[dict[str, Any]] = []
    for name, path, lattices in datasets:
        result = audit_dataset(name, path, lattices)
        summaries.append(result)

        print(f"\n{result['name']}")
        print(f"  cases checked: {result['total']} (skipped: {result['skipped']})")
        print(f"  set-monotonic (consecutive): {result['set_monotonic']}/{result['total']}")
        print(f"  all-pairs set-monotonic:     {result['all_pairs_monotonic']}/{result['total']}")
        print(f"  prefix-monotonic:            {result['prefix_monotonic']}/{result['total']}")
        print(f"  era_search impl mismatches:  {result['impl_mismatches']} (must be 0)")
        avg = {b: round(result["avg_lookup_counts"][b], 1) for b in BUDGETS}
        print(f"  avg lookups: {avg}")

        if result["violation_cases"]:
            print("  SET VIOLATIONS:")
            for v in result["violation_cases"][:5]:
                print(f"    case_id={v['case_id']} qid={v['qid']} size={v['size']}")
                for viol in v["violations"]:
                    n_missing = len(viol["only_in_small"])
                    print(
                        f"      b{viol['b_small']}->{viol['b_large']} "
                        f"counts={viol['counts']} missing_keys={n_missing}"
                    )

        if result["prefix_cases"]:
            print("  PREFIX VIOLATIONS:")
            for v in result["prefix_cases"][:5]:
                print(
                    f"    case_id={v['case_id']} qid={v['qid']} "
                    f"violations={v['prefix_violations']}"
                )

    print("\n" + "=" * 72)
    all_ok = True
    for s in summaries:
        ok = (
            s["set_monotonic"] == s["total"]
            and s["prefix_monotonic"] == s["total"]
            and s["all_pairs_monotonic"] == s["total"]
            and s["impl_mismatches"] == 0
        )
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {s['name']}: "
            f"consecutive {s['set_monotonic']}/{s['total']}, "
            f"all-pairs {s['all_pairs_monotonic']}/{s['total']}, "
            f"prefix {s['prefix_monotonic']}/{s['total']}, "
            f"impl_mismatch {s['impl_mismatches']}"
        )

    print("OVERALL:", "ALL PASS" if all_ok else "FAILURES FOUND")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
