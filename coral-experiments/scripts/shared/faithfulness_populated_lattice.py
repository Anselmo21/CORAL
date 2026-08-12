"""Faithfulness of aggregated sub-lattices under SRA vs ERA × {Borda, MCT}.

For each size-10 subsampled case, populate every non-singleton, non-full subset
of the case's start_node with an estimated ranking under 4 methods:

  - SRA-*: at node S, voters = |S| singleton rankings for c ∈ S. Aggregate
           via pyflagr <method>.
  - ERA-*: populate bottom-up. Node S with |S| >= 2 uses its |S| immediate
           children (size |S|-1) as voters, each represented as a score vector
           (Borda-normalized ranking for the singletons; max-normalized
           aggregated scores for internal nodes — matching `_get_child_voters`
           + `_store_estimate(propagate_scores=True)` in search.rank).
           At each node the voters (score vectors) are collapsed to rankings
           via score-sort and fed to pyflagr <method>.

Compare each estimated ranking to the ground-truth LLM ranking rbc[key].

Metrics per node: top-1 match, Kendall tau vs true.
Aggregation: mean across all nodes across all cases per dataset.

Usage:
    conda run -n coral python3 -m scripts.shared.faithfulness_populated_lattice
    conda run -n coral python3 -m scripts.shared.faithfulness_populated_lattice --datasets msmarco_criteria
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from search.flagr import flagr_aggregate, make_era_aggregator
from search.heuristics import max_jump_cutoff, softmax_entropy
from search.lattice import combo_key, parse_combo_key
from search.rank import _era_phase1_down, _normalize_max, _ranking_to_score_vector

DATASETS = {
    "msmarco_criteria": "data/llama31-8b/msmarco-criteria-lattices.json",
    "msmarco_pragma":   "data/llama31-8b/msmarco-pragmas-lattices.json",
    "curated_criteria":  "data/llama31-8b/curated-criteria-lattices.json",
    "curated_pragma":    "data/llama31-8b/curated-pragmas-lattices.json",
}

METHODS = ["borda", "mct"]
START_SIZE = 10
ERA_BUDGET = 4  # used by the era_*_b4 variants (real era_search Phase 1)
OUT_PATH = "logged_results/faithfulness_populated_lattice.json"


def _load_lattices(path: str, qids: set[str]) -> dict[str, tuple[list[str], dict]]:
    raw = json.load(open(path))
    out = {}
    for qid, q in raw.items():
        if qid == "_meta" or qid not in qids or q.get("halted"):
            continue
        crit = list(q.get("criteria") or [])
        sorted_crit = sorted(crit)
        rbc = {combo_key(parse_combo_key(k, sorted_crit)): v
               for k, v in (q.get("raw_answers") or {}).items()}
        if crit and combo_key(crit) in rbc:
            out[qid] = (crit, rbc)
    return out


def _kendall(pred: list[str], true: list[str]) -> float:
    common = [a for a in true if a in pred]
    if len(common) < 2:
        return float("nan")
    true_rank = {a: i for i, a in enumerate(true)}
    pred_rank = {a: i for i, a in enumerate(pred)}
    tau, _ = kendalltau([true_rank[a] for a in common],
                        [pred_rank[a] for a in common])
    return float(tau) if tau is not None else float("nan")


def _pad(ranking: list[str], answers: list[str]) -> list[str]:
    if not ranking:
        return []
    missing = [a for a in answers if a not in ranking]
    return ranking + missing


def _sra_populate(start: list[str], singleton_rankings: dict[str, list[str]],
                  answers: list[str], method: str,
                  ) -> dict[str, tuple[list[str], dict[str, float]]]:
    """SRA populate: at every node, voters are |S| singleton rankings."""
    out: dict[str, tuple[list[str], dict[str, float]]] = {}
    for size in range(2, len(start)):
        for subset in combinations(start, size):
            voters = {c: singleton_rankings[c] for c in subset
                      if c in singleton_rankings}
            if not voters:
                continue
            ranking, scores = flagr_aggregate(voters, answers, method=method)
            out[combo_key(subset)] = (_pad(ranking, answers), scores)
    return out


def _era_populate(start: list[str], singleton_rankings: dict[str, list[str]],
                  answers: list[str], method: str,
                  ) -> dict[str, tuple[list[str], dict[str, float]]]:
    """ERA populate (bottom-up, propagate_scores=True semantics).

    Node S with |S| >= 2 aggregates its |S| immediate children. Each child
    contributes a score vector: singletons use Borda from their true ranking;
    internal nodes reuse the max-normalized aggregated scores stored below.
    """
    # child_scores[key] = score vector at that node (used as a voter for parents).
    child_scores: dict[str, dict[str, float]] = {}
    for c in start:
        if c in singleton_rankings:
            child_scores[combo_key([c])] = _normalize_max(
                _ranking_to_score_vector(singleton_rankings[c], answers))

    out: dict[str, tuple[list[str], dict[str, float]]] = {}
    for size in range(2, len(start)):
        for subset in combinations(start, size):
            voter_scores: dict[str, dict[str, float]] = {}
            for i in range(len(subset)):
                child = subset[:i] + subset[i + 1:]
                ck = combo_key(child)
                if ck in child_scores:
                    voter_scores[ck] = child_scores[ck]
            if not voter_scores:
                continue
            voter_rankings = {
                k: sorted(answers, key=lambda a: (-sv.get(a, 0.0), a))
                for k, sv in voter_scores.items()
            }
            ranking, scores = flagr_aggregate(voter_rankings, answers, method=method)
            if not ranking:
                continue
            padded = _pad(ranking, answers)
            out[combo_key(subset)] = (padded, scores)
            child_scores[combo_key(subset)] = _normalize_max(scores)
    return out


def _era_phase1_populate(start: list[str], singleton_rankings: dict[str, list[str]],
                          answers: list[str], method: str,
                          rbc: dict[str, list[str]], budget: int,
                          ) -> dict[str, tuple[list[str], dict[str, float]]]:
    """Real ERA Phase-1 with a budget cap.

    Returns only nodes that ended up as *estimates* (i.e. ERA did NOT spend
    budget on them). Nodes promoted to `known` by budget spend are excluded so
    the reported faithfulness reflects extrapolation quality, not ground truth.
    """
    known = {c: singleton_rankings[c] for c in start if c in singleton_rankings}
    if "" in rbc:
        known[""] = rbc[""]
    estimated_scores: dict[str, dict[str, float]] = {}
    entropies: dict[str, float] = {}

    _era_phase1_down(
        start_node=start,
        rankings_by_combo=rbc,
        answers=answers,
        known=known,
        estimated_scores=estimated_scores,
        entropies=entropies,
        budget_remaining=budget,
        aggregator_fn=make_era_aggregator(method),
        entropy_fn=softmax_entropy,
        cutoff_fn=max_jump_cutoff,
        propagate_scores=True,
    )

    out: dict[str, tuple[list[str], dict[str, float]]] = {}
    for key, scores in estimated_scores.items():
        if key in known:
            continue  # promoted → ground truth, not an estimate
        ranking = sorted(scores, key=lambda a: (-scores[a], a))
        out[key] = (ranking, scores)
    return out


def _era_borda_b4(start, singletons, answers, method):
    return _era_phase1_populate(start, singletons, answers, "borda", _CURRENT_RBC[0], ERA_BUDGET)


def _era_mct_b4(start, singletons, answers, method):
    return _era_phase1_populate(start, singletons, answers, "mct", _CURRENT_RBC[0], ERA_BUDGET)


# ERA Phase 1 needs the full rbc (for the budget-spend LLM lookups). Threaded
# in via a per-case slot instead of changing every populate_fn signature.
_CURRENT_RBC: list[dict[str, list[str]]] = [None]  # type: ignore


VARIANTS = [
    ("sra_borda",    "borda", _sra_populate),
    ("sra_mct",      "mct",   _sra_populate),
    ("era_borda",    "borda", _era_populate),
    ("era_mct",      "mct",   _era_populate),
    ("era_borda_b4", "borda", _era_borda_b4),
    ("era_mct_b4",   "mct",   _era_mct_b4),
]


def _run_dataset(name: str, path: str) -> dict:
    cases = json.loads(Path(f"logged_results/{name}/cases.json").read_text())
    lat = _load_lattices(path, {str(c["qid"]) for c in cases})

    stats = {v: {
        "n": 0, "top1_hits": 0, "kendall_sum": 0.0, "kendall_n": 0,
        # majority = true top-1 == baseline; cf = true top-1 != baseline
        "maj_n": 0, "maj_top1_hits": 0, "maj_false_flip": 0,
        "cf_n": 0, "cf_top1_hits": 0, "cf_detected": 0,
    } for v, _, _ in VARIANTS}

    t0 = time.time()
    total_cases = 0
    for i, case in enumerate(cases):
        qid = str(case["qid"])
        if qid not in lat:
            continue
        if len(case["start_node"]) != START_SIZE:
            continue
        total_cases += 1
        _crit, rbc = lat[qid]
        start = case["start_node"]
        answers = list(rbc[combo_key(start)])
        baseline_top1 = answers[0]
        singleton_rankings = {c: rbc[c] for c in start if c in rbc}
        if not singleton_rankings:
            continue

        _CURRENT_RBC[0] = rbc
        for vname, method, populate_fn in VARIANTS:
            populated = populate_fn(start, singleton_rankings, answers, method)
            st = stats[vname]
            for key, (pred, _scores) in populated.items():
                true_rank = rbc.get(key)
                if not true_rank or not pred:
                    continue
                st["n"] += 1
                pred_top1 = pred[0]
                true_top1 = true_rank[0]
                hit = pred_top1 == true_top1
                if hit:
                    st["top1_hits"] += 1
                if true_top1 == baseline_top1:
                    st["maj_n"] += 1
                    if hit:
                        st["maj_top1_hits"] += 1
                    if pred_top1 != baseline_top1:
                        st["maj_false_flip"] += 1
                else:
                    st["cf_n"] += 1
                    if hit:
                        st["cf_top1_hits"] += 1
                    if pred_top1 != baseline_top1:
                        st["cf_detected"] += 1
                tau = _kendall(pred, true_rank)
                if not np.isnan(tau):
                    st["kendall_sum"] += tau
                    st["kendall_n"] += 1

        if (i + 1) % 5 == 0:
            print(f"  {name}: {i + 1}/{len(cases)} cases  ({time.time() - t0:.0f}s)",
                  flush=True)

    def _pct(num, den):
        return (100.0 * num / den) if den else float("nan")

    summary = {}
    for vname, _, _ in VARIANTS:
        st = stats[vname]
        n = st["n"]
        summary[vname] = {
            "nodes": n,
            "top1_acc": _pct(st["top1_hits"], n),
            "kendall_tau_mean": (st["kendall_sum"] / st["kendall_n"])
                                if st["kendall_n"] else float("nan"),
            "kendall_n": st["kendall_n"],
            "majority": {
                "n": st["maj_n"],
                "top1_acc": _pct(st["maj_top1_hits"], st["maj_n"]),
                "false_flip_rate": _pct(st["maj_false_flip"], st["maj_n"]),
            },
            "counterfactual": {
                "n": st["cf_n"],
                "top1_acc": _pct(st["cf_top1_hits"], st["cf_n"]),
                "detection_rate": _pct(st["cf_detected"], st["cf_n"]),
            },
        }
    summary["_meta"] = {"cases": total_cases, "elapsed_s": round(time.time() - t0, 1)}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    out = Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {}

    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        s = _run_dataset(name, DATASETS[name])
        existing[name] = s
        out.write_text(json.dumps(existing, indent=2))
        print(f"  cases={s['_meta']['cases']}  elapsed={s['_meta']['elapsed_s']}s  -> {out}",
              flush=True)
        for v, _, _ in VARIANTS:
            r = s[v]
            m = r["majority"]
            c = r["counterfactual"]
            print(f"  {v:<13s}  n={r['nodes']:>5d}  top1={r['top1_acc']:5.2f}%  "
                  f"τ={r['kendall_tau_mean']:+.3f}  "
                  f"|  maj n={m['n']:>4d} acc={m['top1_acc']:5.2f}% ff={m['false_flip_rate']:5.2f}%  "
                  f"|  cf n={c['n']:>4d} acc={c['top1_acc']:5.2f}% det={c['detection_rate']:5.2f}%",
                  flush=True)

    hdr = "{:<20s} | ".format("dataset") + " | ".join(
        f"{v:>15s}" for v, _, _ in VARIANTS)
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name in args.datasets:
        s = existing[name]
        cells = []
        for v, _, _ in VARIANTS:
            r = s[v]
            cells.append(f"{r['top1_acc']:5.1f}% τ={r['kendall_tau_mean']:+.2f}")
        print(f"{name:<20s} | " + " | ".join(f"{c:>15s}" for c in cells))


if __name__ == "__main__":
    main()
