"""Re-aggregate `data/gemini_eval/gemini_eval_results.json` over a deduplicated
subset of qids, after discovering that gemini-created-dataset.json reuses the
same criteria/answers/sources across ids 5..60 in groups of 5-6 entries.

Deduplicated subset (13 lattices):
  - ids 2, 3, 4:    three truly-distinct lattices (qid=1 is halted)
  - ids 5..14:      one representative from each of the 10 reused-template groups
    (group N = ids {N, N+10, N+20, ...})

Writes:
  data/gemini_eval/dedup/gemini_eval_results_dedup.json — filtered aggregate
  plots/gemini_eval_dedup/{accuracy_and_cost, cost_per_query, outcomes}.png
"""
from __future__ import annotations

import json
import os
import shutil

# Lattices that are independent samples.
DEDUP_QIDS: list[str] = [str(i) for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

FULL_RESULTS_PATH = "data/gemini_eval/gemini_eval_results.json"
OUT_RESULTS_PATH = "data/gemini_eval/dedup/gemini_eval_results_dedup.json"


def filter_results() -> dict:
    with open(FULL_RESULTS_PATH) as f:
        full = json.load(f)
    keep = [r for r in full["results"] if r["query_id"] in DEDUP_QIDS]
    print(f"Loaded {len(full['results'])} results, kept {len(keep)} after dedup filter.")
    print(f"  Kept qids: {sorted([r['query_id'] for r in keep], key=int)}")
    return {**full, "n_cases": len(keep), "results": keep, "dedup_qids": DEDUP_QIDS}


def write_filtered(d: dict) -> None:
    os.makedirs(os.path.dirname(OUT_RESULTS_PATH), exist_ok=True)
    with open(OUT_RESULTS_PATH, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Wrote {OUT_RESULTS_PATH}")


def regenerate_plots() -> None:
    """Reuse plot_gemini_eval.py by temporarily swapping RESULTS_PATH."""
    # Easiest: monkey-patch the constants on the plotting module.
    import plot_gemini_eval as pge
    from plot_style import apply_plot_style
    apply_plot_style()
    out_dir = "plots/gemini_eval_dedup"
    os.makedirs(out_dir, exist_ok=True)
    # Point the plot module at the dedup JSON.
    pge.RESULTS_PATH = OUT_RESULTS_PATH
    results = pge._load_results()
    pge.plot_accuracy_and_cost(results, out_dir)
    pge.plot_cost_per_query(results, out_dir)
    pge.plot_outcomes(results, out_dir)


def print_summary(d: dict) -> None:
    results = d["results"]
    n = len(results)
    bfs_found = sum(1 for r in results if r["bfs"]["found"])
    print(f"\n{'=' * 100}")
    print(f"  GEMINI EVAL (DEDUPLICATED) — {n} queries (top-down, 10 criteria)")
    print(f"  BFS found CF in {bfs_found}/{n}")
    print(f"{'=' * 100}\n")

    def _stats(label, picker):
        ms = [picker(r) for r in results if picker(r) is not None]
        if not ms:
            return
        c = sum(1 for m in ms if m["outcome"] == "correct")
        cost = sum(m["calls"] + m.get("budget_used", 0) for m in ms) / len(ms)
        print(f"  {label:<22} | {c}/{len(ms)}  ({c/len(ms):>4.0%}) | avg cost {cost:>7.1f}")

    bfs_avg = sum(r["bfs"]["calls"] for r in results) / n
    print(f"  {'BFS (baseline)':<22} | {n}/{n}  (100%) | avg cost {bfs_avg:>7.1f}")
    print("  " + "-" * 60)
    _stats("SRA", lambda r: r.get("sra"))
    print("  " + "-" * 60)
    for b in [4, 8, 16, 32, 64, 128, 256]:
        lab = f"era_fixed_{b}"
        _stats(f"ERA (budget={b})", lambda r, l=lab: r.get("era", {}).get(l))
    print("  " + "-" * 60)
    for p in [10, 20, 30, 40, 50]:
        lab = f"era_pct_{p}"
        b = max(1, int(2**10 * p / 100))
        _stats(f"ERA ({p}% → b={b})", lambda r, l=lab: r.get("era", {}).get(l))


def main() -> None:
    d = filter_results()
    write_filtered(d)
    print_summary(d)
    regenerate_plots()


if __name__ == "__main__":
    main()
