"""Compare singleton-ranking diversity: regen vs original msmarco-pragma.

For each dataset, per-qid:
  n_distinct_singletons / 10
  plurality class share
  top-1 == baseline top-1 rate

Bottom-line numbers go to plots/18_lattices_cf_layer2to8/diversity_compare.txt
and a side-by-side bar plot diversity_compare.png.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.shared.plot_style import apply_plot_style
from scripts.pragma.msmarco.run_subsample import _load_lattices as _load_orig
from scripts.pragma.msmarco.run_subsample import _load_all_qids as _orig_qids
from scripts.pragma.msmarco_regen.run_subsample import _load_lattices as _load_regen
from scripts.pragma.msmarco_regen.run_subsample import _load_all_qids as _regen_qids
from search.lattice import combo_key

OUT_DIR = "plots/18_lattices_cf_layer2to8"


def measure(name: str, lats: dict[str, tuple[list[str], dict[str, list[str]]]]) -> dict:
    distinct = []
    plurality = []
    top1_agree = []
    n_answers = []
    for qid, (crit, rbc) in lats.items():
        baseline = rbc[combo_key(crit)]
        baseline_top1 = baseline[0]
        n_answers.append(len(baseline))
        singletons = [tuple(rbc[c]) for c in crit if c in rbc]
        cnt = Counter(singletons)
        distinct.append(len(cnt))
        plurality.append(cnt.most_common(1)[0][1] / len(singletons))
        top1_agree.append(sum(1 for s in singletons if s and s[0] == baseline_top1) / len(singletons))
    return {
        "name": name,
        "n_qids": len(lats),
        "answer_space_median": median(n_answers),
        "distinct_median": median(distinct),
        "distinct_over_10_median": median(distinct) / 10,
        "plurality_median": median(plurality),
        "plurality_mean": mean(plurality),
        "top1_agree_median": median(top1_agree),
        "top1_agree_mean": mean(top1_agree),
        "per_qid": list(zip(lats.keys(), distinct, plurality, top1_agree, n_answers)),
    }


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    orig = measure("original msmarco-pragma",  _load_orig(_orig_qids()))
    regen = measure("regen (18 lattices)",     _load_regen(_regen_qids()))

    lines: list[str] = []
    lines.append(f"\n  Singleton diversity: regen (18 lattices) vs original msmarco-pragma\n")
    lines.append(f"  {'metric':<35}  {'original':>15}  {'regen':>15}")
    lines.append("  " + "-" * 70)
    for label, key in [
        ("n qids",                          "n_qids"),
        ("answer-space size (median)",      "answer_space_median"),
        ("distinct singletons / 10 (median)", "distinct_over_10_median"),
        ("plurality share (median)",        "plurality_median"),
        ("plurality share (mean)",          "plurality_mean"),
        ("top-1 == baseline (median)",      "top1_agree_median"),
        ("top-1 == baseline (mean)",        "top1_agree_mean"),
    ]:
        ov = orig[key]
        rv = regen[key]
        lines.append(f"  {label:<35}  {ov:>15.3f}  {rv:>15.3f}")

    summary = "\n".join(lines)
    print(summary)
    with open(f"{OUT_DIR}/diversity_compare.txt", "w") as f:
        f.write(summary + "\n")

    # Bar plot: per-dataset distribution of plurality_share + distinct/10
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    o_pl = [p for _, _, p, _, _ in orig["per_qid"]]
    r_pl = [p for _, _, p, _, _ in regen["per_qid"]]
    o_d = [d / 10 for _, d, _, _, _ in orig["per_qid"]]
    r_d = [d / 10 for _, d, _, _, _ in regen["per_qid"]]

    bins = np.linspace(0, 1, 11)
    axes[0].hist(o_pl, bins=bins, alpha=0.6, color="#d62728",
                 label=f"original (n={len(o_pl)})", edgecolor="black", linewidth=0.4)
    axes[0].hist(r_pl, bins=bins, alpha=0.7, color="#1f77b4",
                 label=f"regen (n={len(r_pl)})", edgecolor="black", linewidth=0.4)
    axes[0].axvline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Plurality-class share (per qid)")
    axes[0].set_ylabel("# qids")
    axes[0].set_title("Plurality share distribution")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=9)

    axes[1].hist(o_d, bins=bins, alpha=0.6, color="#d62728",
                 label=f"original (n={len(o_d)})", edgecolor="black", linewidth=0.4)
    axes[1].hist(r_d, bins=bins, alpha=0.7, color="#1f77b4",
                 label=f"regen (n={len(r_d)})", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Distinct singletons / 10 (per qid)")
    axes[1].set_ylabel("# qids")
    axes[1].set_title("Singleton diversity distribution")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=9)

    fig.suptitle("Pragma voter diversity — regen vs original", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/diversity_compare.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUT_DIR}/diversity_compare.png")
    print(f"Summary → {OUT_DIR}/diversity_compare.txt")


if __name__ == "__main__":
    main()
