"""Phase 2 diagnostic for a pragma case where ERA(b=16) prunes 0 nodes.

Picks sonnet_pragma qid=6 k=10 (case_id 154 in paper_results.json), reruns
ERA(b=16) with instrumentation, then for every Phase-2 layer plots:
    - bars of softmax-entropy per node (sorted desc), with max_jump_cutoff line
    - the score-margin (baseline − best other) of every "displacement=0" node
      (the only nodes ERA's rule even considers pruning)
    - a per-layer text annotation:
        "n_unknown = N | high_entropy = H | displacement>=1 = D | pruned = 0"

Why it matters: when pruned_nodes = 0, every layer's nodes either (a) show
displacement >= 1 (proxy already thinks they're CFs), or (b) sit above the
entropy cliff. Neither branch triggers the pruning rule.

Output: reports/era_pragma_pruning/figures/qid6_k10_phase2_layers.png
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.pragma.sonnet.run_subsample import _load_pragma_lattices
from scripts.shared.plot_style import apply_plot_style
from search.heuristics import max_jump_cutoff, softmax_entropy, score_margin
from search.lattice import combo_key, iter_search_layers
from search.rank import _aggregate_with_scores, _get_child_rankings, _era_phase1_down

OUT_DIR = "reports/era_pragma_pruning/figures"

QID    = "6"
BUDGET = 16


def reach_layers(criteria, rbc, start_node, budget) -> list[dict]:
    aggregator  = "combsum"
    entropy_fn  = softmax_entropy
    cutoff_fn   = max_jump_cutoff
    start_key   = combo_key(start_node)
    baseline    = rbc[start_key]
    baseline_answer = baseline[0]
    answers     = list(baseline)
    s = len(start_node)

    known: dict[str, list[str]] = {c: rbc[c] for c in criteria}
    estimated: dict[str, list[str]] = {}
    entropies: dict[str, float] = {}

    # Real Phase 1
    budget_rem, _ = _era_phase1_down(
        start_node, rbc, answers,
        known, estimated, entropies, budget,
        aggregator, entropy_fn, cutoff_fn,
    )

    # Phase 2 walk: collect per-layer state
    per_layer: list[dict] = []
    found = False
    for layer_size, subsets in iter_search_layers(criteria, start_node, "down"):
        if found:
            break
        unknown: list[tuple[str, int, float, dict]] = []
        for subset in subsets:
            key = combo_key(subset)
            if key in known:
                continue
            if key not in estimated:
                continue
            voters = _get_child_rankings(subset, known, estimated)
            ranking, scores = _aggregate_with_scores(voters, answers, aggregator)
            disp = ranking.index(baseline_answer) if baseline_answer in ranking else len(ranking)
            ent = entropies.get(key, 0.0)
            margin = score_margin(scores, baseline_answer)
            unknown.append((key, disp, ent, {"margin": margin, "scores": scores}))
        if not unknown:
            continue

        ents_only = [(k, e) for k, _, e, _ in unknown]
        high_ent_keys = set(cutoff_fn(ents_only, len(unknown)))
        kept = [(k, d, e, ex) for k, d, e, ex in unknown
                if not (d < 1 and k not in high_ent_keys)]
        pruned = [(k, d, e, ex) for k, d, e, ex in unknown
                  if  (d < 1 and k not in high_ent_keys)]

        per_layer.append({
            "layer": layer_size,
            "unknown": unknown,
            "high_ent_keys": high_ent_keys,
            "pruned": pruned,
            "kept": kept,
        })

        kept.sort(key=lambda x: (-x[1], x[2]))
        for key, _, _, _ in kept:
            gt = rbc.get(key)
            if gt:
                known[key] = gt
            if gt and gt[0] != baseline_answer:
                found = True
                break

    return per_layer


def plot_layers(per_layer: list[dict], qid: str, k: int, out: str) -> None:
    n = len(per_layer)
    n_cols = 2   # entropy panel + margin panel side by side
    fig, axes = plt.subplots(n, n_cols, figsize=(14, 2.4 * n),
                              squeeze=False)
    for i, info in enumerate(per_layer):
        ax_ent = axes[i, 0]
        ax_mar = axes[i, 1]
        layer = info["layer"]
        unknown = info["unknown"]
        high_ent_keys = info["high_ent_keys"]

        # sort by entropy desc
        sorted_nodes = sorted(unknown, key=lambda r: r[2], reverse=True)
        ents = [r[2] for r in sorted_nodes]
        disps = [r[1] for r in sorted_nodes]
        keys  = [r[0] for r in sorted_nodes]
        margins = [r[3]["margin"] for r in sorted_nodes]

        # cutoff index from max_jump_cutoff (the highest-ent prefix that "passes")
        if len(ents) > 1:
            gaps = [ents[j] - ents[j+1] for j in range(len(ents) - 1)]
            cutoff = max(range(len(gaps)), key=lambda j: (gaps[j], j)) + 1
        else:
            cutoff = len(ents)

        x = np.arange(len(ents))
        bar_colors = ["#2ca02c" if d >= 1 else "#9467bd" for d in disps]
        ax_ent.bar(x, ents, color=bar_colors, width=1.0, alpha=0.8)
        ax_ent.axvline(cutoff - 0.5, color="red", linestyle="--", linewidth=1.5,
                       label=f"max-jump @ {cutoff}")
        n_disp_ge_1   = sum(1 for d in disps if d >= 1)
        n_high_ent    = sum(1 for k_ in keys if k_ in high_ent_keys)
        n_pruned      = len(info["pruned"])
        ax_ent.set_title(
            f"L{layer} — n={len(unknown)}  "
            f"high-ent={n_high_ent}  disp≥1={n_disp_ge_1}  pruned={n_pruned}",
            fontsize=9, fontweight="bold",
        )
        ax_ent.set_xlabel("nodes sorted by entropy ↓")
        ax_ent.set_ylabel("softmax H")
        ax_ent.grid(True, alpha=0.3, axis="y")
        ax_ent.legend(fontsize=7, loc="upper right")

        # margin panel: ONLY plot displacement=0 nodes (the prune candidates)
        disp0 = [(k_, m_, e_) for k_, d_, e_, ex_, m_ in
                  [(k, d, e, ex, ex["margin"]) for k, d, e, ex in sorted_nodes]
                  if d_ == 0]
        if disp0:
            keys0 = [t[0] for t in disp0]
            ms    = [t[1] for t in disp0]
            es    = [t[2] for t in disp0]
            # order: same as entropy panel
            order = sorted(range(len(ms)), key=lambda j: -es[j])
            ms_sorted = [ms[j] for j in order]
            keys_sorted = [keys0[j] for j in order]
            in_high   = [k_ in high_ent_keys for k_ in keys_sorted]
            bar_c = ["#1f77b4" if h else "#c1c1c1" for h in in_high]
            ax_mar.bar(np.arange(len(ms_sorted)), ms_sorted, color=bar_c, width=1.0, alpha=0.85)
            ax_mar.axhline(0, color="black", linewidth=0.6)
            ax_mar.set_title(
                f"L{layer} — disp=0 nodes (margin = baseline − best other)  "
                f"n_disp0 = {len(ms_sorted)}",
                fontsize=9,
            )
            ax_mar.set_xlabel("disp=0 nodes (entropy order)")
            ax_mar.set_ylabel("score margin")
            ax_mar.grid(True, alpha=0.3, axis="y")
        else:
            ax_mar.text(0.5, 0.5, "no displacement=0 nodes\n(no prune candidates)",
                        ha="center", va="center", transform=ax_mar.transAxes,
                        fontsize=9, color="gray")
            ax_mar.set_xticks([]); ax_mar.set_yticks([])

    fig.suptitle(
        f"Sonnet pragma qid={qid}, k={k}, ERA(b={BUDGET}) — "
        f"Phase 2 by layer  (pruned_nodes = 0)",
        fontweight="bold", fontsize=12,
    )
    legend_text = (
        "purple bars = displacement=0 (proxy says baseline still wins)   "
        "green bars = displacement≥1 (proxy says answer flipped)   "
        "blue margin bars = node IS in high-entropy set"
    )
    fig.text(0.5, 0.005, legend_text, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    lats = _load_pragma_lattices([QID])
    criteria, rbc = lats[QID]
    start_node = list(criteria)

    per_layer = reach_layers(criteria, rbc, start_node, BUDGET)
    print(f"Captured {len(per_layer)} Phase 2 layers for qid={QID}, k={len(start_node)}")
    for info in per_layer:
        print(f"  L{info['layer']:>2}: n_unknown={len(info['unknown']):>3}  "
              f"high_ent={len(info['high_ent_keys']):>3}  "
              f"pruned={len(info['pruned']):>3}")

    out = f"{OUT_DIR}/qid6_k10_phase2_layers.png"
    plot_layers(per_layer, QID, len(start_node), out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
